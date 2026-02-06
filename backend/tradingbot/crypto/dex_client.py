"""
Uniswap V3 DEX Client.

Implements ExecutionClient interface for decentralized exchange trading.
Supports Ethereum, Polygon, and Arbitrum networks.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from ..execution.interfaces import ExecutionClient, OrderRequest, OrderAck, OrderFill

logger = logging.getLogger(__name__)

# Optional web3 dependency
try:
    from web3 import Web3
    from web3.middleware import ExtraDataToPOAMiddleware
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    logger.info("web3 not available. DEX features disabled. Install with: pip install web3")


class Chain(Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"


# Default RPC URLs (users should provide their own)
DEFAULT_RPC_URLS = {
    Chain.ETHEREUM: "https://eth.llamarpc.com",
    Chain.POLYGON: "https://polygon-rpc.com",
    Chain.ARBITRUM: "https://arb1.arbitrum.io/rpc",
}

# Uniswap V3 Router addresses
UNISWAP_V3_ROUTER = {
    Chain.ETHEREUM: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    Chain.POLYGON: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    Chain.ARBITRUM: "0xE592427A0AEce92De3Edee1F18E0157C05861564",
}

# Common token addresses (Ethereum mainnet)
COMMON_TOKENS = {
    Chain.ETHEREUM: {
        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
    },
    Chain.POLYGON: {
        "WMATIC": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
        "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
        "WETH": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",
        "DAI": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
    },
    Chain.ARBITRUM: {
        "WETH": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        "USDC": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        "USDT": "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
        "ARB": "0x912CE59144191C1204E64559FE8253a0e49E6548",
    },
}

# Minimal ERC20 ABI for approval and balance checking
ERC20_ABI = [
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": False, "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
]

# Minimal Uniswap V3 Router ABI
UNISWAP_V3_ROUTER_ABI = [
    {
        "inputs": [{
            "components": [
                {"name": "tokenIn", "type": "address"},
                {"name": "tokenOut", "type": "address"},
                {"name": "fee", "type": "uint24"},
                {"name": "recipient", "type": "address"},
                {"name": "deadline", "type": "uint256"},
                {"name": "amountIn", "type": "uint256"},
                {"name": "amountOutMinimum", "type": "uint256"},
                {"name": "sqrtPriceLimitX96", "type": "uint160"},
            ],
            "name": "params",
            "type": "tuple",
        }],
        "name": "exactInputSingle",
        "outputs": [{"name": "amountOut", "type": "uint256"}],
        "stateMutability": "payable",
        "type": "function",
    },
]


@dataclass
class SwapParams:
    """Parameters for a token swap."""
    token_in: str  # Token address or symbol
    token_out: str  # Token address or symbol
    amount_in: Decimal
    slippage_pct: float = 0.5  # Default 0.5%
    fee_tier: int = 3000  # 0.3% pool fee (3000 = 0.3%)
    deadline_minutes: int = 20


class UniswapV3Client(ExecutionClient):
    """
    Uniswap V3 DEX execution client.

    Implements ExecutionClient for decentralized exchange trading.
    Supports Ethereum, Polygon, and Arbitrum networks.
    """

    def __init__(
        self,
        chain: Chain = Chain.ETHEREUM,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        wallet_address: Optional[str] = None,
        default_slippage_pct: float = 0.5,
    ):
        super().__init__()
        self.chain = chain
        self.default_slippage_pct = default_slippage_pct
        self._private_key = private_key
        self._wallet_address = wallet_address
        self._web3: Optional[Any] = None
        self._router_contract: Optional[Any] = None

        if HAS_WEB3 and rpc_url:
            self._initialize_web3(rpc_url)
        elif HAS_WEB3:
            default_url = DEFAULT_RPC_URLS.get(chain)
            if default_url:
                self._initialize_web3(default_url)

    def _initialize_web3(self, rpc_url: str) -> None:
        """Initialize Web3 connection."""
        if not HAS_WEB3:
            return

        self._web3 = Web3(Web3.HTTPProvider(rpc_url))

        # Add POA middleware for Polygon/Arbitrum
        if self.chain in (Chain.POLYGON, Chain.ARBITRUM):
            self._web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        # Initialize router contract
        router_address = UNISWAP_V3_ROUTER.get(self.chain)
        if router_address:
            self._router_contract = self._web3.eth.contract(
                address=Web3.to_checksum_address(router_address),
                abi=UNISWAP_V3_ROUTER_ABI,
            )

        if self._private_key and not self._wallet_address:
            account = self._web3.eth.account.from_key(self._private_key)
            self._wallet_address = account.address

    def validate_connection(self) -> bool:
        """Validate blockchain connection."""
        if not HAS_WEB3 or not self._web3:
            return False
        try:
            return self._web3.is_connected()
        except Exception:
            return False

    def place_order(self, req: OrderRequest) -> OrderAck:
        """
        Place a swap order on Uniswap V3.

        Maps OrderRequest to a DEX swap:
        - symbol is interpreted as "TOKEN_IN/TOKEN_OUT" (e.g., "WETH/USDC")
        - side "buy" means buy token_out with token_in
        - side "sell" means sell token_in for token_out
        """
        if not HAS_WEB3:
            return OrderAck(
                client_order_id=req.client_order_id,
                broker_order_id=None,
                accepted=False,
                reason="web3 not installed",
            )

        if not self._web3 or not self._web3.is_connected():
            return OrderAck(
                client_order_id=req.client_order_id,
                broker_order_id=None,
                accepted=False,
                reason="Not connected to blockchain",
            )

        if not self._private_key:
            return OrderAck(
                client_order_id=req.client_order_id,
                broker_order_id=None,
                accepted=False,
                reason="No private key configured",
            )

        try:
            # Parse symbol (e.g., "WETH/USDC")
            parts = req.symbol.split('/')
            if len(parts) != 2:
                return OrderAck(
                    client_order_id=req.client_order_id,
                    broker_order_id=None,
                    accepted=False,
                    reason=f"Invalid symbol format: {req.symbol}. Use TOKEN_IN/TOKEN_OUT",
                )

            token_in_symbol, token_out_symbol = parts
            token_in = self._resolve_token_address(token_in_symbol)
            token_out = self._resolve_token_address(token_out_symbol)

            if not token_in or not token_out:
                return OrderAck(
                    client_order_id=req.client_order_id,
                    broker_order_id=None,
                    accepted=False,
                    reason=f"Unknown token: {token_in_symbol if not token_in else token_out_symbol}",
                )

            # Get token decimals
            token_in_contract = self._web3.eth.contract(
                address=Web3.to_checksum_address(token_in),
                abi=ERC20_ABI,
            )
            decimals = token_in_contract.functions.decimals().call()
            amount_in_wei = int(Decimal(str(req.qty)) * Decimal(10 ** decimals))

            # Check allowance and approve if needed
            self._ensure_approval(token_in, amount_in_wei)

            # Calculate minimum output (slippage protection)
            min_amount_out = 0  # In production, use quote to calculate
            if req.limit_price:
                token_out_contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(token_out),
                    abi=ERC20_ABI,
                )
                out_decimals = token_out_contract.functions.decimals().call()
                min_amount_out = int(
                    Decimal(str(req.qty)) * Decimal(str(req.limit_price))
                    * Decimal(10 ** out_decimals)
                    * Decimal(1 - self.default_slippage_pct / 100)
                )

            # Build swap transaction
            deadline = int(time.time()) + 1200  # 20 minutes
            swap_params = {
                'tokenIn': Web3.to_checksum_address(token_in),
                'tokenOut': Web3.to_checksum_address(token_out),
                'fee': 3000,  # 0.3% pool
                'recipient': Web3.to_checksum_address(self._wallet_address),
                'deadline': deadline,
                'amountIn': amount_in_wei,
                'amountOutMinimum': min_amount_out,
                'sqrtPriceLimitX96': 0,
            }

            # Build and sign transaction
            tx = self._router_contract.functions.exactInputSingle(swap_params).build_transaction({
                'from': self._wallet_address,
                'nonce': self._web3.eth.get_transaction_count(self._wallet_address),
                'gas': 300000,
                'gasPrice': self._web3.eth.gas_price,
            })

            signed_tx = self._web3.eth.account.sign_transaction(tx, self._private_key)
            tx_hash = self._web3.eth.send_raw_transaction(signed_tx.raw_transaction)

            broker_order_id = tx_hash.hex()

            # Store order data
            self._orders[broker_order_id] = {
                "client_order_id": req.client_order_id,
                "broker_order_id": broker_order_id,
                "symbol": req.symbol,
                "qty": req.qty,
                "side": req.side,
                "type": "market",
                "status": "pending",
                "tx_hash": broker_order_id,
            }
            self._client_to_broker[req.client_order_id] = broker_order_id

            logger.info(f"DEX swap submitted: {req.symbol} qty={req.qty} tx={broker_order_id}")

            return OrderAck(
                client_order_id=req.client_order_id,
                broker_order_id=broker_order_id,
                accepted=True,
            )
        except Exception as e:
            logger.error(f"DEX swap failed: {e}")
            return OrderAck(
                client_order_id=req.client_order_id,
                broker_order_id=None,
                accepted=False,
                reason=str(e),
            )

    def _resolve_token_address(self, symbol: str) -> Optional[str]:
        """Resolve token symbol to address."""
        tokens = COMMON_TOKENS.get(self.chain, {})
        # Check if it's already an address
        if symbol.startswith("0x") and len(symbol) == 42:
            return symbol
        return tokens.get(symbol.upper())

    def _ensure_approval(self, token_address: str, amount: int) -> None:
        """Ensure token approval for router."""
        if not self._web3 or not self._wallet_address:
            return

        token_contract = self._web3.eth.contract(
            address=Web3.to_checksum_address(token_address),
            abi=ERC20_ABI,
        )

        router_address = UNISWAP_V3_ROUTER.get(self.chain)
        if not router_address:
            return

        current_allowance = token_contract.functions.allowance(
            Web3.to_checksum_address(self._wallet_address),
            Web3.to_checksum_address(router_address),
        ).call()

        if current_allowance < amount:
            max_uint256 = 2**256 - 1
            approve_tx = token_contract.functions.approve(
                Web3.to_checksum_address(router_address),
                max_uint256,
            ).build_transaction({
                'from': self._wallet_address,
                'nonce': self._web3.eth.get_transaction_count(self._wallet_address),
                'gas': 100000,
                'gasPrice': self._web3.eth.gas_price,
            })

            signed = self._web3.eth.account.sign_transaction(approve_tx, self._private_key)
            tx_hash = self._web3.eth.send_raw_transaction(signed.raw_transaction)
            self._web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            logger.info(f"Token approved: {token_address}")

    def get_token_balance(self, token_symbol: str) -> Decimal:
        """Get token balance for wallet."""
        if not HAS_WEB3 or not self._web3 or not self._wallet_address:
            return Decimal("0")

        token_address = self._resolve_token_address(token_symbol)
        if not token_address:
            return Decimal("0")

        try:
            token_contract = self._web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=ERC20_ABI,
            )
            balance = token_contract.functions.balanceOf(
                Web3.to_checksum_address(self._wallet_address)
            ).call()
            decimals = token_contract.functions.decimals().call()
            return Decimal(balance) / Decimal(10 ** decimals)
        except Exception as e:
            logger.error(f"Error getting balance for {token_symbol}: {e}")
            return Decimal("0")

    def get_eth_balance(self) -> Decimal:
        """Get native ETH/MATIC balance."""
        if not HAS_WEB3 or not self._web3 or not self._wallet_address:
            return Decimal("0")

        try:
            balance_wei = self._web3.eth.get_balance(
                Web3.to_checksum_address(self._wallet_address)
            )
            return Decimal(balance_wei) / Decimal(10 ** 18)
        except Exception as e:
            logger.error(f"Error getting ETH balance: {e}")
            return Decimal("0")

    def get_supported_tokens(self) -> Dict[str, str]:
        """Get supported tokens for the current chain."""
        return COMMON_TOKENS.get(self.chain, {})

    def estimate_gas(self, token_in: str, token_out: str, amount: Decimal) -> Optional[int]:
        """Estimate gas for a swap."""
        if not HAS_WEB3 or not self._web3:
            return None
        try:
            return 300000  # Approximate gas for Uniswap V3 swap
        except Exception:
            return None
