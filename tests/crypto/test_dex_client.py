"""Tests for DEX client."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from decimal import Decimal

from backend.tradingbot.execution.interfaces import OrderRequest, OrderAck
from backend.tradingbot.crypto.dex_client import (
    UniswapV3Client,
    Chain,
    COMMON_TOKENS,
    SwapParams,
    HAS_WEB3,
)


class TestUniswapV3Client:
    """Test UniswapV3Client."""

    def test_init_without_web3(self):
        """Client should initialize even without web3."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        assert client.chain == Chain.ETHEREUM

    def test_validate_connection_no_web3(self):
        """Should return False when web3 not connected."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        if not HAS_WEB3:
            assert client.validate_connection() is False

    def test_place_order_no_web3(self):
        """Should reject order when web3 not available."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        client._web3 = None
        client._private_key = None

        req = OrderRequest(
            client_order_id="test-1",
            symbol="WETH/USDC",
            qty=1.0,
            side="buy",
            type="market",
        )
        ack = client.place_order(req)
        assert ack.accepted is False

    def test_resolve_token_address(self):
        """Should resolve known token symbols."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        addr = client._resolve_token_address("WETH")
        assert addr == COMMON_TOKENS[Chain.ETHEREUM]["WETH"]

    def test_resolve_unknown_token(self):
        """Should return None for unknown tokens."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        addr = client._resolve_token_address("UNKNOWN_TOKEN_XYZ")
        assert addr is None

    def test_resolve_raw_address(self):
        """Should pass through raw addresses."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        addr = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        result = client._resolve_token_address(addr)
        assert result == addr

    def test_get_supported_tokens(self):
        """Should return tokens for chain."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        tokens = client.get_supported_tokens()
        assert "WETH" in tokens
        assert "USDC" in tokens

    def test_polygon_chain(self):
        """Should support Polygon chain."""
        client = UniswapV3Client(chain=Chain.POLYGON)
        tokens = client.get_supported_tokens()
        assert "WMATIC" in tokens

    def test_invalid_symbol_format(self):
        """Should reject invalid symbol format."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        client._web3 = MagicMock()
        client._web3.is_connected.return_value = True
        client._private_key = "0x" + "a" * 64

        req = OrderRequest(
            client_order_id="test-2",
            symbol="INVALID",
            qty=1.0,
            side="buy",
            type="market",
        )
        with patch('backend.tradingbot.crypto.dex_client.HAS_WEB3', True):
            ack = client.place_order(req)
        assert ack.accepted is False
        assert "Invalid symbol format" in ack.reason

    @pytest.mark.skipif(not HAS_WEB3, reason="web3 not installed")
    def test_get_token_balance_no_connection(self):
        """Should return 0 when not connected."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        client._web3 = None
        balance = client.get_token_balance("WETH")
        assert balance == Decimal("0")

    def test_estimate_gas(self):
        """Should return gas estimate."""
        client = UniswapV3Client(chain=Chain.ETHEREUM)
        gas = client.estimate_gas("WETH", "USDC", Decimal("1.0"))
        if HAS_WEB3:
            assert gas is not None
            assert gas > 0


class TestChain:
    """Test Chain enum."""

    def test_chains(self):
        assert Chain.ETHEREUM.value == "ethereum"
        assert Chain.POLYGON.value == "polygon"
        assert Chain.ARBITRUM.value == "arbitrum"


class TestSwapParams:
    """Test SwapParams dataclass."""

    def test_defaults(self):
        params = SwapParams(
            token_in="WETH",
            token_out="USDC",
            amount_in=Decimal("1.0"),
        )
        assert params.slippage_pct == 0.5
        assert params.fee_tier == 3000
        assert params.deadline_minutes == 20
