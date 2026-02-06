"""
Cryptocurrency Trading Module

Provides crypto trading functionality via Alpaca:
- 24/7 crypto market access
- BTC, ETH, SOL, and other major cryptos
- Dip buying strategies
"""

from .alpaca_crypto_client import (
    AlpacaCryptoClient,
    CryptoAsset,
    CryptoQuote,
    CryptoTrade,
    CryptoBar,
    CryptoPosition,
    CryptoOrder,
    CryptoMarketHours,
    create_crypto_client,
)
from .crypto_dip_bot import (
    CryptoDipBot,
    CryptoDipBotConfig,
    DipSignal,
    DipSeverity,
    DipBotState,
    create_dip_bot,
)
from .dex_client import (
    UniswapV3Client,
    Chain,
    SwapParams,
    COMMON_TOKENS,
    UNISWAP_V3_ROUTER,
)
from .wallet_manager import (
    WalletManager,
    WalletInfo,
)

__all__ = [
    # Client
    "AlpacaCryptoClient",
    "CryptoAsset",
    "CryptoQuote",
    "CryptoTrade",
    "CryptoBar",
    "CryptoPosition",
    "CryptoOrder",
    "CryptoMarketHours",
    "create_crypto_client",
    # Dip Bot
    "CryptoDipBot",
    "CryptoDipBotConfig",
    "DipSignal",
    "DipSeverity",
    "DipBotState",
    "create_dip_bot",
    # DEX Client
    "UniswapV3Client",
    "Chain",
    "SwapParams",
    "COMMON_TOKENS",
    "UNISWAP_V3_ROUTER",
    # Wallet Manager
    "WalletManager",
    "WalletInfo",
]
