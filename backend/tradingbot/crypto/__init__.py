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
]
