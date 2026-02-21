"""
Data Sources Module.

Provides market data from multiple providers:
- Yahoo Finance (free)
- Polygon.io (paid)
- Alpaca Markets (free tier available)

Usage:
    from backend.tradingbot.data.sources import (
        YahooDataSource,
        PolygonDataSource,
        AlpacaDataSource,
        DataSourceManager,
    )

    # Create data source
    yahoo = YahooDataSource()
    await yahoo.connect()

    # Get historical bars
    bars = await yahoo.get_bars("AAPL", DataResolution.DAILY, start, end)

    # Get quote
    quote = await yahoo.get_quote("AAPL")

    # Use manager for multi-source access
    manager = DataSourceManager()
    manager.add_source(yahoo)
    manager.add_source(AlpacaDataSource(api_key, api_secret))

    bars = await manager.get_bars("AAPL", DataResolution.DAILY, start)
"""

from .base import (
    DataResolution,
    AssetType,
    OHLCV,
    Quote,
    Trade,
    OptionChain,
    FundamentalData,
    IDataSource,
    CachedDataSource,
)

from .yahoo import YahooDataSource
from .polygon import PolygonDataSource
from .alpaca import AlpacaDataSource
from .fred import FREDDataSource
from .dark_pool import DarkPoolDataSource

__all__ = [
    "OHLCV",
    "AlpacaDataSource",
    "AssetType",
    "CachedDataSource",
    "DarkPoolDataSource",
    # Types
    "DataResolution",
    # Alternative data
    "FREDDataSource",
    "FundamentalData",
    # Base
    "IDataSource",
    "OptionChain",
    "PolygonDataSource",
    "Quote",
    "Trade",
    # Providers
    "YahooDataSource",
]
