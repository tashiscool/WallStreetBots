"""
Stock Screener Plugins Module.

Ported from freqtrade's pairlist plugins, adapted for stock/options trading.

Provides dynamic stock screening based on:
- Volume and liquidity
- Price movement (gainers/losers)
- Technical indicators
- Fundamental metrics
- Custom filters

Usage:
    from backend.tradingbot.plugins.screeners import (
        ScreenerManager,
        VolumeScreener,
        MomentumScreener,
        TechnicalScreener,
    )

    manager = ScreenerManager(data_client)
    manager.add_screener(VolumeScreener(min_volume=1_000_000))
    manager.add_screener(MomentumScreener(min_change_pct=5.0))

    stocks = await manager.screen()
"""

from .base import (
    IScreener,
    CompositeScreener,
    SortScreener,
    SortOrder,
    ScreenerResult,
    StockData,
)

from .volume import (
    VolumeScreener,
    DollarVolumeScreener,
    RelativeVolumeScreener,
    VolumeSpike,
)

from .momentum import (
    MomentumScreener,
    GapScreener,
    BreakoutScreener,
    IntradayMomentumScreener,
    ReversalScreener,
)

from .technical import (
    TechnicalScreener,
    RSIScreener,
    MovingAverageScreener,
    VolatilityScreener,
    SupportResistanceScreener,
    PriceRangeScreener,
)

from .fundamental import (
    FundamentalScreener,
    MarketCapScreener,
    EarningsScreener,
    DividendScreener,
    SectorScreener,
    OptionsScreener,
)

from .manager import (
    ScreenerManager,
    ScreenerPipeline,
    PresetScreenerManager,
)

__all__ = [
    "BreakoutScreener",
    "CompositeScreener",
    "DividendScreener",
    "DollarVolumeScreener",
    "EarningsScreener",
    # Fundamental screeners
    "FundamentalScreener",
    "GapScreener",
    # Base
    "IScreener",
    "IntradayMomentumScreener",
    "MarketCapScreener",
    # Momentum screeners
    "MomentumScreener",
    "MovingAverageScreener",
    "OptionsScreener",
    "PresetScreenerManager",
    "PriceRangeScreener",
    "RSIScreener",
    "RelativeVolumeScreener",
    "ReversalScreener",
    # Manager
    "ScreenerManager",
    "ScreenerPipeline",
    "ScreenerResult",
    "SectorScreener",
    "SortOrder",
    "SortScreener",
    "StockData",
    "SupportResistanceScreener",
    # Technical screeners
    "TechnicalScreener",
    "VolatilityScreener",
    # Volume screeners
    "VolumeScreener",
    "VolumeSpike",
]
