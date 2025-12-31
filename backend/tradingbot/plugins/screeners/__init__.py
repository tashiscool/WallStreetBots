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
    # Base
    "IScreener",
    "CompositeScreener",
    "SortScreener",
    "SortOrder",
    "ScreenerResult",
    "StockData",
    # Volume screeners
    "VolumeScreener",
    "DollarVolumeScreener",
    "RelativeVolumeScreener",
    "VolumeSpike",
    # Momentum screeners
    "MomentumScreener",
    "GapScreener",
    "BreakoutScreener",
    "IntradayMomentumScreener",
    "ReversalScreener",
    # Technical screeners
    "TechnicalScreener",
    "RSIScreener",
    "MovingAverageScreener",
    "VolatilityScreener",
    "SupportResistanceScreener",
    "PriceRangeScreener",
    # Fundamental screeners
    "FundamentalScreener",
    "MarketCapScreener",
    "EarningsScreener",
    "DividendScreener",
    "SectorScreener",
    "OptionsScreener",
    # Manager
    "ScreenerManager",
    "ScreenerPipeline",
    "PresetScreenerManager",
]
