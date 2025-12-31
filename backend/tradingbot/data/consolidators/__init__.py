"""
Data Consolidators Module.

Ported from QuantConnect/LEAN's consolidator framework.
Consolidators aggregate tick/bar data into higher timeframe bars.

Usage:
    from backend.tradingbot.data.consolidators import (
        ConsolidatorManager,
        ConsolidatorFactory,
        MinuteConsolidator,
        DailyConsolidator,
        RenkoConsolidator,
    )

    # Create manager
    manager = ConsolidatorManager()

    # Add consolidators
    manager.add_consolidator("AAPL", MinuteConsolidator("AAPL", minutes=5))
    manager.add_consolidator("AAPL", DailyConsolidator("AAPL"))

    # Register indicator
    manager.register_indicator("AAPL", my_sma_indicator)

    # Process data
    manager.process_bar("AAPL", bar_data)
"""

from .base import (
    Bar,
    BarType,
    IDataConsolidator,
    Resolution,
    Tick,
    BarConsolidatorBase,
    TickConsolidatorBase,
    IdentityDataConsolidator,
    FilteredIdentityConsolidator,
)

from .period import (
    TimePeriodConsolidator,
    MinuteConsolidator,
    HourlyConsolidator,
    DailyConsolidator,
    WeeklyConsolidator,
    MonthlyConsolidator,
    TickTimePeriodConsolidator,
    MarketHoursConsolidator,
    SessionConsolidator,
)

from .count import (
    TickCountConsolidator,
    VolumeConsolidator,
    VolumeBarConsolidator,
    DollarVolumeConsolidator,
    TradeCountBarConsolidator,
)

from .renko import (
    RenkoBrick,
    RenkoType,
    RenkoConsolidator,
    ClassicRenkoConsolidator,
    RangeConsolidator,
    TickRangeConsolidator,
)

from .manager import (
    ConsolidatorManager,
    ConsolidatorFactory,
    ConsolidatorPresets,
)

__all__ = [
    # Base
    "Bar",
    "BarType",
    "IDataConsolidator",
    "Resolution",
    "Tick",
    "BarConsolidatorBase",
    "TickConsolidatorBase",
    "IdentityDataConsolidator",
    "FilteredIdentityConsolidator",
    # Period-based
    "TimePeriodConsolidator",
    "MinuteConsolidator",
    "HourlyConsolidator",
    "DailyConsolidator",
    "WeeklyConsolidator",
    "MonthlyConsolidator",
    "TickTimePeriodConsolidator",
    "MarketHoursConsolidator",
    "SessionConsolidator",
    # Count-based
    "TickCountConsolidator",
    "VolumeConsolidator",
    "VolumeBarConsolidator",
    "DollarVolumeConsolidator",
    "TradeCountBarConsolidator",
    # Renko/Range
    "RenkoBrick",
    "RenkoType",
    "RenkoConsolidator",
    "ClassicRenkoConsolidator",
    "RangeConsolidator",
    "TickRangeConsolidator",
    # Manager
    "ConsolidatorManager",
    "ConsolidatorFactory",
    "ConsolidatorPresets",
]
