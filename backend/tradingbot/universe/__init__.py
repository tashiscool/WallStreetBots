"""
Universe Selection Module for WallStreetBots.

Provides dynamic asset filtering using a two-stage coarse/fine selection process:
1. Coarse Selection: Fast filter on price, volume, market cap
2. Fine Selection: Detailed filter on fundamentals, technicals, custom criteria

Also includes point-in-time membership tracking.
"""

from .point_in_time import (
    UniverseProvider,
    Membership,
)

from .base import (
    IUniverseSelectionModel,
    UniverseSelectionResult,
    SecurityData,
    SecurityType,
    CompositeUniverseSelection,
    UnionUniverseSelection,
    ScheduledUniverseSelection,
)

from .coarse import (
    CoarseSelectionFilter,
    VolumeUniverseSelection,
    PriceUniverseSelection,
    DollarVolumeUniverseSelection,
    SpreadUniverseSelection,
    OptionsableUniverseSelection,
    ETFUniverseSelection,
)

from .fine import (
    FineSelectionFilter,
    MarketCapUniverseSelection,
    SectorUniverseSelection,
    FundamentalUniverseSelection,
    DividendAristocratsSelection,
    SP500UniverseSelection,
)

from .technical import (
    TechnicalUniverseSelection,
    MomentumUniverseSelection,
    VolatilityUniverseSelection,
    TrendUniverseSelection,
    OversoldUniverseSelection,
    OverboughtUniverseSelection,
)

from .manager import (
    UniverseManager,
    PresetUniverseManager,
)

__all__ = [
    # Coarse filters
    "CoarseSelectionFilter",
    "CompositeUniverseSelection",
    "DividendAristocratsSelection",
    "DollarVolumeUniverseSelection",
    "ETFUniverseSelection",
    # Fine filters
    "FineSelectionFilter",
    "FundamentalUniverseSelection",
    # Base
    "IUniverseSelectionModel",
    "MarketCapUniverseSelection",
    # Point-in-time
    "Membership",
    "MomentumUniverseSelection",
    "OptionsableUniverseSelection",
    "OverboughtUniverseSelection",
    "OversoldUniverseSelection",
    "PresetUniverseManager",
    "PriceUniverseSelection",
    "SP500UniverseSelection",
    "ScheduledUniverseSelection",
    "SectorUniverseSelection",
    "SecurityData",
    "SecurityType",
    "SpreadUniverseSelection",
    # Technical filters
    "TechnicalUniverseSelection",
    "TrendUniverseSelection",
    "UnionUniverseSelection",
    # Manager
    "UniverseManager",
    "UniverseProvider",
    "UniverseSelectionResult",
    "VolatilityUniverseSelection",
    "VolumeUniverseSelection",
]
