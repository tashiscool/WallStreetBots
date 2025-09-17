"""Data module for WallStreetBots."""

from . import providers
from .providers import MarketDataClient, BarSpec, CorporateActionsAdjuster
from .quality import DataQualityMonitor, OutlierDetector, QualityCheckResult

# Create module attributes for backward compatibility
client = providers
corporate_actions = providers

__all__ = [
    "BarSpec",
    "CorporateActionsAdjuster",
    "DataQualityMonitor",
    "MarketDataClient",
    "OutlierDetector",
    "QualityCheckResult",
]
