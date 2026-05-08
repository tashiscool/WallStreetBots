"""Production Trading System.

This package contains production - ready trading strategies and infrastructure
for live trading with real broker integration, live data feeds, and comprehensive
risk management.

Components:
- core: Core production infrastructure (integration, manager, CLI)
- strategies: Production - ready trading strategies
- data: Live data integration and providers
- tests: Comprehensive test suite for production components
"""

from .core.production_integration import ProductionIntegrationManager
from .core.production_strategy_manager import (
    ProductionStrategyManager,
    ProductionStrategyManagerConfig,
)
__all__ = [
    "ProductionEarningsProtection",
    "ProductionIndexBaseline",
    "ProductionIntegrationManager",
    "ProductionStrategyManager",
    "ProductionStrategyManagerConfig",
    "ProductionWSBDipBot",
]


def __getattr__(name):
    if name in {
        "ProductionEarningsProtection",
        "ProductionIndexBaseline",
        "ProductionWSBDipBot",
    }:
        from . import strategies

        value = getattr(strategies, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
