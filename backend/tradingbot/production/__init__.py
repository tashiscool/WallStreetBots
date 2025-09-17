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
# Import strategies with fallbacks
try:
    from .strategies import (
        ProductionEarningsProtection,
        ProductionIndexBaseline,
        ProductionWSBDipBot,
    )
except ImportError:
    ProductionEarningsProtection = ProductionIndexBaseline = ProductionWSBDipBot = None

__all__ = [
    "ProductionEarningsProtection",
    "ProductionIndexBaseline",
    "ProductionIntegrationManager",
    "ProductionStrategyManager",
    "ProductionStrategyManagerConfig",
    "ProductionWSBDipBot",
]
