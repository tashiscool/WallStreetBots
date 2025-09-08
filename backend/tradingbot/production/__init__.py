"""
Production Trading System

This package contains production-ready trading strategies and infrastructure
for live trading with real broker integration, live data feeds, and comprehensive
risk management.

Components:
- core: Core production infrastructure (integration, manager, CLI)
- strategies: Production-ready trading strategies
- data: Live data integration and providers
- tests: Comprehensive test suite for production components
"""

from .core.production_integration import ProductionIntegrationManager
from .core.production_strategy_manager import ProductionStrategyManager, ProductionStrategyManagerConfig
from .strategies.production_wsb_dip_bot import ProductionWSBDipBot
from .strategies.production_earnings_protection import ProductionEarningsProtection
from .strategies.production_index_baseline import ProductionIndexBaseline

__all__ = [
    'ProductionIntegrationManager',
    'ProductionStrategyManager',
    'ProductionStrategyManagerConfig',
    'ProductionWSBDipBot',
    'ProductionEarningsProtection',
    'ProductionIndexBaseline',
]


