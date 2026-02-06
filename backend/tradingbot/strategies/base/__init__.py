"""Base strategy classes and interfaces.

This module provides abstract base classes and interfaces that all
trading strategies must implement, ensuring consistency and
interoperability across the system.
"""

from .base_strategy import BaseStrategy, StrategyConfig, StrategyResult
from .production_strategy import ProductionStrategy, ProductionStrategyConfig
from .risk_managed_strategy import RiskManagedStrategy, RiskConfig
from .lifecycle_mixin import LifecycleMixin
from .framework_strategy import FrameworkStrategy

__all__ = [
    "BaseStrategy",
    "FrameworkStrategy",
    "LifecycleMixin",
    "ProductionStrategy",
    "ProductionStrategyConfig",
    "RiskConfig",
    "RiskManagedStrategy",
    "StrategyConfig",
    "StrategyResult",
]



