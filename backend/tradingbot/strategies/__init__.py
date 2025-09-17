"""Trading strategies package.

This package contains all trading strategies organized into:
- base: Abstract base classes and interfaces
- implementations: Concrete strategy implementations  
- production: Production-ready strategy wrappers
"""

# Import base classes
from .base import (
    BaseStrategy,
    StrategyConfig,
    StrategyResult,
    ProductionStrategy,
    ProductionStrategyConfig,
    RiskManagedStrategy,
    RiskConfig,
)

# Import implementations for backward compatibility
from .implementations import (
    wsb_dip_bot,
    earnings_protection,
    wheel_strategy,
    momentum_weeklies,
    debit_spreads,
    leaps_tracker,
    swing_trading,
    spx_credit_spreads,
    lotto_scanner,
)

# Create module attributes for backward compatibility
earnings_protection = earnings_protection
leaps_tracker = leaps_tracker
spx_credit_spreads = spx_credit_spreads
swing_trading = swing_trading
lotto_scanner = lotto_scanner
wheel_strategy = wheel_strategy
wsb_dip_bot = wsb_dip_bot

__all__ = [
    # Base classes
    "BaseStrategy",
    "StrategyConfig",
    "StrategyResult", 
    "ProductionStrategy",
    "ProductionStrategyConfig",
    "RiskManagedStrategy",
    "RiskConfig",
    
    # Implementations (for backward compatibility)
    "wsb_dip_bot",
    "earnings_protection",
    "wheel_strategy",
    "momentum_weeklies",
    "debit_spreads",
    "leaps_tracker",
    "swing_trading",
    "spx_credit_spreads",
    "lotto_scanner",
]