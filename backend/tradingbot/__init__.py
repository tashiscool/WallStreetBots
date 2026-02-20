"""Options Trading System - Implementing the Successful 240% Playbook.

This module implements a comprehensive options trading system based on the successful
240% GOOGL call trade, with proper risk management to prevent existential bets.

Key Components:
- Black - Scholes options pricing and Greeks calculation
- Market regime detection (bull pullback reversal signals)
- Sophisticated position sizing and risk management
- Systematic exit planning with profit targets and stop losses
- Real - time alerts and execution checklists
- Comprehensive scenario analysis

The system transforms the successful but risky trade into a repeatable,
risk - managed strategy suitable for consistent profitable trading.

Usage:
    from backend.tradingbot import IntegratedTradingSystem, TradingConfig

    config = TradingConfig(account_size=500000, max_position_risk_pct=0.10)
    system = IntegratedTradingSystem(config)

    # Calculate trade for a ticker
    trade = system.calculate_trade_for_ticker("GOOGL", 207.0, 0.28)

    # Get portfolio status
    status = system.get_portfolio_status()
"""


from .alert_system import (
    Alert,
    AlertPriority,
    AlertType,
    ChecklistItem,
    ExecutionChecklist,
    ExecutionChecklistManager,
    TradingAlertSystem,
)
from .exit_planning import (
    ExitLevel,
    ExitReason,
    ExitSignal,
    ExitStrategy,
    ScenarioAnalyzer,
    ScenarioResult,
)
from .market_regime import (
    MarketRegime,
    MarketRegimeFilter,
    MarketSignal,
    SignalGenerator,
    SignalType,
    TechnicalAnalysis,
    TechnicalIndicators,
)
from .options_calculator import (
    BlackScholesCalculator,
    OptionsSetup,
    OptionsTradeCalculator,
    TradeCalculation,
    validate_successful_trade,
)
from .risk_management import (
    KellyCalculator,
    PortfolioRisk,
    Position,
    PositionSizer,
    PositionStatus,
    RiskLevel,
    RiskManager,
    RiskParameters,
)
from .trading_system import IntegratedTradingSystem, SystemState, TradingConfig

__version__ = "1.0.0"
__author__ = "WallStreetBots Team"

__all__ = [
    "Alert",
    "AlertPriority",
    "AlertType",
    # Options pricing
    "BlackScholesCalculator",
    "ChecklistItem",
    "ExecutionChecklist",
    "ExecutionChecklistManager",
    "ExitLevel",
    "ExitReason",
    "ExitSignal",
    # Exit planning
    "ExitStrategy",
    # Main system
    "IntegratedTradingSystem",
    "KellyCalculator",
    "MarketRegime",
    # Market analysis
    "MarketRegimeFilter",
    "MarketSignal",
    "OptionsSetup",
    "OptionsTradeCalculator",
    "PortfolioRisk",
    "Position",
    "PositionSizer",
    "PositionStatus",
    "RiskLevel",
    # Risk management
    "RiskManager",
    "RiskParameters",
    "ScenarioAnalyzer",
    "ScenarioResult",
    "SignalGenerator",
    "SignalType",
    "SystemState",
    "TechnicalAnalysis",
    "TechnicalIndicators",
    "TradeCalculation",
    # Alerts and execution
    "TradingAlertSystem",
    "TradingConfig",
    "validate_successful_trade",
]
