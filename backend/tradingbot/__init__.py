"""
Options Trading System - Implementing the Successful 240% Playbook

This module implements a comprehensive options trading system based on the successful
240% GOOGL call trade, with proper risk management to prevent existential bets.

Key Components:
- Black-Scholes options pricing and Greeks calculation
- Market regime detection (bull pullback reversal signals)
- Sophisticated position sizing and risk management
- Systematic exit planning with profit targets and stop losses
- Real-time alerts and execution checklists
- Comprehensive scenario analysis

The system transforms the successful but risky trade into a repeatable,
risk-managed strategy suitable for consistent profitable trading.

Usage:
    from backend.tradingbot import IntegratedTradingSystem, TradingConfig

    config = TradingConfig(account_size=500000, max_position_risk_pct=0.10)
    system = IntegratedTradingSystem(config)

    # Calculate trade for a ticker
    trade = system.calculate_trade_for_ticker("GOOGL", 207.0, 0.28)

    # Get portfolio status
    status = system.get_portfolio_status()
"""

from .options_calculator import (
    BlackScholesCalculator,
    OptionsTradeCalculator,
    OptionsSetup,
    TradeCalculation,
    validate_successful_trade
)

from .market_regime import (
    MarketRegimeFilter,
    SignalGenerator,
    TechnicalIndicators,
    MarketSignal,
    MarketRegime,
    SignalType,
    TechnicalAnalysis
)

from .risk_management import (
    RiskManager,
    PositionSizer,
    KellyCalculator,
    Position,
    PositionStatus,
    RiskParameters,
    PortfolioRisk,
    RiskLevel
)

from .exit_planning import (
    ExitStrategy,
    ScenarioAnalyzer,
    ExitSignal,
    ExitReason,
    ExitLevel,
    ScenarioResult
)

from .alert_system import (
    TradingAlertSystem,
    ExecutionChecklistManager,
    Alert,
    AlertType,
    AlertPriority,
    ChecklistItem,
    ExecutionChecklist
)

from .trading_system import (
    IntegratedTradingSystem,
    TradingConfig,
    SystemState
)

__version__ = "1.0.0"
__author__ = "WallStreetBots Team"

__all__ = [
    # Main system
    "IntegratedTradingSystem",
    "TradingConfig",
    "SystemState",

    # Options pricing
    "BlackScholesCalculator",
    "OptionsTradeCalculator",
    "OptionsSetup",
    "TradeCalculation",
    "validate_successful_trade",

    # Market analysis
    "MarketRegimeFilter",
    "SignalGenerator",
    "TechnicalIndicators",
    "MarketSignal",
    "MarketRegime",
    "SignalType",
    "TechnicalAnalysis",

    # Risk management
    "RiskManager",
    "PositionSizer",
    "KellyCalculator",
    "Position",
    "PositionStatus",
    "RiskParameters",
    "PortfolioRisk",
    "RiskLevel",

    # Exit planning
    "ExitStrategy",
    "ScenarioAnalyzer",
    "ExitSignal",
    "ExitReason",
    "ExitLevel",
    "ScenarioResult",

    # Alerts and execution
    "TradingAlertSystem",
    "ExecutionChecklistManager",
    "Alert",
    "AlertType",
    "AlertPriority",
    "ChecklistItem",
    "ExecutionChecklist"
]
