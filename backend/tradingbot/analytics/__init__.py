"""Analytics Package-Advanced Performance Analytics and Market Regime Adaptation

This package provides comprehensive analytics capabilities including:
- Advanced performance metrics (Sharpe ratio, max drawdown, etc.)
- Market regime detection and strategy adaptation
- Integration with WallStreetBots trading strategies
"""

from .advanced_analytics import (
    AdvancedAnalytics,
    DrawdownPeriod,
    PerformanceMetrics,
    analyze_performance,
)
from .market_regime_adapter import (
    AdaptationLevel,
    MarketRegimeAdapter,
    RegimeAdaptationConfig,
    StrategyAdaptation,
    adapt_strategies_to_market,
)

__all__ = [
    "AdaptationLevel",
    "AdvancedAnalytics",
    "DrawdownPeriod",
    "MarketRegimeAdapter",
    "PerformanceMetrics",
    "RegimeAdaptationConfig",
    "StrategyAdaptation",
    "adapt_strategies_to_market",
    "analyze_performance",
]
