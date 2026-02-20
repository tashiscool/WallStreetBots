"""Analytics Package - Advanced Performance Analytics and Market Regime Adaptation.

This package provides comprehensive analytics capabilities including:
- Advanced performance metrics (Sharpe ratio, max drawdown, etc.)
- Market regime detection and strategy adaptation
- Portfolio statistics (20+ metrics inspired by Nautilus Trader)
- Trading filters (news events, time-of-day, volatility, liquidity)
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
from .statistics import (
    Trade,
    PortfolioStatistic,
    WinRate,
    ProfitFactor,
    Expectancy,
    SharpeRatio,
    SortinoRatio,
    CalmarRatio,
    MaxDrawdown,
    CAGR,
    ReturnsVolatility,
    ReturnsAverage,
    ReturnsAvgWin,
    ReturnsAvgLoss,
    RiskReturnRatio,
    WinnerAverage,
    WinnerMax,
    WinnerMin,
    LoserAverage,
    LoserMax,
    LoserMin,
    LongRatio,
    PayoffRatio,
    TradeCount,
    MaxConsecutiveWins,
    MaxConsecutiveLosses,
    DEFAULT_STATISTICS,
)
from .portfolio_analyzer import (
    PortfolioAnalyzer,
    AnalysisResult,
    create_analyzer_with_trades,
    create_analyzer_with_returns,
)
from .filters import (
    NewsImpact,
    NewsEvent,
    TradingFilter,
    EconomicNewsEventFilter,
    TimeOfDayFilter,
    VolatilityFilter,
    LiquidityFilter,
    CompositeFilter,
    RateLimitFilter,
)

__all__ = [
    "CAGR",
    "DEFAULT_STATISTICS",
    # Existing
    "AdaptationLevel",
    "AdvancedAnalytics",
    "AnalysisResult",
    "CalmarRatio",
    "CompositeFilter",
    "DrawdownPeriod",
    "EconomicNewsEventFilter",
    "Expectancy",
    "LiquidityFilter",
    "LongRatio",
    "LoserAverage",
    "LoserMax",
    "LoserMin",
    "MarketRegimeAdapter",
    "MaxConsecutiveLosses",
    "MaxConsecutiveWins",
    "MaxDrawdown",
    "NewsEvent",
    # Filters
    "NewsImpact",
    "PayoffRatio",
    "PerformanceMetrics",
    # Analyzer
    "PortfolioAnalyzer",
    "PortfolioStatistic",
    "ProfitFactor",
    "RateLimitFilter",
    "RegimeAdaptationConfig",
    "ReturnsAverage",
    "ReturnsAvgLoss",
    "ReturnsAvgWin",
    "ReturnsVolatility",
    "RiskReturnRatio",
    "SharpeRatio",
    "SortinoRatio",
    "StrategyAdaptation",
    "TimeOfDayFilter",
    # Statistics (from Nautilus concepts)
    "Trade",
    "TradeCount",
    "TradingFilter",
    "VolatilityFilter",
    "WinRate",
    "WinnerAverage",
    "WinnerMax",
    "WinnerMin",
    "adapt_strategies_to_market",
    "analyze_performance",
    "create_analyzer_with_returns",
    "create_analyzer_with_trades",
]
