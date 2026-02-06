"""
Algorithm Framework - Ported from QuantConnect LEAN & Nautilus Trader.

This module provides a modular approach to algorithm construction with:
- Alpha Models: Signal generation
- Portfolio Construction: Position sizing
- Execution Models: Order execution
- Risk Management: Position limits and stops
- Universe Selection: Security filtering
- Enhanced Risk: Rate limiting, trading states (from Nautilus)
- Data Aggregation: Bar building patterns (from Nautilus)

LEAN components ported under Apache 2.0 License from:
https://github.com/QuantConnect/Lean/tree/master/Algorithm.Framework

Nautilus Trader concepts inspired the Enhanced Risk and Data Aggregation modules.
"""

from .alpha import (
    Insight,
    InsightDirection,
    InsightType,
    AlphaModel,
    EmaCrossAlphaModel,
    RsiAlphaModel,
    MacdAlphaModel,
    MomentumAlphaModel,
    ConstantAlphaModel,
)
from .portfolio import (
    PortfolioTarget,
    PortfolioConstructionModel,
    EqualWeightingPortfolioModel,
    InsightWeightingPortfolioModel,
    MaximumSharpeRatioPortfolioModel,
    MeanVariancePortfolioModel,
    RiskParityPortfolioModel,
    BlackLittermanPortfolioModel,
    NullPortfolioModel,
)
from .execution import (
    OrderType,
    OrderSide,
    TimeInForce,
    OrderTicket,
    ExecutionModel,
    ImmediateExecutionModel,
    VWAPExecutionModel,
    TWAPExecutionModel,
    LimitOrderExecutionModel,
    SpreadExecutionModel,
    StandardDeviationExecutionModel,
    NullExecutionModel,
)
from .risk import (
    RiskManagementResult,
    RiskManagementModel,
    MaximumDrawdownModel,
    MaximumDrawdownPerSecurityModel,
    TrailingStopModel,
    MaximumPositionSizeModel,
    MaximumUnrealizedProfitModel,
    SectorExposureModel,
    CompositeRiskModel,
    NullRiskModel,
)
from .universe import (
    UniverseChangeType,
    UniverseChange,
    SecurityData,
    UniverseSelectionModel,
    ManualUniverseSelectionModel,
    ScheduledUniverseSelectionModel,
    QC500UniverseModel,
    FundamentalUniverseModel,
    LiquidityUniverseModel,
    SectorUniverseModel,
    MomentumUniverseModel,
    VolatilityUniverseModel,
    ETFConstituentsUniverseModel,
    CompositeUniverseModel,
    NullUniverseModel,
    PipelineUniverseSelectionModel,
)
from .enhanced_risk import (
    TradingState,
    RiskEngineConfig,
    RiskCheckResult,
    RateLimiter,
    EnhancedRiskEngine,
)
from .data_aggregation import (
    BarAggregation,
    IntervalType,
    Quote,
    Tick,
    Bar,
    DataEngineConfig,
    BarBuilder,
    TimeBarAggregator,
    TickBarAggregator,
    VolumeBarAggregator,
    DollarBarAggregator,
    BarResampler,
    QuoteAggregator,
    create_time_aggregator,
    create_tick_aggregator,
    create_volume_aggregator,
    create_dollar_aggregator,
)
from .pipeline import (
    Factor,
    AverageDollarVolume,
    Returns,
    Volatility,
    MeanReversion,
    ScreenFilter,
    TopFilter,
    BottomFilter,
    PercentileFilter,
    Pipeline,
)

# Sub-packages with additional models
from . import alpha_models
from . import portfolio_models

__all__ = [
    # Alpha
    "Insight",
    "InsightDirection",
    "InsightType",
    "AlphaModel",
    "EmaCrossAlphaModel",
    "RsiAlphaModel",
    "MacdAlphaModel",
    "MomentumAlphaModel",
    "ConstantAlphaModel",
    # Portfolio
    "PortfolioTarget",
    "PortfolioConstructionModel",
    "EqualWeightingPortfolioModel",
    "InsightWeightingPortfolioModel",
    "MaximumSharpeRatioPortfolioModel",
    "MeanVariancePortfolioModel",
    "RiskParityPortfolioModel",
    "BlackLittermanPortfolioModel",
    "NullPortfolioModel",
    # Execution
    "OrderType",
    "OrderSide",
    "TimeInForce",
    "OrderTicket",
    "ExecutionModel",
    "ImmediateExecutionModel",
    "VWAPExecutionModel",
    "TWAPExecutionModel",
    "LimitOrderExecutionModel",
    "SpreadExecutionModel",
    "StandardDeviationExecutionModel",
    "NullExecutionModel",
    # Risk
    "RiskManagementResult",
    "RiskManagementModel",
    "MaximumDrawdownModel",
    "MaximumDrawdownPerSecurityModel",
    "TrailingStopModel",
    "MaximumPositionSizeModel",
    "MaximumUnrealizedProfitModel",
    "SectorExposureModel",
    "CompositeRiskModel",
    "NullRiskModel",
    # Universe Selection
    "UniverseChangeType",
    "UniverseChange",
    "SecurityData",
    "UniverseSelectionModel",
    "ManualUniverseSelectionModel",
    "ScheduledUniverseSelectionModel",
    "QC500UniverseModel",
    "FundamentalUniverseModel",
    "LiquidityUniverseModel",
    "SectorUniverseModel",
    "MomentumUniverseModel",
    "VolatilityUniverseModel",
    "ETFConstituentsUniverseModel",
    "CompositeUniverseModel",
    "NullUniverseModel",
    "PipelineUniverseSelectionModel",
    # Enhanced Risk (from Nautilus)
    "TradingState",
    "RiskEngineConfig",
    "RiskCheckResult",
    "RateLimiter",
    "EnhancedRiskEngine",
    # Data Aggregation (from Nautilus)
    "BarAggregation",
    "IntervalType",
    "Quote",
    "Tick",
    "Bar",
    "DataEngineConfig",
    "BarBuilder",
    "TimeBarAggregator",
    "TickBarAggregator",
    "VolumeBarAggregator",
    "DollarBarAggregator",
    "BarResampler",
    "QuoteAggregator",
    "create_time_aggregator",
    "create_tick_aggregator",
    "create_volume_aggregator",
    "create_dollar_aggregator",
    # Pipeline
    "Factor",
    "AverageDollarVolume",
    "Returns",
    "Volatility",
    "MeanReversion",
    "ScreenFilter",
    "TopFilter",
    "BottomFilter",
    "PercentileFilter",
    "Pipeline",
    # Sub-packages
    "alpha_models",
    "portfolio_models",
]
