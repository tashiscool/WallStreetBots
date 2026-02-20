"""
Backtesting Module

Provides backtesting capabilities for trading strategies:
- Event-driven backtesting engine with realistic fill models
- Vectorized backtesting for ultra-fast parameter sweeps
- Modular loss functions for hyperparameter optimization
- Optuna-based optimization service
"""

from .backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResults,
    Trade,
    TradeDirection,
    TradeStatus,
    DailySnapshot,
    run_backtest,
)

from .fill_models import (
    IFillModel,
    FillResult,
    FillStatus,
    MarketData,
    ImmediateFillModel,
    SpreadFillModel,
    VolumeSlippageFillModel,
    EquityFillModel,
    OptionsFillModel,
    FillModelFactory,
)

from .progress_monitor import (
    BacktestStatus,
    ProgressEventType,
    ProgressEvent,
    BacktestMetrics,
    ProgressState,
    ETAEstimator,
    ProgressMonitor,
    BacktestProgressManager,
    progress_manager,
    get_progress_manager,
    create_progress_monitor,
)

from .vectorized_engine import (
    VectorizedResult,
    VectorizedBacktestEngine,
)

from .loss_functions import (
    BaseLossFunction,
    SharpeLoss,
    SortinoLoss,
    CalmarLoss,
    ProfitLoss,
    MaxDrawdownLoss,
    CustomWeightedLoss,
    get_loss_function,
)

from .optimization_service import (
    OptimizationObjective,
    SamplerType,
    PrunerType,
    ParameterRange,
    OptimizationConfig,
    TrialResult,
    OptimizationResult,
    OptimizationService,
    get_optimization_service,
)

__all__ = [
    # Backtest engine
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestProgressManager",
    "BacktestResults",
    # Progress Monitor
    "BacktestStatus",
    # Loss Functions
    "BaseLossFunction",
    "CalmarLoss",
    "CustomWeightedLoss",
    "DailySnapshot",
    "ETAEstimator",
    "EquityFillModel",
    "FillModelFactory",
    "FillResult",
    "FillStatus",
    # Fill models
    "IFillModel",
    "ImmediateFillModel",
    "MarketData",
    "MaxDrawdownLoss",
    "OptimizationConfig",
    # Optimization Service
    "OptimizationObjective",
    "OptimizationResult",
    "OptimizationService",
    "OptionsFillModel",
    "ParameterRange",
    "ProfitLoss",
    "ProgressEvent",
    "ProgressEventType",
    "ProgressMonitor",
    "ProgressState",
    "PrunerType",
    "SamplerType",
    "SharpeLoss",
    "SortinoLoss",
    "SpreadFillModel",
    "Trade",
    "TradeDirection",
    "TradeStatus",
    "TrialResult",
    "VectorizedBacktestEngine",
    # Vectorized Engine
    "VectorizedResult",
    "VolumeSlippageFillModel",
    "create_progress_monitor",
    "get_loss_function",
    "get_optimization_service",
    "get_progress_manager",
    "progress_manager",
    "run_backtest",
]
