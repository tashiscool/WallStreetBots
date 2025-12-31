"""
Backtesting Module

Provides backtesting capabilities for trading strategies.
Includes realistic fill models for accurate simulation.
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

__all__ = [
    # Backtest engine
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResults",
    "Trade",
    "TradeDirection",
    "TradeStatus",
    "DailySnapshot",
    "run_backtest",
    # Fill models
    "IFillModel",
    "FillResult",
    "FillStatus",
    "MarketData",
    "ImmediateFillModel",
    "SpreadFillModel",
    "VolumeSlippageFillModel",
    "EquityFillModel",
    "OptionsFillModel",
    "FillModelFactory",
    # Progress Monitor
    "BacktestStatus",
    "ProgressEventType",
    "ProgressEvent",
    "BacktestMetrics",
    "ProgressState",
    "ETAEstimator",
    "ProgressMonitor",
    "BacktestProgressManager",
    "progress_manager",
    "get_progress_manager",
    "create_progress_monitor",
]
