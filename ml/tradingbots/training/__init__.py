"""
Training utilities with best practices for ML trading models.

Provides:
- Walk-forward validation for time series
- Progress tracking with rich console output
- Model checkpointing and early stopping
- RL training with observation normalization
- Comprehensive metrics evaluation
"""

from .training_utils import (
    TrainingConfig,
    TrainingMetrics,
    WalkForwardValidator,
    ProgressTracker,
    ModelCheckpointer,
    EarlyStoppingCallback,
    train_with_best_practices,
    evaluate_trading_model,
    display_training_summary,
    calculate_trading_metrics,
)

from .rl_training import (
    RLTrainingConfig,
    RLTrainingMetrics,
    RLProgressTracker,
    RunningMeanStd,
    train_rl_agent,
    evaluate_agent,
    display_rl_training_summary,
)

__all__ = [
    # General training
    "TrainingConfig",
    "TrainingMetrics",
    "WalkForwardValidator",
    "ProgressTracker",
    "ModelCheckpointer",
    "EarlyStoppingCallback",
    "train_with_best_practices",
    "evaluate_trading_model",
    "display_training_summary",
    "calculate_trading_metrics",
    # RL training
    "RLTrainingConfig",
    "RLTrainingMetrics",
    "RLProgressTracker",
    "RunningMeanStd",
    "train_rl_agent",
    "evaluate_agent",
    "display_rl_training_summary",
]
