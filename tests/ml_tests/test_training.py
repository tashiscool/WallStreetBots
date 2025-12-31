"""
Tests for Training Utilities

Tests cover:
- Walk-forward validation
- Progress tracking
- Early stopping
- Model checkpointing
- Trading metrics calculation
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from ml.tradingbots.training import (
    TrainingConfig,
    TrainingMetrics,
    WalkForwardValidator,
    ProgressTracker,
    EarlyStoppingCallback,
    ModelCheckpointer,
    calculate_trading_metrics,
    RLTrainingConfig,
    RunningMeanStd,
)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_config_defaults(self):
        """Test training config defaults."""
        config = TrainingConfig()
        assert config.validation_strategy == "walk_forward"
        assert config.train_ratio == 0.8
        assert config.epochs == 100
        assert config.early_stopping
        assert config.early_stopping_patience == 10


class TestTrainingMetrics:
    """Tests for TrainingMetrics."""

    def test_metrics_creation(self):
        """Test metrics object creation."""
        metrics = TrainingMetrics()
        assert len(metrics.train_losses) == 0
        assert len(metrics.val_losses) == 0

    def test_metrics_add_epoch(self):
        """Test adding epoch metrics."""
        metrics = TrainingMetrics()
        metrics.add_epoch(
            train_loss=0.5,
            val_loss=0.6,
            train_acc=0.8,
            val_acc=0.75,
            epoch_time=1.5,
        )

        assert len(metrics.train_losses) == 1
        assert metrics.train_losses[0] == 0.5
        assert metrics.val_losses[0] == 0.6
        assert metrics.epoch_times[0] == 1.5

    def test_metrics_best_val_loss(self):
        """Test best validation loss property."""
        metrics = TrainingMetrics()
        metrics.add_epoch(train_loss=0.5, val_loss=0.6)
        metrics.add_epoch(train_loss=0.4, val_loss=0.5)
        metrics.add_epoch(train_loss=0.3, val_loss=0.55)

        assert metrics.best_val_loss == 0.5
        assert metrics.best_epoch == 1

    def test_metrics_summary(self):
        """Test metrics summary."""
        metrics = TrainingMetrics()
        metrics.add_epoch(train_loss=0.5, val_loss=0.6, epoch_time=1.0)
        metrics.add_epoch(train_loss=0.4, val_loss=0.5, epoch_time=1.0)

        summary = metrics.get_summary()

        assert summary['epochs_trained'] == 2
        assert summary['best_epoch'] == 1
        assert summary['best_val_loss'] == 0.5


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator."""

    def test_validator_creation(self):
        """Test validator creation."""
        validator = WalkForwardValidator(n_splits=5, train_ratio=0.8)
        assert validator.n_splits == 5
        assert validator.train_ratio == 0.8

    def test_validator_split(self):
        """Test walk-forward splits."""
        validator = WalkForwardValidator(n_splits=3)
        splits = validator.split(data_length=300)

        assert len(splits) > 0
        for train_idx, val_idx in splits:
            # Train indices should come before validation
            assert train_idx.max() < val_idx.min()
            # No overlap
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_validator_expanding_splits(self):
        """Test expanding window splits."""
        validator = WalkForwardValidator(n_splits=3)
        splits = validator.get_expanding_splits(data_length=300)

        assert len(splits) > 0
        # Each subsequent split should have more training data
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1]


class TestEarlyStopping:
    """Tests for EarlyStoppingCallback."""

    def test_early_stopping_creation(self):
        """Test early stopping creation."""
        callback = EarlyStoppingCallback(patience=5, min_delta=1e-4)
        assert callback.patience == 5
        assert callback.min_delta == 1e-4
        assert callback.best_loss == float('inf')

    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        callback = EarlyStoppingCallback(patience=3)

        # First call - should improve
        should_stop = callback(epoch=0, val_loss=0.5)
        assert not should_stop
        assert callback.best_loss == 0.5

        # Second call - improvement
        should_stop = callback(epoch=1, val_loss=0.4)
        assert not should_stop
        assert callback.best_loss == 0.4

    def test_early_stopping_no_improvement(self):
        """Test early stopping triggers after no improvement."""
        callback = EarlyStoppingCallback(patience=2)

        callback(epoch=0, val_loss=0.5)
        callback(epoch=1, val_loss=0.6)  # No improvement
        should_stop = callback(epoch=2, val_loss=0.6)  # No improvement

        assert should_stop
        assert callback.should_stop


class TestModelCheckpointer:
    """Tests for ModelCheckpointer."""

    def test_checkpointer_creation(self):
        """Test checkpointer creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = ModelCheckpointer(
                checkpoint_dir=tmpdir,
                save_best_only=True,
            )

            assert checkpointer.checkpoint_dir == tmpdir
            assert checkpointer.save_best_only

    def test_checkpointer_saves_best(self):
        """Test checkpointer saves best model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = ModelCheckpointer(
                checkpoint_dir=tmpdir,
                save_best_only=True,
            )

            # First save - best
            model_state = {"weights": torch.randn(10)}
            path = checkpointer.save(model_state, epoch=0, val_loss=0.5)
            assert path is not None
            assert os.path.exists(path)

            # Second save - not best
            path = checkpointer.save(model_state, epoch=1, val_loss=0.6)
            assert path is None  # Shouldn't save

            # Third save - new best
            path = checkpointer.save(model_state, epoch=2, val_loss=0.4)
            assert path is not None


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_progress_tracker_creation(self):
        """Test progress tracker creation."""
        tracker = ProgressTracker(total_epochs=10, use_tqdm=False)
        assert tracker.total_epochs == 10

    def test_progress_tracker_context_manager(self):
        """Test progress tracker as context manager."""
        with ProgressTracker(total_epochs=5, use_tqdm=False) as tracker:
            for epoch in range(5):
                tracker.update(
                    epoch=epoch + 1,
                    train_loss=0.5 - epoch * 0.1,
                    val_loss=0.6 - epoch * 0.1,
                    epoch_time=1.0,
                )

        # Should complete without error


class TestTradingMetricsCalculation:
    """Tests for calculate_trading_metrics."""

    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        # Simulate daily returns
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0003  # Slight positive bias

        metrics = calculate_trading_metrics(returns)

        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics

    def test_calculate_metrics_sharpe(self):
        """Test Sharpe ratio calculation."""
        # Consistent positive returns
        returns = np.ones(252) * 0.001  # 0.1% daily

        metrics = calculate_trading_metrics(returns)

        # With no volatility, Sharpe should be very high
        assert metrics['sharpe_ratio'] > 5

    def test_calculate_metrics_drawdown(self):
        """Test max drawdown calculation."""
        # Returns that create a drawdown
        returns = np.array([0.1, 0.1, -0.3, -0.2, 0.1])

        metrics = calculate_trading_metrics(returns)

        assert metrics['max_drawdown'] < 0  # Drawdown is negative

    def test_calculate_metrics_win_rate(self):
        """Test win rate calculation."""
        returns = np.array([0.01, -0.01, 0.02, 0.01, -0.02])

        metrics = calculate_trading_metrics(returns)

        assert metrics['win_rate'] == 0.6  # 3 out of 5 positive

    def test_calculate_metrics_empty(self):
        """Test metrics with empty returns."""
        returns = np.array([])

        metrics = calculate_trading_metrics(returns)

        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['win_rate'] == 0.0


class TestRLTrainingConfig:
    """Tests for RLTrainingConfig."""

    def test_rl_config_defaults(self):
        """Test RL training config defaults."""
        config = RLTrainingConfig()
        assert config.total_timesteps == 100000
        assert config.normalize_observations  # Should be True!
        assert config.normalize_rewards


class TestRunningMeanStd:
    """Tests for RunningMeanStd (observation normalization)."""

    def test_running_mean_std_creation(self):
        """Test RunningMeanStd creation."""
        rms = RunningMeanStd(shape=(10,))
        assert rms.mean.shape == (10,)
        assert rms.var.shape == (10,)

    def test_running_mean_std_update(self):
        """Test running statistics update."""
        rms = RunningMeanStd(shape=(3,))

        # Update with batch of data
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        rms.update(data)

        assert rms.mean[0] > 0
        assert rms.count > 0

    def test_running_mean_std_normalize(self):
        """Test normalization."""
        rms = RunningMeanStd(shape=(3,))

        # Update with data
        data = np.random.randn(100, 3) * 10 + 50
        rms.update(data)

        # Normalize new sample
        sample = np.array([50, 50, 50])
        normalized = rms.normalize(sample)

        # Normalized should be close to 0 for mean values
        assert np.abs(normalized).mean() < 1.0
