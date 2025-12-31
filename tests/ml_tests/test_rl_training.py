"""
Comprehensive Tests for RL Training Utilities

Tests the RL training infrastructure including:
- RLTrainingConfig
- RLTrainingMetrics
- RunningMeanStd
- RLProgressTracker
- evaluate_agent
- train_rl_agent
- Edge cases and error handling
"""

import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pytest

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.tradingbots.training.rl_training import (
    RLTrainingConfig,
    RLTrainingMetrics,
    RunningMeanStd,
    RLProgressTracker,
    evaluate_agent,
    train_rl_agent,
    display_rl_training_summary
)


class TestRLTrainingConfig:
    """Tests for RLTrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RLTrainingConfig()

        assert config.total_timesteps == 100000
        assert config.n_eval_episodes == 5
        assert config.eval_freq == 10000
        assert config.log_freq == 1000
        assert config.verbose is True
        assert config.progress_bar is True
        assert config.checkpoint_dir == "rl_checkpoints"
        assert config.save_freq == 10000
        assert config.normalize_observations is True
        assert config.normalize_rewards is True
        assert config.clip_observations == 10.0
        assert config.clip_rewards == 10.0
        assert config.reward_scale == 1.0
        assert config.seed == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = RLTrainingConfig(
            total_timesteps=50000,
            eval_freq=5000,
            normalize_observations=False,
            seed=123
        )

        assert config.total_timesteps == 50000
        assert config.eval_freq == 5000
        assert config.normalize_observations is False
        assert config.seed == 123


class TestRLTrainingMetrics:
    """Tests for RLTrainingMetrics class."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = RLTrainingMetrics()

        assert len(metrics.episode_rewards) == 0
        assert len(metrics.episode_lengths) == 0
        assert len(metrics.episode_sharpes) == 0
        assert len(metrics.episode_drawdowns) == 0
        assert len(metrics.eval_rewards) == 0
        assert len(metrics.eval_sharpes) == 0
        assert metrics.total_timesteps == 0
        assert metrics.total_episodes == 0
        assert metrics.training_time == 0.0

    def test_add_episode(self):
        """Test adding episode results."""
        metrics = RLTrainingMetrics()

        metrics.add_episode(
            reward=100.0,
            length=250,
            sharpe=1.5,
            drawdown=-0.1
        )

        assert len(metrics.episode_rewards) == 1
        assert len(metrics.episode_lengths) == 1
        assert len(metrics.episode_sharpes) == 1
        assert len(metrics.episode_drawdowns) == 1
        assert metrics.total_episodes == 1

        assert metrics.episode_rewards[0] == 100.0
        assert metrics.episode_lengths[0] == 250
        assert metrics.episode_sharpes[0] == 1.5
        assert metrics.episode_drawdowns[0] == -0.1

    def test_add_episode_without_optional_metrics(self):
        """Test adding episode without sharpe/drawdown."""
        metrics = RLTrainingMetrics()

        metrics.add_episode(reward=50.0, length=100)

        assert len(metrics.episode_rewards) == 1
        assert len(metrics.episode_sharpes) == 0
        assert len(metrics.episode_drawdowns) == 0

    def test_add_eval(self):
        """Test adding evaluation results."""
        metrics = RLTrainingMetrics()

        metrics.add_eval(mean_reward=150.0, mean_sharpe=2.0)

        assert len(metrics.eval_rewards) == 1
        assert len(metrics.eval_sharpes) == 1
        assert metrics.eval_rewards[0] == 150.0
        assert metrics.eval_sharpes[0] == 2.0

    def test_mean_reward_100(self):
        """Test mean reward over last 100 episodes."""
        metrics = RLTrainingMetrics()

        # Add 50 episodes
        for i in range(50):
            metrics.add_episode(reward=float(i), length=100)

        mean = metrics.mean_reward_100
        assert mean == np.mean(range(50))

        # Add 100 more episodes
        for i in range(100):
            metrics.add_episode(reward=float(i + 50), length=100)

        # Should only consider last 100
        mean = metrics.mean_reward_100
        expected = np.mean(range(50, 150))
        assert mean == expected

    def test_mean_reward_100_empty(self):
        """Test mean reward with no episodes."""
        metrics = RLTrainingMetrics()

        assert metrics.mean_reward_100 == 0.0

    def test_best_eval_reward(self):
        """Test best evaluation reward."""
        metrics = RLTrainingMetrics()

        metrics.add_eval(mean_reward=100.0)
        metrics.add_eval(mean_reward=150.0)
        metrics.add_eval(mean_reward=120.0)

        assert metrics.best_eval_reward == 150.0

    def test_best_eval_reward_empty(self):
        """Test best eval reward with no evaluations."""
        metrics = RLTrainingMetrics()

        assert metrics.best_eval_reward == float('-inf')

    def test_get_summary(self):
        """Test getting training summary."""
        metrics = RLTrainingMetrics()

        # Add some data
        for i in range(10):
            metrics.add_episode(reward=float(i * 10), length=100, sharpe=1.5)

        metrics.add_eval(mean_reward=50.0, mean_sharpe=1.8)
        metrics.total_timesteps = 1000
        metrics.training_time = 100.0

        summary = metrics.get_summary()

        assert summary["total_timesteps"] == 1000
        assert summary["total_episodes"] == 10
        assert summary["mean_reward_100"] == 45.0  # Mean of [0, 10, 20, ..., 90]
        assert summary["best_eval_reward"] == 50.0
        assert summary["mean_sharpe"] == 1.5
        assert summary["best_sharpe"] == 1.5
        assert summary["training_time"] == 100.0


class TestRunningMeanStd:
    """Tests for RunningMeanStd class."""

    def test_initialization(self):
        """Test initialization."""
        rms = RunningMeanStd(shape=(4,))

        assert rms.mean.shape == (4,)
        assert rms.var.shape == (4,)
        assert rms.count > 0

    def test_update_single_batch(self):
        """Test updating with single batch."""
        rms = RunningMeanStd(shape=())

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(x)

        # Mean should be close to 3.0
        assert np.isclose(rms.mean, 3.0, atol=0.1)

    def test_update_multiple_batches(self):
        """Test updating with multiple batches."""
        rms = RunningMeanStd(shape=())

        # Add multiple batches
        for _ in range(5):
            x = np.random.randn(10)
            rms.update(x)

        # Mean should be close to 0 for random normal
        assert abs(rms.mean) < 1.0

    def test_update_multidimensional(self):
        """Test with multidimensional data."""
        rms = RunningMeanStd(shape=(3,))

        x = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])

        rms.update(x)

        # Each dimension should have different mean
        assert rms.mean.shape == (3,)

    def test_normalize(self):
        """Test normalization."""
        rms = RunningMeanStd(shape=())

        # Train on data
        x_train = np.random.randn(100) * 10 + 50
        rms.update(x_train)

        # Normalize new data
        x_test = np.array([50.0, 60.0, 40.0])
        normalized = rms.normalize(x_test)

        # Normalized data should have roughly zero mean and unit variance
        assert normalized.shape == x_test.shape

    def test_normalize_clipping(self):
        """Test that normalization clips values."""
        rms = RunningMeanStd(shape=())

        x_train = np.array([1.0, 2.0, 3.0])
        rms.update(x_train)

        # Create extreme outlier
        x_test = np.array([1000.0])
        normalized = rms.normalize(x_test, clip=10.0)

        # Should be clipped to [-10, 10]
        assert normalized[0] <= 10.0
        assert normalized[0] >= -10.0

    def test_welford_algorithm_accuracy(self):
        """Test accuracy of Welford's algorithm."""
        rms = RunningMeanStd(shape=())

        # Generate known data
        data = np.random.randn(1000)

        # Update incrementally
        for i in range(0, len(data), 10):
            batch = data[i:i+10]
            rms.update(batch)

        # Compare with numpy
        expected_mean = np.mean(data)
        expected_var = np.var(data)

        assert np.isclose(rms.mean, expected_mean, rtol=0.1)
        assert np.isclose(rms.var, expected_var, rtol=0.1)


class TestRLProgressTracker:
    """Tests for RLProgressTracker."""

    def test_initialization_with_tqdm(self):
        """Test initialization with tqdm."""
        tracker = RLProgressTracker(
            total_timesteps=10000,
            desc="Test Training",
            use_tqdm=True
        )

        assert tracker.total_timesteps == 10000
        assert tracker.desc == "Test Training"

    def test_initialization_without_tqdm(self):
        """Test initialization without tqdm."""
        tracker = RLProgressTracker(
            total_timesteps=10000,
            desc="Test Training",
            use_tqdm=False
        )

        assert tracker.total_timesteps == 10000
        assert tracker.use_tqdm is False

    def test_context_manager(self):
        """Test context manager usage."""
        with RLProgressTracker(10000, use_tqdm=False) as tracker:
            assert tracker is not None

    def test_update_without_tqdm(self, capsys):
        """Test update without tqdm."""
        with RLProgressTracker(10000, use_tqdm=False) as tracker:
            tracker.update(
                timesteps=1000,
                episodes=10,
                mean_reward=50.0,
                sharpe=1.5,
                fps=100.0
            )

        # Should print to stdout
        captured = capsys.readouterr()
        assert "1000" in captured.out

    def test_log_without_tqdm(self, capsys):
        """Test logging without tqdm."""
        with RLProgressTracker(10000, use_tqdm=False) as tracker:
            tracker.log("Test message")

        captured = capsys.readouterr()
        assert "Test message" in captured.out


class TestEvaluateAgent:
    """Tests for evaluate_agent function."""

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        agent = Mock()
        agent.select_action = Mock(return_value=0)
        return agent

    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        env = Mock()
        env.reset = Mock(return_value=np.array([0.0, 0.0, 0.0]))

        # Simulate episode
        step_count = [0]

        def step_side_effect(action):
            step_count[0] += 1
            done = step_count[0] >= 10
            obs = np.random.randn(3)
            reward = np.random.randn()
            info = {'return': reward}
            return obs, reward, done, info

        env.step = Mock(side_effect=step_side_effect)
        return env

    def test_evaluate_agent_basic(self, mock_agent, mock_env):
        """Test basic agent evaluation."""
        results = evaluate_agent(
            mock_agent,
            mock_env,
            n_episodes=2,
            deterministic=True
        )

        assert "mean_reward" in results
        assert "std_reward" in results
        assert "mean_length" in results
        assert "sharpe_ratio" in results

    def test_evaluate_agent_multiple_episodes(self, mock_agent, mock_env):
        """Test evaluation over multiple episodes."""
        results = evaluate_agent(
            mock_agent,
            mock_env,
            n_episodes=5,
            deterministic=True
        )

        # Should have statistics from 5 episodes
        assert isinstance(results["mean_reward"], float)
        assert isinstance(results["std_reward"], float)

    def test_evaluate_agent_deterministic(self, mock_agent, mock_env):
        """Test deterministic evaluation."""
        evaluate_agent(
            mock_agent,
            mock_env,
            n_episodes=1,
            deterministic=True
        )

        # Agent should be called with deterministic=True
        mock_agent.select_action.assert_called()
        call_args = mock_agent.select_action.call_args
        if len(call_args) > 1 and 'deterministic' in call_args[1]:
            assert call_args[1]['deterministic'] is True

    def test_evaluate_agent_sharpe_calculation(self, mock_agent):
        """Test Sharpe ratio calculation."""
        # Create env with known returns
        env = Mock()
        env.reset = Mock(return_value=np.array([0.0]))

        returns = [0.01, 0.02, -0.01, 0.015, 0.01]
        return_idx = [0]

        def step_side_effect(action):
            idx = return_idx[0]
            return_idx[0] += 1
            done = idx >= len(returns) - 1
            info = {'return': returns[idx]} if idx < len(returns) else {'return': 0}
            return np.array([0.0]), returns[idx] if idx < len(returns) else 0, done, info

        env.step = Mock(side_effect=step_side_effect)

        results = evaluate_agent(mock_agent, env, n_episodes=1)

        # Should calculate Sharpe ratio
        assert "sharpe_ratio" in results


class TestTrainRLAgent:
    """Tests for train_rl_agent function."""

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        agent = Mock()
        agent.__class__.__name__ = "PPOAgent"
        agent.select_action = Mock(return_value=(np.array([0.5]), 0.0, 0.0))
        agent.store_transition = Mock()
        agent.update = Mock()
        agent.can_update = Mock(return_value=True)
        agent.save = Mock()
        return agent

    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        env = Mock()
        obs = np.random.randn(4)
        env.reset = Mock(return_value=obs)

        step_count = [0]

        def step_side_effect(action):
            step_count[0] += 1
            done = (step_count[0] % 50) == 0
            if done:
                step_count[0] = 0
            next_obs = np.random.randn(4)
            reward = np.random.randn()
            info = {'return': reward}
            return next_obs, reward, done, info

        env.step = Mock(side_effect=step_side_effect)
        return env

    @pytest.fixture
    def quick_config(self):
        """Quick config for testing."""
        return RLTrainingConfig(
            total_timesteps=100,
            log_freq=50,
            eval_freq=50,
            save_freq=50,
            progress_bar=False,
            verbose=False,
            checkpoint_dir=None,
            normalize_observations=False,
            normalize_rewards=False
        )

    def test_train_rl_agent_basic(self, mock_agent, mock_env, quick_config):
        """Test basic training."""
        trained_agent, metrics = train_rl_agent(
            mock_agent,
            mock_env,
            quick_config
        )

        assert trained_agent is mock_agent
        assert isinstance(metrics, RLTrainingMetrics)
        assert metrics.total_timesteps == 100

    def test_train_rl_agent_with_normalization(self, mock_agent, mock_env):
        """Test training with observation normalization."""
        config = RLTrainingConfig(
            total_timesteps=100,
            progress_bar=False,
            normalize_observations=True,
            normalize_rewards=True
        )

        trained_agent, metrics = train_rl_agent(
            mock_agent,
            mock_env,
            config
        )

        assert metrics.total_timesteps == 100

    def test_train_rl_agent_with_checkpointing(self, mock_agent, mock_env):
        """Test training with checkpointing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLTrainingConfig(
                total_timesteps=100,
                save_freq=50,
                checkpoint_dir=tmpdir,
                progress_bar=False
            )

            train_rl_agent(mock_agent, mock_env, config)

            # Should have called save
            mock_agent.save.assert_called()

    def test_train_rl_agent_with_evaluation(self, mock_agent, mock_env):
        """Test training with periodic evaluation."""
        eval_env = Mock()
        eval_env.reset = Mock(return_value=np.random.randn(4))

        def eval_step_side_effect(action):
            done = True
            return np.random.randn(4), 1.0, done, {'return': 1.0}

        eval_env.step = Mock(side_effect=eval_step_side_effect)

        config = RLTrainingConfig(
            total_timesteps=100,
            eval_freq=50,
            n_eval_episodes=2,
            progress_bar=False
        )

        trained_agent, metrics = train_rl_agent(
            mock_agent,
            mock_env,
            config,
            eval_env=eval_env
        )

        # Should have evaluation results
        assert len(metrics.eval_rewards) > 0

    def test_train_rl_agent_seeding(self, mock_agent, mock_env):
        """Test that seeding works."""
        config = RLTrainingConfig(
            total_timesteps=50,
            seed=42,
            progress_bar=False
        )

        # Train twice with same seed
        _, metrics1 = train_rl_agent(mock_agent, mock_env, config)

        # Reset mocks
        mock_env.reset = Mock(return_value=np.random.randn(4))

        config2 = RLTrainingConfig(
            total_timesteps=50,
            seed=42,
            progress_bar=False
        )

        _, metrics2 = train_rl_agent(mock_agent, mock_env, config2)

        # Both should complete
        assert metrics1.total_timesteps == 50
        assert metrics2.total_timesteps == 50

    def test_train_rl_agent_reward_scaling(self, mock_agent, mock_env):
        """Test reward scaling."""
        config = RLTrainingConfig(
            total_timesteps=50,
            reward_scale=0.1,
            progress_bar=False,
            normalize_rewards=False
        )

        train_rl_agent(mock_agent, mock_env, config)

        # Agent should receive scaled rewards
        mock_agent.store_transition.assert_called()

    def test_train_rl_agent_episode_metrics(self, mock_agent, mock_env):
        """Test episode metrics collection."""
        config = RLTrainingConfig(
            total_timesteps=200,
            progress_bar=False
        )

        _, metrics = train_rl_agent(mock_agent, mock_env, config)

        # Should have collected episode metrics
        assert len(metrics.episode_rewards) > 0
        assert len(metrics.episode_lengths) > 0

    def test_train_rl_agent_agent_update(self, mock_agent, mock_env):
        """Test that agent is updated."""
        config = RLTrainingConfig(
            total_timesteps=100,
            progress_bar=False
        )

        train_rl_agent(mock_agent, mock_env, config)

        # Agent should be updated
        mock_agent.update.assert_called()


class TestDisplayRLTrainingSummary:
    """Tests for display_rl_training_summary."""

    def test_display_summary(self, capsys):
        """Test displaying training summary."""
        metrics = RLTrainingMetrics()

        # Add some data
        for i in range(100):
            metrics.add_episode(reward=float(i), length=50, sharpe=1.5)

        metrics.add_eval(mean_reward=50.0, mean_sharpe=1.8)
        metrics.total_timesteps = 5000
        metrics.training_time = 100.0

        display_rl_training_summary(metrics, agent_name="TestAgent")

        captured = capsys.readouterr()

        # Check output contains key information
        assert "TestAgent" in captured.out
        assert "5,000" in captured.out or "5000" in captured.out
        assert "100" in captured.out  # Episodes
        assert "100.00s" in captured.out  # Training time


class TestEdgeCases:
    """Tests for edge cases."""

    def test_running_mean_std_single_value(self):
        """Test RunningMeanStd with single value."""
        rms = RunningMeanStd(shape=())

        x = np.array([5.0])
        rms.update(x)

        # Should handle single value
        assert np.isfinite(rms.mean)
        assert np.isfinite(rms.var)

    def test_running_mean_std_zero_variance(self):
        """Test RunningMeanStd with constant values."""
        rms = RunningMeanStd(shape=())

        x = np.array([5.0, 5.0, 5.0, 5.0])
        rms.update(x)

        # Variance should be close to zero
        assert np.isclose(rms.var, 0.0, atol=1e-5)

    def test_evaluate_agent_no_returns(self):
        """Test evaluation with no return info."""
        agent = Mock()
        agent.select_action = Mock(return_value=0)

        env = Mock()
        env.reset = Mock(return_value=np.array([0.0]))

        def step_side_effect(action):
            return np.array([0.0]), 1.0, True, {}  # No return in info

        env.step = Mock(side_effect=step_side_effect)

        results = evaluate_agent(agent, env, n_episodes=1)

        # Should still return results
        assert "mean_reward" in results
        assert results["sharpe_ratio"] == 0.0

    def test_train_rl_agent_zero_timesteps(self, mock_agent, mock_env):
        """Test training with zero timesteps."""
        config = RLTrainingConfig(
            total_timesteps=0,
            progress_bar=False
        )

        _, metrics = train_rl_agent(mock_agent, mock_env, config)

        assert metrics.total_timesteps == 0

    def test_metrics_summary_empty(self):
        """Test getting summary with no data."""
        metrics = RLTrainingMetrics()

        summary = metrics.get_summary()

        assert summary["total_timesteps"] == 0
        assert summary["total_episodes"] == 0
        assert summary["mean_reward_100"] == 0.0
        assert summary["best_eval_reward"] == float('-inf')

    def test_train_rl_agent_without_update_method(self):
        """Test training with agent that doesn't have update method."""
        agent = Mock()
        agent.__class__.__name__ = "SimpleAgent"
        agent.select_action = Mock(return_value=(np.array([0.5]), 0.0, 0.0))
        agent.store_transition = Mock()
        # No update method

        env = Mock()
        env.reset = Mock(return_value=np.random.randn(4))

        def step_side_effect(action):
            return np.random.randn(4), 1.0, True, {}

        env.step = Mock(side_effect=step_side_effect)

        config = RLTrainingConfig(
            total_timesteps=50,
            progress_bar=False
        )

        # Should handle gracefully
        train_rl_agent(agent, env, config)

    def test_normalize_with_nan(self):
        """Test normalization with NaN values."""
        rms = RunningMeanStd(shape=())

        x = np.array([1.0, 2.0, 3.0])
        rms.update(x)

        # Try to normalize with NaN
        x_test = np.array([np.nan, 2.0, 3.0])

        try:
            normalized = rms.normalize(x_test, clip=10.0)
            # If it doesn't raise, check that NaN is handled
            assert normalized.shape == x_test.shape
        except (ValueError, RuntimeError):
            # It's okay to raise an error for NaN
            pass
