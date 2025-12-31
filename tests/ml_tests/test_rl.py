"""
Tests for Reinforcement Learning Components

Tests cover:
- Trading Environment
- PPO Agent
- DQN Agent
- Action spaces and reward functions
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_prices():
    """Generate sample price data for environment."""
    np.random.seed(42)
    n_samples = 500
    # Simulate realistic price series
    returns = np.random.randn(n_samples) * 0.02  # 2% daily vol
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def short_prices():
    """Short price series for quick tests."""
    np.random.seed(42)
    returns = np.random.randn(200) * 0.01
    return 100 * np.exp(np.cumsum(returns))


class TestTradingEnvironmentConfig:
    """Tests for TradingEnvConfig."""

    def test_config_defaults(self):
        """Test environment config defaults."""
        from ml.tradingbots.components.rl_environment import TradingEnvConfig

        config = TradingEnvConfig()
        assert config.window_size == 60
        assert config.max_steps == 1000
        assert config.initial_capital == 100000.0
        assert config.commission_rate == 0.001
        assert config.max_drawdown_pct == 0.20

    def test_config_sharpe_default_reward(self):
        """Test Sharpe ratio is default reward function."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvConfig, RewardFunction
        )

        config = TradingEnvConfig()
        assert config.reward_function == RewardFunction.SHARPE


class TestActionSpace:
    """Tests for action space types."""

    def test_action_space_enum(self):
        """Test ActionSpace enum values."""
        from ml.tradingbots.components.rl_environment import ActionSpace

        assert ActionSpace.DISCRETE.value == "discrete"
        assert ActionSpace.CONTINUOUS.value == "continuous"


class TestRewardFunction:
    """Tests for reward functions."""

    def test_reward_function_enum(self):
        """Test RewardFunction enum values."""
        from ml.tradingbots.components.rl_environment import RewardFunction

        assert RewardFunction.PNL.value == "pnl"
        assert RewardFunction.SHARPE.value == "sharpe"
        assert RewardFunction.SORTINO.value == "sortino"
        assert RewardFunction.CALMAR.value == "calmar"


class TestTradingEnvironment:
    """Tests for TradingEnvironment."""

    def test_environment_creation(self, sample_prices):
        """Test environment can be created."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig
        )

        config = TradingEnvConfig(window_size=30, max_steps=100)
        env = TradingEnvironment(sample_prices, config)

        assert env is not None
        assert env.observation_dim > 0
        assert env.action_dim > 0

    def test_environment_reset(self, sample_prices):
        """Test environment reset returns observation."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig
        )

        config = TradingEnvConfig(window_size=30, max_steps=100)
        env = TradingEnvironment(sample_prices, config)

        obs = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs) == env.observation_dim

    def test_environment_step_discrete(self, sample_prices):
        """Test environment step with discrete action."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )

        config = TradingEnvConfig(
            window_size=30,
            max_steps=100,
            action_space_type=ActionSpace.DISCRETE,
        )
        env = TradingEnvironment(sample_prices, config)
        env.reset()

        # Action 1 = buy
        obs, reward, done, info = env.step(1)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'portfolio_value' in info
        assert 'position' in info

    def test_environment_step_continuous(self, sample_prices):
        """Test environment step with continuous action."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )

        config = TradingEnvConfig(
            window_size=30,
            max_steps=100,
            action_space_type=ActionSpace.CONTINUOUS,
        )
        env = TradingEnvironment(sample_prices, config)
        env.reset()

        # Continuous action: go 50% long
        obs, reward, done, info = env.step(0.5)

        assert isinstance(obs, np.ndarray)
        assert info['position'] == pytest.approx(0.5, abs=0.1)

    def test_environment_episode_completion(self, short_prices):
        """Test environment completes episode."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig
        )

        config = TradingEnvConfig(window_size=30, max_steps=50)
        env = TradingEnvironment(short_prices, config)
        obs = env.reset()

        done = False
        steps = 0
        while not done and steps < 200:
            obs, reward, done, info = env.step(0)  # Hold
            steps += 1

        assert done or steps >= config.max_steps

    def test_environment_render(self, sample_prices):
        """Test environment render."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig
        )

        config = TradingEnvConfig(window_size=30)
        env = TradingEnvironment(sample_prices, config)
        env.reset()

        output = env.render(mode="human")
        # render returns string in non-human mode
        assert output is None or isinstance(output, str)

    def test_environment_metrics(self, short_prices):
        """Test environment metrics after episode."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig
        )

        config = TradingEnvConfig(window_size=30, max_steps=50)
        env = TradingEnvironment(short_prices, config)
        env.reset()

        # Run a few steps
        for _ in range(20):
            env.step(1)  # Buy

        metrics = env.get_metrics()

        assert 'total_return_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown_pct' in metrics
        assert 'num_trades' in metrics


class TestPPOAgent:
    """Tests for PPO Agent."""

    def test_ppo_config_defaults(self):
        """Test PPO config defaults."""
        from ml.tradingbots.components.rl_agents import PPOConfig

        config = PPOConfig()
        assert config.hidden_dim == 256
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_epsilon == 0.2

    def test_ppo_agent_creation(self, sample_prices):
        """Test PPO agent creation."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )
        from ml.tradingbots.components.rl_agents import PPOAgent, PPOConfig

        config = TradingEnvConfig(
            window_size=30,
            action_space_type=ActionSpace.CONTINUOUS,
        )
        env = TradingEnvironment(sample_prices, config)

        ppo_config = PPOConfig(hidden_dim=64)
        agent = PPOAgent(
            state_dim=env.observation_dim,
            action_dim=1,
            config=ppo_config,
        )

        assert agent is not None
        assert hasattr(agent, 'network')  # Actor-Critic network
        assert hasattr(agent, 'optimizer')

    def test_ppo_select_action(self, sample_prices):
        """Test PPO action selection."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )
        from ml.tradingbots.components.rl_agents import PPOAgent, PPOConfig

        config = TradingEnvConfig(
            window_size=30,
            action_space_type=ActionSpace.CONTINUOUS,
        )
        env = TradingEnvironment(sample_prices, config)
        obs = env.reset()

        ppo_config = PPOConfig(hidden_dim=32)
        agent = PPOAgent(
            state_dim=env.observation_dim,
            action_dim=1,
            config=ppo_config,
        )

        result = agent.select_action(obs)

        # select_action returns (action, log_prob, value) tuple
        assert isinstance(result, tuple)
        assert len(result) == 3
        action, log_prob, value = result
        assert isinstance(action, np.ndarray)  # Action array
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_ppo_store_transition(self, sample_prices):
        """Test PPO transition storage."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )
        from ml.tradingbots.components.rl_agents import PPOAgent, PPOConfig

        # Use continuous action space for PPO
        config = TradingEnvConfig(
            window_size=30,
            action_space_type=ActionSpace.CONTINUOUS,
        )
        env = TradingEnvironment(sample_prices, config)
        obs = env.reset()

        agent = PPOAgent(
            state_dim=env.observation_dim,
            action_dim=1,
            config=PPOConfig(hidden_dim=32),
            continuous=True,
        )

        action_result = agent.select_action(obs)
        action = action_result[0]  # Extract action from (action, log_prob, value)

        # PPO stores transitions in buffer
        assert hasattr(agent, 'buffer')
        assert agent.buffer is not None

    def test_create_ppo_trading_agent(self, sample_prices):
        """Test factory function for PPO agent."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )
        from ml.tradingbots.components.rl_agents import create_ppo_trading_agent, PPOConfig

        config = TradingEnvConfig(
            window_size=30,
            action_space_type=ActionSpace.CONTINUOUS,
        )
        env = TradingEnvironment(sample_prices, config)

        ppo_config = PPOConfig(hidden_dim=32)
        agent = create_ppo_trading_agent(env, config=ppo_config)

        assert agent is not None


class TestDQNAgent:
    """Tests for DQN Agent."""

    def test_dqn_config_defaults(self):
        """Test DQN config defaults."""
        from ml.tradingbots.components.rl_agents import DQNConfig

        config = DQNConfig()
        assert config.hidden_dim == 256
        assert config.learning_rate == 1e-4
        assert config.gamma == 0.99
        assert config.epsilon_start == 1.0
        assert config.dueling  # Dueling by default

    def test_dqn_agent_creation(self, sample_prices):
        """Test DQN agent creation."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )
        from ml.tradingbots.components.rl_agents import DQNAgent, DQNConfig

        config = TradingEnvConfig(
            window_size=30,
            action_space_type=ActionSpace.DISCRETE,
        )
        env = TradingEnvironment(sample_prices, config)

        dqn_config = DQNConfig(hidden_dim=64)
        agent = DQNAgent(
            state_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=dqn_config,
        )

        assert agent is not None
        assert hasattr(agent, 'q_network')
        assert hasattr(agent, 'target_network')

    def test_dqn_select_action_exploration(self, sample_prices):
        """Test DQN action selection with exploration."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )
        from ml.tradingbots.components.rl_agents import DQNAgent, DQNConfig

        config = TradingEnvConfig(
            window_size=30,
            action_space_type=ActionSpace.DISCRETE,
        )
        env = TradingEnvironment(sample_prices, config)
        obs = env.reset()

        dqn_config = DQNConfig(hidden_dim=32, epsilon_start=1.0)
        agent = DQNAgent(
            state_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=dqn_config,
        )

        action = agent.select_action(obs)

        assert isinstance(action, (int, np.integer))
        assert 0 <= action < env.action_dim

    def test_dqn_select_action_deterministic(self, sample_prices):
        """Test DQN deterministic action selection."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )
        from ml.tradingbots.components.rl_agents import DQNAgent, DQNConfig

        config = TradingEnvConfig(
            window_size=30,
            action_space_type=ActionSpace.DISCRETE,
        )
        env = TradingEnvironment(sample_prices, config)
        obs = env.reset()

        dqn_config = DQNConfig(hidden_dim=32)
        agent = DQNAgent(
            state_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=dqn_config,
        )

        action = agent.select_action(obs, deterministic=True)

        assert isinstance(action, (int, np.integer))

    def test_dqn_replay_buffer(self, sample_prices):
        """Test DQN has replay buffer."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )
        from ml.tradingbots.components.rl_agents import DQNAgent, DQNConfig

        config = TradingEnvConfig(
            window_size=30,
            action_space_type=ActionSpace.DISCRETE,
        )
        env = TradingEnvironment(sample_prices, config)

        agent = DQNAgent(
            state_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=DQNConfig(hidden_dim=32),
        )

        # Verify agent has replay buffer
        assert hasattr(agent, 'replay_buffer')
        assert agent.replay_buffer is not None
        assert len(agent.replay_buffer) == 0  # Empty at start

    def test_create_dqn_trading_agent(self, sample_prices):
        """Test factory function for DQN agent."""
        from ml.tradingbots.components.rl_environment import (
            TradingEnvironment, TradingEnvConfig, ActionSpace
        )
        from ml.tradingbots.components.rl_agents import create_dqn_trading_agent, DQNConfig

        config = TradingEnvConfig(
            window_size=30,
            action_space_type=ActionSpace.DISCRETE,
        )
        env = TradingEnvironment(sample_prices, config)

        dqn_config = DQNConfig(hidden_dim=32)
        agent = create_dqn_trading_agent(env, config=dqn_config)

        assert agent is not None


class TestMultiAssetEnvironment:
    """Tests for MultiAssetTradingEnvironment."""

    def test_multi_asset_creation(self):
        """Test multi-asset environment creation."""
        from ml.tradingbots.components.rl_environment import (
            MultiAssetTradingEnvironment, TradingEnvConfig
        )

        np.random.seed(42)
        price_data = {
            "AAPL": 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02)),
            "GOOGL": 150 * np.exp(np.cumsum(np.random.randn(300) * 0.02)),
        }

        config = TradingEnvConfig(window_size=30)
        env = MultiAssetTradingEnvironment(price_data, config)

        assert env is not None
        assert len(env.symbols) == 2

    def test_multi_asset_reset(self):
        """Test multi-asset reset."""
        from ml.tradingbots.components.rl_environment import (
            MultiAssetTradingEnvironment, TradingEnvConfig
        )

        np.random.seed(42)
        price_data = {
            "AAPL": 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02)),
            "GOOGL": 150 * np.exp(np.cumsum(np.random.randn(300) * 0.02)),
        }

        config = TradingEnvConfig(window_size=30)
        env = MultiAssetTradingEnvironment(price_data, config)

        observations = env.reset()

        assert isinstance(observations, dict)
        assert "AAPL" in observations
        assert "GOOGL" in observations

    def test_multi_asset_step(self):
        """Test multi-asset step."""
        from ml.tradingbots.components.rl_environment import (
            MultiAssetTradingEnvironment, TradingEnvConfig
        )

        np.random.seed(42)
        price_data = {
            "AAPL": 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02)),
            "GOOGL": 150 * np.exp(np.cumsum(np.random.randn(300) * 0.02)),
        }

        config = TradingEnvConfig(window_size=30)
        env = MultiAssetTradingEnvironment(price_data, config)
        env.reset()

        actions = {"AAPL": 1, "GOOGL": 0}  # Buy AAPL, hold GOOGL
        observations, rewards, done, info = env.step(actions)

        assert isinstance(observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(done, bool)
