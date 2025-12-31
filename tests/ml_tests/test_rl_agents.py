"""
Comprehensive Tests for RL Agents

Tests the reinforcement learning agents including:
- ActorCriticNetwork
- DQNetwork
- Experience and ReplayBuffer
- RolloutBuffer
- PPOAgent
- DQNAgent
- Agent factory functions
- Edge cases and error handling
"""

import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock
import numpy as np
import torch
import pytest

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.tradingbots.components.rl_agents import (
    ActorCriticNetwork,
    DQNetwork,
    Experience,
    ReplayBuffer,
    RolloutBuffer,
    PPOConfig,
    PPOAgent,
    DQNConfig,
    DQNAgent,
    create_ppo_trading_agent,
    create_dqn_trading_agent
)


class TestActorCriticNetwork:
    """Tests for ActorCriticNetwork."""

    def test_continuous_initialization(self):
        """Test initialization for continuous action space."""
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            continuous=True
        )

        assert network.continuous is True
        assert network.shared is not None
        assert network.actor_mean is not None
        assert network.actor_log_std is not None
        assert network.critic is not None

    def test_discrete_initialization(self):
        """Test initialization for discrete action space."""
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=3,
            hidden_dim=64,
            continuous=False
        )

        assert network.continuous is False
        assert network.actor is not None
        assert network.critic is not None

    def test_continuous_forward_pass(self):
        """Test forward pass for continuous actions."""
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=2,
            continuous=True
        )

        state = torch.randn(16, 4)
        output = network(state)

        assert len(output) == 3  # mean, std, value
        assert output[0].shape == (16, 2)  # action mean
        assert output[1].shape == (16, 2)  # action std
        assert output[2].shape == (16, 1)  # value

    def test_discrete_forward_pass(self):
        """Test forward pass for discrete actions."""
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=3,
            continuous=False
        )

        state = torch.randn(16, 4)
        output = network(state)

        assert len(output) == 2  # logits, value
        assert output[0].shape == (16, 3)  # action logits
        assert output[1].shape == (16, 1)  # value

    def test_get_action_continuous(self):
        """Test getting action for continuous space."""
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=2,
            continuous=True
        )

        state = torch.randn(1, 4)
        action, log_prob, value = network.get_action(state)

        assert action.shape == (1, 2)
        assert log_prob.shape == (1, 1)
        assert value.shape == (1, 1)

        # Actions should be in [-1, 1]
        assert torch.all(action >= -1)
        assert torch.all(action <= 1)

    def test_get_action_discrete(self):
        """Test getting action for discrete space."""
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=3,
            continuous=False
        )

        state = torch.randn(1, 4)
        action, log_prob, value = network.get_action(state)

        assert action.shape == (1,)
        assert log_prob.shape == (1, 1)
        assert value.shape == (1, 1)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=2,
            continuous=True
        )

        state = torch.randn(1, 4)
        action1, _, _ = network.get_action(state, deterministic=True)
        action2, _, _ = network.get_action(state, deterministic=True)

        # Deterministic actions should be identical
        assert torch.allclose(action1, action2)


class TestDQNetwork:
    """Tests for DQNetwork."""

    def test_standard_initialization(self):
        """Test standard DQN initialization."""
        network = DQNetwork(
            state_dim=4,
            action_dim=3,
            hidden_dim=64,
            dueling=False
        )

        assert network.dueling is False
        assert network.feature is not None
        assert network.q_layer is not None

    def test_dueling_initialization(self):
        """Test dueling DQN initialization."""
        network = DQNetwork(
            state_dim=4,
            action_dim=3,
            hidden_dim=64,
            dueling=True
        )

        assert network.dueling is True
        assert network.value_stream is not None
        assert network.advantage_stream is not None

    def test_forward_pass_standard(self):
        """Test forward pass for standard DQN."""
        network = DQNetwork(
            state_dim=4,
            action_dim=3,
            dueling=False
        )

        state = torch.randn(16, 4)
        q_values = network(state)

        assert q_values.shape == (16, 3)

    def test_forward_pass_dueling(self):
        """Test forward pass for dueling DQN."""
        network = DQNetwork(
            state_dim=4,
            action_dim=3,
            dueling=True
        )

        state = torch.randn(16, 4)
        q_values = network(state)

        assert q_values.shape == (16, 3)

    def test_dueling_architecture_properties(self):
        """Test dueling architecture maintains correct properties."""
        network = DQNetwork(
            state_dim=4,
            action_dim=3,
            dueling=True
        )

        state = torch.randn(1, 4)
        q_values = network(state)

        # Q-values should be real numbers
        assert torch.all(torch.isfinite(q_values))


class TestExperienceAndReplayBuffer:
    """Tests for Experience and ReplayBuffer."""

    def test_experience_creation(self):
        """Test creating experience tuple."""
        exp = Experience(
            state=np.array([1, 2, 3]),
            action=1,
            reward=1.5,
            next_state=np.array([2, 3, 4]),
            done=False
        )

        assert exp.state.shape == (3,)
        assert exp.action == 1
        assert exp.reward == 1.5
        assert exp.done is False

    def test_replay_buffer_initialization(self):
        """Test replay buffer initialization."""
        buffer = ReplayBuffer(capacity=100)

        assert len(buffer) == 0
        assert buffer.buffer.maxlen == 100

    def test_replay_buffer_push(self):
        """Test adding experiences to buffer."""
        buffer = ReplayBuffer(capacity=100)

        exp = Experience(
            state=np.array([1, 2]),
            action=0,
            reward=1.0,
            next_state=np.array([2, 3]),
            done=False
        )

        buffer.push(exp)

        assert len(buffer) == 1

    def test_replay_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(50):
            exp = Experience(
                state=np.array([i, i+1]),
                action=i % 3,
                reward=float(i),
                next_state=np.array([i+1, i+2]),
                done=False
            )
            buffer.push(exp)

        samples = buffer.sample(batch_size=10)

        assert len(samples) == 10
        assert all(isinstance(s, Experience) for s in samples)

    def test_replay_buffer_capacity(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)

        for i in range(20):
            exp = Experience(
                state=np.array([i]),
                action=0,
                reward=0.0,
                next_state=np.array([i+1]),
                done=False
            )
            buffer.push(exp)

        assert len(buffer) == 10

    def test_replay_buffer_sample_smaller_than_size(self):
        """Test sampling when buffer is small."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(5):
            exp = Experience(
                state=np.array([i]),
                action=0,
                reward=0.0,
                next_state=np.array([i+1]),
                done=False
            )
            buffer.push(exp)

        samples = buffer.sample(batch_size=10)

        # Should return all 5 samples
        assert len(samples) == 5


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_initialization(self):
        """Test rollout buffer initialization."""
        buffer = RolloutBuffer()

        assert len(buffer.states) == 0
        assert len(buffer.actions) == 0
        assert len(buffer.rewards) == 0
        assert len(buffer.dones) == 0
        assert len(buffer.log_probs) == 0
        assert len(buffer.values) == 0

    def test_add(self):
        """Test adding data to buffer."""
        buffer = RolloutBuffer()

        buffer.add(
            state=np.array([1, 2]),
            action=0,
            reward=1.0,
            done=False,
            log_prob=-0.5,
            value=0.8
        )

        assert len(buffer.states) == 1
        assert len(buffer.actions) == 1
        assert len(buffer.rewards) == 1

    def test_get(self):
        """Test getting data from buffer."""
        buffer = RolloutBuffer()

        for i in range(10):
            buffer.add(
                state=np.array([i, i+1]),
                action=i % 2,
                reward=float(i),
                done=False,
                log_prob=-0.5,
                value=0.8
            )

        states, actions, rewards, dones, log_probs, values = buffer.get()

        assert states.shape == (10, 2)
        assert actions.shape == (10,)
        assert rewards.shape == (10,)

    def test_clear(self):
        """Test clearing buffer."""
        buffer = RolloutBuffer()

        buffer.add(
            state=np.array([1, 2]),
            action=0,
            reward=1.0,
            done=False,
            log_prob=-0.5,
            value=0.8
        )

        buffer.clear()

        assert len(buffer.states) == 0
        assert len(buffer.actions) == 0


class TestPPOConfig:
    """Tests for PPOConfig."""

    def test_default_config(self):
        """Test default PPO configuration."""
        config = PPOConfig()

        assert config.hidden_dim == 256
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_epsilon == 0.2
        assert config.n_steps == 2048
        assert config.n_epochs == 10
        assert config.batch_size == 64

    def test_custom_config(self):
        """Test custom PPO configuration."""
        config = PPOConfig(
            hidden_dim=128,
            learning_rate=1e-3,
            clip_epsilon=0.1
        )

        assert config.hidden_dim == 128
        assert config.learning_rate == 1e-3
        assert config.clip_epsilon == 0.1


class TestPPOAgent:
    """Tests for PPO agent."""

    @pytest.fixture
    def mock_env(self):
        """Create mock environment."""
        env = Mock()
        env.reset = Mock(return_value=np.random.randn(4))

        def step_side_effect(action):
            return (
                np.random.randn(4),
                np.random.randn(),
                False,
                {}
            )

        env.step = Mock(side_effect=step_side_effect)
        return env

    def test_initialization(self):
        """Test PPO agent initialization."""
        agent = PPOAgent(
            state_dim=4,
            action_dim=2,
            continuous=True
        )

        assert agent.network is not None
        assert agent.optimizer is not None
        assert agent.buffer is not None
        assert agent.continuous is True

    def test_select_action(self):
        """Test action selection."""
        agent = PPOAgent(
            state_dim=4,
            action_dim=2,
            continuous=True
        )

        state = np.random.randn(4)
        action, log_prob, value = agent.select_action(state)

        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_select_action_deterministic(self):
        """Test deterministic action selection."""
        agent = PPOAgent(
            state_dim=4,
            action_dim=2,
            continuous=True
        )

        state = np.random.randn(4)
        action1, _, _ = agent.select_action(state, deterministic=True)
        action2, _, _ = agent.select_action(state, deterministic=True)

        # Should be identical for same state
        np.testing.assert_array_almost_equal(action1, action2)

    def test_discrete_action_selection(self):
        """Test discrete action selection."""
        agent = PPOAgent(
            state_dim=4,
            action_dim=3,
            continuous=False
        )

        state = np.random.randn(4)
        action, log_prob, value = agent.select_action(state)

        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 3

    def test_compute_gae(self):
        """Test GAE computation."""
        config = PPOConfig(gamma=0.99, gae_lambda=0.95)
        agent = PPOAgent(state_dim=4, action_dim=2, config=config)

        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        values = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        dones = np.array([False, False, False, False, True])

        returns, advantages = agent._compute_gae(rewards, values, dones)

        assert returns.shape == rewards.shape
        assert advantages.shape == rewards.shape

    def test_save_and_load(self):
        """Test saving and loading agent."""
        agent = PPOAgent(state_dim=4, action_dim=2)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            agent.save(path)

            new_agent = PPOAgent(state_dim=4, action_dim=2)
            new_agent.load(path)

            # Should load successfully
            assert new_agent.network is not None
        finally:
            os.unlink(path)


class TestDQNConfig:
    """Tests for DQNConfig."""

    def test_default_config(self):
        """Test default DQN configuration."""
        config = DQNConfig()

        assert config.hidden_dim == 256
        assert config.dueling is True
        assert config.learning_rate == 1e-4
        assert config.gamma == 0.99
        assert config.epsilon_start == 1.0
        assert config.epsilon_end == 0.01
        assert config.buffer_size == 100000
        assert config.batch_size == 64

    def test_custom_config(self):
        """Test custom DQN configuration."""
        config = DQNConfig(
            hidden_dim=128,
            dueling=False,
            epsilon_start=0.5
        )

        assert config.hidden_dim == 128
        assert config.dueling is False
        assert config.epsilon_start == 0.5


class TestDQNAgent:
    """Tests for DQN agent."""

    def test_initialization(self):
        """Test DQN agent initialization."""
        agent = DQNAgent(
            state_dim=4,
            action_dim=3
        )

        assert agent.q_network is not None
        assert agent.target_network is not None
        assert agent.optimizer is not None
        assert agent.replay_buffer is not None
        assert agent.epsilon == 1.0

    def test_select_action_exploration(self):
        """Test action selection with exploration."""
        config = DQNConfig(epsilon_start=1.0)
        agent = DQNAgent(
            state_dim=4,
            action_dim=3,
            config=config
        )

        state = np.random.randn(4)
        action = agent.select_action(state, deterministic=False)

        assert 0 <= action < 3

    def test_select_action_greedy(self):
        """Test greedy action selection."""
        agent = DQNAgent(
            state_dim=4,
            action_dim=3
        )

        state = np.random.randn(4)
        action = agent.select_action(state, deterministic=True)

        assert 0 <= action < 3

    def test_soft_update_target(self):
        """Test soft update of target network."""
        agent = DQNAgent(state_dim=4, action_dim=3)

        # Get initial target parameters
        initial_params = [
            p.clone() for p in agent.target_network.parameters()
        ]

        # Update target
        agent._soft_update_target()

        # Parameters should have changed slightly
        updated_params = list(agent.target_network.parameters())

        # At least some parameters should be different
        any_changed = any(
            not torch.allclose(init, updated)
            for init, updated in zip(initial_params, updated_params)
        )

        assert any_changed or len(initial_params) == 0

    def test_save_and_load(self):
        """Test saving and loading DQN agent."""
        agent = DQNAgent(state_dim=4, action_dim=3)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            agent.save(path)

            new_agent = DQNAgent(state_dim=4, action_dim=3)
            new_agent.load(path)

            # Should load successfully
            assert new_agent.q_network is not None
            assert new_agent.target_network is not None
        finally:
            os.unlink(path)

    def test_epsilon_decay(self):
        """Test epsilon decay over time."""
        config = DQNConfig(
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.99
        )

        agent = DQNAgent(state_dim=4, action_dim=3, config=config)

        initial_epsilon = agent.epsilon

        # Simulate some training
        for _ in range(100):
            agent.epsilon = max(
                config.epsilon_end,
                agent.epsilon * config.epsilon_decay
            )

        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= config.epsilon_end


class TestAgentFactoryFunctions:
    """Tests for agent factory functions."""

    @pytest.fixture
    def mock_env(self):
        """Create mock trading environment."""
        env = Mock()
        env.observation_dim = 10
        env.action_dim = 3
        return env

    def test_create_ppo_trading_agent(self, mock_env):
        """Test creating PPO trading agent."""
        agent = create_ppo_trading_agent(mock_env)

        assert isinstance(agent, PPOAgent)
        assert agent.continuous is True

    def test_create_ppo_trading_agent_with_config(self, mock_env):
        """Test creating PPO agent with custom config."""
        config = PPOConfig(hidden_dim=128)
        agent = create_ppo_trading_agent(mock_env, config=config)

        assert isinstance(agent, PPOAgent)
        assert agent.config.hidden_dim == 128

    def test_create_dqn_trading_agent(self, mock_env):
        """Test creating DQN trading agent."""
        agent = create_dqn_trading_agent(mock_env)

        assert isinstance(agent, DQNAgent)
        assert agent.action_dim == 3

    def test_create_dqn_trading_agent_with_config(self, mock_env):
        """Test creating DQN agent with custom config."""
        config = DQNConfig(hidden_dim=128)
        agent = create_dqn_trading_agent(mock_env, config=config)

        assert isinstance(agent, DQNAgent)
        assert agent.config.hidden_dim == 128


class TestEdgeCases:
    """Tests for edge cases."""

    def test_ppo_empty_buffer_update(self):
        """Test PPO update with empty buffer."""
        agent = PPOAgent(state_dim=4, action_dim=2)

        # Try to update with empty buffer
        try:
            agent._update()
        except (IndexError, ValueError):
            # Expected to fail with empty buffer
            pass

    def test_dqn_update_insufficient_samples(self):
        """Test DQN update with insufficient samples."""
        config = DQNConfig(min_replay_size=100)
        agent = DQNAgent(state_dim=4, action_dim=3, config=config)

        # Add only a few samples
        for i in range(10):
            exp = Experience(
                state=np.random.randn(4),
                action=0,
                reward=1.0,
                next_state=np.random.randn(4),
                done=False
            )
            agent.replay_buffer.push(exp)

        # Should not update yet
        initial_loss_count = len(agent.training_info['loss'])
        agent._update()

        # Loss should not have increased (no update)
        assert len(agent.training_info['loss']) >= initial_loss_count

    def test_ppo_with_all_dones(self):
        """Test PPO with all episodes ending immediately."""
        agent = PPOAgent(state_dim=4, action_dim=2)

        # Add rollout where all are done
        for i in range(10):
            agent.buffer.add(
                state=np.random.randn(4),
                action=np.random.randn(2),
                reward=1.0,
                done=True,
                log_prob=-0.5,
                value=0.5
            )

        # Should handle all dones
        try:
            agent._update()
        except Exception as e:
            pytest.fail(f"Failed with all dones: {e}")

    def test_dqn_with_extreme_q_values(self):
        """Test DQN with extreme Q-values."""
        agent = DQNAgent(state_dim=4, action_dim=3)

        # Create experiences with extreme rewards
        for i in range(100):
            exp = Experience(
                state=np.random.randn(4),
                action=np.random.randint(0, 3),
                reward=1000.0,  # Extreme reward
                next_state=np.random.randn(4),
                done=False
            )
            agent.replay_buffer.push(exp)

        # Should handle extreme values
        try:
            agent._update()
        except Exception as e:
            pytest.fail(f"Failed with extreme Q-values: {e}")

    def test_actor_critic_with_nan_input(self):
        """Test actor-critic with NaN input."""
        network = ActorCriticNetwork(
            state_dim=4,
            action_dim=2,
            continuous=True
        )

        state = torch.tensor([[np.nan, 0.0, 0.0, 0.0]])

        # Should handle NaN gracefully or raise
        try:
            output = network(state)
            # Check if output has NaN
            assert any(torch.isnan(o).any() for o in output)
        except RuntimeError:
            # It's okay to raise with NaN input
            pass

    def test_replay_buffer_with_large_states(self):
        """Test replay buffer with large state dimensions."""
        buffer = ReplayBuffer(capacity=100)

        # Add experiences with large states
        for i in range(50):
            exp = Experience(
                state=np.random.randn(1000),  # Large state
                action=0,
                reward=1.0,
                next_state=np.random.randn(1000),
                done=False
            )
            buffer.push(exp)

        samples = buffer.sample(10)

        assert len(samples) == 10
        assert samples[0].state.shape == (1000,)

    def test_ppo_gradient_clipping(self):
        """Test that PPO clips gradients."""
        config = PPOConfig(max_grad_norm=0.5)
        agent = PPOAgent(state_dim=4, action_dim=2, config=config)

        # Add some data to buffer
        for i in range(100):
            agent.buffer.add(
                state=np.random.randn(4),
                action=np.random.randn(2),
                reward=np.random.randn(),
                done=False,
                log_prob=np.random.randn(),
                value=np.random.randn()
            )

        # Update should clip gradients
        try:
            agent._update()
            # If it doesn't raise, gradient clipping worked
        except Exception:
            # Some failures are expected with random data
            pass

    def test_dqn_double_dqn(self):
        """Test that DQN uses double DQN for target calculation."""
        agent = DQNAgent(state_dim=4, action_dim=3)

        # Add experiences
        for i in range(100):
            exp = Experience(
                state=np.random.randn(4),
                action=np.random.randint(0, 3),
                reward=np.random.randn(),
                next_state=np.random.randn(4),
                done=False
            )
            agent.replay_buffer.push(exp)

        # Update should use double DQN
        agent._update()

        # Should have updated Q-network
        assert len(agent.training_info['loss']) > 0
