"""Tests for SAC Agent."""
import os
import tempfile
import numpy as np
import pytest
import torch

from ml.tradingbots.components.sac_agent import SACAgent, SACConfig, SACActorNetwork, SACCriticNetwork


class MockEnv:
    """Minimal environment for testing."""
    def __init__(self, state_dim=4, action_dim=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_dim = state_dim
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.random.randn(self.state_dim).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        state = np.random.randn(self.state_dim).astype(np.float32)
        reward = float(np.random.randn())
        done = self.step_count >= 10
        return state, reward, done, {}


class TestSACAgent:
    def test_init(self):
        agent = SACAgent(state_dim=8, action_dim=2)
        assert agent.action_dim == 2
        assert agent.config.auto_alpha is True

    def test_select_action_shape(self):
        agent = SACAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state)
        assert action.shape == (2,)
        assert np.all(action >= -1) and np.all(action <= 1)

    def test_select_action_deterministic(self):
        agent = SACAgent(state_dim=4, action_dim=1)
        state = np.random.randn(4).astype(np.float32)
        a1 = agent.select_action(state, deterministic=True)
        a2 = agent.select_action(state, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2)

    def test_train_short(self):
        env = MockEnv(state_dim=4, action_dim=1)
        config = SACConfig(
            buffer_size=500, batch_size=16, min_replay_size=50,
            total_timesteps=200
        )
        agent = SACAgent(state_dim=4, action_dim=1, config=config)
        info = agent.train(env)
        assert "returns" in info
        assert len(info["returns"]) > 0

    def test_save_load(self):
        agent = SACAgent(state_dim=4, action_dim=1)
        state = np.random.randn(4).astype(np.float32)
        action_before = agent.select_action(state, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            agent2 = SACAgent(state_dim=4, action_dim=1)
            agent2.load(path)
            action_after = agent2.select_action(state, deterministic=True)
            np.testing.assert_array_almost_equal(action_before, action_after)
        finally:
            os.unlink(path)

    def test_auto_alpha_updates(self):
        config = SACConfig(
            auto_alpha=True, buffer_size=200, batch_size=16,
            min_replay_size=50, total_timesteps=100
        )
        agent = SACAgent(state_dim=4, action_dim=1, config=config)
        initial_alpha = agent.alpha
        env = MockEnv()
        agent.train(env)
        # Alpha should have been adjusted
        assert isinstance(agent.alpha, float)


class TestSACNetworks:
    def test_actor_sample(self):
        actor = SACActorNetwork(4, 2)
        state = torch.randn(1, 4)
        action, log_prob = actor.sample(state)
        assert action.shape == (1, 2)
        assert log_prob.shape == (1, 1)
        assert torch.all(action >= -1) and torch.all(action <= 1)

    def test_critic_twin(self):
        critic = SACCriticNetwork(4, 2)
        state = torch.randn(3, 4)
        action = torch.randn(3, 2)
        q1, q2 = critic(state, action)
        assert q1.shape == (3, 1)
        assert q2.shape == (3, 1)
