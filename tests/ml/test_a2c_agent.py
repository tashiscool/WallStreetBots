"""Tests for A2C Agent."""
import os
import tempfile
import numpy as np
import pytest

from ml.tradingbots.components.a2c_agent import A2CAgent, A2CConfig


class MockEnv:
    def __init__(self, state_dim=4, action_dim=1):
        self.state_dim = state_dim
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.random.randn(self.state_dim).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        return np.random.randn(self.state_dim).astype(np.float32), float(np.random.randn()), self.step_count >= 10, {}


class TestA2CAgent:
    def test_init(self):
        agent = A2CAgent(state_dim=8, action_dim=2)
        assert agent.continuous is True

    def test_select_action_continuous(self):
        agent = A2CAgent(state_dim=4, action_dim=2, continuous=True)
        state = np.random.randn(4).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        assert action.shape == (2,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_select_action_discrete(self):
        agent = A2CAgent(state_dim=4, action_dim=3, continuous=False)
        state = np.random.randn(4).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 3

    def test_train_short(self):
        env = MockEnv()
        config = A2CConfig(n_steps=5, total_timesteps=100)
        agent = A2CAgent(state_dim=4, action_dim=1, config=config)
        info = agent.train(env)
        assert "returns" in info

    def test_save_load(self):
        agent = A2CAgent(state_dim=4, action_dim=1)
        state = np.random.randn(4).astype(np.float32)
        a1, _, _ = agent.select_action(state, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            agent2 = A2CAgent(state_dim=4, action_dim=1)
            agent2.load(path)
            a2, _, _ = agent2.select_action(state, deterministic=True)
            np.testing.assert_array_almost_equal(a1, a2)
        finally:
            os.unlink(path)
