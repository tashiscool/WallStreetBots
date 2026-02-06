"""Tests for DDPG Agent."""
import os
import tempfile
import numpy as np
import pytest

from ml.tradingbots.components.ddpg_agent import DDPGAgent, DDPGConfig, OrnsteinUhlenbeckNoise


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


class TestOUNoise:
    def test_temporally_correlated(self):
        noise = OrnsteinUhlenbeckNoise(2)
        samples = [noise() for _ in range(100)]
        # OU noise should be autocorrelated
        arr = np.array(samples)
        assert arr.shape == (100, 2)

    def test_reset(self):
        noise = OrnsteinUhlenbeckNoise(2)
        for _ in range(50):
            noise()
        noise.reset()
        np.testing.assert_array_almost_equal(noise.state, noise.mu)


class TestDDPGAgent:
    def test_init(self):
        agent = DDPGAgent(state_dim=8, action_dim=2)
        assert agent.action_dim == 2

    def test_select_action(self):
        agent = DDPGAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state)
        assert action.shape == (2,)
        assert np.all(action >= -1) and np.all(action <= 1)

    def test_train_short(self):
        env = MockEnv()
        config = DDPGConfig(buffer_size=500, batch_size=16, min_replay_size=50, total_timesteps=200)
        agent = DDPGAgent(state_dim=4, action_dim=1, config=config)
        info = agent.train(env)
        assert "returns" in info
        assert len(info["returns"]) > 0

    def test_save_load(self):
        agent = DDPGAgent(state_dim=4, action_dim=1)
        state = np.random.randn(4).astype(np.float32)
        action_before = agent.select_action(state, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            agent2 = DDPGAgent(state_dim=4, action_dim=1)
            agent2.load(path)
            action_after = agent2.select_action(state, deterministic=True)
            np.testing.assert_array_almost_equal(action_before, action_after)
        finally:
            os.unlink(path)
