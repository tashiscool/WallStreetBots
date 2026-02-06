"""Tests for Meta-Learning Components (Phase 8)."""
import os
import tempfile
import numpy as np
import pytest

from ml.tradingbots.components.meta_learning import (
    RegimeDetector, RegimeAwareAgent, RegimeAwareConfig,
    TransferLearningTrainer, MultiTaskRLAgent,
)


class MockAgent:
    """Minimal mock agent for testing."""
    def __init__(self, state_dim=4, action_dim=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trained = False

    def select_action(self, state, deterministic=False):
        return np.random.randn(self.action_dim)

    def train(self, env, total_timesteps=100, callback=None):
        self.trained = True
        return {"returns": [float(np.random.randn()) for _ in range(10)]}

    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump({"saved": True}, f)

    def load(self, path):
        pass


class MockEnv:
    """Minimal mock environment."""
    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        return np.zeros(4, dtype=np.float32), 1.0, self.step_count >= 10, {}


# --- RegimeDetector Tests ---

class TestRegimeDetector:
    def test_init(self):
        detector = RegimeDetector()
        assert detector.n_regimes == 3
        assert detector.lookback == 60

    def test_low_vol_regime(self):
        detector = RegimeDetector(lookback=30, vol_thresholds=(0.10, 0.25))
        # Very low volatility returns
        returns = np.random.normal(0, 0.001, 60)  # ~1.6% annualized vol
        regime = detector.detect(returns)
        assert regime == 0

    def test_normal_regime(self):
        detector = RegimeDetector(lookback=30, vol_thresholds=(0.10, 0.25))
        # Normal volatility
        returns = np.random.normal(0, 0.01, 60)  # ~15.9% annualized vol
        regime = detector.detect(returns)
        assert regime == 1

    def test_high_vol_regime(self):
        detector = RegimeDetector(lookback=30, vol_thresholds=(0.10, 0.25))
        # Very high volatility
        returns = np.random.normal(0, 0.03, 60)  # ~47.6% annualized vol
        regime = detector.detect(returns)
        assert regime == 2

    def test_insufficient_data_defaults_to_normal(self):
        detector = RegimeDetector(lookback=60)
        returns = np.random.normal(0, 0.01, 10)  # Too short
        regime = detector.detect(returns)
        assert regime == 1


# --- RegimeAwareAgent Tests ---

class TestRegimeAwareAgent:
    def test_init(self):
        agent = RegimeAwareAgent(
            create_agent_fn=lambda: MockAgent(),
            config=RegimeAwareConfig(n_regimes=3),
        )
        assert len(agent.specialists) == 3
        assert agent.generalist is not None
        assert agent.current_regime == 1

    def test_select_action(self):
        agent = RegimeAwareAgent(create_agent_fn=lambda: MockAgent())
        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state)
        assert isinstance(action, np.ndarray)

    def test_select_action_deterministic(self):
        agent = RegimeAwareAgent(create_agent_fn=lambda: MockAgent())
        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state, deterministic=True)
        assert isinstance(action, np.ndarray)

    def test_update_regime(self):
        agent = RegimeAwareAgent(
            create_agent_fn=lambda: MockAgent(),
            config=RegimeAwareConfig(n_regimes=3),
        )
        # High vol returns
        returns = np.random.normal(0, 0.03, 100)
        regime = agent.update_regime(returns)
        assert regime in [0, 1, 2]
        assert agent.current_regime == regime

    def test_train(self):
        agent = RegimeAwareAgent(
            create_agent_fn=lambda: MockAgent(),
            config=RegimeAwareConfig(n_regimes=2),
        )
        env = MockEnv()
        results = agent.train(env, total_timesteps=100)
        assert "generalist" in results
        assert "specialist_0" in results
        assert "specialist_1" in results

    def test_save_load(self):
        agent = RegimeAwareAgent(
            create_agent_fn=lambda: MockAgent(),
            config=RegimeAwareConfig(n_regimes=2),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "generalist.pt"))
            assert os.path.exists(os.path.join(tmpdir, "specialist_0.pt"))
            assert os.path.exists(os.path.join(tmpdir, "specialist_1.pt"))

            agent2 = RegimeAwareAgent(
                create_agent_fn=lambda: MockAgent(),
                config=RegimeAwareConfig(n_regimes=2),
            )
            agent2.load(tmpdir)  # Should not raise

    def test_blend_weight(self):
        config = RegimeAwareConfig(blend_weight=0.5)
        agent = RegimeAwareAgent(
            create_agent_fn=lambda: MockAgent(),
            config=config,
        )
        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state, deterministic=False)
        assert isinstance(action, np.ndarray)


# --- TransferLearningTrainer Tests ---

class TestTransferLearningTrainer:
    def test_init(self):
        trainer = TransferLearningTrainer(create_agent_fn=lambda: MockAgent())
        assert trainer.agent is None
        assert trainer.freeze_layers == 0

    def test_pretrain(self):
        trainer = TransferLearningTrainer(create_agent_fn=lambda: MockAgent())
        envs = [MockEnv(), MockEnv()]
        agent = trainer.pretrain(envs, timesteps_per_env=50)
        assert agent is not None
        assert agent.trained

    def test_fine_tune(self):
        trainer = TransferLearningTrainer(create_agent_fn=lambda: MockAgent())
        # Pretrain first
        trainer.pretrain([MockEnv()], timesteps_per_env=50)
        # Fine-tune
        agent = trainer.fine_tune(MockEnv(), total_timesteps=20)
        assert agent is not None

    def test_fine_tune_without_pretrain(self):
        trainer = TransferLearningTrainer(create_agent_fn=lambda: MockAgent())
        agent = trainer.fine_tune(MockEnv(), total_timesteps=20)
        assert agent is not None
        assert agent.trained


# --- MultiTaskRLAgent Tests ---

class TestMultiTaskRLAgent:
    def test_init(self):
        envs = {"task_a": MockEnv(), "task_b": MockEnv()}
        agent = MultiTaskRLAgent(
            create_agent_fn=lambda: MockAgent(),
            task_envs=envs,
        )
        assert len(agent.task_envs) == 2
        assert "task_a" in agent.task_metrics
        assert "task_b" in agent.task_metrics

    def test_train(self):
        envs = {"task_a": MockEnv(), "task_b": MockEnv()}
        agent = MultiTaskRLAgent(
            create_agent_fn=lambda: MockAgent(),
            task_envs=envs,
            steps_per_task=50,
        )
        metrics = agent.train(total_rounds=2)
        assert "task_a" in metrics
        assert "task_b" in metrics
        # Each task should have metrics from 2 rounds
        assert len(metrics["task_a"]) == 2
        assert len(metrics["task_b"]) == 2

    def test_select_action(self):
        envs = {"task": MockEnv()}
        agent = MultiTaskRLAgent(
            create_agent_fn=lambda: MockAgent(),
            task_envs=envs,
        )
        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state)
        assert isinstance(action, np.ndarray)

    def test_save_load(self):
        envs = {"task": MockEnv()}
        agent = MultiTaskRLAgent(
            create_agent_fn=lambda: MockAgent(),
            task_envs=envs,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            assert os.path.exists(path)
            agent.load(path)  # Should not raise
        finally:
            os.unlink(path)
