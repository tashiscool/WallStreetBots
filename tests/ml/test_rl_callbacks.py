"""Tests for RL Training Callbacks."""
import os
import tempfile
import numpy as np
import pytest

from ml.tradingbots.training.callbacks import (
    BaseCallback, CallbackList, EvalCallback, CheckpointCallback,
    EarlyStoppingRLCallback, TradingMetricsCallback,
)


class MockAgent:
    def select_action(self, state, deterministic=False):
        return np.array([0.5])

    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump({"saved": True}, f)


class MockEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        return np.zeros(4, dtype=np.float32), 1.0, self.step_count >= 5, {"return": 0.01}


class TestCallbackList:
    def test_runs_all_callbacks(self):
        calls = []

        class TrackingCallback(BaseCallback):
            def __init__(self, name):
                super().__init__()
                self.name = name
            def _on_step(self, locals_):
                calls.append(self.name)
                return True

        cb = CallbackList([TrackingCallback("a"), TrackingCallback("b")])
        cb.init_callback(MockAgent(), MockEnv())
        cb.on_step({"timestep": 1})
        assert calls == ["a", "b"]

    def test_stops_if_callback_returns_false(self):
        class StopCallback(BaseCallback):
            def _on_step(self, locals_):
                return False

        class ContinueCallback(BaseCallback):
            def _on_step(self, locals_):
                return True

        cb = CallbackList([StopCallback(), ContinueCallback()])
        cb.init_callback(MockAgent(), MockEnv())
        assert cb.on_step({}) is False


class TestEvalCallback:
    def test_evaluation(self):
        eval_env = MockEnv()
        cb = EvalCallback(eval_env, n_eval_episodes=2, eval_freq=1, verbose=0)
        cb.init_callback(MockAgent(), MockEnv())
        cb.on_step({"timestep": 1})
        assert len(cb.eval_history) == 1
        assert cb.best_mean_reward > -np.inf

    def test_saves_best_model(self):
        eval_env = MockEnv()
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = EvalCallback(
                eval_env, n_eval_episodes=1, eval_freq=1,
                best_model_save_path=tmpdir, verbose=0,
            )
            cb.init_callback(MockAgent(), MockEnv())
            cb.on_step({"timestep": 1})
            assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))


class TestCheckpointCallback:
    def test_saves_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = CheckpointCallback(save_freq=1, save_path=tmpdir, verbose=0)
            cb.init_callback(MockAgent(), MockEnv())
            cb.on_step({"timestep": 1})
            files = os.listdir(tmpdir)
            assert any("rl_model" in f for f in files)


class TestEarlyStoppingRLCallback:
    def test_stops_at_threshold(self):
        cb = EarlyStoppingRLCallback(reward_threshold=5.0, check_freq=1, verbose=0)
        cb.init_callback(MockAgent(), MockEnv())
        for i in range(15):
            cb.on_episode_end({"episode_reward": 10.0})
        result = cb.on_step({"timestep": 15})
        assert result is False

    def test_continues_below_threshold(self):
        cb = EarlyStoppingRLCallback(reward_threshold=100.0, check_freq=1, verbose=0)
        cb.init_callback(MockAgent(), MockEnv())
        for i in range(15):
            cb.on_episode_end({"episode_reward": 1.0})
        result = cb.on_step({"timestep": 15})
        assert result is True


class TestTradingMetricsCallback:
    def test_tracks_episode_metrics(self):
        cb = TradingMetricsCallback(log_freq=100, verbose=0)
        cb.init_callback(MockAgent(), MockEnv())
        cb.on_step({"info": {"return": 0.01}, "timestep": 1})
        cb.on_episode_end({"episode_reward": 10.0})
        metrics = cb.get_metrics()
        assert len(metrics["episode_returns"]) == 1
