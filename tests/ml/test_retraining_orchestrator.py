"""Tests for RetrainingOrchestrator — end-to-end retraining pipeline."""

import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.tradingbots.training.model_registry import ModelRegistry
from ml.tradingbots.training.retraining_policy import RetrainingPolicy
from ml.tradingbots.training.retraining_orchestrator import (
    RetrainingOrchestrator,
    RetrainingResult,
)


@dataclass
class MockDriftAlert:
    severity: str = "critical"
    message: str = "test drift"


@dataclass
class MockMetrics:
    mean_reward: float = 100.0
    sharpe_ratio: float = 1.5
    max_drawdown: float = 0.08
    total_reward: float = 500.0


class MockAgent:
    """Minimal mock RL agent."""

    def __init__(self):
        self.saved_path = None

    def select_action(self, state, deterministic=False):
        return 0

    def save(self, path):
        self.saved_path = path
        with open(path, "w") as f:
            f.write("mock-checkpoint")

    def load(self, path):
        pass


class MockEnv:
    """Minimal mock environment."""

    def __init__(self):
        self._step = 0

    def reset(self):
        self._step = 0
        return [0.0] * 10

    def step(self, action):
        self._step += 1
        done = self._step >= 5
        return [0.0] * 10, 1.0, done, {}


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(str(tmp_path / "registry"))


@pytest.fixture
def policy():
    return RetrainingPolicy(
        schedule_interval_days=1,
        cooldown_hours=0,
        min_drift_alerts=1,
        min_sharpe_improvement=-999,  # Accept anything
        max_drawdown_increase=999,
    )


@pytest.fixture
def orchestrator(registry, policy):
    return RetrainingOrchestrator(
        registry=registry,
        policy=policy,
        state_dim=10,
        action_dim=3,
    )


class TestNoDriftSkip:
    def test_skips_when_no_trigger(self, registry):
        """With strict policy and recent retrain, should skip."""
        strict_policy = RetrainingPolicy(
            schedule_interval_days=30,
            cooldown_hours=24,
            min_drift_alerts=10,
        )
        orch = RetrainingOrchestrator(registry=registry, policy=strict_policy)

        # Register + promote an active model so there's a recent retrain time
        dummy = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        dummy.write(b"data")
        dummy.close()
        v = registry.register("ppo", dummy.name)
        registry.promote(v.version_id)

        result = orch.check_and_retrain("ppo", MockEnv)
        assert result.success is False
        os.unlink(dummy.name)


class TestFullPipeline:
    @patch("ml.tradingbots.training.retraining_orchestrator.RetrainingOrchestrator._train")
    def test_first_model_auto_promotes(self, mock_train, orchestrator):
        """First model with no active should auto-promote."""
        agent = MockAgent()
        mock_train.return_value = (agent, MockMetrics())

        result = orchestrator.check_and_retrain("ppo", MockEnv)
        assert result.success is True
        assert result.promoted is True
        assert result.new_version is not None
        assert orchestrator.registry.get_active() is not None

    @patch("ml.tradingbots.training.retraining_orchestrator.RetrainingOrchestrator._train")
    def test_retrain_with_existing_active(self, mock_train, orchestrator):
        """Retrain when an active model already exists."""
        agent = MockAgent()
        mock_train.return_value = (agent, MockMetrics())

        # First run — auto-promote
        result1 = orchestrator.check_and_retrain("ppo", MockEnv)
        assert result1.promoted is True

        # Simulate drift alerts so second run triggers
        orchestrator.drift_monitor = MagicMock()
        orchestrator.drift_monitor.get_drift_summary.return_value = {
            "recent_alerts": [MockDriftAlert(severity="critical")]
        }

        result2 = orchestrator.check_and_retrain("ppo", MockEnv)
        assert result2.new_version is not None


class TestShadowTest:
    def test_shadow_test_returns_metrics(self, orchestrator):
        candidate = MockAgent()
        env = MockEnv()

        with patch.object(orchestrator.registry, "load_agent", return_value=MockAgent()):
            result = orchestrator._shadow_test(
                candidate_agent=candidate,
                current_version_id="v_0001",
                env=env,
                n_episodes=3,
            )

        assert "candidate" in result
        assert "current" in result
        assert "sharpe_ratio" in result["candidate"]
        assert "mean_reward" in result["candidate"]

    def test_shadow_test_handles_load_failure(self, orchestrator):
        candidate = MockAgent()
        env = MockEnv()

        with patch.object(orchestrator.registry, "load_agent", side_effect=Exception("fail")):
            result = orchestrator._shadow_test(candidate, "v_0001", env)

        assert result["candidate"] == {}
        assert result["current"] == {}


class TestCandidateRejection:
    @patch("ml.tradingbots.training.retraining_orchestrator.RetrainingOrchestrator._train")
    def test_candidate_rejected_by_validation(self, mock_train, registry):
        """Strict policy rejects poor candidate."""
        strict_policy = RetrainingPolicy(
            schedule_interval_days=0,
            cooldown_hours=0,
            min_sharpe_improvement=10.0,  # Impossible to meet
            max_drawdown_increase=0.0,
        )
        orch = RetrainingOrchestrator(registry=registry, policy=strict_policy)

        # Create an active model first
        dummy = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        dummy.write(b"data")
        dummy.close()
        v = registry.register("ppo", dummy.name, validation_metrics={"sharpe_ratio": 5.0, "max_drawdown": 0.01})
        registry.promote(v.version_id)

        agent = MockAgent()
        mock_train.return_value = (agent, MockMetrics(sharpe_ratio=1.0))

        result = orch.check_and_retrain("ppo", MockEnv)
        assert result.promoted is False
        os.unlink(dummy.name)


class TestRetrainingResult:
    def test_dataclass_defaults(self):
        r = RetrainingResult(success=True, reason="ok")
        assert r.old_version is None
        assert r.new_version is None
        assert r.duration_seconds == 0.0
        assert r.promoted is False


class TestHelpers:
    def test_quick_sharpe(self):
        rewards = [10.0, 12.0, 11.0, 13.0, 9.0]
        sharpe = RetrainingOrchestrator._quick_sharpe(rewards)
        assert isinstance(sharpe, float)
        assert sharpe > 0

    def test_quick_sharpe_empty(self):
        assert RetrainingOrchestrator._quick_sharpe([]) == 0.0

    def test_quick_drawdown(self):
        rewards = [10.0, -5.0, 3.0, -8.0, 2.0]
        dd = RetrainingOrchestrator._quick_drawdown(rewards)
        assert dd > 0

    def test_quick_drawdown_empty(self):
        assert RetrainingOrchestrator._quick_drawdown([]) == 0.0
