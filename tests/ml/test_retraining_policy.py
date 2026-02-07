"""Tests for RetrainingPolicy and RetrainingDecisionEngine."""

import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.tradingbots.training.retraining_policy import (
    RetrainingDecisionEngine,
    RetrainingPolicy,
)


@dataclass
class MockDriftAlert:
    """Mimics DriftAlert from drift_monitor."""
    severity: str = "warning"
    message: str = "test alert"


@pytest.fixture
def policy():
    return RetrainingPolicy(
        schedule_interval_days=7,
        drift_severity_threshold="warning",
        min_drift_alerts=2,
        cooldown_hours=24,
        min_sharpe_improvement=0.1,
        min_oos_performance=0.7,
        max_drawdown_increase=0.05,
    )


@pytest.fixture
def engine(policy):
    return RetrainingDecisionEngine(policy)


class TestShouldRetrain:
    def test_no_previous_training(self, engine):
        should, reason = engine.should_retrain([], last_retrain_time=None)
        assert should is True
        assert "No previous training" in reason

    def test_schedule_trigger(self, engine):
        old = datetime.utcnow() - timedelta(days=10)
        should, reason = engine.should_retrain([], last_retrain_time=old)
        assert should is True
        assert "Scheduled" in reason

    def test_schedule_not_yet_due(self, engine):
        recent = datetime.utcnow() - timedelta(days=3)
        should, reason = engine.should_retrain([], last_retrain_time=recent)
        assert should is False
        assert "No trigger" in reason

    def test_drift_alert_trigger(self, engine):
        recent = datetime.utcnow() - timedelta(days=3)
        alerts = [MockDriftAlert(severity="warning"), MockDriftAlert(severity="critical")]
        should, reason = engine.should_retrain(alerts, last_retrain_time=recent)
        assert should is True
        assert "Drift detected" in reason

    def test_drift_below_threshold(self, engine):
        recent = datetime.utcnow() - timedelta(days=3)
        alerts = [MockDriftAlert(severity="warning")]  # Only 1, need 2
        should, reason = engine.should_retrain(alerts, last_retrain_time=recent)
        assert should is False

    def test_cooldown_blocks_retrain(self, engine):
        very_recent = datetime.utcnow() - timedelta(hours=6)
        alerts = [MockDriftAlert(severity="critical")] * 5
        should, reason = engine.should_retrain(alerts, last_retrain_time=very_recent)
        assert should is False
        assert "Cooldown" in reason

    def test_critical_only_policy(self):
        policy = RetrainingPolicy(drift_severity_threshold="critical", min_drift_alerts=1)
        engine = RetrainingDecisionEngine(policy)
        recent = datetime.utcnow() - timedelta(days=3)

        # Warning alone shouldn't trigger
        should, _ = engine.should_retrain(
            [MockDriftAlert(severity="warning")],
            last_retrain_time=recent,
        )
        assert should is False

        # Critical should trigger
        should, _ = engine.should_retrain(
            [MockDriftAlert(severity="critical")],
            last_retrain_time=recent,
        )
        assert should is True

    def test_record_retrain(self, engine):
        engine.record_retrain()
        # Now cooldown should be active
        should, reason = engine.should_retrain([])
        assert should is False
        assert "Cooldown" in reason


class TestValidateCandidate:
    def test_candidate_passes(self, engine):
        old = {"sharpe_ratio": 1.0, "max_drawdown": 0.10}
        new = {"sharpe_ratio": 1.2, "max_drawdown": 0.11}
        passes, reason = engine.validate_candidate(old, new)
        assert passes is True
        assert "passes" in reason

    def test_insufficient_sharpe_improvement(self, engine):
        old = {"sharpe_ratio": 1.0, "max_drawdown": 0.10}
        new = {"sharpe_ratio": 1.05, "max_drawdown": 0.10}
        passes, reason = engine.validate_candidate(old, new)
        assert passes is False
        assert "Sharpe improvement" in reason

    def test_drawdown_increase_too_large(self, engine):
        old = {"sharpe_ratio": 1.0, "max_drawdown": 0.10}
        new = {"sharpe_ratio": 1.2, "max_drawdown": 0.20}
        passes, reason = engine.validate_candidate(old, new)
        assert passes is False
        assert "Drawdown" in reason

    def test_oos_performance_from_report_dict(self, engine):
        old = {"sharpe_ratio": 1.0, "max_drawdown": 0.10}
        new = {"sharpe_ratio": 1.2, "max_drawdown": 0.10}
        wf = {"oos_sharpe": 0.5}  # Below 0.7 threshold
        passes, reason = engine.validate_candidate(old, new, wf_report=wf)
        assert passes is False
        assert "OOS" in reason

    def test_oos_performance_from_report_object(self, engine):
        @dataclass
        class MockReport:
            oos_sharpe: float = 0.8

        old = {"sharpe_ratio": 1.0, "max_drawdown": 0.10}
        new = {"sharpe_ratio": 1.2, "max_drawdown": 0.10}
        passes, _ = engine.validate_candidate(old, new, wf_report=MockReport())
        assert passes is True

    def test_multiple_failures(self, engine):
        old = {"sharpe_ratio": 1.0, "max_drawdown": 0.10}
        new = {"sharpe_ratio": 0.9, "max_drawdown": 0.25}
        passes, reason = engine.validate_candidate(old, new)
        assert passes is False
        assert "Sharpe" in reason
        assert "Drawdown" in reason

    def test_empty_old_metrics(self, engine):
        """First model with no old metrics should pass easily."""
        old = {}
        new = {"sharpe_ratio": 0.5, "max_drawdown": 0.10}
        passes, reason = engine.validate_candidate(old, new)
        # sharpe improvement: 0.5 - 0 = 0.5 > 0.1 ✓
        # drawdown increase: 0.10 - 0 = 0.10 > 0.05 ✗
        assert passes is False  # drawdown gate still applies
