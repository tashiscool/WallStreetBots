"""
Retraining Policy — Business rules for when to retrain and how to validate candidates.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RetrainingPolicy:
    """Configuration for automated retraining decisions."""

    # Schedule-based triggers
    schedule_interval_days: int = 7

    # Drift-based triggers
    drift_severity_threshold: str = "warning"  # 'warning' or 'critical'
    min_drift_alerts: int = 2  # alerts within window before triggering

    # Cooldown
    cooldown_hours: int = 24  # minimum hours between retraining runs

    # Validation gates for promoting a candidate
    min_sharpe_improvement: float = 0.1
    min_oos_performance: float = 0.7  # min out-of-sample metric
    max_drawdown_increase: float = 0.05  # max additional drawdown allowed


class RetrainingDecisionEngine:
    """Decides *if* retraining should occur and *if* a candidate passes validation."""

    def __init__(self, policy: Optional[RetrainingPolicy] = None) -> None:
        self.policy = policy or RetrainingPolicy()
        self._last_retrain: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Trigger decision
    # ------------------------------------------------------------------

    def should_retrain(
        self,
        alerts: List[Any],
        last_retrain_time: Optional[datetime] = None,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """Determine whether retraining should be triggered.

        Args:
            alerts: List of ``DriftAlert`` objects from the drift monitor.
            last_retrain_time: When the last retraining completed.
            current_time: Override for testing (default ``datetime.utcnow()``).

        Returns:
            ``(should_retrain, reason)``
        """
        now = current_time or datetime.utcnow()
        last = last_retrain_time or self._last_retrain

        # Check cooldown
        if last is not None:
            elapsed = now - last
            if elapsed < timedelta(hours=self.policy.cooldown_hours):
                return False, f"Cooldown active ({elapsed} < {self.policy.cooldown_hours}h)"

        # Check schedule
        if last is not None:
            days_since = (now - last).days
            if days_since >= self.policy.schedule_interval_days:
                return True, f"Scheduled retrain ({days_since} days since last)"

        # Check drift alerts
        severity_order = {"warning": 1, "critical": 2}
        threshold_level = severity_order.get(self.policy.drift_severity_threshold, 1)

        qualifying = [
            a for a in alerts
            if severity_order.get(getattr(a, "severity", ""), 0) >= threshold_level
        ]

        if len(qualifying) >= self.policy.min_drift_alerts:
            return True, f"Drift detected ({len(qualifying)} alerts >= {self.policy.min_drift_alerts})"

        # No trigger
        if last is None:
            return True, "No previous training recorded"

        return False, "No trigger conditions met"

    # ------------------------------------------------------------------
    # Validation gate
    # ------------------------------------------------------------------

    def validate_candidate(
        self,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        wf_report: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """Validate a candidate model against the current active model.

        Args:
            old_metrics: Metrics from the current active model.
            new_metrics: Metrics from the candidate model.
            wf_report: Optional walk-forward report for OOS metrics.

        Returns:
            ``(passes, reason)``
        """
        reasons: List[str] = []

        # Sharpe improvement
        old_sharpe = old_metrics.get("sharpe_ratio", 0.0)
        new_sharpe = new_metrics.get("sharpe_ratio", 0.0)
        sharpe_diff = new_sharpe - old_sharpe

        if sharpe_diff < self.policy.min_sharpe_improvement:
            reasons.append(
                f"Sharpe improvement {sharpe_diff:.3f} < {self.policy.min_sharpe_improvement}"
            )

        # Max drawdown check
        old_dd = old_metrics.get("max_drawdown", 0.0)
        new_dd = new_metrics.get("max_drawdown", 0.0)
        dd_increase = new_dd - old_dd

        if dd_increase > self.policy.max_drawdown_increase:
            reasons.append(
                f"Drawdown increase {dd_increase:.3f} > {self.policy.max_drawdown_increase}"
            )

        # OOS performance from walk-forward report
        if wf_report is not None:
            oos_perf = getattr(wf_report, "oos_sharpe", None)
            if oos_perf is None:
                # Try dict-style access
                oos_perf = (
                    wf_report.get("oos_sharpe")
                    if isinstance(wf_report, dict)
                    else None
                )
            if oos_perf is not None and oos_perf < self.policy.min_oos_performance:
                reasons.append(
                    f"OOS performance {oos_perf:.3f} < {self.policy.min_oos_performance}"
                )

        if reasons:
            return False, "; ".join(reasons)

        return True, "Candidate passes all validation gates"

    def record_retrain(self, when: Optional[datetime] = None) -> None:
        """Record that a retraining was performed."""
        self._last_retrain = when or datetime.utcnow()
