"""
Retraining Orchestrator — End-to-end pipeline:
    drift_monitor → policy → train_rl_agent → walk_forward → registry
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .model_registry import ModelRegistry
from .retraining_policy import RetrainingDecisionEngine, RetrainingPolicy

logger = logging.getLogger(__name__)


@dataclass
class RetrainingResult:
    """Outcome of a retraining attempt."""

    success: bool
    reason: str
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    old_metrics: Dict[str, float] = field(default_factory=dict)
    new_metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    promoted: bool = False


class RetrainingOrchestrator:
    """Connects drift detection → retraining → validation → promotion."""

    def __init__(
        self,
        registry: ModelRegistry,
        policy: Optional[RetrainingPolicy] = None,
        drift_monitor: Optional[Any] = None,
        state_dim: int = 10,
        action_dim: int = 3,
    ) -> None:
        self.registry = registry
        self.engine = RetrainingDecisionEngine(policy)
        self.drift_monitor = drift_monitor
        self.state_dim = state_dim
        self.action_dim = action_dim

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check_and_retrain(
        self,
        agent_type: str,
        env_factory: Callable[[], Any],
        data_provider: Optional[Callable] = None,
        training_config: Optional[Any] = None,
    ) -> RetrainingResult:
        """Full pipeline: check drift → retrain → validate → promote/reject.

        Args:
            agent_type: RL agent type (e.g. 'ppo', 'dqn').
            env_factory: Callable returning a new environment instance.
            data_provider: Optional callable for refreshing training data.
            training_config: Optional ``RLTrainingConfig``.

        Returns:
            ``RetrainingResult`` describing the outcome.
        """
        start = time.time()

        # 1. Gather drift alerts
        alerts = self._get_drift_alerts()

        # 2. Decide whether to retrain
        current = self.registry.get_active()
        last_retrain = None
        if current:
            last_retrain = datetime.fromisoformat(current.created_at)

        should, reason = self.engine.should_retrain(
            alerts, last_retrain_time=last_retrain
        )

        if not should:
            return RetrainingResult(
                success=False,
                reason=reason,
                duration_seconds=time.time() - start,
            )

        logger.info("Retraining triggered: %s", reason)

        # 3. Refresh data if provider given
        if data_provider is not None:
            data_provider()

        # 4. Train new model
        try:
            agent, metrics = self._train(agent_type, env_factory, training_config)
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            return RetrainingResult(
                success=False,
                reason=f"Training error: {exc}",
                duration_seconds=time.time() - start,
            )

        new_metrics = self._extract_metrics(metrics)

        # 5. Save & register candidate
        import tempfile, os

        save_dir = tempfile.mkdtemp(prefix="wsb_retrain_")
        save_path = os.path.join(save_dir, "checkpoint.pt")
        agent.save(save_path)

        candidate = self.registry.register(
            agent_type=agent_type,
            model_path=save_path,
            training_metrics=new_metrics,
        )

        # 6. Shadow test if current active exists
        old_metrics: Dict[str, float] = {}
        if current is not None:
            old_metrics = current.validation_metrics or current.training_metrics
            env = env_factory()
            shadow_result = self._shadow_test(
                candidate_agent=agent,
                current_version_id=current.version_id,
                env=env,
            )
            candidate_metrics = shadow_result.get("candidate", new_metrics)
            current_metrics = shadow_result.get("current", old_metrics)
        else:
            candidate_metrics = new_metrics
            current_metrics = {}

        # 7. Validate candidate
        passes, val_reason = self.engine.validate_candidate(
            current_metrics, candidate_metrics
        )

        if passes or current is None:
            self.registry.promote(candidate.version_id)
            self.engine.record_retrain()
            return RetrainingResult(
                success=True,
                reason=val_reason if passes else "First model — auto-promoted",
                old_version=current.version_id if current else None,
                new_version=candidate.version_id,
                old_metrics=current_metrics,
                new_metrics=candidate_metrics,
                duration_seconds=time.time() - start,
                promoted=True,
            )
        else:
            logger.warning("Candidate rejected: %s", val_reason)
            return RetrainingResult(
                success=False,
                reason=val_reason,
                old_version=current.version_id if current else None,
                new_version=candidate.version_id,
                old_metrics=current_metrics,
                new_metrics=candidate_metrics,
                duration_seconds=time.time() - start,
                promoted=False,
            )

    # ------------------------------------------------------------------
    # Shadow testing
    # ------------------------------------------------------------------

    def _shadow_test(
        self,
        candidate_agent: Any,
        current_version_id: str,
        env: Any,
        n_episodes: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Run A/B comparison between candidate and current active.

        Returns dict with ``candidate`` and ``current`` metric dicts.
        """
        from .rl_training import evaluate_agent

        # Load current model
        try:
            current_agent = self.registry.load_agent(
                current_version_id,
                self.state_dim,
                self.action_dim,
            )
        except Exception as exc:
            logger.warning("Could not load current agent for shadow test: %s", exc)
            return {"candidate": {}, "current": {}}

        # Evaluate both
        candidate_rewards = self._evaluate_episodes(candidate_agent, env, n_episodes)
        current_rewards = self._evaluate_episodes(current_agent, env, n_episodes)

        candidate_metrics = {
            "mean_reward": sum(candidate_rewards) / max(len(candidate_rewards), 1),
            "sharpe_ratio": self._quick_sharpe(candidate_rewards),
            "max_drawdown": self._quick_drawdown(candidate_rewards),
        }
        current_metrics = {
            "mean_reward": sum(current_rewards) / max(len(current_rewards), 1),
            "sharpe_ratio": self._quick_sharpe(current_rewards),
            "max_drawdown": self._quick_drawdown(current_rewards),
        }

        logger.info(
            "Shadow test: candidate sharpe=%.3f  current sharpe=%.3f",
            candidate_metrics["sharpe_ratio"],
            current_metrics["sharpe_ratio"],
        )

        return {"candidate": candidate_metrics, "current": current_metrics}

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def register_with_scheduler(
        self,
        scheduler: Any,
        agent_type: str,
        env_factory: Callable[[], Any],
        data_provider: Optional[Callable] = None,
        name: str = "auto_retrain",
    ) -> None:
        """Register periodic retraining with a ``TradingScheduler``."""

        def _retrain_callback():
            return self.check_and_retrain(agent_type, env_factory, data_provider)

        interval = self.engine.policy.schedule_interval_days
        scheduler.schedule(
            name=name,
            date_rule=None,
            time_rule=None,
            callback=_retrain_callback,
        )
        logger.info(
            "Registered auto-retrain '%s' (every %d days)", name, interval
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_drift_alerts(self) -> List[Any]:
        if self.drift_monitor is None:
            return []
        summary = self.drift_monitor.get_drift_summary()
        return summary.get("recent_alerts", [])

    def _train(
        self, agent_type: str, env_factory: Callable, config: Optional[Any]
    ) -> Tuple[Any, Any]:
        from ..components.rl_agents import create_rl_agent
        from .rl_training import RLTrainingConfig, train_rl_agent

        env = env_factory()
        agent = create_rl_agent(agent_type, self.state_dim, self.action_dim)
        cfg = config or RLTrainingConfig(total_timesteps=10000)
        trained_agent, metrics = train_rl_agent(agent, env, cfg)
        return trained_agent, metrics

    def _extract_metrics(self, metrics: Any) -> Dict[str, float]:
        if isinstance(metrics, dict):
            return metrics
        result: Dict[str, float] = {}
        for attr in ("mean_reward", "sharpe_ratio", "max_drawdown", "total_reward"):
            val = getattr(metrics, attr, None)
            if val is not None:
                result[attr] = float(val)
        return result

    def _evaluate_episodes(
        self, agent: Any, env: Any, n_episodes: int
    ) -> List[float]:
        rewards: List[float] = []
        for _ in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            total = 0.0
            done = False
            steps = 0
            while not done and steps < 1000:
                action = agent.select_action(state, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
                result = env.step(action)
                if len(result) == 5:
                    state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    state, reward, done, _ = result
                total += float(reward)
                steps += 1
            rewards.append(total)
        return rewards

    @staticmethod
    def _quick_sharpe(rewards: List[float]) -> float:
        if len(rewards) < 2:
            return 0.0
        import math

        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
        std = math.sqrt(var) if var > 0 else 1e-8
        return mean / std

    @staticmethod
    def _quick_drawdown(rewards: List[float]) -> float:
        if not rewards:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in rewards:
            cumulative += r
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)
        return max_dd
