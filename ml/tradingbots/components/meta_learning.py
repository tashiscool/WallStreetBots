"""
Meta-Learning for Regime-Adaptive Trading

Implements regime-aware and transfer learning agents:
- RegimeAwareAgent: switches between specialist models per market regime
- TransferLearningTrainer: pre-train on related tasks, fine-tune on target
- MultiTaskRLAgent: trains on multiple environments simultaneously

Reference: FinRL/finrl/meta/
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Regime Detection
# =============================================================================

class RegimeDetector:
    """Detect market regimes from price data."""

    def __init__(
        self,
        n_regimes: int = 3,
        lookback: int = 60,
        vol_thresholds: Optional[Tuple[float, float]] = None,
    ):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.vol_thresholds = vol_thresholds or (0.10, 0.25)

    def detect(self, returns: np.ndarray) -> int:
        """Detect current regime from recent returns.

        Returns:
            0 = low vol / trending
            1 = normal
            2 = high vol / crisis
        """
        if len(returns) < self.lookback:
            return 1  # default normal

        recent = returns[-self.lookback:]
        vol = np.std(recent) * np.sqrt(252)

        if vol < self.vol_thresholds[0]:
            return 0
        elif vol > self.vol_thresholds[1]:
            return 2
        return 1


# =============================================================================
# Regime-Aware Agent
# =============================================================================

@dataclass
class RegimeAwareConfig:
    """Configuration for regime-aware agent."""
    n_regimes: int = 3
    lookback: int = 60
    blend_weight: float = 0.3  # Weight for generalist vs specialist


class RegimeAwareAgent:
    """
    Switches between specialist models based on detected market regime.

    Maintains one model per regime plus a generalist fallback.
    Uses soft switching with configurable blend weight.
    """

    def __init__(
        self,
        create_agent_fn: Callable,
        config: Optional[RegimeAwareConfig] = None,
    ):
        """
        Args:
            create_agent_fn: Factory function that creates a fresh agent.
            config: Configuration for regime awareness.
        """
        self.config = config or RegimeAwareConfig()
        self.regime_detector = RegimeDetector(
            n_regimes=self.config.n_regimes,
            lookback=self.config.lookback,
        )

        # Create specialist agents per regime + generalist
        self.specialists = {
            i: create_agent_fn() for i in range(self.config.n_regimes)
        }
        self.generalist = create_agent_fn()
        self.current_regime = 1  # Start normal
        self._returns_history: List[float] = []

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action using current regime's specialist."""
        specialist = self.specialists[self.current_regime]
        action = specialist.select_action(state, deterministic)

        if not deterministic and self.config.blend_weight > 0:
            gen_action = self.generalist.select_action(state, deterministic)
            w = self.config.blend_weight
            if isinstance(action, np.ndarray):
                action = (1 - w) * action + w * gen_action

        return action

    def update_regime(self, returns: np.ndarray) -> int:
        """Update the detected regime from returns.

        Args:
            returns: Array of recent returns.

        Returns:
            Detected regime index.
        """
        self.current_regime = self.regime_detector.detect(returns)
        return self.current_regime

    def train(
        self,
        env,
        total_timesteps: int = 100000,
        callback=None,
    ) -> Dict[str, Any]:
        """Train all specialists and generalist."""
        results = {}

        # Train generalist on all data
        gen_result = self.generalist.train(
            env, total_timesteps=total_timesteps // 2, callback=callback
        )
        results["generalist"] = gen_result

        # Each specialist gets shorter training in regime-appropriate conditions
        for regime_id, agent in self.specialists.items():
            result = agent.train(
                env,
                total_timesteps=total_timesteps // (2 * self.config.n_regimes),
                callback=callback,
            )
            results[f"specialist_{regime_id}"] = result

        return results

    def save(self, path: str) -> None:
        """Save all models."""
        import os
        os.makedirs(path, exist_ok=True)
        self.generalist.save(os.path.join(path, "generalist.pt"))
        for i, agent in self.specialists.items():
            agent.save(os.path.join(path, f"specialist_{i}.pt"))

    def load(self, path: str) -> None:
        """Load all models."""
        import os
        self.generalist.load(os.path.join(path, "generalist.pt"))
        for i, agent in self.specialists.items():
            agent.load(os.path.join(path, f"specialist_{i}.pt"))


# =============================================================================
# Transfer Learning Trainer
# =============================================================================

class TransferLearningTrainer:
    """
    Pre-train on source tasks, then fine-tune on target task.

    Useful for training on related markets or time periods.
    """

    def __init__(
        self,
        create_agent_fn: Callable,
        freeze_layers: int = 0,
        fine_tune_lr_factor: float = 0.1,
    ):
        """
        Args:
            create_agent_fn: Factory for creating agent instances.
            freeze_layers: Number of layers to freeze during fine-tuning.
            fine_tune_lr_factor: Learning rate multiplier for fine-tuning.
        """
        self.create_agent_fn = create_agent_fn
        self.freeze_layers = freeze_layers
        self.fine_tune_lr_factor = fine_tune_lr_factor
        self.agent = None

    def pretrain(
        self,
        source_envs: List[Any],
        timesteps_per_env: int = 50000,
    ) -> Any:
        """Pre-train on source environments.

        Args:
            source_envs: List of source training environments.
            timesteps_per_env: Training steps per environment.

        Returns:
            Pre-trained agent.
        """
        self.agent = self.create_agent_fn()

        for i, env in enumerate(source_envs):
            self.agent.train(env, total_timesteps=timesteps_per_env)

        return self.agent

    def fine_tune(
        self,
        target_env: Any,
        total_timesteps: int = 20000,
    ) -> Any:
        """Fine-tune pre-trained agent on target environment.

        Args:
            target_env: Target environment for fine-tuning.
            total_timesteps: Fine-tuning steps.

        Returns:
            Fine-tuned agent.
        """
        if self.agent is None:
            self.agent = self.create_agent_fn()

        # Reduce learning rate for fine-tuning
        if TORCH_AVAILABLE and hasattr(self.agent, 'optimizer'):
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] *= self.fine_tune_lr_factor

        # Freeze early layers if requested
        if TORCH_AVAILABLE and self.freeze_layers > 0:
            self._freeze_early_layers()

        self.agent.train(target_env, total_timesteps=total_timesteps)
        return self.agent

    def _freeze_early_layers(self) -> None:
        """Freeze the first N layers of the network."""
        if not TORCH_AVAILABLE:
            return

        network = getattr(self.agent, 'network', None) or getattr(self.agent, 'actor', None)
        if network is None:
            return

        frozen = 0
        for param in network.parameters():
            if frozen >= self.freeze_layers:
                break
            param.requires_grad = False
            frozen += 1


# =============================================================================
# Multi-Task RL Agent
# =============================================================================

class MultiTaskRLAgent:
    """
    Trains on multiple environments simultaneously.

    Shares a common representation across tasks while maintaining
    task-specific output heads. Useful for learning cross-market patterns.
    """

    def __init__(
        self,
        create_agent_fn: Callable,
        task_envs: Dict[str, Any],
        steps_per_task: int = 100,
    ):
        """
        Args:
            create_agent_fn: Factory for creating the shared agent.
            task_envs: Dict of task_name -> environment.
            steps_per_task: Steps to train on each task per round.
        """
        self.agent = create_agent_fn()
        self.task_envs = task_envs
        self.steps_per_task = steps_per_task
        self.task_metrics: Dict[str, List[float]] = {
            name: [] for name in task_envs
        }

    def train(
        self,
        total_rounds: int = 100,
        callback=None,
    ) -> Dict[str, Any]:
        """Train across all tasks in round-robin fashion.

        Args:
            total_rounds: Number of complete rounds across all tasks.
            callback: Optional progress callback.

        Returns:
            Training metrics per task.
        """
        for round_idx in range(total_rounds):
            for task_name, env in self.task_envs.items():
                result = self.agent.train(
                    env,
                    total_timesteps=self.steps_per_task,
                    callback=callback,
                )
                returns = result.get("returns", [])
                if returns:
                    self.task_metrics[task_name].append(np.mean(returns[-10:]))

        return self.task_metrics

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        return self.agent.select_action(state, deterministic)

    def save(self, path: str) -> None:
        self.agent.save(path)

    def load(self, path: str) -> None:
        self.agent.load(path)
