"""
RL Training Callbacks

Lifecycle hooks for RL training: early stopping, checkpointing,
evaluation, and trading-specific metrics tracking.

Reference: stable-baselines3/stable_baselines3/common/callbacks.py
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class BaseCallback(ABC):
    """Base class for training callbacks."""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.n_calls = 0
        self.num_timesteps = 0
        self.training_env = None
        self.agent = None

    def init_callback(self, agent, env) -> None:
        self.agent = agent
        self.training_env = env

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        """Called at the start of training."""
        pass

    def on_step(self, locals_: Dict[str, Any]) -> bool:
        """Called after each environment step.

        Returns:
            False to stop training early, True to continue.
        """
        self.n_calls += 1
        self.num_timesteps = locals_.get("timestep", self.n_calls)
        return self._on_step(locals_)

    @abstractmethod
    def _on_step(self, locals_: Dict[str, Any]) -> bool:
        pass

    def on_episode_end(self, locals_: Dict[str, Any]) -> None:
        """Called at the end of each episode."""
        pass

    def on_training_end(self, locals_: Dict[str, Any]) -> None:
        """Called at the end of training."""
        pass


class CallbackList(BaseCallback):
    """Runs multiple callbacks in sequence."""

    def __init__(self, callbacks: List[BaseCallback]):
        super().__init__()
        self.callbacks = callbacks

    def init_callback(self, agent, env) -> None:
        super().init_callback(agent, env)
        for cb in self.callbacks:
            cb.init_callback(agent, env)

    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_training_start(locals_)

    def _on_step(self, locals_: Dict[str, Any]) -> bool:
        continue_training = True
        for cb in self.callbacks:
            if not cb.on_step(locals_):
                continue_training = False
        return continue_training

    def on_episode_end(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_episode_end(locals_)

    def on_training_end(self, locals_: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_training_end(locals_)


class EvalCallback(BaseCallback):
    """Evaluate the agent periodically and save the best model."""

    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.eval_history: List[Dict[str, float]] = []

    def _on_step(self, locals_: Dict[str, Any]) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        rewards = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = self.agent.select_action(obs, deterministic=self.deterministic)
                if isinstance(action, tuple):
                    action = action[0]
                obs, reward, done, _ = self.eval_env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        self.eval_history.append({
            "timestep": self.num_timesteps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        })

        if self.verbose >= 1:
            print(
                f"Eval @ step {self.num_timesteps}: "
                f"reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.best_model_save_path and hasattr(self.agent, "save"):
                path = os.path.join(self.best_model_save_path, "best_model.pt")
                os.makedirs(self.best_model_save_path, exist_ok=True)
                self.agent.save(path)
                if self.verbose >= 1:
                    print(f"  New best model saved (reward={mean_reward:.2f})")

        return True


class CheckpointCallback(BaseCallback):
    """Save model checkpoints at regular intervals."""

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "checkpoints",
        name_prefix: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self, locals_: Dict[str, Any]) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        if hasattr(self.agent, "save"):
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.pt",
            )
            self.agent.save(path)
            if self.verbose >= 1:
                print(f"Checkpoint saved: {path}")

        return True


class EarlyStoppingRLCallback(BaseCallback):
    """Stop training when reward plateaus."""

    def __init__(
        self,
        reward_threshold: Optional[float] = None,
        patience: int = 10,
        min_delta: float = 0.01,
        check_freq: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self._episode_rewards: List[float] = []
        self._best_mean_reward = -np.inf
        self._no_improvement_count = 0

    def on_episode_end(self, locals_: Dict[str, Any]) -> None:
        reward = locals_.get("episode_reward", 0.0)
        self._episode_rewards.append(reward)

    def _on_step(self, locals_: Dict[str, Any]) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        if len(self._episode_rewards) < 10:
            return True

        mean_reward = np.mean(self._episode_rewards[-100:])

        # Check absolute threshold
        if self.reward_threshold is not None and mean_reward >= self.reward_threshold:
            if self.verbose >= 1:
                print(
                    f"Early stopping: reward {mean_reward:.2f} >= "
                    f"threshold {self.reward_threshold:.2f}"
                )
            return False

        # Check improvement
        if mean_reward > self._best_mean_reward + self.min_delta:
            self._best_mean_reward = mean_reward
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            if self.verbose >= 1:
                print(
                    f"Early stopping: no improvement for "
                    f"{self.patience} checks (best={self._best_mean_reward:.2f})"
                )
            return False

        return True


class TradingMetricsCallback(BaseCallback):
    """Track trading-specific metrics during RL training."""

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_returns: List[float] = []
        self._episode_sharpes: List[float] = []
        self._episode_drawdowns: List[float] = []
        self._current_returns: List[float] = []

    def _on_step(self, locals_: Dict[str, Any]) -> bool:
        info = locals_.get("info", {})
        if "return" in info:
            self._current_returns.append(info["return"])

        if self.n_calls % self.log_freq == 0 and self.verbose >= 1:
            self._log_metrics()

        return True

    def on_episode_end(self, locals_: Dict[str, Any]) -> None:
        reward = locals_.get("episode_reward", 0.0)
        self._episode_returns.append(reward)

        if self._current_returns:
            returns = np.array(self._current_returns)
            sharpe = (
                np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            )
            self._episode_sharpes.append(sharpe)

            # Max drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / (running_max + 1e-8)
            self._episode_drawdowns.append(np.min(drawdowns))

        self._current_returns = []

    def _log_metrics(self) -> None:
        if not self._episode_returns:
            return

        recent = self._episode_returns[-50:]
        msg = f"Step {self.num_timesteps}: reward={np.mean(recent):.2f}"

        if self._episode_sharpes:
            msg += f", sharpe={np.mean(self._episode_sharpes[-50:]):.3f}"
        if self._episode_drawdowns:
            msg += f", max_dd={np.mean(self._episode_drawdowns[-50:]):.3f}"

        print(msg)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "episode_returns": self._episode_returns,
            "episode_sharpes": self._episode_sharpes,
            "episode_drawdowns": self._episode_drawdowns,
        }
