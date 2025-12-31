"""
Reinforcement Learning Training Utilities

Provides training loops with best practices for PPO and DQN agents:
- Progress tracking with live metrics
- Episode logging with trading statistics
- Periodic evaluation
- Model checkpointing
- Normalized observations
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class RLTrainingConfig:
    """Configuration for RL training."""
    # Training
    total_timesteps: int = 100000
    n_eval_episodes: int = 5
    eval_freq: int = 10000  # Evaluate every N steps

    # Logging
    log_freq: int = 1000  # Log every N steps
    verbose: bool = True
    progress_bar: bool = True

    # Checkpointing
    checkpoint_dir: str = "rl_checkpoints"
    save_freq: int = 10000

    # Normalization (critical for RL!)
    normalize_observations: bool = True
    normalize_rewards: bool = True
    clip_observations: float = 10.0
    clip_rewards: float = 10.0

    # Reward scaling
    reward_scale: float = 1.0

    # Reproducibility
    seed: Optional[int] = 42


@dataclass
class RLTrainingMetrics:
    """Tracks RL training metrics."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_sharpes: List[float] = field(default_factory=list)
    episode_drawdowns: List[float] = field(default_factory=list)

    eval_rewards: List[float] = field(default_factory=list)
    eval_sharpes: List[float] = field(default_factory=list)

    total_timesteps: int = 0
    total_episodes: int = 0
    training_time: float = 0.0

    def add_episode(
        self,
        reward: float,
        length: int,
        sharpe: Optional[float] = None,
        drawdown: Optional[float] = None,
    ) -> None:
        """Add episode results."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if sharpe is not None:
            self.episode_sharpes.append(sharpe)
        if drawdown is not None:
            self.episode_drawdowns.append(drawdown)
        self.total_episodes += 1

    def add_eval(
        self,
        mean_reward: float,
        mean_sharpe: Optional[float] = None,
    ) -> None:
        """Add evaluation results."""
        self.eval_rewards.append(mean_reward)
        if mean_sharpe is not None:
            self.eval_sharpes.append(mean_sharpe)

    @property
    def mean_reward_100(self) -> float:
        """Mean reward over last 100 episodes."""
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards[-100:])

    @property
    def best_eval_reward(self) -> float:
        """Best evaluation reward."""
        return max(self.eval_rewards) if self.eval_rewards else float('-inf')

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "mean_reward_100": self.mean_reward_100,
            "best_eval_reward": self.best_eval_reward,
            "mean_sharpe": np.mean(self.episode_sharpes) if self.episode_sharpes else 0,
            "best_sharpe": max(self.episode_sharpes) if self.episode_sharpes else 0,
            "training_time": self.training_time,
        }


class RunningMeanStd:
    """
    Welford's online algorithm for computing running mean and std.

    Critical for observation/reward normalization in RL.
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 1 else 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize input using running statistics."""
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


class RLProgressTracker:
    """Rich progress tracking for RL training."""

    def __init__(
        self,
        total_timesteps: int,
        desc: str = "Training RL Agent",
        use_tqdm: bool = True,
    ):
        self.total_timesteps = total_timesteps
        self.desc = desc
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.pbar = None

    def __enter__(self):
        if self.use_tqdm:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc=self.desc,
                unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        else:
            print("\n" + "=" * 80)
            print(f"  {self.desc}")
            print("=" * 80)
            print(f"{'Steps':>10} | {'Episodes':>8} | {'Mean Reward':>12} | "
                  f"{'Sharpe':>8} | {'FPS':>6}")
            print("-" * 80)
        return self

    def __exit__(self, *args):
        if self.pbar:
            self.pbar.close()
        else:
            print("-" * 80)
            print("Training complete!")
            print("=" * 80 + "\n")

    def update(
        self,
        timesteps: int,
        episodes: int,
        mean_reward: float,
        sharpe: Optional[float] = None,
        fps: Optional[float] = None,
        is_eval: bool = False,
    ) -> None:
        """Update progress display."""
        if self.use_tqdm:
            self.pbar.n = timesteps
            self.pbar.refresh()
            self.pbar.set_postfix({
                "ep": episodes,
                "reward": f"{mean_reward:.2f}",
                "sharpe": f"{sharpe:.3f}" if sharpe else "N/A",
            })
        else:
            status = "[EVAL]" if is_eval else ""
            sharpe_str = f"{sharpe:.4f}" if sharpe else "N/A"
            fps_str = f"{fps:.0f}" if fps else "N/A"
            print(f"{timesteps:>10} | {episodes:>8} | {mean_reward:>12.2f} | "
                  f"{sharpe_str:>8} | {fps_str:>6} {status}")

    def log(self, message: str) -> None:
        """Log a message."""
        if self.use_tqdm:
            tqdm.write(message)
        else:
            print(f"  >> {message}")


def evaluate_agent(
    agent: Any,
    env: Any,
    n_episodes: int = 5,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate an RL agent.

    Args:
        agent: Trained RL agent
        env: Trading environment
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    all_returns = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action = agent.select_action(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if 'return' in info:
                all_returns.append(info['return'])

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    # Calculate trading metrics
    returns = np.array(all_returns) if all_returns else np.array([])
    if len(returns) > 0:
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "sharpe_ratio": sharpe,
    }


def train_rl_agent(
    agent: Any,
    env: Any,
    config: RLTrainingConfig,
    eval_env: Optional[Any] = None,
) -> Tuple[Any, RLTrainingMetrics]:
    """
    Train an RL agent with all best practices.

    Features:
    - Observation normalization (critical for stable training)
    - Progress tracking with live metrics
    - Periodic evaluation
    - Model checkpointing
    - Trading-specific metrics (Sharpe, drawdown)

    Args:
        agent: RL agent (PPO or DQN)
        env: Training environment
        config: Training configuration
        eval_env: Optional separate evaluation environment

    Returns:
        Tuple of (trained_agent, training_metrics)
    """
    # Set seed
    if config.seed is not None:
        np.random.seed(config.seed)

    # Create checkpoint directory
    if config.checkpoint_dir:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Initialize normalization
    obs_rms = None
    reward_rms = None
    if config.normalize_observations:
        obs = env.reset()
        obs_rms = RunningMeanStd(shape=obs.shape)
    if config.normalize_rewards:
        reward_rms = RunningMeanStd(shape=())

    # Initialize metrics
    metrics = RLTrainingMetrics()
    start_time = time.time()

    # Training loop
    timestep = 0
    episode_reward = 0.0
    episode_length = 0
    episode_returns = []

    obs = env.reset()
    if obs_rms:
        obs_rms.update(obs.reshape(1, -1))
        obs = obs_rms.normalize(obs, config.clip_observations)

    with RLProgressTracker(
        config.total_timesteps,
        desc=f"Training {agent.__class__.__name__}",
        use_tqdm=config.progress_bar,
    ) as progress:

        while timestep < config.total_timesteps:
            # Select action
            action = agent.select_action(obs)

            # Step environment
            next_obs, reward, done, info = env.step(action)

            # Normalize observation
            if obs_rms:
                obs_rms.update(next_obs.reshape(1, -1))
                next_obs_norm = obs_rms.normalize(next_obs, config.clip_observations)
            else:
                next_obs_norm = next_obs

            # Scale and clip reward
            scaled_reward = reward * config.reward_scale
            if reward_rms:
                reward_rms.update(np.array([scaled_reward]))
                scaled_reward = reward_rms.normalize(
                    np.array([scaled_reward]),
                    config.clip_rewards
                )[0]

            # Store transition and update agent
            agent.store_transition(obs, action, scaled_reward, next_obs_norm, done)
            if hasattr(agent, 'update') and agent.can_update():
                agent.update()

            # Track episode metrics
            episode_reward += reward
            episode_length += 1
            if 'return' in info:
                episode_returns.append(info['return'])

            timestep += 1
            metrics.total_timesteps = timestep

            # Handle episode end
            if done:
                # Calculate trading metrics for episode
                if episode_returns:
                    returns = np.array(episode_returns)
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                    max_dd = 0.0
                    if len(returns) > 0:
                        cumulative = np.cumprod(1 + returns)
                        running_max = np.maximum.accumulate(cumulative)
                        drawdowns = (cumulative - running_max) / running_max
                        max_dd = np.min(drawdowns)
                else:
                    sharpe = None
                    max_dd = None

                metrics.add_episode(
                    reward=episode_reward,
                    length=episode_length,
                    sharpe=sharpe,
                    drawdown=max_dd,
                )

                # Reset for next episode
                episode_reward = 0.0
                episode_length = 0
                episode_returns = []
                obs = env.reset()
                if obs_rms:
                    obs_rms.update(obs.reshape(1, -1))
                    obs = obs_rms.normalize(obs, config.clip_observations)
            else:
                obs = next_obs_norm

            # Logging
            if timestep % config.log_freq == 0:
                fps = timestep / (time.time() - start_time)
                mean_sharpe = np.mean(metrics.episode_sharpes[-10:]) if metrics.episode_sharpes else None
                progress.update(
                    timesteps=timestep,
                    episodes=metrics.total_episodes,
                    mean_reward=metrics.mean_reward_100,
                    sharpe=mean_sharpe,
                    fps=fps,
                )

            # Evaluation
            if eval_env and timestep % config.eval_freq == 0:
                eval_results = evaluate_agent(agent, eval_env, config.n_eval_episodes)
                metrics.add_eval(
                    mean_reward=eval_results['mean_reward'],
                    mean_sharpe=eval_results['sharpe_ratio'],
                )
                progress.update(
                    timesteps=timestep,
                    episodes=metrics.total_episodes,
                    mean_reward=eval_results['mean_reward'],
                    sharpe=eval_results['sharpe_ratio'],
                    is_eval=True,
                )

            # Checkpointing
            if config.checkpoint_dir and timestep % config.save_freq == 0:
                if hasattr(agent, 'save'):
                    path = os.path.join(
                        config.checkpoint_dir,
                        f"agent_step{timestep}.pt"
                    )
                    agent.save(path)

    metrics.training_time = time.time() - start_time

    # Save final model
    if config.checkpoint_dir and hasattr(agent, 'save'):
        final_path = os.path.join(config.checkpoint_dir, "agent_final.pt")
        agent.save(final_path)

    return agent, metrics


def display_rl_training_summary(
    metrics: RLTrainingMetrics,
    agent_name: str = "RL Agent",
) -> None:
    """Display RL training summary."""
    summary = metrics.get_summary()

    print("\n" + "=" * 70)
    print(f"  RL TRAINING SUMMARY: {agent_name}")
    print("=" * 70)

    print("\n  Training Progress:")
    print(f"    Total timesteps:    {summary['total_timesteps']:,}")
    print(f"    Total episodes:     {summary['total_episodes']:,}")
    print(f"    Training time:      {summary['training_time']:.2f}s")
    print(f"    Steps/second:       {summary['total_timesteps'] / (summary['training_time'] + 1e-8):.0f}")

    print("\n  Performance Metrics:")
    print(f"    Mean reward (100):  {summary['mean_reward_100']:.2f}")
    print(f"    Best eval reward:   {summary['best_eval_reward']:.2f}")
    print(f"    Mean Sharpe:        {summary['mean_sharpe']:.4f}")
    print(f"    Best Sharpe:        {summary['best_sharpe']:.4f}")

    print("\n" + "=" * 70)
