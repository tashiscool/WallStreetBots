"""
Soft Actor-Critic (SAC) Agent for Trading

State-of-the-art for continuous portfolio allocation:
- Twin Q-networks for stable value estimation
- Entropy-regularized for exploration
- Auto-tuned temperature (alpha)

Reference: stable-baselines3/stable_baselines3/sac/sac.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from .rl_agents import ReplayBuffer, Experience


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class SACActorNetwork(nn.Module):
    """Squashed Gaussian policy for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with log probability (reparameterization trick)."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)

        # Log probability with squashing correction
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCriticNetwork(nn.Module):
    """Twin Q-networks for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


@dataclass
class SACConfig:
    """Configuration for SAC agent."""

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    buffer_size: int = 100000
    batch_size: int = 256
    min_replay_size: int = 1000
    total_timesteps: int = 100000
    gradient_steps: int = 1


class SACAgent:
    """
    Soft Actor-Critic for continuous portfolio allocation.

    Uses entropy regularization for robust exploration and twin
    Q-networks for stable critic estimation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        config: Optional[SACConfig] = None,
    ):
        self.config = config or SACConfig()
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.actor = SACActorNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )

        # Twin critics + target
        self.critic = SACCriticNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.critic_target = SACCriticNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )

        # Entropy temperature
        self.log_alpha = torch.tensor(
            np.log(self.config.alpha), dtype=torch.float32, device=self.device,
            requires_grad=self.config.auto_alpha,
        )
        self.target_entropy = -action_dim
        if self.config.auto_alpha:
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.config.learning_rate
            )

        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        self.training_info: Dict[str, List[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha": [],
            "returns": [],
        }

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_t)
        return action.cpu().numpy()[0]

    def train(
        self,
        env,
        total_timesteps: Optional[int] = None,
        callback=None,
    ) -> Dict[str, List[float]]:
        total_timesteps = total_timesteps or self.config.total_timesteps
        timestep = 0
        episode_rewards: List[float] = []
        state = env.reset()
        episode_reward = 0.0

        while timestep < total_timesteps:
            if len(self.replay_buffer) < self.config.min_replay_size:
                action = np.random.uniform(-1, 1, size=self.action_dim)
            else:
                action = self.select_action(state)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            self.replay_buffer.push(
                Experience(state, action, reward, next_state, done)
            )

            state = next_state
            timestep += 1

            if done:
                episode_rewards.append(episode_reward)
                state = env.reset()
                episode_reward = 0.0

            if len(self.replay_buffer) >= self.config.min_replay_size:
                for _ in range(self.config.gradient_steps):
                    self._update()

            if callback:
                callback(timestep, info)

        self.training_info["returns"] = episode_rewards
        return self.training_info

    def _update(self) -> None:
        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.FloatTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).unsqueeze(1).to(self.device)

        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
            target_q = rewards + self.config.gamma * (1 - dones) * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_action, log_prob = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha update ---
        if self.config.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update target
        for tp, p in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            tp.data.copy_(self.config.tau * p.data + (1 - self.config.tau) * tp.data)

        self.training_info["actor_loss"].append(actor_loss.item())
        self.training_info["critic_loss"].append(critic_loss.item())
        self.training_info["alpha"].append(self.alpha)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha,
                "config": self.config,
                "training_info": self.training_info,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha = checkpoint["log_alpha"]
        self.training_info = checkpoint.get("training_info", {})
