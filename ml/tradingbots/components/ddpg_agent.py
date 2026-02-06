"""
Deep Deterministic Policy Gradient (DDPG) Agent for Trading

Continuous action actor-critic with:
- Deterministic policy gradient
- Ornstein-Uhlenbeck exploration noise
- Experience replay and target networks

Reference: stable-baselines3/stable_baselines3/ddpg/ddpg.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .rl_agents import ReplayBuffer, Experience


class OrnsteinUhlenbeckNoise:
    """Temporally correlated noise for smooth exploration."""

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self) -> None:
        self.state = self.mu.copy()

    def __call__(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            len(self.state)
        )
        self.state += dx
        return self.state


class DDPGActorNetwork(nn.Module):
    """Deterministic policy network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DDPGCriticNetwork(nn.Module):
    """Single Q-network for DDPG."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


@dataclass
class DDPGConfig:
    """Configuration for DDPG agent."""

    hidden_dim: int = 256
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005

    # OU Noise
    noise_theta: float = 0.15
    noise_sigma: float = 0.2

    buffer_size: int = 100000
    batch_size: int = 256
    min_replay_size: int = 1000
    total_timesteps: int = 100000


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient for continuous trading.

    Uses Ornstein-Uhlenbeck noise for temporally correlated exploration,
    suitable for sequential trading decisions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        config: Optional[DDPGConfig] = None,
    ):
        self.config = config or DDPGConfig()
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor + target
        self.actor = DDPGActorNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.actor_target = DDPGActorNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )

        # Critic + target
        self.critic = DDPGCriticNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.critic_target = DDPGCriticNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

        # OU Noise
        self.noise = OrnsteinUhlenbeckNoise(
            action_dim,
            theta=self.config.noise_theta,
            sigma=self.config.noise_sigma,
        )

        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        self.training_info: Dict[str, List[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "returns": [],
        }

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if not deterministic:
            action = np.clip(action + self.noise(), -1.0, 1.0)

        return action

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
                self.noise.reset()

            if len(self.replay_buffer) >= self.config.min_replay_size:
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

        # --- Critic update ---
        with torch.no_grad():
            next_action = self.actor_target(next_states)
            target_q = rewards + self.config.gamma * (1 - dones) * self.critic_target(
                next_states, next_action
            )

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(self.config.tau * p.data + (1 - self.config.tau) * tp.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.config.tau * p.data + (1 - self.config.tau) * tp.data)

        self.training_info["actor_loss"].append(actor_loss.item())
        self.training_info["critic_loss"].append(critic_loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": self.config,
                "training_info": self.training_info,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_info = checkpoint.get("training_info", {})
