"""
Twin Delayed DDPG (TD3) Agent for Trading

Addresses overestimation bias in actor-critic methods:
- Twin critics (clipped double Q-learning)
- Delayed policy updates
- Target policy smoothing

Reference: stable-baselines3/stable_baselines3/td3/td3.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .rl_agents import ReplayBuffer, Experience


class TD3ActorNetwork(nn.Module):
    """Deterministic policy for TD3."""

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


class TD3CriticNetwork(nn.Module):
    """Twin Q-networks for TD3."""

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

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


@dataclass
class TD3Config:
    """Configuration for TD3 agent."""

    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    # TD3-specific
    policy_delay: int = 2
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    exploration_noise: float = 0.1

    buffer_size: int = 100000
    batch_size: int = 256
    min_replay_size: int = 1000
    total_timesteps: int = 100000


class TD3Agent:
    """
    Twin Delayed DDPG for continuous trading actions.

    Improves on DDPG with twin critics, delayed policy updates,
    and target policy smoothing for more stable training.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        config: Optional[TD3Config] = None,
    ):
        self.config = config or TD3Config()
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor + target
        self.actor = TD3ActorNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.actor_target = TD3ActorNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )

        # Twin critics + target
        self.critic = TD3CriticNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.critic_target = TD3CriticNetwork(
            state_dim, action_dim, self.config.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )

        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self._update_count = 0

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
            noise = np.random.normal(0, self.config.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)

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

            if len(self.replay_buffer) >= self.config.min_replay_size:
                self._update()

            if callback:
                callback(timestep, info)

        self.training_info["returns"] = episode_rewards
        return self.training_info

    def _update(self) -> None:
        self._update_count += 1
        batch = self.replay_buffer.sample(self.config.batch_size)

        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.FloatTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).unsqueeze(1).to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            # Target policy smoothing
            noise = (
                torch.randn_like(actions) * self.config.target_noise_std
            ).clamp(-self.config.target_noise_clip, self.config.target_noise_clip)
            next_action = (self.actor_target(next_states) + noise).clamp(-1, 1)

            q1_next, q2_next = self.critic_target(next_states, next_action)
            target_q = rewards + self.config.gamma * (1 - dones) * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.training_info["critic_loss"].append(critic_loss.item())

        # --- Delayed policy update ---
        if self._update_count % self.config.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.training_info["actor_loss"].append(actor_loss.item())

            # Soft update targets
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.copy_(self.config.tau * p.data + (1 - self.config.tau) * tp.data)
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(self.config.tau * p.data + (1 - self.config.tau) * tp.data)

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
