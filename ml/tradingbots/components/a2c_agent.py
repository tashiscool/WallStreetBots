"""
Advantage Actor-Critic (A2C) Agent for Trading

Synchronous actor-critic, simpler than PPO:
- Single environment, no clipping
- N-step returns for advantage estimation
- Entropy bonus for exploration

Reference: stable-baselines3/stable_baselines3/a2c/a2c.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

from .rl_agents import RolloutBuffer


class A2CNetwork(nn.Module):
    """Shared actor-critic network for A2C."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = True,
    ):
        super().__init__()
        self.continuous = continuous

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        h = self.shared(state)
        value = self.critic(h)

        if self.continuous:
            mean = torch.tanh(self.actor_mean(h))
            std = self.actor_log_std.exp().expand_as(mean)
            return mean, std, value
        else:
            logits = self.actor(h)
            return logits, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(state)
        value = self.critic(h)

        if self.continuous:
            mean = torch.tanh(self.actor_mean(h))
            std = self.actor_log_std.exp().expand_as(mean)
            if deterministic:
                action = mean
                log_prob = torch.zeros_like(value)
            else:
                dist = Normal(mean, std)
                action = dist.sample().clamp(-1, 1)
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob, value
        else:
            logits = self.actor(h)
            if deterministic:
                action = logits.argmax(dim=-1)
                log_prob = torch.zeros(state.shape[0], 1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action).unsqueeze(-1)
            return action, log_prob, value


@dataclass
class A2CConfig:
    """Configuration for A2C agent."""

    hidden_dim: int = 256
    learning_rate: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 1.0  # A2C typically uses lambda=1
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 5
    normalize_advantage: bool = False
    total_timesteps: int = 100000


class A2CAgent:
    """
    Advantage Actor-Critic for trading decisions.

    Simpler and faster than PPO, good baseline for trading tasks.
    Uses n-step returns and entropy regularization.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        config: Optional[A2CConfig] = None,
        continuous: bool = True,
    ):
        self.config = config or A2CConfig()
        self.continuous = continuous
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = A2CNetwork(
            state_dim, action_dim, self.config.hidden_dim, continuous
        ).to(self.device)

        self.optimizer = optim.RMSprop(
            self.network.parameters(),
            lr=self.config.learning_rate,
            alpha=0.99,
            eps=1e-5,
        )

        self.buffer = RolloutBuffer()

        self.training_info: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "returns": [],
        }

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_t, deterministic)

        if self.continuous:
            return (
                action.cpu().numpy()[0],
                log_prob.cpu().item(),
                value.cpu().item(),
            )
        else:
            return (
                action.cpu().item(),
                log_prob.cpu().item(),
                value.cpu().item(),
            )

    def train(
        self,
        env,
        total_timesteps: Optional[int] = None,
        callback=None,
    ) -> Dict[str, List[float]]:
        total_timesteps = total_timesteps or self.config.total_timesteps
        timestep = 0
        episode_rewards: List[float] = []

        while timestep < total_timesteps:
            state = env.reset()
            episode_reward = 0.0
            done = False

            while not done and len(self.buffer.states) < self.config.n_steps:
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, info = env.step(action)

                self.buffer.add(state, action, reward, done, log_prob, value)
                state = next_state
                episode_reward += reward
                timestep += 1

                if callback:
                    callback(timestep, info)

            if done:
                episode_rewards.append(episode_reward)

            if len(self.buffer.states) >= self.config.n_steps:
                self._update(state, done)
                self.buffer.clear()

        self.training_info["returns"] = episode_rewards
        return self.training_info

    def _update(self, last_state: np.ndarray, last_done: bool) -> None:
        states, actions, rewards, dones, old_log_probs, values = self.buffer.get()

        # Bootstrap value for last state
        with torch.no_grad():
            last_state_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            _, _, last_value = self.network.get_action(last_state_t)
            last_value = 0.0 if last_done else last_value.cpu().item()

        # Compute returns and advantages
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_done = last_done if t == len(rewards) - 1 else dones[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            last_gae = advantages[t]

        returns = advantages + values

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        if self.config.normalize_advantage and len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Forward pass
        if self.continuous:
            actions_t = torch.FloatTensor(actions).to(self.device)
            h = self.network.shared(states_t)
            mean = torch.tanh(self.network.actor_mean(h))
            std = self.network.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions_t).sum(dim=-1)
            entropy = dist.entropy().mean()
            values_pred = self.network.critic(h).squeeze()
        else:
            actions_t = torch.LongTensor(actions).to(self.device)
            h = self.network.shared(states_t)
            logits = self.network.actor(h)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
            values_pred = self.network.critic(h).squeeze()

        policy_loss = -(log_probs * advantages_t).mean()
        value_loss = F.mse_loss(values_pred, returns_t)

        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.training_info["policy_loss"].append(policy_loss.item())
        self.training_info["value_loss"].append(value_loss.item())
        self.training_info["entropy"].append(entropy.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "training_info": self.training_info,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_info = checkpoint.get("training_info", {})
