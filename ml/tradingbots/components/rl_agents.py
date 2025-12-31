"""
Reinforcement Learning Agents for Trading

PPO and DQN agents for dynamic position sizing and trade timing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

from .rl_environment import TradingEnvironment, TradingEnvConfig, ActionSpace


# =============================================================================
# Neural Network Architectures
# =============================================================================

class ActorCriticNetwork(nn.Module):
    """Shared network for PPO Actor-Critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = True,
    ):
        super().__init__()
        self.continuous = continuous

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution params and value."""
        shared_out = self.shared(state)

        if self.continuous:
            action_mean = torch.tanh(self.actor_mean(shared_out))  # [-1, 1]
            action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
            return action_mean, action_std, self.critic(shared_out)
        else:
            action_logits = self.actor(shared_out)
            return action_logits, self.critic(shared_out)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, and value from state."""
        shared_out = self.shared(state)
        value = self.critic(shared_out)

        if self.continuous:
            action_mean = torch.tanh(self.actor_mean(shared_out))
            action_std = torch.exp(self.actor_log_std).expand_as(action_mean)

            if deterministic:
                action = action_mean
                log_prob = torch.zeros_like(action)
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

            return action, log_prob, value
        else:
            action_logits = self.actor(shared_out)

            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
                log_prob = torch.zeros(action.shape[0], 1)
            else:
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action).unsqueeze(-1)

            return action, log_prob, value


class DQNetwork(nn.Module):
    """Deep Q-Network for discrete actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dueling: bool = True,
    ):
        super().__init__()
        self.dueling = dueling

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if dueling:
            # Dueling architecture
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )
        else:
            self.q_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all actions."""
        features = self.feature(state)

        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_layer(features)

        return q_values


# =============================================================================
# Experience Replay
# =============================================================================

@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: Any
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class RolloutBuffer:
    """Rollout buffer for PPO."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self) -> Tuple:
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.values),
        )

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []


# =============================================================================
# PPO Agent
# =============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    # Network
    hidden_dim: int = 256

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Rollout
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64

    # Training loop
    total_timesteps: int = 100000
    eval_freq: int = 1000


class PPOAgent:
    """
    Proximal Policy Optimization agent for continuous position sizing.

    Uses actor-critic architecture with clipped objective for stable training.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        config: Optional[PPOConfig] = None,
        continuous: bool = True,
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: Observation dimension
            action_dim: Action dimension
            config: PPO configuration
            continuous: Use continuous or discrete actions
        """
        self.config = config or PPOConfig()
        self.continuous = continuous
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        self.network = ActorCriticNetwork(
            state_dim,
            action_dim,
            self.config.hidden_dim,
            continuous,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training metrics
        self.training_info = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "returns": [],
        }

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action from policy.

        Args:
            state: Current state
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)

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
        env: TradingEnvironment,
        total_timesteps: Optional[int] = None,
        callback: Optional[callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the PPO agent.

        Args:
            env: Trading environment
            total_timesteps: Total training steps
            callback: Optional callback function

        Returns:
            Training metrics
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        timestep = 0
        episode_rewards = []

        while timestep < total_timesteps:
            # Collect rollout
            state = env.reset()
            episode_reward = 0
            done = False

            while not done and len(self.buffer.states) < self.config.n_steps:
                action, log_prob, value = self.select_action(state)

                # Environment step
                next_state, reward, done, info = env.step(action)

                # Store experience
                self.buffer.add(state, action, reward, done, log_prob, value)

                state = next_state
                episode_reward += reward
                timestep += 1

                if callback:
                    callback(timestep, info)

            if done:
                episode_rewards.append(episode_reward)

            # Update policy when buffer is full
            if len(self.buffer.states) >= self.config.n_steps:
                self._update()
                self.buffer.clear()

        self.training_info["returns"] = episode_rewards
        return self.training_info

    def _update(self) -> None:
        """Update policy using collected rollout."""
        states, actions, rewards, dones, old_log_probs, values = self.buffer.get()

        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, values, dones)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        if self.continuous:
            actions = torch.FloatTensor(actions).to(self.device)
        else:
            actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        for _ in range(self.config.n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Get current policy outputs
                _, new_log_probs, new_values = self.network.get_action(batch_states)

                if self.continuous:
                    # Recalculate log probs for continuous actions
                    shared_out = self.network.shared(batch_states)
                    action_mean = torch.tanh(self.network.actor_mean(shared_out))
                    action_std = torch.exp(self.network.actor_log_std).expand_as(action_mean)
                    dist = Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                    entropy = dist.entropy().mean()
                else:
                    shared_out = self.network.shared(batch_states)
                    action_logits = self.network.actor(shared_out)
                    dist = Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(batch_actions).unsqueeze(-1)
                    entropy = dist.entropy().mean()

                # Ratio for PPO
                ratio = torch.exp(new_log_probs.squeeze() - batch_old_log_probs)

                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # Track metrics
                self.training_info["policy_loss"].append(policy_loss.item())
                self.training_info["value_loss"].append(value_loss.item())
                self.training_info["entropy"].append(entropy.item())

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return returns, advantages

    def save(self, path: str) -> None:
        """Save agent to disk."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_info": self.training_info,
        }, path)

    def load(self, path: str) -> None:
        """Load agent from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_info = checkpoint.get("training_info", {})


# =============================================================================
# DQN Agent
# =============================================================================

@dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    # Network
    hidden_dim: int = 256
    dueling: bool = True

    # Training
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Replay
    buffer_size: int = 100000
    batch_size: int = 64
    min_replay_size: int = 1000

    # Training loop
    total_timesteps: int = 100000
    target_update_freq: int = 100


class DQNAgent:
    """
    Deep Q-Network agent for discrete trade timing decisions.

    Uses dueling architecture with target network for stable training.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[DQNConfig] = None,
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Observation dimension
            action_dim: Number of discrete actions
            config: DQN configuration
        """
        self.config = config or DQNConfig()
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_network = DQNetwork(
            state_dim,
            action_dim,
            self.config.hidden_dim,
            self.config.dueling,
        ).to(self.device)

        self.target_network = DQNetwork(
            state_dim,
            action_dim,
            self.config.hidden_dim,
            self.config.dueling,
        ).to(self.device)

        # Copy weights
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        # Exploration
        self.epsilon = self.config.epsilon_start

        # Training metrics
        self.training_info = {
            "loss": [],
            "q_values": [],
            "returns": [],
        }

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            deterministic: Use greedy policy

        Returns:
            Selected action
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=-1).item()

        return action

    def train(
        self,
        env: TradingEnvironment,
        total_timesteps: Optional[int] = None,
        callback: Optional[callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the DQN agent.

        Args:
            env: Trading environment
            total_timesteps: Total training steps
            callback: Optional callback function

        Returns:
            Training metrics
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        timestep = 0
        episode_rewards = []

        state = env.reset()
        episode_reward = 0

        while timestep < total_timesteps:
            # Select action
            action = self.select_action(state)

            # Environment step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Store experience
            self.replay_buffer.push(Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            ))

            state = next_state
            timestep += 1

            # Reset if done
            if done:
                episode_rewards.append(episode_reward)
                state = env.reset()
                episode_reward = 0

            # Update network
            if len(self.replay_buffer) >= self.config.min_replay_size:
                self._update()

            # Update target network
            if timestep % self.config.target_update_freq == 0:
                self._soft_update_target()

            # Decay epsilon
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay,
            )

            if callback:
                callback(timestep, info)

        self.training_info["returns"] = episode_rewards
        return self.training_info

    def _update(self) -> None:
        """Update Q-network using experience replay."""
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)

        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.config.gamma * next_q * (1 - dones)

        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # Track metrics
        self.training_info["loss"].append(loss.item())
        self.training_info["q_values"].append(current_q.mean().item())

    def _soft_update_target(self) -> None:
        """Soft update target network."""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters(),
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def save(self, path: str) -> None:
        """Save agent to disk."""
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "epsilon": self.epsilon,
            "training_info": self.training_info,
        }, path)

    def load(self, path: str) -> None:
        """Load agent from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.config.epsilon_end)
        self.training_info = checkpoint.get("training_info", {})


# =============================================================================
# Trading Agent Factory
# =============================================================================

def create_ppo_trading_agent(
    env: TradingEnvironment,
    config: Optional[PPOConfig] = None,
) -> PPOAgent:
    """Create PPO agent for continuous position sizing."""
    return PPOAgent(
        state_dim=env.observation_dim,
        action_dim=1,
        config=config,
        continuous=True,
    )


def create_dqn_trading_agent(
    env: TradingEnvironment,
    config: Optional[DQNConfig] = None,
) -> DQNAgent:
    """Create DQN agent for discrete trade timing."""
    return DQNAgent(
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        config=config,
    )
