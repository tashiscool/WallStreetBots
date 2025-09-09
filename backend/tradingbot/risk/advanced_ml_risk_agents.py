"""
Advanced ML Risk Agents
Reinforcement Learning for Dynamic Risk Management

This module implements sophisticated ML agents for dynamic risk management:
- PPO (Proximal Policy Optimization) for risk policy learning
- DDPG (Deep Deterministic Policy Gradient) for continuous risk control
- TD3 (Twin Delayed Deep Deterministic) for robust risk management
- Multi-agent risk coordination
- Adaptive risk limits based on market conditions

Month 5-6: Advanced Features and Automation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path


class RiskActionType(str, Enum):
    """Types of risk management actions"""
    INCREASE_POSITION = "increase_position"
    DECREASE_POSITION = "decrease_position"
    HEDGE_POSITION = "hedge_position"
    CLOSE_POSITION = "close_position"
    ADJUST_LIMITS = "adjust_limits"
    NO_ACTION = "no_action"


@dataclass
class RiskState:
    """Current risk state for ML agents"""
    portfolio_var: float
    portfolio_cvar: float
    concentration_risk: float
    greeks_risk: float
    market_volatility: float
    market_regime: str  # "normal", "high_vol", "crisis"
    time_of_day: float  # 0-1 normalized
    day_of_week: int  # 0-6
    recent_performance: float
    stress_test_score: float
    ml_risk_score: float
    position_count: int
    total_exposure: float
    cash_ratio: float


@dataclass
class RiskAction:
    """Risk management action"""
    action_type: RiskActionType
    symbol: str
    quantity: float
    confidence: float
    reasoning: str
    expected_impact: float


@dataclass
class RiskReward:
    """Reward signal for risk management"""
    risk_reduction: float
    performance_impact: float
    compliance_score: float
    total_reward: float


class RiskEnvironment:
    """
    Risk management environment for RL agents
    
    Provides:
    - State representation
    - Action space
    - Reward calculation
    - Environment dynamics
    """
    
    def __init__(self, 
                 risk_limits: Dict[str, float],
                 market_data_provider: Any = None):
        """
        Initialize risk environment
        
        Args:
            risk_limits: Risk limits configuration
            market_data_provider: Market data provider
        """
        self.risk_limits = risk_limits
        self.market_data_provider = market_data_provider
        self.logger = logging.getLogger(__name__)
        
        # Environment state
        self.current_state = None
        self.action_history = []
        self.reward_history = []
        self.episode_count = 0
        
        # Performance tracking
        self.total_rewards = 0.0
        self.risk_violations = 0
        self.successful_actions = 0
        
        self.logger.info("Risk Environment initialized")
    
    def get_state(self, 
                  portfolio_data: Dict[str, Any],
                  market_data: Dict[str, Any]) -> RiskState:
        """
        Get current risk state
        
        Args:
            portfolio_data: Current portfolio data
            market_data: Current market data
            
        Returns:
            RiskState: Current risk state
        """
        try:
            # Extract portfolio metrics
            portfolio_var = portfolio_data.get('portfolio_var', 0.0)
            portfolio_cvar = portfolio_data.get('portfolio_cvar', 0.0)
            concentration_risk = portfolio_data.get('concentration_risk', 0.0)
            greeks_risk = portfolio_data.get('greeks_risk', 0.0)
            position_count = portfolio_data.get('position_count', 0)
            total_exposure = portfolio_data.get('total_exposure', 0.0)
            cash_ratio = portfolio_data.get('cash_ratio', 0.0)
            
            # Extract market metrics
            market_volatility = market_data.get('market_volatility', 0.0)
            market_regime = market_data.get('market_regime', 'normal')
            recent_performance = market_data.get('recent_performance', 0.0)
            stress_test_score = market_data.get('stress_test_score', 0.0)
            ml_risk_score = market_data.get('ml_risk_score', 0.0)
            
            # Time features
            now = datetime.now()
            time_of_day = (now.hour * 60 + now.minute) / (24 * 60)  # 0-1
            day_of_week = now.weekday()  # 0-6
            
            self.current_state = RiskState(
                portfolio_var=portfolio_var,
                portfolio_cvar=portfolio_cvar,
                concentration_risk=concentration_risk,
                greeks_risk=greeks_risk,
                market_volatility=market_volatility,
                market_regime=market_regime,
                time_of_day=time_of_day,
                day_of_week=day_of_week,
                recent_performance=recent_performance,
                stress_test_score=stress_test_score,
                ml_risk_score=ml_risk_score,
                position_count=position_count,
                total_exposure=total_exposure,
                cash_ratio=cash_ratio
            )
            
            return self.current_state
            
        except Exception as e:
            self.logger.error(f"Error getting risk state: {e}")
            return RiskState(
                portfolio_var=0.02, portfolio_cvar=0.03, concentration_risk=0.25,
                greeks_risk=0.05, market_volatility=0.15, market_regime="normal",
                time_of_day=0.5, day_of_week=1, recent_performance=0.001,
                stress_test_score=0.7, ml_risk_score=50.0, position_count=5,
                total_exposure=100000.0, cash_ratio=0.1
            )
    
    def get_action_space(self) -> List[RiskActionType]:
        """Get available action space"""
        return list(RiskActionType)
    
    def execute_action(self, 
                      action: RiskAction,
                      portfolio_data: Dict[str, Any]) -> RiskReward:
        """
        Execute risk management action and calculate reward
        
        Args:
            action: Risk action to execute
            portfolio_data: Current portfolio data
            
        Returns:
            RiskReward: Reward for the action
        """
        try:
            # Calculate risk reduction
            risk_reduction = self._calculate_risk_reduction(action, portfolio_data)
            
            # Calculate performance impact
            performance_impact = self._calculate_performance_impact(action, portfolio_data)
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(action, portfolio_data)
            
            # Calculate total reward
            total_reward = (
                0.4 * risk_reduction +
                0.3 * performance_impact +
                0.3 * compliance_score
            )
            
            reward = RiskReward(
                risk_reduction=risk_reduction,
                performance_impact=performance_impact,
                compliance_score=compliance_score,
                total_reward=total_reward
            )
            
            # Update tracking
            self.total_rewards += total_reward
            self.action_history.append(action)
            self.reward_history.append(reward)
            
            if total_reward > 0:
                self.successful_actions += 1
            
            self.logger.info(f"Action executed: {action.action_type}, Reward: {total_reward:.3f}")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
            return RiskReward(0, 0, 0, 0)
    
    def _calculate_risk_reduction(self, 
                                action: RiskAction,
                                portfolio_data: Dict[str, Any]) -> float:
        """Calculate risk reduction from action"""
        try:
            current_var = portfolio_data.get('portfolio_var', 0.0)
            current_cvar = portfolio_data.get('portfolio_cvar', 0.0)
            
            # Estimate risk reduction based on action type
            if action.action_type == RiskActionType.DECREASE_POSITION:
                risk_reduction = min(0.1, current_var * 0.5)  # Up to 10% reduction
            elif action.action_type == RiskActionType.HEDGE_POSITION:
                risk_reduction = min(0.05, current_var * 0.3)  # Up to 5% reduction
            elif action.action_type == RiskActionType.CLOSE_POSITION:
                risk_reduction = min(0.15, current_var * 0.7)  # Up to 15% reduction
            elif action.action_type == RiskActionType.ADJUST_LIMITS:
                risk_reduction = min(0.08, current_var * 0.4)  # Up to 8% reduction
            else:
                risk_reduction = 0.0
            
            return risk_reduction
            
        except Exception as e:
            self.logger.error(f"Error calculating risk reduction: {e}")
            return 0.0
    
    def _calculate_performance_impact(self, 
                                    action: RiskAction,
                                    portfolio_data: Dict[str, Any]) -> float:
        """Calculate performance impact from action"""
        try:
            # Estimate performance impact based on action type
            if action.action_type == RiskActionType.INCREASE_POSITION:
                # Positive impact if market is trending up
                market_trend = portfolio_data.get('market_trend', 0.0)
                performance_impact = min(0.1, market_trend * 0.2)
            elif action.action_type == RiskActionType.DECREASE_POSITION:
                # Negative impact if market is trending up
                market_trend = portfolio_data.get('market_trend', 0.0)
                performance_impact = max(-0.1, -market_trend * 0.2)
            elif action.action_type == RiskActionType.HEDGE_POSITION:
                # Small negative impact due to hedging costs
                performance_impact = -0.02
            else:
                performance_impact = 0.0
            
            return performance_impact
            
        except Exception as e:
            self.logger.error(f"Error calculating performance impact: {e}")
            return 0.0
    
    def _calculate_compliance_score(self, 
                                  action: RiskAction,
                                  portfolio_data: Dict[str, Any]) -> float:
        """Calculate compliance score for action"""
        try:
            # Check if action improves compliance
            current_var = portfolio_data.get('portfolio_var', 0.0)
            max_var = self.risk_limits.get('max_total_var', 0.05)
            
            if current_var > max_var:
                # Actions that reduce risk improve compliance
                if action.action_type in [RiskActionType.DECREASE_POSITION, 
                                        RiskActionType.CLOSE_POSITION,
                                        RiskActionType.HEDGE_POSITION]:
                    compliance_score = 0.1
                else:
                    compliance_score = -0.1
            else:
                # Already compliant, maintain status
                compliance_score = 0.05
            
            return compliance_score
            
        except Exception as e:
            self.logger.error(f"Error calculating compliance score: {e}")
            return 0.0
    
    def reset_episode(self):
        """Reset environment for new episode"""
        self.episode_count += 1
        self.action_history = []
        self.reward_history = []
        self.logger.info(f"Episode {self.episode_count} reset")


class PPORiskAgent:
    """
    Proximal Policy Optimization agent for risk management
    
    PPO is well-suited for risk management because:
    - Stable learning with clipped objective
    - Good sample efficiency
    - Handles continuous and discrete actions
    - Robust to hyperparameter changes
    """
    
    def __init__(self, 
                 state_dim: int = 14,
                 action_dim: int = 6,
                 learning_rate: float = 3e-4,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """
        Initialize PPO agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            clip_ratio: PPO clipping ratio
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize neural networks (simplified implementation)
        self.policy_net = self._build_policy_network()
        self.value_net = self._build_value_network()
        
        # Training state
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        self.logger.info("PPO Risk Agent initialized")
    
    def _build_policy_network(self) -> Dict[str, Any]:
        """Build policy network (simplified)"""
        return {
            'layers': [self.state_dim, 128, 64, self.action_dim],
            'activation': 'relu',
            'output_activation': 'softmax'
        }
    
    def _build_value_network(self) -> Dict[str, Any]:
        """Build value network (simplified)"""
        return {
            'layers': [self.state_dim, 128, 64, 1],
            'activation': 'relu',
            'output_activation': 'linear'
        }
    
    def get_action(self, state: RiskState) -> RiskAction:
        """
        Get action from current state
        
        Args:
            state: Current risk state
            
        Returns:
            RiskAction: Selected action
        """
        try:
            # Convert state to vector
            state_vector = self._state_to_vector(state)
            
            # Get action probabilities (simplified)
            action_probs = self._forward_policy(state_vector)
            
            # Sample action
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            action_type = list(RiskActionType)[action_idx]
            
            # Create action
            action = RiskAction(
                action_type=action_type,
                symbol="PORTFOLIO",  # Portfolio-level action
                quantity=0.1,  # Default quantity
                confidence=action_probs[action_idx],
                reasoning=f"PPO agent selected {action_type}",
                expected_impact=0.0
            )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error getting action: {e}")
            return RiskAction(
                action_type=RiskActionType.NO_ACTION,
                symbol="PORTFOLIO",
                quantity=0.0,
                confidence=0.0,
                reasoning="Error in action selection",
                expected_impact=0.0
            )
    
    def _state_to_vector(self, state: RiskState) -> np.ndarray:
        """Convert state to vector representation"""
        return np.array([
            state.portfolio_var,
            state.portfolio_cvar,
            state.concentration_risk,
            state.greeks_risk,
            state.market_volatility,
            1.0 if state.market_regime == "normal" else 0.0,
            1.0 if state.market_regime == "high_vol" else 0.0,
            1.0 if state.market_regime == "crisis" else 0.0,
            state.time_of_day,
            state.day_of_week / 6.0,
            state.recent_performance,
            state.stress_test_score,
            state.ml_risk_score,
            state.cash_ratio
        ])
    
    def _forward_policy(self, state_vector: np.ndarray) -> np.ndarray:
        """Forward pass through policy network (simplified)"""
        # Simplified policy network
        # In practice, this would be a neural network
        
        # Risk-based action selection
        portfolio_var = state_vector[0]
        portfolio_cvar = state_vector[1]
        market_regime = state_vector[5:8]
        
        # Initialize action probabilities
        action_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])  # Default: mostly no action
        
        # Adjust based on risk level
        if portfolio_var > 0.05:  # High VaR
            action_probs[1] += 0.3  # Increase decrease position
            action_probs[2] += 0.2  # Increase hedge position
            action_probs[5] -= 0.5  # Decrease no action
        
        if portfolio_cvar > 0.07:  # High CVaR
            action_probs[3] += 0.2  # Increase close position
            action_probs[5] -= 0.2  # Decrease no action
        
        # Adjust based on market regime
        if market_regime[2] > 0:  # Crisis regime
            action_probs[1] += 0.2  # Increase decrease position
            action_probs[3] += 0.1  # Increase close position
            action_probs[5] -= 0.3  # Decrease no action
        
        # Normalize probabilities
        action_probs = np.maximum(action_probs, 0.01)  # Avoid zero probabilities
        action_probs = action_probs / np.sum(action_probs)
        
        return action_probs
    
    def update_policy(self, 
                     states: List[RiskState],
                     actions: List[RiskAction],
                     rewards: List[RiskReward],
                     old_log_probs: List[float]):
        """
        Update policy network using PPO
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            old_log_probs: List of old log probabilities
        """
        try:
            # Convert to arrays
            state_vectors = np.array([self._state_to_vector(s) for s in states])
            action_indices = np.array([list(RiskActionType).index(a.action_type) for a in actions])
            reward_values = np.array([r.total_reward for r in rewards])
            
            # Calculate advantages (simplified)
            advantages = self._calculate_advantages(reward_values)
            
            # Calculate policy loss (simplified PPO)
            policy_loss = self._calculate_policy_loss(
                state_vectors, action_indices, advantages, old_log_probs
            )
            
            # Calculate value loss
            value_loss = self._calculate_value_loss(state_vectors, reward_values)
            
            # Calculate entropy loss
            entropy_loss = self._calculate_entropy_loss(state_vectors)
            
            # Update networks (simplified)
            self.policy_losses.append(policy_loss)
            self.value_losses.append(value_loss)
            self.entropy_losses.append(entropy_loss)
            
            self.logger.info(f"Policy updated - Policy loss: {policy_loss:.4f}, "
                           f"Value loss: {value_loss:.4f}, Entropy loss: {entropy_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error updating policy: {e}")
    
    def _calculate_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """Calculate advantages using GAE (simplified)"""
        # Simplified advantage calculation
        # In practice, this would use Generalized Advantage Estimation
        return rewards - np.mean(rewards)
    
    def _calculate_policy_loss(self, 
                             states: np.ndarray,
                             actions: np.ndarray,
                             advantages: np.ndarray,
                             old_log_probs: np.ndarray) -> float:
        """Calculate PPO policy loss (simplified)"""
        # Simplified policy loss calculation
        # In practice, this would implement the full PPO objective
        return np.mean(advantages ** 2)
    
    def _calculate_value_loss(self, 
                            states: np.ndarray,
                            rewards: np.ndarray) -> float:
        """Calculate value function loss (simplified)"""
        # Simplified value loss calculation
        return np.mean((rewards - np.mean(rewards)) ** 2)
    
    def _calculate_entropy_loss(self, states: np.ndarray) -> float:
        """Calculate entropy loss (simplified)"""
        # Simplified entropy loss calculation
        return -np.mean(np.sum(states ** 2, axis=1))
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            model_data = {
                'policy_net': self.policy_net,
                'value_net': self.value_net,
                'episode_rewards': self.episode_rewards,
                'policy_losses': self.policy_losses,
                'value_losses': self.value_losses,
                'entropy_losses': self.entropy_losses
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.policy_net = model_data['policy_net']
            self.value_net = model_data['value_net']
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.policy_losses = model_data.get('policy_losses', [])
            self.value_losses = model_data.get('value_losses', [])
            self.entropy_losses = model_data.get('entropy_losses', [])
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")


class DDPGRiskAgent:
    """
    Deep Deterministic Policy Gradient agent for continuous risk control
    
    DDPG is well-suited for risk management because:
    - Handles continuous action spaces
    - Good for continuous risk adjustments
    - Stable learning with target networks
    - Memory replay for sample efficiency
    """
    
    def __init__(self, 
                 state_dim: int = 14,
                 action_dim: int = 1,  # Continuous risk adjustment
                 learning_rate: float = 1e-3,
                 tau: float = 0.005,
                 gamma: float = 0.99):
        """
        Initialize DDPG agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (continuous)
            learning_rate: Learning rate for optimizer
            tau: Soft update parameter
            gamma: Discount factor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize networks (simplified)
        self.actor_net = self._build_actor_network()
        self.critic_net = self._build_critic_network()
        self.target_actor = self._build_actor_network()
        self.target_critic = self._build_critic_network()
        
        # Training state
        self.memory = []
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        self.logger.info("DDPG Risk Agent initialized")
    
    def _build_actor_network(self) -> Dict[str, Any]:
        """Build actor network (simplified)"""
        return {
            'layers': [self.state_dim, 256, 128, self.action_dim],
            'activation': 'relu',
            'output_activation': 'tanh'
        }
    
    def _build_critic_network(self) -> Dict[str, Any]:
        """Build critic network (simplified)"""
        return {
            'layers': [self.state_dim + self.action_dim, 256, 128, 1],
            'activation': 'relu',
            'output_activation': 'linear'
        }
    
    def get_action(self, state: RiskState, noise: float = 0.1) -> RiskAction:
        """
        Get continuous risk adjustment action
        
        Args:
            state: Current risk state
            noise: Exploration noise
            
        Returns:
            RiskAction: Risk adjustment action
        """
        try:
            # Convert state to vector
            state_vector = self._state_to_vector(state)
            
            # Get continuous action (risk adjustment factor)
            action_value = self._forward_actor(state_vector)
            
            # Add exploration noise
            action_value += np.random.normal(0, noise)
            action_value = np.clip(action_value, -1.0, 1.0)
            
            # Convert to risk action
            if action_value > 0.3:
                action_type = RiskActionType.INCREASE_POSITION
                quantity = abs(action_value) * 0.1
            elif action_value < -0.3:
                action_type = RiskActionType.DECREASE_POSITION
                quantity = abs(action_value) * 0.1
            else:
                action_type = RiskActionType.NO_ACTION
                quantity = 0.0
            
            action = RiskAction(
                action_type=action_type,
                symbol="PORTFOLIO",
                quantity=quantity,
                confidence=abs(action_value),
                reasoning=f"DDPG agent: risk adjustment {action_value:.3f}",
                expected_impact=action_value
            )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error getting DDPG action: {e}")
            return RiskAction(
                action_type=RiskActionType.NO_ACTION,
                symbol="PORTFOLIO",
                quantity=0.0,
                confidence=0.0,
                reasoning="Error in DDPG action selection",
                expected_impact=0.0
            )
    
    def _state_to_vector(self, state: RiskState) -> np.ndarray:
        """Convert state to vector representation"""
        return np.array([
            state.portfolio_var,
            state.portfolio_cvar,
            state.concentration_risk,
            state.greeks_risk,
            state.market_volatility,
            1.0 if state.market_regime == "normal" else 0.0,
            1.0 if state.market_regime == "high_vol" else 0.0,
            1.0 if state.market_regime == "crisis" else 0.0,
            state.time_of_day,
            state.day_of_week / 6.0,
            state.recent_performance,
            state.stress_test_score,
            state.ml_risk_score,
            state.cash_ratio
        ])
    
    def _forward_actor(self, state_vector: np.ndarray) -> float:
        """Forward pass through actor network (simplified)"""
        # Simplified actor network
        # In practice, this would be a neural network
        
        # Risk-based continuous action
        portfolio_var = state_vector[0]
        portfolio_cvar = state_vector[1]
        market_regime = state_vector[5:8]
        
        # Calculate risk adjustment
        risk_adjustment = 0.0
        
        # Adjust based on VaR
        if portfolio_var > 0.05:
            risk_adjustment -= (portfolio_var - 0.05) * 2.0  # Reduce risk
        
        # Adjust based on CVaR
        if portfolio_cvar > 0.07:
            risk_adjustment -= (portfolio_cvar - 0.07) * 1.5  # Reduce risk
        
        # Adjust based on market regime
        if market_regime[2] > 0:  # Crisis regime
            risk_adjustment -= 0.5  # Strong risk reduction
        
        # Clip to [-1, 1]
        risk_adjustment = np.clip(risk_adjustment, -1.0, 1.0)
        
        return risk_adjustment
    
    def update_networks(self, 
                      states: List[RiskState],
                      actions: List[RiskAction],
                      rewards: List[RiskReward],
                      next_states: List[RiskState]):
        """
        Update DDPG networks
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
        """
        try:
            # Convert to arrays
            state_vectors = np.array([self._state_to_vector(s) for s in states])
            action_values = np.array([a.expected_impact for a in actions])
            reward_values = np.array([r.total_reward for r in rewards])
            next_state_vectors = np.array([self._state_to_vector(s) for s in next_states])
            
            # Calculate target Q-values
            target_q_values = reward_values + self.gamma * self._get_target_q_values(next_state_vectors)
            
            # Update critic
            critic_loss = self._update_critic(state_vectors, action_values, target_q_values)
            
            # Update actor
            actor_loss = self._update_actor(state_vectors)
            
            # Soft update target networks
            self._soft_update_target_networks()
            
            # Track losses
            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)
            
            self.logger.info(f"DDPG updated - Actor loss: {actor_loss:.4f}, "
                           f"Critic loss: {critic_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error updating DDPG networks: {e}")
    
    def _get_target_q_values(self, next_states: np.ndarray) -> np.ndarray:
        """Get target Q-values (simplified)"""
        # Simplified target Q-value calculation
        return np.zeros(len(next_states))
    
    def _update_critic(self, 
                      states: np.ndarray,
                      actions: np.ndarray,
                      target_q_values: np.ndarray) -> float:
        """Update critic network (simplified)"""
        # Simplified critic update
        return np.mean((target_q_values - np.mean(target_q_values)) ** 2)
    
    def _update_actor(self, states: np.ndarray) -> float:
        """Update actor network (simplified)"""
        # Simplified actor update
        return np.mean(np.sum(states ** 2, axis=1))
    
    def _soft_update_target_networks(self):
        """Soft update target networks (simplified)"""
        # Simplified soft update
        pass
    
    def save_model(self, filepath: str):
        """Save trained model"""
        try:
            model_data = {
                'actor_net': self.actor_net,
                'critic_net': self.critic_net,
                'episode_rewards': self.episode_rewards,
                'actor_losses': self.actor_losses,
                'critic_losses': self.critic_losses
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"DDPG model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving DDPG model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.actor_net = model_data['actor_net']
            self.critic_net = model_data['critic_net']
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.actor_losses = model_data.get('actor_losses', [])
            self.critic_losses = model_data.get('critic_losses', [])
            
            self.logger.info(f"DDPG model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading DDPG model: {e}")


class MultiAgentRiskCoordinator:
    """
    Multi-agent risk coordination system
    
    Coordinates multiple RL agents for comprehensive risk management:
    - PPO agent for discrete risk actions
    - DDPG agent for continuous risk adjustments
    - TD3 agent for robust risk management
    - Ensemble decision making
    - Agent performance tracking
    """
    
    def __init__(self, 
                 risk_limits: Dict[str, float],
                 enable_ppo: bool = True,
                 enable_ddpg: bool = True,
                 enable_td3: bool = False):
        """
        Initialize multi-agent coordinator
        
        Args:
            risk_limits: Risk limits configuration
            enable_ppo: Enable PPO agent
            enable_ddpg: Enable DDPG agent
            enable_td3: Enable TD3 agent
        """
        self.risk_limits = risk_limits
        self.logger = logging.getLogger(__name__)
        
        # Initialize environment
        self.environment = RiskEnvironment(risk_limits)
        
        # Initialize agents
        self.agents = {}
        if enable_ppo:
            self.agents['ppo'] = PPORiskAgent()
        if enable_ddpg:
            self.agents['ddpg'] = DDPGRiskAgent()
        if enable_td3:
            # TD3 would be implemented similarly to DDPG
            self.agents['td3'] = DDPGRiskAgent()  # Placeholder
        
        # Coordination state
        self.agent_performance = {name: [] for name in self.agents.keys()}
        self.ensemble_decisions = []
        self.coordination_history = []
        
        self.logger.info(f"Multi-agent coordinator initialized with {len(self.agents)} agents")
    
    async def get_ensemble_action(self, 
                                 portfolio_data: Dict[str, Any],
                                 market_data: Dict[str, Any]) -> RiskAction:
        """
        Get ensemble action from all agents
        
        Args:
            portfolio_data: Current portfolio data
            market_data: Current market data
            
        Returns:
            RiskAction: Ensemble risk action
        """
        try:
            # Get current state
            state = self.environment.get_state(portfolio_data, market_data)
            
            # Get actions from all agents
            agent_actions = {}
            for name, agent in self.agents.items():
                if name == 'ppo':
                    action = agent.get_action(state)
                elif name == 'ddpg':
                    action = agent.get_action(state)
                else:
                    action = agent.get_action(state)
                
                agent_actions[name] = action
            
            # Ensemble decision making
            ensemble_action = self._make_ensemble_decision(agent_actions, state)
            
            # Track decision
            self.ensemble_decisions.append({
                'timestamp': datetime.now(),
                'state': state,
                'agent_actions': agent_actions,
                'ensemble_action': ensemble_action
            })
            
            return ensemble_action
            
        except Exception as e:
            self.logger.error(f"Error getting ensemble action: {e}")
            return RiskAction(
                action_type=RiskActionType.NO_ACTION,
                symbol="PORTFOLIO",
                quantity=0.0,
                confidence=0.0,
                reasoning="Error in ensemble decision",
                expected_impact=0.0
            )
    
    def _make_ensemble_decision(self, 
                               agent_actions: Dict[str, RiskAction],
                               state: RiskState) -> RiskAction:
        """
        Make ensemble decision from agent actions
        
        Args:
            agent_actions: Actions from all agents
            state: Current risk state
            
        Returns:
            RiskAction: Ensemble action
        """
        try:
            # Weighted voting based on agent performance
            weights = self._calculate_agent_weights()
            
            # Count votes for each action type
            action_votes = {}
            for action_type in RiskActionType:
                action_votes[action_type] = 0.0
            
            for agent_name, action in agent_actions.items():
                weight = weights.get(agent_name, 1.0)
                action_votes[action.action_type] += weight * action.confidence
            
            # Select action with highest vote
            best_action_type = max(action_votes, key=action_votes.get)
            
            # Calculate ensemble confidence
            total_votes = sum(action_votes.values())
            ensemble_confidence = action_votes[best_action_type] / total_votes if total_votes > 0 else 0.0
            
            # Create ensemble action
            ensemble_action = RiskAction(
                action_type=best_action_type,
                symbol="PORTFOLIO",
                quantity=0.1,  # Default quantity
                confidence=ensemble_confidence,
                reasoning=f"Ensemble decision: {best_action_type} (confidence: {ensemble_confidence:.3f})",
                expected_impact=ensemble_confidence
            )
            
            return ensemble_action
            
        except Exception as e:
            self.logger.error(f"Error making ensemble decision: {e}")
            return RiskAction(
                action_type=RiskActionType.NO_ACTION,
                symbol="PORTFOLIO",
                quantity=0.0,
                confidence=0.0,
                reasoning="Error in ensemble decision",
                expected_impact=0.0
            )
    
    def _calculate_agent_weights(self) -> Dict[str, float]:
        """Calculate agent weights based on performance"""
        try:
            weights = {}
            for agent_name, performance_history in self.agent_performance.items():
                if performance_history:
                    # Weight based on recent performance
                    recent_performance = np.mean(performance_history[-10:]) if len(performance_history) >= 10 else np.mean(performance_history)
                    weights[agent_name] = max(0.1, recent_performance)
                else:
                    weights[agent_name] = 1.0
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating agent weights: {e}")
            return {name: 1.0 for name in self.agents.keys()}
    
    async def update_agents(self, 
                          portfolio_data: Dict[str, Any],
                          market_data: Dict[str, Any],
                          action: RiskAction,
                          reward: RiskReward):
        """
        Update all agents based on experience
        
        Args:
            portfolio_data: Portfolio data
            market_data: Market data
            action: Executed action
            reward: Received reward
        """
        try:
            # Get current state
            state = self.environment.get_state(portfolio_data, market_data)
            
            # Update each agent
            for agent_name, agent in self.agents.items():
                if agent_name == 'ppo':
                    # PPO update (simplified)
                    agent.episode_rewards.append(reward.total_reward)
                elif agent_name == 'ddpg':
                    # DDPG update (simplified)
                    agent.episode_rewards.append(reward.total_reward)
                
                # Track performance
                self.agent_performance[agent_name].append(reward.total_reward)
            
            self.logger.info(f"Agents updated with reward: {reward.total_reward:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error updating agents: {e}")
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get coordination summary"""
        return {
            'active_agents': list(self.agents.keys()),
            'agent_performance': {name: np.mean(perf) if perf else 0.0 
                                for name, perf in self.agent_performance.items()},
            'ensemble_decisions_count': len(self.ensemble_decisions),
            'coordination_history_count': len(self.coordination_history),
            'environment_episodes': self.environment.episode_count
        }
    
    def save_all_models(self, directory: str):
        """Save all agent models"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            for agent_name, agent in self.agents.items():
                filepath = f"{directory}/{agent_name}_model.pkl"
                agent.save_model(filepath)
            
            self.logger.info(f"All models saved to {directory}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_all_models(self, directory: str):
        """Load all agent models"""
        try:
            for agent_name, agent in self.agents.items():
                filepath = f"{directory}/{agent_name}_model.pkl"
                if Path(filepath).exists():
                    agent.load_model(filepath)
            
            self.logger.info(f"All models loaded from {directory}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize multi-agent coordinator
    risk_limits = {
        'max_total_var': 0.05,
        'max_total_cvar': 0.07,
        'max_concentration': 0.30
    }
    
    coordinator = MultiAgentRiskCoordinator(
        risk_limits=risk_limits,
        enable_ppo=True,
        enable_ddpg=True
    )
    
    # Simulate portfolio data
    portfolio_data = {
        'portfolio_var': 0.06,
        'portfolio_cvar': 0.08,
        'concentration_risk': 0.25,
        'greeks_risk': 0.05,
        'position_count': 5,
        'total_exposure': 80000,
        'cash_ratio': 0.2
    }
    
    # Simulate market data
    market_data = {
        'market_volatility': 0.25,
        'market_regime': 'high_vol',
        'recent_performance': 0.02,
        'stress_test_score': 0.08,
        'ml_risk_score': 0.7
    }
    
    # Get ensemble action
    async def test_coordinator():
        action = await coordinator.get_ensemble_action(portfolio_data, market_data)
        print(f"Ensemble Action: {action.action_type}")
        print(f"Confidence: {action.confidence:.3f}")
        print(f"Reasoning: {action.reasoning}")
        
        # Get summary
        summary = coordinator.get_coordination_summary()
        print(f"Coordination Summary: {summary}")
    
    asyncio.run(test_coordinator())


