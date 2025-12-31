"""
Reinforcement Learning Trading Environment

OpenAI Gym-compatible environment for training RL agents on trading tasks.
Supports both discrete (DQN) and continuous (PPO) action spaces.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class ActionSpace(Enum):
    """Type of action space."""
    DISCRETE = "discrete"  # Buy, Hold, Sell
    CONTINUOUS = "continuous"  # Position size as float


class RewardFunction(Enum):
    """Type of reward function."""
    PNL = "pnl"  # Simple PnL
    SHARPE = "sharpe"  # Risk-adjusted returns
    SORTINO = "sortino"  # Downside-adjusted returns
    CALMAR = "calmar"  # Drawdown-adjusted returns


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment."""
    # Data settings
    window_size: int = 60  # Observation window
    max_steps: int = 1000  # Maximum steps per episode

    # Action space
    action_space_type: ActionSpace = ActionSpace.DISCRETE
    discrete_actions: List[str] = field(default_factory=lambda: ["hold", "buy", "sell"])
    max_position_size: float = 1.0  # Max fraction of capital

    # Trading costs
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%

    # Risk settings
    initial_capital: float = 100000.0
    max_drawdown_pct: float = 0.20  # Max 20% drawdown

    # Reward settings
    reward_function: RewardFunction = RewardFunction.SHARPE
    reward_scaling: float = 100.0

    # Features
    use_technical_features: bool = True
    use_position_features: bool = True


@dataclass
class EnvState:
    """Current state of the trading environment."""
    step: int
    position: float  # Current position (-1 to 1)
    cash: float
    portfolio_value: float
    current_price: float
    returns_history: List[float] = field(default_factory=list)
    trades_history: List[Dict] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        return self.portfolio_value

    @property
    def is_holding(self) -> bool:
        return abs(self.position) > 0.01


class TradingEnvironment:
    """
    OpenAI Gym-compatible trading environment.

    Supports:
    - Discrete actions (Hold, Buy, Sell) for DQN
    - Continuous actions (position sizing) for PPO
    - Multiple reward functions
    - Technical indicators as features
    """

    def __init__(
        self,
        prices: np.ndarray,
        config: Optional[TradingEnvConfig] = None,
    ):
        """
        Initialize trading environment.

        Args:
            prices: Historical price data
            config: Environment configuration
        """
        self.prices = np.array(prices)
        self.config = config or TradingEnvConfig()

        # Precompute features
        self.features = self._compute_features()

        # State dimensions
        self.observation_dim = self._get_observation_dim()
        self.action_dim = self._get_action_dim()

        # Initialize state
        self.state: Optional[EnvState] = None
        self.done = False

    def _compute_features(self) -> np.ndarray:
        """Compute technical features from prices."""
        features_list = []

        # Normalized price (log returns)
        log_returns = np.diff(np.log(self.prices))
        log_returns = np.insert(log_returns, 0, 0)
        features_list.append(log_returns)

        if self.config.use_technical_features:
            # Simple Moving Averages
            for period in [5, 10, 20]:
                sma = self._rolling_mean(self.prices, period)
                sma_norm = (self.prices - sma) / (sma + 1e-8)
                features_list.append(sma_norm)

            # Volatility
            volatility = self._rolling_std(log_returns, 20)
            features_list.append(volatility)

            # RSI-like momentum
            momentum = self._compute_momentum(self.prices, 14)
            features_list.append(momentum)

            # Price position in range
            high_20 = self._rolling_max(self.prices, 20)
            low_20 = self._rolling_min(self.prices, 20)
            range_pos = (self.prices - low_20) / (high_20 - low_20 + 1e-8)
            features_list.append(range_pos)

        return np.stack(features_list, axis=1)

    def _rolling_mean(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros_like(x)
        for i in range(len(x)):
            start = max(0, i - window + 1)
            result[i] = np.mean(x[start:i + 1])
        return result

    def _rolling_std(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros_like(x)
        for i in range(len(x)):
            start = max(0, i - window + 1)
            result[i] = np.std(x[start:i + 1]) if i >= 1 else 0
        return result

    def _rolling_max(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros_like(x)
        for i in range(len(x)):
            start = max(0, i - window + 1)
            result[i] = np.max(x[start:i + 1])
        return result

    def _rolling_min(self, x: np.ndarray, window: int) -> np.ndarray:
        result = np.zeros_like(x)
        for i in range(len(x)):
            start = max(0, i - window + 1)
            result[i] = np.min(x[start:i + 1])
        return result

    def _compute_momentum(self, x: np.ndarray, period: int) -> np.ndarray:
        result = np.zeros_like(x)
        for i in range(period, len(x)):
            gains = sum(max(0, x[j] - x[j-1]) for j in range(i - period + 1, i + 1))
            losses = sum(max(0, x[j-1] - x[j]) for j in range(i - period + 1, i + 1))
            if losses > 0:
                rs = gains / losses
                result[i] = (100 - 100 / (1 + rs)) / 100 - 0.5
            else:
                result[i] = 0.5
        return result

    def _get_observation_dim(self) -> int:
        """Get observation space dimension."""
        feature_dim = self.features.shape[1] * self.config.window_size

        if self.config.use_position_features:
            feature_dim += 3  # position, cash_ratio, portfolio_return

        return feature_dim

    def _get_action_dim(self) -> int:
        """Get action space dimension."""
        if self.config.action_space_type == ActionSpace.DISCRETE:
            return len(self.config.discrete_actions)
        else:
            return 1  # Continuous position

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = EnvState(
            step=self.config.window_size,
            position=0.0,
            cash=self.config.initial_capital,
            portfolio_value=self.config.initial_capital,
            current_price=self.prices[self.config.window_size],
            returns_history=[],
            trades_history=[],
        )
        self.done = False
        return self._get_observation()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done or self.state is None:
            raise RuntimeError("Environment not initialized or already done")

        old_value = self.state.portfolio_value
        old_position = self.state.position

        # Parse action
        if self.config.action_space_type == ActionSpace.DISCRETE:
            target_position = self._discrete_to_position(action)
        else:
            target_position = np.clip(action, -1.0, 1.0)

        # Execute trade
        self._execute_trade(target_position)

        # Move to next step
        self.state.step += 1
        if self.state.step >= len(self.prices) - 1:
            self.done = True

        # Update prices and portfolio value
        new_price = self.prices[self.state.step]
        self.state.current_price = new_price
        self._update_portfolio_value()

        # Calculate reward
        reward = self._calculate_reward(old_value)

        # Track return
        step_return = (self.state.portfolio_value - old_value) / old_value
        self.state.returns_history.append(step_return)

        # Check max drawdown
        if self._check_max_drawdown():
            self.done = True
            reward -= 10.0  # Penalty for hitting max drawdown

        # Check max steps
        if self.state.step - self.config.window_size >= self.config.max_steps:
            self.done = True

        # Build info dict
        info = {
            "portfolio_value": self.state.portfolio_value,
            "position": self.state.position,
            "step": self.state.step,
            "price": new_price,
            "return": step_return,
        }

        return self._get_observation(), reward, self.done, info

    def _discrete_to_position(self, action: int) -> float:
        """Convert discrete action to target position."""
        action_name = self.config.discrete_actions[action]
        if action_name == "hold":
            return self.state.position
        elif action_name == "buy":
            return min(1.0, self.state.position + 0.5)
        elif action_name == "sell":
            return max(-1.0, self.state.position - 0.5)
        else:
            return 0.0

    def _execute_trade(self, target_position: float) -> None:
        """Execute trade to reach target position."""
        position_change = target_position - self.state.position

        if abs(position_change) < 0.01:
            return  # No significant change

        # Calculate trade value
        trade_value = abs(position_change) * self.state.portfolio_value

        # Apply costs
        commission = trade_value * self.config.commission_rate
        slippage = trade_value * self.config.slippage_rate
        total_cost = commission + slippage

        # Update state
        self.state.cash -= total_cost
        self.state.position = target_position

        # Record trade
        self.state.trades_history.append({
            "step": self.state.step,
            "position_change": position_change,
            "cost": total_cost,
            "price": self.state.current_price,
        })

    def _update_portfolio_value(self) -> None:
        """Update portfolio value based on position and price."""
        # Initial capital at position = 0
        # As price changes, position value changes
        price_return = self.prices[self.state.step] / self.prices[self.state.step - 1] - 1
        position_pnl = self.state.position * price_return * self.state.portfolio_value
        self.state.portfolio_value = self.state.portfolio_value + position_pnl

    def _calculate_reward(self, old_value: float) -> float:
        """Calculate reward based on configured reward function."""
        step_return = (self.state.portfolio_value - old_value) / old_value

        if self.config.reward_function == RewardFunction.PNL:
            reward = step_return

        elif self.config.reward_function == RewardFunction.SHARPE:
            # Incremental Sharpe-like reward
            if len(self.state.returns_history) > 1:
                mean_return = np.mean(self.state.returns_history[-20:])
                std_return = np.std(self.state.returns_history[-20:]) + 1e-8
                reward = step_return / std_return
            else:
                reward = step_return

        elif self.config.reward_function == RewardFunction.SORTINO:
            # Sortino-like reward (penalize downside more)
            if step_return < 0:
                reward = step_return * 2  # Double penalty for losses
            else:
                reward = step_return

        elif self.config.reward_function == RewardFunction.CALMAR:
            # Penalize drawdown
            if self.state.returns_history:
                cumulative = np.cumprod(1 + np.array(self.state.returns_history[-50:]))
                max_cum = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - max_cum) / max_cum
                current_dd = drawdown[-1] if len(drawdown) > 0 else 0
                reward = step_return - abs(current_dd) * 0.1
            else:
                reward = step_return
        else:
            reward = step_return

        return reward * self.config.reward_scaling

    def _check_max_drawdown(self) -> bool:
        """Check if max drawdown exceeded."""
        if not self.state.returns_history:
            return False

        cumulative = np.cumprod(1 + np.array(self.state.returns_history))
        max_cum = np.maximum.accumulate(cumulative)
        current_dd = (max_cum[-1] - cumulative[-1]) / max_cum[-1]

        return current_dd > self.config.max_drawdown_pct

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.state is None:
            raise RuntimeError("Environment not initialized")

        # Window of features
        start_idx = self.state.step - self.config.window_size
        end_idx = self.state.step
        feature_window = self.features[start_idx:end_idx].flatten()

        if self.config.use_position_features:
            # Add position-related features
            position_features = np.array([
                self.state.position,
                self.state.cash / self.config.initial_capital,
                (self.state.portfolio_value - self.config.initial_capital) / self.config.initial_capital,
            ])
            observation = np.concatenate([feature_window, position_features])
        else:
            observation = feature_window

        return observation.astype(np.float32)

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment."""
        if self.state is None:
            return None

        info = (
            f"Step: {self.state.step} | "
            f"Price: ${self.state.current_price:.2f} | "
            f"Position: {self.state.position:.2f} | "
            f"Portfolio: ${self.state.portfolio_value:.2f} | "
            f"Return: {(self.state.portfolio_value / self.config.initial_capital - 1) * 100:.2f}%"
        )

        if mode == "human":
            print(info)
        return info

    def get_metrics(self) -> Dict[str, float]:
        """Get episode performance metrics."""
        if not self.state or not self.state.returns_history:
            return {}

        returns = np.array(self.state.returns_history)
        cumulative = np.cumprod(1 + returns)

        # Calculate metrics
        total_return = (self.state.portfolio_value / self.config.initial_capital - 1) * 100
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        # Max drawdown
        max_cum = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - max_cum) / max_cum
        max_drawdown = np.min(drawdowns) * 100

        # Win rate
        trades = self.state.trades_history
        if trades:
            profitable_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = profitable_trades / len(trades) * 100
        else:
            win_rate = 0

        return {
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "num_trades": len(trades),
            "win_rate_pct": win_rate,
            "final_portfolio_value": self.state.portfolio_value,
        }


class MultiAssetTradingEnvironment:
    """
    Trading environment for multiple assets simultaneously.

    Extends single-asset environment for portfolio-level decisions.
    """

    def __init__(
        self,
        price_data: Dict[str, np.ndarray],
        config: Optional[TradingEnvConfig] = None,
    ):
        """
        Initialize multi-asset environment.

        Args:
            price_data: Dict mapping symbol to price array
            config: Environment configuration
        """
        self.symbols = list(price_data.keys())
        self.price_data = price_data
        self.config = config or TradingEnvConfig()

        # Create individual environments
        self.envs = {
            symbol: TradingEnvironment(prices, config)
            for symbol, prices in price_data.items()
        }

        # Portfolio-level state
        self.positions: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.capital = self.config.initial_capital

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset all environments."""
        observations = {}
        for symbol, env in self.envs.items():
            observations[symbol] = env.reset()
            self.positions[symbol] = 0.0
        self.capital = self.config.initial_capital
        return observations

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """Execute step in all environments."""
        observations = {}
        rewards = {}
        done = False
        info = {}

        for symbol, action in actions.items():
            obs, reward, env_done, env_info = self.envs[symbol].step(action)
            observations[symbol] = obs
            rewards[symbol] = reward
            if env_done:
                done = True
            info[symbol] = env_info

        return observations, rewards, done, info
