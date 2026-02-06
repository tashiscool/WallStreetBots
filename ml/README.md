# ML - Machine Learning for Trading

The `ml/tradingbots/` package provides PyTorch-based machine learning components for algorithmic trading, including reinforcement learning agents, deep learning predictors, and a full training infrastructure. Models are trained independently and saved to disk for use by the Django backend.

## Components Overview

### Reinforcement Learning Agents

Six RL agents are available, all following a consistent pattern: a `Config` dataclass for hyperparameters, a neural network module, and an `Agent` class with `train()`, `select_action()`, `save()`, and `load()` methods.

| Agent | Type | Description |
|-------|------|-------------|
| **PPO** (Proximal Policy Optimization) | Continuous | Stable on-policy algorithm using actor-critic architecture with clipped objective. Best for continuous position sizing. |
| **DQN** (Deep Q-Network) | Discrete | Dueling architecture with target network for discrete trade timing decisions (buy/hold/sell). |
| **SAC** (Soft Actor-Critic) | Continuous | Entropy-regularized off-policy algorithm with twin Q-networks. State-of-the-art for continuous portfolio allocation. |
| **TD3** (Twin Delayed DDPG) | Continuous | Improves on DDPG with twin critics, delayed policy updates, and target policy smoothing for more stable training. |
| **DDPG** (Deep Deterministic Policy Gradient) | Continuous | Uses Ornstein-Uhlenbeck noise for temporally correlated exploration, suitable for sequential trading decisions. |
| **A2C** (Advantage Actor-Critic) | Continuous | Simpler synchronous variant of PPO. Faster training, good baseline for trading tasks. Uses n-step returns and entropy regularization. |

### Deep Learning Predictors

- **LSTMPricePredictor** - LSTM networks for sequential price prediction with ensemble support
- **TransformerPricePredictor** - Transformer architecture for time series forecasting
- **CNNPricePredictor** - Convolutional networks for chart pattern recognition
- **EnsemblePricePredictor** - Combines multiple models (stacking, voting, weighted average)

### Meta-Learning

- **RegimeAwareAgent** - Adapts RL policy based on detected market regimes (bull/bear/sideways)
- **TransferLearningTrainer** - Pre-trains on one market/asset and fine-tunes on another
- **MultiTaskRLAgent** - Trains across multiple assets/objectives simultaneously

## Factory Pattern

The `create_rl_agent()` factory provides a unified interface for creating any RL agent:

```python
from ml.tradingbots.components import create_rl_agent, list_available_agents

# See what's available
print(list_available_agents())
# {'ppo': 'Proximal Policy Optimization - stable on-policy algorithm',
#  'dqn': 'Deep Q-Network - discrete action spaces',
#  'sac': 'Soft Actor-Critic - state-of-the-art for continuous control',
#  'td3': 'Twin Delayed DDPG - twin critics with delayed policy updates',
#  'ddpg': 'Deep Deterministic Policy Gradient - continuous control',
#  'a2c': 'Advantage Actor-Critic - simpler synchronous variant of PPO'}

# Create an agent with default config
agent = create_rl_agent(
    agent_type="sac",
    state_dim=20,
    action_dim=3,
)

# Or pass a custom config
from ml.tradingbots.components import SACConfig

config = SACConfig(hidden_dim=512, learning_rate=1e-4, gamma=0.995)
agent = create_rl_agent("sac", state_dim=20, action_dim=3, config=config)
```

## Training Infrastructure

The `ml/tradingbots/training/` module provides production-grade training utilities.

### Core Training

- **`train_rl_agent()`** - Main training loop with observation normalization, episode tracking, and metrics collection
- **`RLTrainingConfig`** - Configuration for training runs (episodes, batch size, eval frequency, etc.)
- **`RLProgressTracker`** - Rich console progress bars and live metrics display
- **`RunningMeanStd`** - Online observation normalization for stable RL training

### Callbacks

Callbacks hook into the training loop at key points (on step, on episode end, on eval):

| Callback | Purpose |
|----------|---------|
| **`CheckpointCallback`** | Saves model weights at regular intervals |
| **`EarlyStoppingRLCallback`** | Stops training when reward plateaus |
| **`EvalCallback`** | Runs periodic evaluation episodes and tracks best model |
| **`TradingMetricsCallback`** | Logs trading-specific metrics (Sharpe, drawdown, win rate) |
| **`CallbackList`** | Composes multiple callbacks together |

### Walk-Forward Validation

- **`WalkForwardValidator`** - Time-series-aware cross-validation that prevents look-ahead bias
- **`ModelCheckpointer`** - Saves/loads model checkpoints with metadata
- **`EarlyStoppingCallback`** - General early stopping for supervised training

## Usage Example

```python
from ml.tradingbots.components import create_rl_agent, SACConfig, TradingEnvConfig, TradingEnvironment
from ml.tradingbots.training import (
    train_rl_agent,
    RLTrainingConfig,
    CheckpointCallback,
    EarlyStoppingRLCallback,
    TradingMetricsCallback,
    CallbackList,
)

# 1. Create the trading environment
env_config = TradingEnvConfig(
    initial_balance=100_000,
    window_size=20,
    commission=0.001,
)
env = TradingEnvironment(price_data=prices_df, config=env_config)

# 2. Create an RL agent
agent_config = SACConfig(hidden_dim=256, learning_rate=3e-4)
agent = create_rl_agent(
    agent_type="sac",
    state_dim=env.observation_space_dim,
    action_dim=env.action_space_dim,
    config=agent_config,
)

# 3. Set up training callbacks
callbacks = CallbackList([
    CheckpointCallback(save_freq=1000, save_path="./checkpoints/"),
    EarlyStoppingRLCallback(patience=50, min_delta=0.01),
    TradingMetricsCallback(log_freq=10),
])

# 4. Train
training_config = RLTrainingConfig(
    total_episodes=5000,
    eval_frequency=100,
    batch_size=64,
)
metrics = train_rl_agent(
    agent=agent,
    env=env,
    config=training_config,
    callbacks=callbacks,
)

# 5. Save the trained model
agent.save("./models/sac_trading_agent.pt")

# 6. Load later for inference or backtesting
agent.load("./models/sac_trading_agent.pt")
action = agent.select_action(observation, deterministic=True)
```

## Integration with Backtesting

Trained RL agents integrate with the backtesting and optimization pipeline through the backend:

- The **`OptimizationService`** in `backend/tradingbot/backtesting/` can use RL agents as part of strategy optimization
- The **vectorized backtesting engine** supports running RL-based strategies at scale
- Agent checkpoints are saved with `torch.save()` using `weights_only=False` to preserve `Config` dataclasses alongside `state_dict`s (required for PyTorch 2.6+ compatibility)

## Project Structure

```
ml/
├── tradingbots/
│   ├── components/              # ML model implementations
│   │   ├── rl_agents.py        # PPO, DQN agents + factory (create_rl_agent)
│   │   ├── sac_agent.py        # Soft Actor-Critic
│   │   ├── td3_agent.py        # Twin Delayed DDPG
│   │   ├── ddpg_agent.py       # Deep Deterministic Policy Gradient
│   │   ├── a2c_agent.py        # Advantage Actor-Critic
│   │   ├── rl_environment.py   # Gym-compatible trading environment
│   │   ├── meta_learning.py    # Regime detection, transfer learning
│   │   ├── lstm_predictor.py   # LSTM price prediction
│   │   ├── transformer_predictor.py  # Transformer prediction
│   │   ├── cnn_predictor.py    # CNN pattern recognition
│   │   ├── ensemble_predictor.py    # Ensemble methods
│   │   ├── hiddenmarkov.py     # HMM regime detection
│   │   └── portfoliomanager.py # Portfolio management
│   └── training/                # Training infrastructure
│       ├── training_utils.py   # Walk-forward validation, checkpointing
│       ├── rl_training.py      # RL training loop + metrics
│       └── callbacks.py        # Training callbacks
└── README.md
```

## Testing

Tests for the ML package live in `tests/ml/`. Note: `tests/ml/` intentionally has **no `__init__.py`** file. Adding one would create a namespace conflict that shadows the top-level `ml/` package, breaking all ML imports. Pytest does not require `__init__.py` files in test directories.
