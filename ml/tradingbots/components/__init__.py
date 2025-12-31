"""
ML Trading Bot Components

This module provides machine learning components for trading analysis:
- Hidden Markov Models (HMM) for regime detection
- Monte Carlo simulations for risk analysis
- LSTM deep learning for price prediction
- Transformer models for time series prediction
- CNN models for pattern recognition
- Ensemble methods combining multiple models
- Reinforcement Learning agents (PPO/DQN) for trading
"""

# Optional imports that depend on alpaca_trade_api
try:
    from .hiddenmarkov import HMM, DataManager
    _HAS_HMM = True
except ImportError:
    HMM = None
    DataManager = None
    _HAS_HMM = False

from .lstm_predictor import (
    LSTMConfig,
    LSTMModel,
    LSTMDataManager,
    LSTMPricePredictor,
    LSTMEnsemble,
)
from .lstm_signal_calculator import LSTMSignalCalculator
from .naiveportfoliomanager import NaiveHMMPortfolioUpdate
from .portfoliomanager import PortfolioManager

# Transformer components
from .transformer_predictor import (
    TransformerConfig,
    TransformerModel,
    TransformerDataManager,
    TransformerPricePredictor,
)

# CNN components
from .cnn_predictor import (
    CNNConfig,
    CNNModel,
    CNNDataManager,
    CNNPricePredictor,
)

# Ensemble components
from .ensemble_predictor import (
    EnsembleConfig,
    EnsembleMethod,
    EnsemblePrediction,
    EnsemblePricePredictor,
    EnsembleHyperparameterTuner,
)

# RL Environment components
from .rl_environment import (
    TradingEnvConfig,
    EnvState,
    TradingEnvironment,
    MultiAssetTradingEnvironment,
    ActionSpace,
    RewardFunction,
)

# RL Agent components
from .rl_agents import (
    PPOConfig,
    DQNConfig,
    PPOAgent,
    DQNAgent,
    create_ppo_trading_agent,
    create_dqn_trading_agent,
)

__all__ = [
    # LSTM
    "LSTMConfig",
    "LSTMModel",
    "LSTMDataManager",
    "LSTMPricePredictor",
    "LSTMEnsemble",
    "LSTMSignalCalculator",
    # Transformer
    "TransformerConfig",
    "TransformerModel",
    "TransformerDataManager",
    "TransformerPricePredictor",
    # CNN
    "CNNConfig",
    "CNNModel",
    "CNNDataManager",
    "CNNPricePredictor",
    # Ensemble
    "EnsembleConfig",
    "EnsembleMethod",
    "EnsemblePrediction",
    "EnsemblePricePredictor",
    "EnsembleHyperparameterTuner",
    # RL Environment
    "TradingEnvConfig",
    "EnvState",
    "TradingEnvironment",
    "MultiAssetTradingEnvironment",
    "ActionSpace",
    "RewardFunction",
    # RL Agents
    "PPOConfig",
    "DQNConfig",
    "PPOAgent",
    "DQNAgent",
    "create_ppo_trading_agent",
    "create_dqn_trading_agent",
    # Portfolio
    "NaiveHMMPortfolioUpdate",
    "PortfolioManager",
]
