"""
ML Trading Bot Components

This module provides machine learning components for trading analysis:
- Hidden Markov Models (HMM) for regime detection
- Monte Carlo simulations for risk analysis
- LSTM deep learning for price prediction
- Transformer models for time series prediction
- CNN models for pattern recognition
- Ensemble methods combining multiple models
- Reinforcement Learning agents (PPO/DQN/SAC/TD3/DDPG/A2C) for trading
- Meta-learning for regime-adaptive trading
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
    AGENT_REGISTRY,
    list_available_agents,
    create_rl_agent,
)

# Additional RL agents
from .sac_agent import SACConfig, SACAgent
from .td3_agent import TD3Config, TD3Agent
from .ddpg_agent import DDPGConfig, DDPGAgent
from .a2c_agent import A2CConfig, A2CAgent

# Meta-learning components
from .meta_learning import (
    RegimeDetector,
    RegimeAwareConfig,
    RegimeAwareAgent,
    TransferLearningTrainer,
    MultiTaskRLAgent,
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
    "AGENT_REGISTRY",
    "list_available_agents",
    "create_rl_agent",
    # Additional RL Agents
    "SACConfig",
    "SACAgent",
    "TD3Config",
    "TD3Agent",
    "DDPGConfig",
    "DDPGAgent",
    "A2CConfig",
    "A2CAgent",
    # Meta-learning
    "RegimeDetector",
    "RegimeAwareConfig",
    "RegimeAwareAgent",
    "TransferLearningTrainer",
    "MultiTaskRLAgent",
    # Portfolio
    "NaiveHMMPortfolioUpdate",
    "PortfolioManager",
]
