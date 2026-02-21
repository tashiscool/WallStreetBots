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

# Optional imports that depend on Alpaca SDK support
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
    "AGENT_REGISTRY",
    "A2CAgent",
    "A2CConfig",
    "ActionSpace",
    # CNN
    "CNNConfig",
    "CNNDataManager",
    "CNNModel",
    "CNNPricePredictor",
    "DDPGAgent",
    "DDPGConfig",
    "DQNAgent",
    "DQNConfig",
    # Ensemble
    "EnsembleConfig",
    "EnsembleHyperparameterTuner",
    "EnsembleMethod",
    "EnsemblePrediction",
    "EnsemblePricePredictor",
    "EnvState",
    # LSTM
    "LSTMConfig",
    "LSTMDataManager",
    "LSTMEnsemble",
    "LSTMModel",
    "LSTMPricePredictor",
    "LSTMSignalCalculator",
    "MultiAssetTradingEnvironment",
    "MultiTaskRLAgent",
    # Portfolio
    "NaiveHMMPortfolioUpdate",
    "PPOAgent",
    # RL Agents
    "PPOConfig",
    "PortfolioManager",
    "RegimeAwareAgent",
    "RegimeAwareConfig",
    # Meta-learning
    "RegimeDetector",
    "RewardFunction",
    "SACAgent",
    # Additional RL Agents
    "SACConfig",
    "TD3Agent",
    "TD3Config",
    # RL Environment
    "TradingEnvConfig",
    "TradingEnvironment",
    "TransferLearningTrainer",
    # Transformer
    "TransformerConfig",
    "TransformerDataManager",
    "TransformerModel",
    "TransformerPricePredictor",
    "create_dqn_trading_agent",
    "create_ppo_trading_agent",
    "create_rl_agent",
    "list_available_agents",
]
