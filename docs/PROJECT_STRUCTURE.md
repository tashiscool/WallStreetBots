# WallStreetBots Project Structure

## Directory Structure

```
WallStreetBots/
├── scripts/                        # Setup & run scripts
│   ├── setup.sh                   # One-command project setup
│   ├── run.sh                     # Start dev server
│   └── setup_postgres.sh         # Optional PostgreSQL setup
│
├── docs/                           # Documentation
│   ├── user-guides/               # User documentation
│   ├── production/                # Production guides
│   └── strategies/                # Strategy documentation
│
├── tests/                         # Test Suite (5,500+ tests)
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── production/                # Production tests
│   ├── sentiment/                 # NLP sentiment tests
│   ├── services/                  # Copy trading & service tests
│   ├── analysis/                  # PDF report tests
│   ├── options/                   # Options tests
│   ├── crypto/                    # Crypto & DEX tests
│   ├── ml/                        # ML/RL agent tests
│   └── fixtures/                  # Test utilities
│
├── backend/tradingbot/            # Core Trading System
│   ├── config/                   # Centralized Configuration
│   │   ├── base.py              # Shared settings
│   │   └── environments/        # Dev/Test/Prod configs
│   │
│   ├── strategies/              # Trading Strategies
│   │   ├── base/               # Abstract base classes + FrameworkStrategy
│   │   ├── implementations/    # Strategy logic (10+ strategies)
│   │   └── production/         # Production wrappers
│   │
│   ├── framework/              # Algorithm Framework (LEAN-inspired)
│   │   ├── alpha_models/       # Alpha signal generators
│   │   ├── portfolio_models/   # Portfolio construction (HRP, Black-Litterman, etc.)
│   │   └── universe.py         # Universe selection & pipeline
│   │
│   ├── sentiment/              # NLP Sentiment Engine
│   │   ├── scoring/            # VADER, FinBERT, ensemble scorers
│   │   ├── sources/            # Reddit, Twitter, SEC EDGAR, aggregator
│   │   └── pipeline.py         # Async sentiment processing
│   │
│   ├── risk/                   # Risk Management
│   │   ├── engines/            # VaR, CVaR calculations
│   │   ├── compliance/         # Regulatory checks
│   │   └── monitoring/         # Real-time monitoring
│   │
│   ├── options/                # Options Trading
│   │   ├── pricing_engine.py   # Black-Scholes + Greeks
│   │   ├── advanced_spreads.py # Iron Condor, Butterfly, etc.
│   │   └── payoff_visualizer.py # P&L diagrams & Greeks dashboard
│   │
│   ├── crypto/                 # Cryptocurrency Trading
│   │   ├── alpaca_crypto_client.py  # CEX integration
│   │   ├── dex_client.py       # Uniswap V3 DEX integration
│   │   └── wallet_manager.py   # Encrypted wallet management
│   │
│   ├── analysis/               # Analytics & Reports
│   │   ├── tearsheet.py        # Performance tearsheets
│   │   └── pdf_report.py       # PDF report generation
│   │
│   ├── execution/              # Trade execution
│   ├── data/                   # Data management
│   ├── models/                 # Django models
│   ├── indicators/             # Technical indicators
│   └── backtesting/            # Backtesting engine + optimizer
│
├── backend/auth0login/           # Auth & API Layer
│   ├── services/                # Business logic services
│   │   ├── copy_trading_service.py    # Copy/social trading
│   │   ├── strategy_builder_service.py # Strategy builder API
│   │   └── report_delivery_service.py  # Report delivery
│   └── api_views.py            # REST API endpoints
│
└── ml/tradingbots/               # Machine Learning
    ├── components/              # RL agents (PPO, DQN, SAC, TD3, DDPG, A2C)
    │   ├── rl_agents.py        # PPO, DQN + agent factory
    │   ├── sac_agent.py        # Soft Actor-Critic
    │   ├── td3_agent.py        # Twin Delayed DDPG
    │   ├── ddpg_agent.py       # Deep Deterministic PG
    │   ├── a2c_agent.py        # Advantage Actor-Critic
    │   ├── rl_environment.py   # Gym-compatible trading env
    │   ├── meta_learning.py    # Regime-aware & transfer learning
    │   ├── lstm_predictor.py   # LSTM price prediction
    │   ├── transformer_predictor.py  # Transformer prediction
    │   ├── cnn_predictor.py    # CNN pattern recognition
    │   └── ensemble_predictor.py    # Ensemble methods
    └── training/                # Training infrastructure & callbacks
        ├── training_utils.py   # Walk-forward, checkpointing, metrics
        ├── rl_training.py      # RL-specific training loop
        └── callbacks.py        # Eval, checkpoint, early stopping callbacks
```

## Key Benefits

- **Single Source of Truth** - No duplicate strategies
- **Logical Organization** - Clear module separation
- **Environment Isolation** - Dev/Test/Prod configs
- **Easy Navigation** - Intuitive file locations
- **Scalable Structure** - Follows Python best practices
- **Full RL Suite** - Six RL algorithms with unified factory API
- **Framework Pipeline** - Alpha -> Portfolio -> Execution model composition
- **Comprehensive Testing** - 5,500+ tests across unit, integration, ML, and production

## Import Examples

```python
# Strategies
from backend.tradingbot.strategies.base import BaseStrategy
from backend.tradingbot.strategies.base.framework_strategy import FrameworkStrategy
from backend.tradingbot.strategies.implementations import WSBDipBot
from backend.tradingbot.strategies.production import ProductionWSBDipBot

# Framework (LEAN-inspired pipeline)
from backend.tradingbot.framework.alpha_models import AlphaModel
from backend.tradingbot.framework.portfolio_models import PortfolioConstructionModel
from backend.tradingbot.framework.universe import PipelineUniverseSelectionModel

# Risk Management
from backend.tradingbot.risk.engines import RiskEngine
from backend.tradingbot.risk.monitoring import CircuitBreaker

# Options Trading
from backend.tradingbot.options.pricing_engine import OptionsPricingEngine
from backend.tradingbot.options.advanced_spreads import IronCondor

# Data Management
from backend.tradingbot.data.providers import MarketDataClient
from backend.tradingbot.data.quality import DataQualityMonitor

# Backtesting & Optimization
from backend.tradingbot.backtesting import BacktestEngine
from backend.tradingbot.backtesting.optimization_service import OptimizationService

# Configuration
from backend.tradingbot.config import get_config, get_trading_config

# ML - RL Agents (factory pattern)
from ml.tradingbots.components import create_rl_agent, list_available_agents
from ml.tradingbots.components import PPOAgent, DQNAgent, SACAgent, TD3Agent

# ML - Training
from ml.tradingbots.training import train_rl_agent, RLTrainingConfig
from ml.tradingbots.training import CheckpointCallback, EarlyStoppingRLCallback

# ML - Predictors
from ml.tradingbots.components import LSTMPricePredictor, TransformerPricePredictor
from ml.tradingbots.components import EnsemblePricePredictor, CNNPricePredictor

# ML - Meta-learning
from ml.tradingbots.components import RegimeAwareAgent, TransferLearningTrainer
```

This structure organizes the project into a maintainable, scalable trading system with clear separation between the Django backend, ML components, and test infrastructure.
