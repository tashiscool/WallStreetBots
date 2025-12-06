# 🚀 WallStreetBots - Institutional-Grade Algorithmic Trading System

<div align="center">

## **Production-Ready Trading Platform with Advanced Risk Management**
### *Sophisticated WSB-style strategies • Institutional risk controls • Real broker integration*

**✅ 10+ Complete Trading Strategies** • **✅ Advanced VaR/CVaR Risk Models** • **✅ ML Risk Agents** • **✅ Multi-Asset Support**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2+-green.svg)](https://djangoproject.com)
[![Tests](https://img.shields.io/badge/Tests-2420+-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

A **comprehensive, institutional-grade trading system** implementing WSB-style strategies with **sophisticated risk management**, **real-time monitoring**, and **production-ready architecture**. 

**🎯 What It Does:** Automatically finds trading opportunities, places trades, manages risk, and tracks performance - like having a professional trader working for you 24/7.

**🛡️ Safety First:** Built-in risk management protects your capital with multiple safety layers including position limits, stop losses, and circuit breakers.

**📚 New to Trading?** Start with our [5-Minute Quick Start](docs/QUICK_START.md) or read [How It Works](docs/HOW_IT_WORKS.md) for a simple explanation!

## 🏆 **Key Capabilities**

### **📈 Trading Strategies (10+ Complete)**
- **WSB Dip Bot** - Momentum-based dip buying with volume confirmation
- **Earnings Protection** - Options-based earnings play protection
- **Wheel Strategy** - Cash-secured puts and covered calls automation
- **Index Baseline** - SPY/QQQ tracking with enhanced returns
- **Momentum Weeklies** - Weekly options momentum trading
- **Debit Spreads** - Automated options spread construction
- **LEAPS Tracker** - Long-term equity anticipation securities
- **Swing Trading** - Multi-timeframe technical analysis
- **Credit Spreads** - SPX credit spread automation
- **Lotto Scanner** - High-probability lottery ticket identification

### **🛡️ Advanced Risk Management**
- **Multi-Method VaR** - Historical, Parametric, Monte Carlo, Extreme Value Theory
- **Liquidity-Adjusted VaR** - Bid-ask spread and slippage modeling
- **Stress Testing** - 6 regulatory scenarios (2008 crisis, flash crash, COVID-19, etc.)
- **ML Risk Agents** - PPO & DDPG reinforcement learning for dynamic risk control
- **Circuit Breakers** - Automated trading halts on drawdown/error thresholds
- **Multi-Asset Risk** - Cross-asset correlation modeling (equity, crypto, forex, commodities)
- **Signal Validation** - Alpha validation gates with comprehensive testing framework

### **⚖️ Regulatory Compliance**
- **FCA/CFTC Compliance** - Full regulatory compliance with audit trails
- **Wash Sale Tracking** - Tax-efficient position management
- **Position Reconciliation** - Daily broker vs. internal position verification
- **Compliance Monitoring** - Real-time rule checking and violation alerts
- **Audit Trail** - Complete transaction and decision logging

### **🤖 Machine Learning & Analytics**
- **Market Regime Detection** - Bull/bear/sideways market identification
- **Portfolio Optimization** - ML-driven rebalancing with cost-benefit analysis
- **Advanced Analytics** - Sharpe ratio, Sortino ratio, max drawdown, VaR analysis
- **Predictive Risk Models** - ML-based volatility and risk forecasting
- **Performance Attribution** - Alpha, beta, information ratio vs benchmarks
- **Signal Validation** - Real-time alpha validation with parameter tracking
- **Data Quality Control** - Automated data validation and quality metrics

## 📚 **Complete Documentation**

### **🎯 New to Trading? Start Here!**
- **[📖 How It Works](docs/HOW_IT_WORKS.md)** - **Simple explanation of everything** ⭐ **READ THIS FIRST!**
- **[🚀 5-Minute Quick Start](docs/QUICK_START.md)** - **Get trading in 5 minutes!** ⭐ **START HERE!**
- **[📋 User Guide Summary](docs/USER_GUIDE_SUMMARY.md)** - **Navigation guide to all docs** ⭐ **FIND WHAT YOU NEED!**

### **📊 Getting Started**
- **[🚀 Getting Started Guide](docs/user-guides/GETTING_STARTED_REAL.md)** - Detailed step-by-step setup
- **[🔧 Launcher Guide](docs/user-guides/LAUNCHER_GUIDE.md)** - How to run the system easily
- **[🎯 User Guide](docs/user-guides/README.md)** - Complete system overview & capabilities

### **📈 Going Deeper**
- **[📊 Strategy Tuning](docs/strategies/STRATEGY_TUNING_GUIDE.md)** - How to optimize strategies
- **[📈 Project Structure](docs/PROJECT_STRUCTURE.md)** - Understanding the codebase
- **[🏭 Production Guide](docs/production/PRODUCTION_MODULES.md)** - Enterprise deployment

## ⚡ **Quick Start (5 Minutes to Paper Trading!)**

### **Step 1: Get Free Alpaca Account** (2 minutes)
1. Go to [alpaca.markets](https://alpaca.markets) and sign up (free, no money needed!)
2. Navigate to "Paper Trading" → "API Keys"
3. Copy your **API Key** and **Secret Key** (keep these safe!)

### **Step 2: Install & Setup** (2 minutes)
```bash
# Clone the repository
git clone https://github.com/yourusername/WallStreetBots.git
cd WallStreetBots

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install alpaca-py>=0.42.0

# Setup database
python manage.py migrate
```

### **Step 3: Configure API Keys** (1 minute)
```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your keys:
# ALPACA_API_KEY=your_paper_api_key_here
# ALPACA_SECRET_KEY=your_paper_secret_key_here
```

**Or manually edit `.env` file:**
- Open `.env` in any text editor
- Replace `your_paper_api_key_here` with your actual API key
- Replace `your_paper_secret_key_here` with your actual secret key
- Save the file

### **Step 4: Test Connection** (30 seconds)
```bash
# Verify everything works
python test_env_keys.py

# You should see:
# ✅ Connection: True - API validated successfully
# 💰 Account value: $100,000.00 (paper trading)
```

### **Step 5: Start Paper Trading!** (Ready to go!)
```bash
# Use the easy launcher (RECOMMENDED)
python run_wallstreetbots.py

# Select option 1: "Start Simple Trading Bot (Paper Trading)"
# The system will start automatically!
```

**That's it!** You're now paper trading with $100,000 fake money. 🎉

### **What Happens Next?**
- System scans markets every few minutes
- Finds trading opportunities automatically
- Places trades (with fake money)
- Monitors positions and exits when targets are hit
- Tracks performance for you to review

**📚 New to trading?** Read [How It Works](docs/HOW_IT_WORKS.md) for a simple explanation!

**🚀 Ready for more?** Check [Getting Started Guide](docs/user-guides/GETTING_STARTED_REAL.md) for detailed instructions!

## 🏛️ How It's Built (For the Technically Curious)

```
backend/tradingbot/
├── production/          # Production deployment and core systems
│   ├── core/           # Production strategy manager and integration
│   ├── strategies/     # Production-optimized strategy implementations
│   ├── data/           # Production data integration and management
│   └── tests/          # Production-specific test suite
├── strategies/          # Strategy framework and implementations
│   ├── base/           # Base strategy classes and interfaces
│   ├── implementations/ # Core strategy algorithms
│   └── production/     # Production wrapper implementations
├── risk/               # Comprehensive risk management
│   ├── engines/        # VaR, stress testing, and risk calculation engines
│   ├── managers/       # Risk management coordination and integration
│   ├── compliance/     # Regulatory compliance and reporting
│   └── monitoring/     # Real-time risk monitoring and alerts
├── validation/         # Signal validation and quality assurance
│   ├── gates/          # Alpha validation gates and filters
│   ├── metrics/        # Validation metrics and reporting
│   └── adapters/       # Integration adapters for validation pipeline
├── data/               # Data management and providers
│   ├── providers/      # Market data source integrations
│   └── quality/        # Data validation and quality assurance
├── core/               # Core trading infrastructure
├── config/             # Configuration management
│   └── environments/   # Environment-specific configurations
├── common/             # Shared utilities and imports
├── analytics/          # Performance analysis and reporting
├── monitoring/         # System health and operational monitoring
├── execution/          # Trade execution and order management
├── accounting/         # Portfolio accounting and reconciliation
├── models/             # Database models and data structures
└── phases/             # Development phase implementations
```

## 🎯 **What Makes This Production-Ready?**

### **🏭 Enterprise-Grade Infrastructure**
- **Async Architecture** - Non-blocking, high-performance trading engine
- **Production Strategy Manager** - Orchestrates multiple strategies simultaneously
- **Circuit Breakers** - Automatic trading halts on loss/error thresholds
- **Real-time Monitoring** - Health checks, system metrics, alert system
- **Database Integration** - SQLite/PostgreSQL with audit trails
- **Multi-Environment Support** - Development, testing, production configurations
- **Signal Validation Pipeline** - Automated alpha validation with parameter tracking
- **Quality Assurance** - Comprehensive data quality monitoring and validation

### **📊 Advanced Market Data & Analytics**
- **Multi-Source Data** - Alpaca, Polygon, IEX, Yahoo Finance integration
- **Options Pricing Engine** - Complete Black-Scholes implementation with Greeks
- **Market Regime Detection** - Bull/bear/sideways market identification
- **Technical Indicators** - 50+ technical analysis indicators
- **Earnings Calendar** - Corporate earnings and ex-dividend tracking
- **Social Sentiment** - WSB/Reddit sentiment integration
- **Real-time Data Validation** - Automated quality checks and anomaly detection
- **Corporate Actions** - Stock splits, dividends, mergers handling

### **⚡ Execution & Operations**
- **Smart Order Routing** - Optimal execution with slippage minimization
- **Replay Protection** - Exactly-once order execution guarantees
- **Shadow Trading** - Risk-free strategy validation in production
- **Automated Rebalancing** - ML-driven portfolio optimization
- **Tax Optimization** - Wash sale avoidance and tax-efficient positioning
- **Performance Tracking** - Real-time P&L and risk-adjusted metrics

### **🔒 Security & Compliance**
- **API Key Management** - Secure credential storage and rotation
- **Audit Logging** - Complete transaction and decision audit trails
- **Compliance Checks** - PDT, SSR, halt monitoring, position limits
- **Risk Limits** - Portfolio-level and position-level risk controls
- **Data Encryption** - Secure data storage and transmission
- **Access Controls** - Role-based permissions and authentication

## 🧪 **Thoroughly Tested (2420+ Tests)**

```bash
# Run all tests (2420+ tests)
python -m pytest tests/ --tb=short -q

# Run by category
python -m pytest tests/unit/          # Unit tests (strategy logic, risk models)
python -m pytest tests/integration/   # Integration tests (end-to-end workflows)
python -m pytest tests/production/    # Production tests (deployment readiness)
python -m pytest tests/risk/          # Risk management validation
python -m pytest tests/strategies/    # Strategy implementation tests
python -m pytest tests/validation/    # Signal validation framework tests
python -m pytest tests/core/          # Core system functionality tests

# Run specific test types
python -m pytest tests/ -k "risk"     # All risk-related tests
python -m pytest tests/ -k "wsb"      # WSB strategy tests
python -m pytest tests/ -k "var"      # VaR calculation tests

# Generate coverage report
python -m pytest tests/ --cov=backend --cov-report=html
```

### **Test Coverage Areas**
- **Strategy Logic** - All 10+ trading strategies with edge cases
- **Risk Management** - VaR, CVaR, stress testing, ML agents
- **Market Data** - Data quality, corporate actions, real-time feeds
- **Execution** - Order routing, replay protection, shadow trading
- **Compliance** - Regulatory checks, audit trails, position limits
- **Performance** - Analytics, regime detection, portfolio optimization
- **Signal Validation** - Alpha validation gates, parameter tracking, quality metrics
- **Data Quality** - Validation framework, quality monitoring, automated testing

## 🚀 Easy Ways to Run the System

### Quick Start Options
```bash
# Cross-platform Python launcher
python run_wallstreetbots.py

# Windows (.bat launcher)
run_wallstreetbots.bat

# macOS/Linux (shell script)
./run_wallstreetbots.sh

# Production CLI
python run.py validate  # Validate configuration
python run.py status    # Show current status
```

### Automated Deployment
- **macOS**: Use launchd for automatic startup - see [macOS Setup Guide](examples/macos/README.md)
- **Windows**: Use Task Scheduler with `run_wallstreetbots.bat`
- **Linux**: Use systemd or cron jobs with `run_wallstreetbots.sh`

### API Key Setup
1. Copy `.env.example` to `.env`
2. Add your Alpaca API keys:
   ```bash
   ALPACA_API_KEY=your-key-here
   ALPACA_SECRET_KEY=your-secret-here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
   ```
3. Test configuration: `python test_env_keys.py`

## 📈 **Ready for Production Use**

### **Enterprise Deployment Options**
- **Docker Containers** - Containerized deployment with health checks
- **Kubernetes** - Scalable orchestration with auto-scaling
- **Cloud Platforms** - AWS, GCP, Azure deployment guides
- **CI/CD Pipelines** - Automated testing and deployment
- **Monitoring Stack** - Prometheus, Grafana, alerting integration

See [Production Guide](docs/production/PRODUCTION_MODULES.md) for complete deployment instructions.

## 🔬 **Advanced Features (For Developers)**

### **Advanced Risk Models**
```python
# Multi-method VaR calculation with ML enhancements
var_engine = AdvancedVaREngine(portfolio_value=1000000)
var_suite = var_engine.calculate_var_suite(
    returns=returns,
    confidence_levels=[0.95, 0.99, 0.999],
    methods=["parametric", "historical", "monte_carlo", "evt"]
)
```

### **ML Risk Agents**
```python
# PPO agent for dynamic risk management
from backend.tradingbot.risk.agents import PPORiskAgent
agent = PPORiskAgent()
risk_action = agent.get_risk_action(market_state, portfolio_state)
```

### **Strategy Framework**
```python
# Production strategy with async execution
from backend.tradingbot.strategies import wsb_dip_bot
strategy = await wsb_dip_bot.create_production_strategy()
signals = await strategy.generate_signals(market_data)
```

### **Real-time Risk Monitoring**
```python
# Comprehensive risk dashboard
dashboard = RiskDashboard2025(portfolio_value=1000000)
risk_data = dashboard.get_risk_dashboard_data(portfolio)
```

### **Signal Validation Framework**
```python
# Alpha validation with quality gates
from backend.validation import AlphaValidationGate, ValidationCriteria
validation_gate = AlphaValidationGate()
criteria = ValidationCriteria(min_sharpe=1.2, max_drawdown=0.15)
validated_signals = validation_gate.validate_signals(signals, criteria)
```

### **Data Quality Monitoring**
```python
# Automated data quality validation
from backend.tradingbot.data.quality import DataQualityMonitor
quality_monitor = DataQualityMonitor()
quality_report = quality_monitor.validate_market_data(market_data)
```

## 📊 **What's Included - At a Glance**

| Component | Status | Features |
|-----------|--------|----------|
| **Trading Strategies** | ✅ Complete | 10+ strategies, async execution, real-time signals |
| **Risk Management** | ✅ Institutional | Multi-method VaR, ML agents, stress testing |
| **Market Data** | ✅ Production | Multi-source feeds, real-time processing |
| **Execution Engine** | ✅ Enterprise | Smart routing, replay protection, audit trails |
| **Compliance** | ✅ Regulatory | FCA/CFTC compliant, wash sale tracking |
| **Analytics** | ✅ Advanced | ML regime detection, performance attribution |
| **Infrastructure** | ✅ Scalable | Async architecture, auto-scaling, monitoring |
| **Signal Validation** | ✅ Production | Alpha validation gates, quality assurance |
| **Data Quality** | ✅ Automated | Real-time validation, quality monitoring |

## 🤝 Want to Contribute?

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add comprehensive tests for new functionality
4. Ensure all 2420+ tests pass (`python -m pytest tests/`)
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

## ⚠️ **Important Disclaimers**

### **Risk Warning**
This software is for **educational and research purposes**. Trading involves **substantial risk of loss**. Past performance does **not** guarantee future results.

### **Before Using Real Money**
1. ✅ **Test extensively** in paper trading mode (30+ days minimum)
2. ✅ **Understand** how each strategy works
3. ✅ **Verify** all risk controls are working
4. ✅ **Start small** with tiny positions (1-2% max)
5. ✅ **Never risk** more than you can afford to lose completely

### **Regulatory Compliance**
Ensure compliance with local financial regulations before deployment. This system is a tool - you are responsible for following all applicable laws and regulations.

### **Remember**
- 🎯 **This is not a get-rich-quick scheme**
- 🎯 **Success requires learning, practice, and discipline**
- 🎯 **Markets can be unpredictable**
- 🎯 **Always prioritize capital preservation over profits**

**🚀 Ready to start?** Begin with [5-Minute Quick Start](docs/QUICK_START.md)!

**📚 Want to understand it?** Read [How It Works](docs/HOW_IT_WORKS.md)!

**💡 Need help?** Check the [Getting Started Guide](docs/user-guides/GETTING_STARTED_REAL.md)!

</div>



