# 🏗️ WallStreetBots - Production Trading System

A comprehensive, production-ready trading system implementing WSB-style strategies with real broker integration, live data feeds, and comprehensive risk management.

## 📚 Documentation

- **[Getting Started](docs/user-guides/GETTING_STARTED_REAL.md)** - Complete setup guide
- **[User Guide](docs/user-guides/README.md)** - Main user documentation  
- **[Launcher Guide](docs/user-guides/LAUNCHER_GUIDE.md)** - How to run the system
- **[Production Guide](docs/production/PRODUCTION_MODULES.md)** - Production deployment
- **[Strategy Tuning](docs/strategies/STRATEGY_TUNING_GUIDE.md)** - Strategy optimization
- **[Production Improvements](docs/production/PRODUCTION_IMPROVEMENTS.md)** - Latest enhancements

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the system
python run_wallstreetbots.py

# Run tests
python -m pytest tests/ --tb=short -q
```

## 🏛️ Architecture

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

## 📊 Features

- **10+ Production Strategies** - WSB Dip Bot, Earnings Protection, Wheel Strategy, etc.
- **Real Broker Integration** - Alpaca API with live trading
- **Advanced Risk Management** - VaR, CVaR, position sizing
- **Live Data Feeds** - Real-time market data and options chains
- **Comprehensive Testing** - 1800+ tests with 75%+ coverage
- **Production Monitoring** - Health checks, alerts, logging

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/integration/   # Integration tests  
python -m pytest tests/production/    # Production tests
```

## 📈 Production Deployment

See [Production Guide](docs/production/PRODUCTION_MODULES.md) for complete deployment instructions including Docker, Kubernetes, and CI/CD setup.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⚠️ Risk Warning**: This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results.
