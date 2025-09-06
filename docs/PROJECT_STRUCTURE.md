# 🗂️ WallStreetBots Project Structure

## 📁 **Clean, Production-Ready File Organization**

This document outlines the reorganized, production-ready file structure for the WallStreetBots trading system.

## 🏗️ **Root Level Structure**

```
WallStreetBots/
├── 📁 backend/                    # Main application backend
│   ├── 📁 tradingbot/            # Core trading system
│   │   ├── 📁 core/              # Production infrastructure
│   │   ├── 📁 strategies/        # Individual trading strategies
│   │   ├── 📁 phases/            # Phase-specific implementations
│   │   ├── 📁 tests/             # Backend-specific tests
│   │   └── 📄 [core files]       # Core trading components
│   ├── 📁 home/                  # Django home app
│   ├── 📁 auth0login/            # Authentication system
│   ├── 📄 production_runner.py   # Main production entry point
│   └── 📄 [Django config files]  # Django settings, URLs, etc.
├── 📁 docs/                      # All project documentation
├── 📁 tests/                     # Comprehensive test suite
│   ├── 📁 strategies/            # Strategy-specific tests
│   ├── 📁 phases/                # Phase-specific tests
│   ├── 📁 core/                  # Core component tests
│   └── 📁 integration/           # Integration tests
├── 📁 scripts/                   # Setup and utility scripts
├── 📁 config/                    # Configuration files
├── 📁 utils/                     # Shared utilities
├── 📄 README.md                  # Main project documentation
├── 📄 requirements.txt           # Python dependencies
├── 📄 pyproject.toml             # Project metadata
└── 📄 [Docker/CI files]          # Deployment configuration
```

## 🔧 **Backend Trading System (`backend/tradingbot/`)**

### Core Infrastructure (`backend/tradingbot/core/`)
- `production_config.py` - Production configuration management
- `production_logging.py` - Advanced logging system
- `production_models.py` - Database models
- `production_models_simple.py` - Dataclass fallbacks
- `production_database.py` - Database management
- `trading_interface.py` - Broker integration interface
- `data_providers.py` - Market data providers
- `production_wheel_strategy.py` - Wheel strategy implementation
- `production_debit_spreads.py` - Debit spreads strategy
- `production_spx_spreads.py` - SPX credit spreads strategy
- `production_index_baseline.py` - Index baseline strategy
- `production_earnings_protection.py` - Earnings protection strategy
- `production_swing_trading.py` - Swing trading strategy
- `production_momentum_weeklies.py` - Momentum weeklies strategy
- `production_lotto_scanner.py` - Lotto scanner strategy
- `production_leaps_tracker.py` - LEAPS tracker strategy
- `production_wsb_dip_bot.py` - WSB dip bot strategy

### Trading Strategies (`backend/tradingbot/strategies/`)
- `momentum_weeklies.py` - Original momentum weeklies scanner
- `debit_spreads.py` - Original debit spreads scanner
- `leaps_tracker.py` - Original LEAPS tracker
- `lotto_scanner.py` - Original lotto scanner
- `wheel_strategy.py` - Original wheel strategy
- `wsb_dip_bot.py` - Original WSB dip bot
- `swing_trading.py` - Original swing trading
- `spx_credit_spreads.py` - Original SPX credit spreads
- `earnings_protection.py` - Original earnings protection
- `index_baseline.py` - Original index baseline

### Phase Implementations (`backend/tradingbot/phases/`)
- `phase1_demo.py` - Phase 1 demonstration
- `phase2_integration.py` - Phase 2 integration
- `phase3_integration.py` - Phase 3 integration
- `phase4_backtesting.py` - Phase 4 backtesting engine
- `phase4_optimization.py` - Phase 4 optimization engine
- `phase4_monitoring.py` - Phase 4 monitoring system
- `phase4_deployment.py` - Phase 4 deployment system
- `phase4_production.py` - Phase 4 production orchestrator

### Core Trading Components
- `alert_system.py` - Alert and notification system
- `apimanagers.py` - API management (Alpaca, etc.)
- `dip_scanner.py` - Real-time dip scanner
- `exact_clone.py` - Exact clone protocol
- `exit_planning.py` - Exit strategy planning
- `market_regime.py` - Market regime detection
- `options_calculator.py` - Options pricing calculator
- `production_scanner.py` - Production scanner
- `risk_management.py` - Risk management system
- `synchronization.py` - Data synchronization
- `trading_system.py` - Main trading system

## 📚 **Documentation (`docs/`)**

- `PRODUCTION_ROADMAP.md` - Complete production roadmap
- `PHASE1_README.md` - Phase 1 implementation guide
- `PHASE2_README.md` - Phase 2 implementation guide
- `PHASE3_IMPLEMENTATION_SUMMARY.md` - Phase 3 summary
- `PHASE4_IMPLEMENTATION_SUMMARY.md` - Phase 4 summary
- `README_REAL_MONEY_TRADING.md` - Real money trading guide
- `README_EXACT_CLONE.md` - Exact clone protocol docs
- `README_OPTIONS_SYSTEM.md` - Options system documentation
- `COMPLETE_IMPLEMENTATION_SUMMARY.md` - Complete implementation summary
- `FINAL_PROJECT_SUMMARY.md` - Final project summary

## 🧪 **Test Suite (`tests/`)**

### Strategy Tests (`tests/strategies/`)
- `test_earnings_protection.py`
- `test_index_baseline.py`
- `test_leaps_tracker.py`
- `test_spx_credit_spreads.py`
- `test_swing_trading.py`

### Phase Tests (`tests/phases/`)
- `test_phase1_basic.py`
- `test_phase2_basic.py`
- `test_phase3_comprehensive.py`
- `test_phase4_comprehensive.py`

### Core Tests (`tests/core/`)
- `test_alert_system.py`
- `test_dip_scanner.py`

### Integration Tests (`tests/integration/`)
- `test_all_wsb_strategies.py`
- `test_django_setup.py`

## 🛠️ **Scripts (`scripts/`)**

- `setup_phase1.py` - Phase 1 setup script
- `setup_for_real_trading.py` - Real trading setup
- `quick_test_wsb_modules.py` - Quick module testing

## ⚙️ **Configuration (`config/`)**

- `requirements_phase1.txt` - Phase 1 specific requirements

## 🔧 **Utilities (`utils/`)**

- `error_handling.py` - Error handling utilities
- `yfinance_hardening.py` - yfinance hardening functions

## 🚀 **Key Benefits of This Organization**

### ✅ **Clean Separation of Concerns**
- **Core Infrastructure**: All production-ready components in `core/`
- **Strategies**: Both original and production versions clearly separated
- **Phases**: Phase-specific implementations organized by development stage
- **Tests**: Comprehensive test coverage organized by component type

### ✅ **Scalable Structure**
- Easy to add new strategies in `strategies/`
- Phase implementations can be extended in `phases/`
- Core infrastructure is centralized and reusable
- Test organization mirrors the code structure

### ✅ **Production Ready**
- Clear entry points (`production_runner.py`)
- Comprehensive documentation in `docs/`
- All configuration centralized
- Clean separation between development and production code

### ✅ **Maintainable**
- Related files are grouped together
- Clear naming conventions
- Easy to navigate and understand
- Follows Python best practices

## 🎯 **Usage Examples**

### Running Production System
```bash
python backend/production_runner.py --live --strategies wheel,debit_spreads
```

### Running Tests
```bash
pytest tests/strategies/  # Test all strategies
pytest tests/phases/      # Test all phases
pytest tests/core/        # Test core components
```

### Accessing Documentation
```bash
# All documentation is in docs/
ls docs/
cat docs/PRODUCTION_ROADMAP.md
```

This organization provides a clean, scalable, and production-ready structure that makes the WallStreetBots trading system easy to understand, maintain, and extend.
