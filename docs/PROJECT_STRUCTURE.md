# 🏗️ WallStreetBots Project Structure

## 📁 Organized Directory Structure

```
WallStreetBots/
├── docs/                           # 📚 Documentation
│   ├── user-guides/               # User documentation
│   ├── production/                # Production guides  
│   └── strategies/                # Strategy documentation
│
├── tests/                         # 🧪 Test Suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── production/                # Production tests
│   └── fixtures/                  # Test utilities
│
├── backend/tradingbot/            # 🏛️ Core Trading System
│   ├── config/                   # ⚙️ Centralized Configuration
│   │   ├── base.py              # Shared settings
│   │   └── environments/        # Dev/Test/Prod configs
│   │
│   ├── strategies/              # 📈 Trading Strategies
│   │   ├── base/               # Abstract base classes
│   │   ├── implementations/    # Strategy logic
│   │   └── production/         # Production wrappers
│   │
│   ├── risk/                   # 🛡️ Risk Management
│   │   ├── engines/            # VaR, CVaR calculations
│   │   ├── compliance/         # Regulatory checks
│   │   └── monitoring/         # Real-time monitoring
│   │
│   └── data/                   # 📊 Data Management
│       ├── providers/          # Market data sources
│       └── quality/            # Data validation
```

## 🎯 Key Benefits

- ✅ **Single Source of Truth** - No duplicate strategies
- ✅ **Logical Organization** - Clear module separation
- ✅ **Environment Isolation** - Dev/Test/Prod configs
- ✅ **Easy Navigation** - Intuitive file locations
- ✅ **Scalable Structure** - Follows Python best practices

## 🔄 Import Examples

```python
# Strategies
from backend.tradingbot.strategies.base import BaseStrategy
from backend.tradingbot.strategies.implementations import WSBDipBot
from backend.tradingbot.strategies.production import ProductionWSBDipBot

# Risk Management
from backend.tradingbot.risk.engines import RiskEngine
from backend.tradingbot.risk.monitoring import CircuitBreaker

# Data Management
from backend.tradingbot.data.providers import MarketDataClient
from backend.tradingbot.data.quality import DataQualityMonitor

# Configuration
from backend.tradingbot.config import get_config, get_trading_config
```

This structure transforms the complex project into a well-organized, maintainable, and scalable trading system.