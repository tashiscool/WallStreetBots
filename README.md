# WallStreetBots - Options Trading Systems

This repository contains comprehensive options trading systems that implement successful WSB-style strategies with proper risk management.

## 🎯 Quick Start - WSB Dip Bot

The **WSB Dip Bot** (`wsb_dip_bot.py`) implements the exact pattern that produces viral WSB gains:

```bash
# Install dependencies
pip install -r wsb_requirements.txt

# Find today's setups (after market close)
python wsb_dip_bot.py scan-eod --account-size 450000 --risk-pct 1.0 --use-options-chain

# Live hunt during market hours
python wsb_dip_bot.py scan-intraday --poll-seconds 120 --max-minutes 120 \
  --account-size 450000 --risk-pct 0.90 --use-options-chain

# Plan exact trade for specific ticker
python wsb_dip_bot.py plan --ticker GOOGL --spot 207 --account-size 450000 --risk-pct 0.90

# Monitor your position for exits
python wsb_dip_bot.py monitor --ticker GOOGL --expiry 2025-10-17 --strike 220 --entry-prem 4.70 \
  --target-mult 3.0 --delta-target 0.60 --poll-seconds 60
```

## 📊 What It Does

**Scans for**: Hard dip ≤ -3% after +10% run over 10 days on mega-caps  
**Builds**: Exact ~5% OTM, ~30 DTE call positions  
**Exits**: At 3x profit or Δ≥0.60 (the WSB screenshot formula)  
**Risk**: Configurable 10-100% account deployment

## 🏗️ Repository Structure

```
├── wsb_dip_bot.py              # Main WSB strategy scanner
├── wsb_requirements.txt        # Dependencies for WSB bot
├── backend/tradingbot/         # Django-integrated trading modules
│   ├── options_calculator.py   # Black-Scholes pricing engine
│   ├── market_regime.py        # Market regime detection
│   ├── risk_management.py      # Position sizing & Kelly Criterion
│   ├── exit_planning.py        # Systematic exit strategies
│   ├── alert_system.py         # Multi-channel alerts
│   ├── exact_clone.py          # Exact replica of successful trade
│   └── production_scanner.py   # Production-ready integrated scanner
├── CLAUDE.md                   # Development guide
├── README_OPTIONS_SYSTEM.md    # Comprehensive system documentation
└── README_EXACT_CLONE.md       # Exact clone implementation guide
```

## 🚨 Risk Warning

The WSB Dip Bot implements high-risk strategies that can result in significant losses:

- ⚠️ **Can lose 70-100% of position** on single trade
- ⚠️ **No diversification** - single ticker, single expiry
- ⚠️ **Time decay risk** - short-dated options lose value quickly
- ⚠️ **IV crush risk** - volatility collapse can destroy gains

**Use only with money you can afford to lose completely.**

## 📈 Strategy Background

Based on analysis of a successful 240% options trade:
- **Original**: 950 contracts, $446,500 cost, 95% account risk → $1.07M profit
- **Our Implementation**: Risk-configurable while maintaining the core edge

The system captures the exact momentum continuation pattern that produces WSB viral gains while offering risk controls.

## 🛠️ Advanced Features

### Multiple Trading Systems:
1. **WSB Dip Bot** - Pure WSB pattern replication
2. **Risk-Managed System** - Same edge with Kelly Criterion sizing
3. **Exact Clone** - Perfect replica of original risky approach

### Production Features:
- Real-time market data via yfinance
- Actual options chain integration
- Implied volatility calculations
- CSV/JSON output for record keeping
- Multi-channel alerting system

## 📚 Documentation

- **[WSB Dip Bot Guide](README_EXACT_CLONE.md)** - Complete WSB implementation
- **[Options System Overview](README_OPTIONS_SYSTEM.md)** - Full system documentation
- **[Development Guide](CLAUDE.md)** - Technical implementation details

## ⚖️ Legal Disclaimer

This software is for educational and research purposes only. Trading options involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any trading losses incurred using this software.

## 🤝 Contributing

Contributions welcome via issues and pull requests. Please ensure any trading strategies include appropriate risk warnings.
