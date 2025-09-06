# WallStreetBots - Comprehensive WSB Trading Strategies

This repository contains a complete suite of options trading systems implementing proven WSB-style strategies with sophisticated risk management and systematic approaches.

## 🚀 Quick Start - WSB Trading Strategies

### 1. WSB Dip Bot - The Original Viral Strategy
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

### 2. Momentum Weeklies Scanner - Intraday Reversals
Detect intraday reversals and news momentum for weekly options plays:

```bash
# Single scan for momentum signals
python momentum_weeklies.py --output text

# Continuous scanning (5-minute intervals)
python momentum_weeklies.py --continuous --min-volume-spike 3.0

# JSON output for programmatic use
python momentum_weeklies.py --output json
```

### 3. Debit Call Spreads - Reduced Risk Strategy
More repeatable than naked calls with reduced theta/IV risk:

```bash
# Scan for spread opportunities
python debit_spreads.py --min-days 20 --max-days 60 --limit 10

# Save results to CSV
python debit_spreads.py --save-csv spreads.csv --min-risk-reward 1.5

# Filter by risk-reward ratio
python debit_spreads.py --min-risk-reward 2.0 --output json
```

### 4. LEAPS Secular Winners - Long-term Growth
Long-term positions on secular growth trends with systematic profit-taking:

```bash
# Scan for LEAPS candidates
python leaps_tracker.py scan --min-score 70 --limit 15

# View portfolio status
python leaps_tracker.py portfolio

# Update positions
python leaps_tracker.py update

# Save candidates to CSV
python leaps_tracker.py scan --save-csv leaps_candidates.csv
```

### 5. 0DTE/Earnings Lotto Scanner - High Risk/High Reward
High-risk, high-reward plays with strict position sizing:

```bash
# Scan 0DTE opportunities
python lotto_scanner.py 0dte --account-size 10000 --max-risk-pct 1.0

# Scan earnings plays
python lotto_scanner.py earnings --account-size 10000 --max-risk-pct 0.5

# Scan both types
python lotto_scanner.py both --account-size 10000 --output json
```

### 6. Wheel Strategy - Income Generation
Consistent income generation on volatile names with positive expectancy:

```bash
# Scan wheel candidates
python wheel_strategy.py scan --min-return 10 --limit 15

# View wheel portfolio
python wheel_strategy.py portfolio

# Update positions
python wheel_strategy.py update

# Save candidates to CSV
python wheel_strategy.py scan --save-csv wheel_candidates.csv
```

## 📊 Strategy Overview

### WSB Dip Bot
**Scans for**: Hard dip ≤ -3% after +10% run over 10 days on mega-caps  
**Builds**: Exact ~5% OTM, ~30 DTE call positions  
**Exits**: At 3x profit or Δ≥0.60 (the WSB screenshot formula)  
**Risk**: Configurable 10-100% account deployment

### Momentum Weeklies Scanner
**Scans for**: Intraday reversals with 3x+ volume spikes on mega-caps
**Builds**: Weekly options 2-5% OTM based on momentum strength
**Exits**: Quick profit-taking same/next day
**Risk**: 1-3% account risk per play

### Debit Call Spreads
**Scans for**: Bullish trends with favorable risk/reward ratios
**Builds**: Call spreads with 1.2+ risk/reward and 20%+ max profit
**Exits**: At max profit or breakeven
**Risk**: Reduced theta/IV exposure vs naked calls

### LEAPS Secular Winners
**Scans for**: Secular growth themes (AI, Cloud, EVs, Fintech, etc.)
**Builds**: 12-24 month LEAPS 10-20% OTM on quality names
**Exits**: Systematic scale-out at 2x, 3x, 4x returns
**Risk**: Long-term capital deployment with diversification

### 0DTE/Earnings Lotto Scanner
**Scans for**: High-volatility 0DTE and earnings plays
**Builds**: Strict position sizing (0.5-1% account risk)
**Exits**: 50% stop loss, 3-5x profit targets
**Risk**: Extreme risk with disciplined position sizing

### Wheel Strategy
**Scans for**: Quality names with decent volatility and dividends
**Builds**: Cash-secured puts → covered calls cycle
**Exits**: Assignment → call away → repeat
**Risk**: Income generation with positive expectancy

## 🏗️ Repository Structure

```
├── wsb_dip_bot.py              # Main WSB strategy scanner
├── momentum_weeklies.py         # Intraday reversal scanner
├── debit_spreads.py            # Call spread opportunity finder
├── leaps_tracker.py            # Long-term secular winners tracker
├── lotto_scanner.py            # 0DTE/earnings high-risk scanner
├── wheel_strategy.py           # Covered calls/wheel income strategy
├── wsb_requirements.txt         # Dependencies for all WSB bots
├── backend/tradingbot/         # Django-integrated trading modules
│   ├── options_calculator.py   # Black-Scholes pricing engine
│   ├── market_regime.py        # Market regime detection
│   ├── risk_management.py      # Position sizing & Kelly Criterion
│   ├── exit_planning.py        # Systematic exit strategies
│   ├── alert_system.py         # Multi-channel alerts
│   ├── exact_clone.py          # Exact replica of successful trade
│   ├── production_scanner.py   # Production-ready integrated scanner
│   ├── dip_scanner.py          # Dip detection algorithms
│   ├── trading_system.py       # Core trading system integration
│   ├── test_production_scanner.py # Production scanner tests
│   └── test_suite.py           # Comprehensive test suite
├── CLAUDE.md                   # Development guide
├── README_OPTIONS_SYSTEM.md    # Comprehensive system documentation
└── README_EXACT_CLONE.md       # Exact clone implementation guide
```

## 🚨 Risk Warning

These trading strategies implement various risk levels and can result in significant losses:

### High Risk Strategies (WSB Dip Bot, Momentum Weeklies, Lotto Scanner):
- ⚠️ **Can lose 70-100% of position** on single trade
- ⚠️ **No diversification** - single ticker, single expiry
- ⚠️ **Time decay risk** - short-dated options lose value quickly
- ⚠️ **IV crush risk** - volatility collapse can destroy gains

### Medium Risk Strategies (Debit Spreads, LEAPS):
- ⚠️ **Limited upside** vs naked options
- ⚠️ **Assignment risk** on short legs
- ⚠️ **Liquidity risk** on both strikes
- ⚠️ **Long-term capital commitment** for LEAPS

### Lower Risk Strategies (Wheel Strategy):
- ⚠️ **Assignment risk** - may be forced to buy/sell shares
- ⚠️ **Limited upside** on covered calls
- ⚠️ **Dividend risk** if holding shares

**Use only with money you can afford to lose completely. Past performance does not guarantee future results.**

## 📈 Strategy Background

Based on analysis of a successful 240% options trade:
- **Original**: 950 contracts, $446,500 cost, 95% account risk → $1.07M profit
- **Our Implementation**: Risk-configurable while maintaining the core edge

The system captures the exact momentum continuation pattern that produces WSB viral gains while offering risk controls.

## 🛠️ Advanced Features

### Six Complete Trading Systems:
1. **WSB Dip Bot** - Pure WSB pattern replication
2. **Momentum Weeklies** - Intraday reversal detection
3. **Debit Call Spreads** - Reduced risk spread strategies
4. **LEAPS Secular Winners** - Long-term growth themes
5. **0DTE/Earnings Lotto** - High-risk/high-reward plays
6. **Wheel Strategy** - Income generation system

### Production Features:
- **Real-time market data** via yfinance API
- **Actual options chain integration** with live pricing
- **Black-Scholes pricing** and implied volatility calculations
- **Risk management** with Kelly Criterion position sizing
- **Portfolio tracking** with JSON persistence
- **Multiple output formats** (JSON, CSV, text)
- **Multi-channel alerting** system
- **Comprehensive testing** suite
- **Market regime detection** for adaptive strategies
- **Systematic exit planning** with profit targets

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/tashiscool/WallStreetBots.git
cd WallStreetBots

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r wsb_requirements.txt

# Test installation
python momentum_weeklies.py --help
python debit_spreads.py --help
python leaps_tracker.py --help
python lotto_scanner.py --help
python wheel_strategy.py --help
python wsb_dip_bot.py --help
```

### Virtual Environment (Recommended)
Always use a virtual environment to avoid dependency conflicts:
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r wsb_requirements.txt

# Run scripts using venv python
venv/bin/python momentum_weeklies.py --output text
```

## 📚 Documentation

- **[WSB Dip Bot Guide](README_EXACT_CLONE.md)** - Complete WSB implementation
- **[Options System Overview](README_OPTIONS_SYSTEM.md)** - Full system documentation
- **[Development Guide](CLAUDE.md)** - Technical implementation details

## ⚖️ Legal Disclaimer

This software is for educational and research purposes only. Trading options involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any trading losses incurred using this software.

## 🤝 Contributing

Contributions welcome via issues and pull requests. Please ensure any trading strategies include appropriate risk warnings.
