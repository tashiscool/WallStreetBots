# WallStreetBots - Comprehensive WSB Trading Strategies

This repository contains a complete suite of options trading systems implementing proven WSB-style strategies with sophisticated risk management and systematic approaches.

## 🚀 Quick Start - WSB Trading Strategies

### 1. WSB Dip Bot - The Original Viral Strategy
The **WSB Dip Bot** (`wsb_dip_bot.py`) implements the exact pattern that produces viral WSB gains:

```bash
# Install dependencies
pip install -r requirements.txt

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

### 7. Enhanced Swing Trading - Fast Profit-Taking
Fast breakout and momentum trades with same-day exit discipline:

```bash
# Scan for swing opportunities
python swing_trading.py scan --min-strength 70

# Monitor active trades for exit signals
python swing_trading.py monitor

# Continuous scanning with monitoring
python swing_trading.py continuous --max-expiry-days 21

# JSON output for integration
python swing_trading.py scan --output json --min-strength 60
```

## 📊 Strategy Overview - WSB "Actually Works" Collection

Based on r/WallStreetBets community analysis of consistently profitable strategies:

### 1. WSB Dip Bot ✅ **IMPLEMENTED**
**WSB Pattern**: Momentum continuation on mega-caps after sharp red days
**Scans for**: Hard dip ≤ -3% after +10% run over 10 days on mega-caps  
**Builds**: Exact ~5% OTM, ~30 DTE call positions  
**Exits**: At 3x profit or Δ≥0.60 (the WSB screenshot formula)  
**Risk**: Configurable 10-100% account deployment
**WSB Success**: Original pattern produced 240% returns ($446K → $1.07M)

### 2. Wheel Strategy (Premium Selling) ✅ **FULLY IMPLEMENTED** 
**WSB Pattern**: Most consistent WSB income strategy on liquid names & ETFs
**Scans for**: Quality names with decent volatility and dividends
**Builds**: Cash-secured puts (~30-45 DTE, ~0.30 delta) → covered calls cycle
**Exits**: Assignment → call away → repeat (theta decay income)
**Risk**: Income generation with positive expectancy
**WSB Success**: Multi-year "theta gang" income, especially on SPY/QQQ/PLTR

### 3. LEAPS Secular Winners ✅ **IMPLEMENTED** (🔄 Enhancement Needed)
**WSB Pattern**: "Buy time on high-beta winners" with rolling rules
**Scans for**: Secular growth themes (AI, Cloud, EVs, Fintech, etc.)
**Builds**: 12-24 month LEAPS 10-20% OTM on quality names
**Exits**: Systematic scale-out at 2x, 3x, 4x returns
**Risk**: Long-term capital deployment with diversification
**WSB Success**: Less screen-time stress, better odds than scalping
**🔄 TODO**: Add golden/death cross timing signals for entries

### 4. Momentum Weeklies Scanner ✅ **IMPLEMENTED**
**WSB Pattern**: Breakout swing with disciplined profit-taking
**Scans for**: Intraday reversals with 3x+ volume spikes on mega-caps
**Builds**: Weekly options 2-5% OTM based on momentum strength
**Exits**: Quick profit-taking same/next day (≈1 month max expiry)
**Risk**: 1-3% account risk per play
**WSB Success**: Fast profit-taking keeps theta manageable

### 5. Debit Call Spreads ✅ **IMPLEMENTED**
**WSB Pattern**: Defined-risk alternative to naked calls
**Scans for**: Bullish trends with favorable risk/reward ratios
**Builds**: Call spreads with 1.2+ risk/reward and 20%+ max profit
**Exits**: At max profit or breakeven
**Risk**: Reduced theta/IV exposure vs naked calls
**WSB Success**: More repeatable than naked options

### 6. 0DTE/Earnings Lotto Scanner ✅ **IMPLEMENTED** 
**WSB Pattern**: High-risk lottery plays with strict position sizing
**Scans for**: High-volatility 0DTE and earnings plays
**Builds**: Strict position sizing (0.5-1% account risk)
**Exits**: 50% stop loss, 3-5x profit targets
**Risk**: Extreme risk with disciplined position sizing
**WSB Warning**: Where most accounts blow up without discipline

## 🏗️ **IMPLEMENTATION STATUS:**

### ✅ **FULLY IMPLEMENTED & TESTED**
1. **WSB Dip Bot** - Exact WSB pattern replication with 240% gain methodology
2. **Momentum Weeklies Scanner** - Intraday reversal detection with volume analysis
3. **Debit Call Spreads** - Defined-risk spread strategies with Black-Scholes pricing
4. **LEAPS Tracker** - Long-term secular winners with systematic profit-taking
5. **Lotto Scanner** - 0DTE/earnings high-risk plays with strict position sizing
6. **Wheel Strategy** - Premium selling income generation (CSPs → CCs)
7. **Enhanced Swing Trading** - Fast breakout/momentum trades with same-day exits
8. **Backend Trading System** - Complete Django-integrated infrastructure with 118 tests

### 🔄 **MISSING WSB "WINNERS" - TODO LIST:**

### 9. SPX/SPY 0DTE Credit Spreads ❌ **TODO: IMPLEMENT**
**WSB Pattern**: Most cited "actually profitable" 0DTE strategy
**Strategy**: Sell ~30-delta defined-risk strangles/credit spreads at open
**Exits**: Auto-close at ~25% profit target (high win rate)
**Risk**: Occasional max-loss weeks, prefer SPX for tax/cash settlement
**Implementation Plan**: 
- Create `spx_credit_spreads.py` scanner
- Focus on SPX/SPY with defined risk
- Auto-close profit targets
- Track win rates and max loss periods

### 10. Earnings IV Crush Protection ❌ **TODO: IMPLEMENT**
**WSB Pattern**: Avoid lotto buying, structure around IV
**Strategy**: Deep ITM options or balanced hedges for earnings
**Problem**: Long straddles/strangles get crushed by IV collapse
**Implementation Plan**:
- Create `earnings_protection.py` module  
- Focus on IV-resistant structures
- Deep ITM options for earnings plays
- Calendar spreads to reduce IV risk

### 11. Index Fund Baseline Comparison ❌ **TODO: ADD**
**WSB Pattern**: "Boring baseline" that beats most WSB strategies
**Strategy**: SPY/VTI buy-and-hold comparison
**Purpose**: Reality check for all active strategies
**Implementation Plan**:
- Create `index_baseline.py` tracker
- Compare all strategy performance vs SPY/VTI
- Show risk-adjusted returns
- Humble pie for overconfident traders

## ⚠️ **WSB WARNINGS - What Usually Loses:**
- ❌ Naked strangles without defined risk (tail risk wipes out gains)
- ❌ 0DTE OTM lotto buys without exit plan
- ❌ Earnings lottos "for the move" (IV crush kills profits)
- ❌ No position sizing or stop rules

## 🏗️ Repository Structure

```
├── wsb_dip_bot.py              # ✅ Main WSB dip-after-run strategy
├── wheel_strategy.py           # ✅ Premium selling (CSPs → CCs)
├── leaps_tracker.py            # ✅ Long-term secular winners (needs enhancement)
├── momentum_weeklies.py        # ✅ Intraday reversal scanner
├── debit_spreads.py            # ✅ Defined-risk call spreads
├── lotto_scanner.py            # ✅ 0DTE/earnings lottery plays
├── spx_credit_spreads.py       # ❌ TODO: SPX 0DTE credit spreads
├── earnings_protection.py      # ❌ TODO: IV crush protection strategies
├── swing_trading.py            # ✅ Enhanced breakout swing trading
├── index_baseline.py           # ❌ TODO: SPY/VTI baseline comparison
├── wsb_requirements.txt        # Dependencies for all WSB bots
├── backend/tradingbot/         # ✅ Django-integrated trading modules (FULLY TESTED)
│   ├── options_calculator.py   # ✅ Black-Scholes pricing engine (36 behavioral tests)
│   ├── market_regime.py        # ✅ Market regime detection (19 accuracy tests)
│   ├── risk_management.py      # ✅ Position sizing & Kelly Criterion (20 mathematical tests)
│   ├── exit_planning.py        # ✅ Systematic exit strategies
│   ├── alert_system.py         # ✅ Multi-channel alerts (8 integration tests)
│   ├── exact_clone.py          # ✅ Exact replica of successful trade
│   ├── production_scanner.py   # ✅ Production-ready integrated scanner (34 tests)
│   ├── dip_scanner.py          # ✅ Dip detection algorithms (7 tests)
│   ├── trading_system.py       # ✅ Core trading system integration (14 tests)
│   ├── test_options_calculator.py # ✅ 36 behavioral verification tests
│   ├── test_market_regime_verification.py # ✅ 19 mathematical accuracy tests
│   ├── test_risk_management_verification.py # ✅ 20 Kelly Criterion tests
│   ├── test_strategy_smoke.py  # ✅ 19 smoke tests (all strategies)
│   ├── test_production_scanner.py # ✅ Production scanner test suite
│   └── test_suite.py           # ✅ Master test suite (118 total tests)
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
- **Comprehensive testing** suite (118 behavioral verification tests)
- **Market regime detection** for adaptive strategies
- **Systematic exit planning** with profit targets

### 🧪 Comprehensive Testing Infrastructure ✅ **FULLY IMPLEMENTED**
- **118 Total Tests** across all trading modules
- **Behavioral Verification Tests** - Test actual strategy behavior, not just smoke tests
- **Mathematical Accuracy Tests** - Verify Black-Scholes, Kelly Criterion, technical analysis formulas
- **Model Validation** - Ensure options pricing accuracy with put-call parity verification
- **Risk Management Tests** - Validate position sizing and risk calculations
- **Strategy Integration Tests** - End-to-end testing of complete trading workflows
- **Production Scanner Tests** - Full test coverage for live market scanning
- **Continuous Integration Ready** - All tests pass with 100% success rate

Run the full test suite:
```bash
# Run all 118 tests
venv/bin/python -m pytest backend/tradingbot/ -v

# Run specific test categories
venv/bin/python -m pytest backend/tradingbot/test_options_calculator.py -v      # 36 BS tests
venv/bin/python -m pytest backend/tradingbot/test_market_regime_verification.py -v  # 19 TA tests  
venv/bin/python -m pytest backend/tradingbot/test_risk_management_verification.py -v # 20 Kelly tests
venv/bin/python -m pytest backend/tradingbot/test_strategy_smoke.py -v         # 19 smoke tests
```

## 🎯 When to Use Each Strategy

### **WSB Dip Bot** - The Original Viral Strategy
**Use When:**
- You want to replicate the exact WSB pattern that produces viral gains
- You're comfortable with high risk (can lose 70-100% of position)
- You have capital you can afford to lose completely
- You want to go "all-in" on single trades (like the original 240% gain)

**Market Conditions:**
- Mega-caps (AAPL, MSFT, GOOGL, META, NVDA, TSLA) in bull market
- Stock has run up +10%+ over 10 days, then pulls back hard (-3%+)
- You can monitor positions actively (1-2 day holds max)

**Risk Level:** ⚠️⚠️⚠️ **EXTREME** - Can lose entire position

---

### **Momentum Weeklies Scanner** - Intraday Reversals
**Use When:**
- You want to catch intraday momentum moves
- You prefer shorter timeframes (same/next day exits)
- You want to trade weekly options for quick profits
- You can monitor markets actively during trading hours

**Market Conditions:**
- High volume days with 3x+ average volume spikes
- Intraday reversals on mega-caps
- Strong momentum continuation patterns
- Market volatility is elevated

**Risk Level:** ⚠️⚠️⚠️ **HIGH** - Quick time decay risk

---

### **Debit Call Spreads** - Reduced Risk Strategy
**Use When:**
- You want bullish exposure with limited risk
- You prefer defined risk over unlimited upside
- You want to reduce theta decay and IV crush exposure
- You're looking for more consistent, repeatable profits

**Market Conditions:**
- Bullish trend with favorable risk/reward ratios
- Moderate volatility (not too high, not too low)
- You want to avoid the "all-or-nothing" nature of naked calls
- You prefer systematic, disciplined approaches

**Risk Level:** ⚠️⚠️ **MEDIUM** - Limited upside but controlled risk

---

### **LEAPS Secular Winners** - Long-term Growth
**Use When:**
- You want to invest in long-term secular trends
- You have patience for 12-24 month timeframes
- You want to capture major thematic moves (AI, Cloud, EVs, etc.)
- You prefer systematic profit-taking over timing exits

**Market Conditions:**
- Secular growth themes are in favor
- You want to avoid short-term market noise
- Quality companies with strong fundamentals
- You can commit capital for extended periods

**Risk Level:** ⚠️⚠️ **MEDIUM** - Long-term capital commitment

---

### **0DTE/Earnings Lotto Scanner** - High Risk/High Reward
**Use When:**
- You want to gamble on high-volatility events
- You're comfortable with most trades expiring worthless
- You want strict position sizing discipline
- You're looking for the few big winners to pay for many losers

**Market Conditions:**
- High volatility periods (earnings, major events)
- You want to trade 0DTE options for maximum leverage
- You can accept that 80-90% of trades will lose money
- You have strict risk management discipline

**Risk Level:** ⚠️⚠️⚠️ **EXTREME** - Most trades expire worthless

---

### **Wheel Strategy** - Income Generation
**Use When:**
- You want consistent income generation
- You prefer lower-risk, systematic approaches
- You want to generate returns in sideways markets
- You're comfortable with assignment risk

**Market Conditions:**
- Quality companies with decent volatility
- You want income generation over capital appreciation
- You prefer positive expected value over time
- You want to avoid the stress of timing entries/exits

**Risk Level:** ⚠️ **LOWER** - Income generation with assignment risk

---

## 🤔 Quick Strategy Selector

**Ask yourself these questions:**

1. **What's your risk tolerance?**
   - **Extreme risk** → WSB Dip Bot or Lotto Scanner
   - **High risk** → Momentum Weeklies
   - **Medium risk** → Debit Spreads or LEAPS
   - **Lower risk** → Wheel Strategy

2. **What's your time horizon?**
   - **Same/next day** → Momentum Weeklies or Lotto Scanner
   - **1-2 days** → WSB Dip Bot
   - **Weeks to months** → Debit Spreads
   - **12-24 months** → LEAPS
   - **Ongoing income** → Wheel Strategy

3. **What's your trading style?**
   - **"All-in" gambler** → WSB Dip Bot
   - **Active day trader** → Momentum Weeklies
   - **Systematic trader** → Debit Spreads
   - **Long-term investor** → LEAPS
   - **Income seeker** → Wheel Strategy
   - **Lottery player** → Lotto Scanner

4. **What's the market condition?**
   - **Bull market with dips** → WSB Dip Bot
   - **High volatility** → Momentum Weeklies or Lotto Scanner
   - **Steady uptrend** → Debit Spreads
   - **Secular themes** → LEAPS
   - **Sideways market** → Wheel Strategy

---

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
pip install -r requirements.txt

# Optional: Install with console scripts for easier CLI usage
pip install -e .

# Test installation
python momentum_weeklies.py --help
python debit_spreads.py --help
python leaps_tracker.py --help
python lotto_scanner.py --help
python wheel_strategy.py --help
python wsb_dip_bot.py --help

# Or use console scripts (if installed with -e .)
wsb-dip-bot --help
momentum-weeklies --help
debit-spreads --help
leaps-tracker --help
lotto-scanner --help
wheel-strategy --help
```

### Configuration
```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys and settings
nano .env
```

### Docker Setup
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual container
docker build -t wallstreetbots .
docker run -it --env-file .env wallstreetbots
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
pip install -r requirements.txt

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
