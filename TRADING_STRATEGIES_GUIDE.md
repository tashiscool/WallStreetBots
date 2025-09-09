# WallStreetBots Trading Strategies Guide

*Complete documentation of all trading strategies implemented in the system*

---

## Overview

WallStreetBots implements **10 distinct trading strategies** inspired by successful WallStreetBets (WSB) tactics. Each strategy is designed with professional risk management and systematic execution. These strategies range from high-frequency day trading to long-term secular trend investing.

---

## Strategy Categories

### 🎯 **High-Frequency Strategies (0-3 days)**
1. WSB Dip-After-Run Bot
2. Momentum Weeklies Scanner  
3. SPX/SPY 0DTE Credit Spreads
4. Enhanced Swing Trading

### 📈 **Medium-Term Strategies (1-8 weeks)**  
5. Debit Call Spreads
6. Earnings Protection Strategy
7. Covered Calls/Wheel Strategy
8. 0DTE/Earnings Lotto Scanner

### 📊 **Long-Term Strategies (3+ months)**
9. LEAPS Secular Winners Tracker
10. Index Fund Baseline Tracker

---

# Detailed Strategy Breakdown

## 1. 🎯 WSB Dip-After-Run Bot

**The signature WSB strategy - buying the dip after big runs**

### How It Works:
- **Pattern Detection**: Identifies stocks that ran up 15%+ in 3-5 days, then dipped 3-8%
- **Entry Signal**: Buys ~5% OTM calls with ~30 days to expiration (DTE) on the dip day
- **Position Size**: Configurable risk (typically 1-2% of account)
- **Exit Criteria**: 3x profit OR delta ≥ 0.60 OR time/loss stops

### Example:
```
NVIDIA runs from $800 → $920 (+15%) over 4 days
Stock dips to $875 (-5% from peak)
Bot buys $920 calls expiring in 30 days
Target: 300% profit or delta above 0.60
```

### Risk Management:
- Maximum 2% of account per trade
- Automatic stop-loss at 50% of premium
- Time decay protection (exits if theta too high)

---

## 2. 📱 Momentum Weeklies Scanner

**Catches intraday reversals and news momentum for weekly options**

### How It Works:
- **Real-Time Scanning**: Monitors mega-cap stocks for sudden reversals
- **Signal Types**: Bullish reversals, news momentum, technical breakouts
- **Time Frame**: Weekly expiries (typically Friday)
- **Entry**: ATM or slightly OTM calls/puts based on direction

### Example:
```
TSLA drops 4% on FUD, then reverses +2% in 1 hour
High volume spike (3x average)
Bot flags weekly $250 calls
Target: Quick 50-100% gain same day
```

### Key Features:
- Volume spike detection (2x+ average)
- News sentiment analysis integration
- Same-day exit discipline
- Multiple profit target levels (25%, 50%, 100%)

---

## 3. 💰 SPX/SPY 0DTE Credit Spreads  

**The "most profitable 0DTE strategy" according to WSB**

### How It Works:
- **Daily Execution**: Sells ~30 delta credit spreads at market open
- **Spread Types**: Put credit spreads, call credit spreads, iron condors
- **Target**: 25% profit by market close
- **Risk**: Defined risk spreads (max loss = spread width - credit)

### Example:
```
SPY at $450 at open
Sell $440/$435 put credit spread (5-point spread)
Collect $1.25 credit per spread
Max profit: $125, Max loss: $375
Target close at 25% profit ($0.94)
```

### Advantages:
- High win rate (~70-80%)
- Defined risk
- Time decay advantage
- Consistent income generation

---

## 4. 🏃‍♂️ Enhanced Swing Trading

**Fast profit-taking swing trades with same-day exit discipline**

### How It Works:
- **Pattern Recognition**: Breakouts, momentum continuation, reversal patterns
- **Time Frame**: 1-3 days maximum hold
- **Options**: 7-30 DTE calls/puts
- **Management**: Multiple profit targets with partial exits

### Example:
```
AAPL breaks above $180 resistance on volume
Buy $185 calls expiring next Friday
Profit targets: 25% (exit 1/3), 50% (exit 1/3), 100% (exit 1/3)
Stop loss: 30% of premium paid
```

### Key Features:
- Breakout confirmation with volume
- Strength scoring algorithm (0-100)
- Multiple profit target system
- Same-day exit preference

---

## 5. 📊 Debit Call Spreads

**More repeatable than naked calls with reduced risk**

### How It Works:
- **Structure**: Buy ATM call, sell OTM call (vertical spread)
- **Benefits**: Lower cost, reduced theta/IV risk
- **Selection**: Strong trend stocks with good liquidity
- **Management**: Target 50% max profit, 30-45 DTE

### Example:
```
GOOGL trending upward, IV rank 40th percentile
Buy $2800 call, sell $2850 call (45 DTE)
Net debit: $25, Max profit: $25, Breakeven: $2825
Risk/Reward: 1:1 with 60% win probability
```

### Advantages:
- Defined risk and reward
- Less sensitive to IV changes
- Lower capital requirement than naked calls
- Higher win rate than naked options

---

## 6. 🏢 Earnings Protection Strategy

**Avoid the #1 WSB mistake - IV crush after earnings**

### How It Works:
- **Problem**: Traditional earnings plays get crushed by IV collapse
- **Solution**: IV-resistant structures and hedging
- **Strategies**: Deep ITM options, calendar spreads, protective hedges
- **Timing**: 1-3 days before earnings announcement

### Strategy Types:

#### A. **Deep ITM Protection**
```
AAPL earnings next week, stock at $180
Buy $160 calls (deep ITM) instead of ATM
Less IV sensitivity, more intrinsic value
```

#### B. **Calendar Spreads**
```
Sell front-month ATM calls, buy back-month ATM calls
Profit from IV crush on front month
```

#### C. **Protective Hedges**  
```
Long stock position + protective puts
Or long calls + protective puts (synthetic collar)
```

### Key Features:
- Real-time IV percentile calculations
- Earnings calendar integration
- Expected move calculations from straddle prices
- IV crush risk assessment

---

## 7. 🔄 Covered Calls/Wheel Strategy

**Consistent income generation on volatile stocks**

### How It Works:
- **Phase 1**: Sell cash-secured puts (collect premium)
- **Phase 2**: If assigned, sell covered calls on shares
- **Phase 3**: If called away, start over with puts
- **Goal**: Collect premium consistently while building positions

### Example Wheel Cycle:
```
Week 1: Sell PLTR $20 puts for $0.50 premium
Week 2: Assigned 100 shares at $20
Week 3: Sell $22 calls for $0.75 premium  
Week 4: Called away at $22
Total profit: $0.50 + $0.75 + $2 = $3.25 (16.25% return)
```

### Selection Criteria:
- High IV rank (>50th percentile)
- Stocks you don't mind owning
- Strong liquidity in options
- ~30-45 DTE for optimal theta decay

### Management:
- Close puts at 25% profit or 7 DTE
- Close calls at 25% profit or 7 DTE
- Roll options if needed to avoid assignment

---

## 8. 🎲 0DTE/Earnings Lotto Scanner

**High-risk, high-reward plays with strict position sizing**

### How It Works:
- **Target**: Very OTM options before major catalysts
- **Risk**: Extreme (total loss possible)
- **Position Size**: Maximum 0.25% of account per trade
- **Catalysts**: Earnings, FDA approvals, product launches

### Example Scenarios:

#### A. **0DTE Lottery**
```
SPY at $450, buy $455 calls expiring today for $0.10
Need 1.2% move in SPY to break even
Potential return: 1000%+ if SPY rallies 3%
```

#### B. **Earnings Lottery**
```
NVDA earnings after close, stock at $800
Buy $850 calls expiring Friday for $2.00
Need 6.25% move to break even
Potential return: 500%+ if beats estimates significantly
```

### Strict Rules:
- Never risk more than 0.25% of account
- Pre-defined profit targets (200-500%)
- Cut losses quickly (50% of premium)
- Maximum 2 lotto plays per week

---

## 9. 📈 LEAPS Secular Winners Tracker  

**Long-term positions on secular growth trends**

### How It Works:
- **Time Frame**: 1-3 year LEAPS options
- **Themes**: AI/ML, Cloud Computing, EV/Battery Tech, Renewable Energy
- **Management**: Systematic profit-taking (25%, 50%, 100% gains)
- **Rebalancing**: Quarterly review and position adjustments

### Secular Themes Tracked:

#### A. **Artificial Intelligence & Machine Learning**
- **Tickers**: NVDA, AMD, GOOGL, MSFT, TSLA
- **Drivers**: GPU demand, AI software, autonomous vehicles
- **Time Horizon**: 3-5 years

#### B. **Cloud Infrastructure & SaaS**
- **Tickers**: MSFT, AMZN, CRM, SNOW, DDOG  
- **Drivers**: Digital transformation, remote work, data growth
- **Time Horizon**: 3-7 years

#### C. **Electric Vehicles & Battery Tech**
- **Tickers**: TSLA, F, GM, RIVN, ALB (lithium)
- **Drivers**: Government mandates, cost parity, charging infrastructure
- **Time Horizon**: 5-10 years

### Example LEAPS Position:
```
Theme: AI/ML Growth
Ticker: NVDA
Entry: Jan 2025 $400 LEAPS at $800 stock price
Cost: $450 per contract
Profit targets: 25% ($562.50), 50% ($675), 100% ($900)
Stop loss: 40% of premium ($270)
```

### Management Rules:
- Take profits systematically at predetermined levels
- Rebalance quarterly based on theme strength
- Maximum 20% of portfolio in LEAPS
- Diversify across themes and time horizons

---

## 10. 📊 Index Fund Baseline Tracker

**Performance validation against passive investing**

### How It Works:
- **Benchmark**: All active strategies against SPY, VTI, QQQ
- **Metrics**: Total return, Sharpe ratio, maximum drawdown, alpha
- **Goal**: Prove active trading beats "just buy and hold"
- **Reporting**: Monthly and quarterly performance reviews

### Key Metrics Tracked:

#### A. **Return Metrics**
- Absolute returns vs. benchmarks
- Risk-adjusted returns (Sharpe ratio)
- Alpha generation over time

#### B. **Risk Metrics**  
- Maximum drawdown comparison
- Volatility vs. benchmarks
- Win rate and profit factor

#### C. **Efficiency Metrics**
- Returns per unit of risk
- Time spent vs. passive returns
- Transaction costs vs. excess returns

### Example Performance Report:
```
Period: Q3 2024
Strategy Portfolio: +18.5%
SPY Benchmark: +12.2%  
VTI Benchmark: +11.8%
QQQ Benchmark: +15.1%

Alpha vs SPY: +6.3%
Alpha vs VTI: +6.7%
Alpha vs QQQ: +3.4%

Sharpe Ratio: 2.1 vs 1.4 (SPY)
Max Drawdown: -8.2% vs -5.1% (SPY)
```

---

# Risk Management Framework

## Universal Risk Controls

### 1. **Position Sizing**
- Maximum 2% of account per individual trade
- Maximum 10% of account per strategy
- Maximum 25% of account in all active positions

### 2. **Loss Limits**  
- Individual trade stop loss: 50% of premium (options)
- Daily loss limit: 1% of account
- Weekly loss limit: 3% of account
- Monthly loss limit: 8% of account

### 3. **Profit Taking**
- Systematic profit targets at 25%, 50%, 100%, 200%
- Trailing stops on profitable positions
- Time decay management (close options with <7 DTE if not ITM)

### 4. **Diversification**
- No more than 20% in any single underlying
- Spread across multiple strategies
- Mix of directional and non-directional trades

---

# Technology Integration

## Data Sources
- **Real-time**: Alpaca Markets API
- **Historical**: Yahoo Finance, Alpha Vantage  
- **Options**: Polygon.io, TDAmeritrade
- **Earnings**: Multiple calendar providers with failover

## Execution Platform
- **Primary**: Alpaca Markets (commission-free)
- **Backup**: Interactive Brokers API
- **Paper Trading**: Full simulation environment

## Risk Systems
- Real-time position monitoring
- Automatic stop-loss execution  
- Portfolio heat monitoring
- Margin requirement calculations

---

# Performance Expectations

## Realistic Targets

### **Conservative Portfolio** (Lower risk)
- Target Annual Return: 15-25%
- Expected Win Rate: 55-65%
- Maximum Drawdown: <15%
- Sharpe Ratio: >1.5

### **Aggressive Portfolio** (Higher risk)
- Target Annual Return: 25-40%  
- Expected Win Rate: 50-60%
- Maximum Drawdown: <25%
- Sharpe Ratio: >1.2

### **Balanced Portfolio** (Recommended)
- Target Annual Return: 20-30%
- Expected Win Rate: 55-65%
- Maximum Drawdown: <18%
- Sharpe Ratio: >1.4

---

# Getting Started

## Phase 1: Paper Trading (1-3 months)
1. Enable all strategies in simulation mode
2. Track performance vs. benchmarks
3. Learn strategy behaviors and risk characteristics
4. Optimize position sizing and risk parameters

## Phase 2: Small Capital Testing (3-6 months)  
1. Start with $10,000-25,000 real money
2. Run 2-3 strategies simultaneously
3. Focus on risk management and discipline
4. Gradually add complexity

## Phase 3: Full Deployment (6+ months)
1. Scale to full capital allocation
2. Enable all relevant strategies
3. Implement systematic rebalancing
4. Continuous performance monitoring

---

*Remember: Past performance doesn't guarantee future results. All strategies involve substantial risk of loss. Never risk more than you can afford to lose.*