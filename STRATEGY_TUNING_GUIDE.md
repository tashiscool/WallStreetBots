# 🎯 Strategy Tuning Guide - Make Money by Optimizing Parameters

## 📋 **Overview**

Each WallStreetBots strategy has configurable parameters that dramatically affect performance. This guide shows you **exactly what to tweak** to optimize profitability for current market conditions.

**Key Principle**: Start conservative, test extensively, then gradually optimize based on actual performance data.

---

## 🚀 **1. WSB Dip Bot - Most Popular Strategy**

### **What It Does**
Buys calls on stocks that dip after big runs (classic WSB pattern)

### **Key Parameters to Tune**

#### **Entry Conditions** (`production_wsb_dip_bot.py`)
```python
# How big a run to require before watching for dips
min_run_percentage = 0.10  # Default: 10%

# Conservative (fewer signals, higher quality)
min_run_percentage = 0.15  # Require 15% run

# Aggressive (more signals, lower quality)
min_run_percentage = 0.08  # Accept 8% runs

# How big a dip to wait for
min_dip_percentage = 0.05  # Default: 5%

# More patient (wait for bigger dips)
min_dip_percentage = 0.08  # Wait for 8% dip

# More aggressive (enter smaller dips)
min_dip_percentage = 0.03  # Enter on 3% dips
```

#### **Position Sizing**
```python
# Risk per position
max_position_risk = 0.05  # Default: 5% of account

# Conservative
max_position_risk = 0.02  # 2% per trade

# Aggressive (if proven profitable)
max_position_risk = 0.08  # 8% per trade
```

#### **Exit Rules**
```python
# Profit taking
profit_target_multiplier = 3.0  # Default: 3x premium

# Take profits faster (higher win rate, smaller gains)
profit_target_multiplier = 2.0  # 2x premium

# Hold for bigger gains (lower win rate, bigger wins)
profit_target_multiplier = 4.0  # 4x premium

# Time-based exit
max_hold_days = 21  # Default: 21 days

# Shorter holds (reduce theta decay)
max_hold_days = 14

# Longer holds (let winners run)
max_hold_days = 30
```

### **Market Condition Adaptations**

#### **Bull Market Settings** (SPY trending up)
```python
min_run_percentage = 0.08    # Lower bar, more opportunities
min_dip_percentage = 0.03    # Enter smaller dips
profit_target_multiplier = 2.5  # Take profits faster
max_position_risk = 0.06     # Slightly more aggressive
```

#### **Bear Market Settings** (SPY trending down)
```python
min_run_percentage = 0.20    # Much higher bar
min_dip_percentage = 0.10    # Wait for bigger dips
profit_target_multiplier = 4.0  # Hold for bigger gains
max_position_risk = 0.03     # Much more conservative
```

#### **Sideways Market Settings** (choppy, no clear trend)
```python
min_run_percentage = 0.12    # Moderate requirements
min_dip_percentage = 0.06    # Moderate dip requirement
profit_target_multiplier = 2.0  # Take profits quickly
max_position_risk = 0.04     # Conservative sizing
```

---

## 📊 **2. Earnings Protection Strategy**

### **What It Does**
Protects against IV crush around earnings using spreads and hedges

### **Key Parameters**
```python
# IV percentile threshold (when to trade)
min_iv_percentile = 50  # Default: trade when IV > 50th percentile

# Conservative (only trade high IV)
min_iv_percentile = 70  # Only trade when IV > 70th percentile

# Aggressive (trade more often)
min_iv_percentile = 30  # Trade when IV > 30th percentile

# Days before earnings to enter
entry_window_days = 7  # Default: 7 days before

# Enter closer to earnings (higher IV, more risk)
entry_window_days = 3

# Enter further from earnings (lower IV, less risk)
entry_window_days = 14
```

---

## ⚡ **3. Momentum Weeklies Strategy**

### **What It Does**
Quick momentum plays with weekly options for same-day exits

### **Key Parameters**
```python
# Volume spike requirement
min_volume_spike = 2.0  # Default: 2x average volume

# Conservative (wait for bigger spikes)
min_volume_spike = 3.0  # 3x average volume

# Aggressive (catch smaller moves)
min_volume_spike = 1.5  # 1.5x average volume

# Price movement requirement
min_price_move = 0.02  # Default: 2% move

# Conservative (bigger moves only)
min_price_move = 0.03  # 3% move

# Aggressive (smaller moves)
min_price_move = 0.015  # 1.5% move

# Maximum hold time
max_hold_hours = 6  # Default: 6 hours max

# Quicker exits (reduce risk)
max_hold_hours = 4

# Longer holds (let momentum run)
max_hold_hours = 8
```

---

## 🛡️ **4. Index Baseline Strategy**

### **What It Does**
"Boring" baseline strategy that beats most WSB plays through systematic SPY/QQQ exposure

### **Key Parameters**
```python
# Rebalancing frequency
rebalance_days = 30  # Default: monthly rebalancing

# More frequent (capture moves faster)
rebalance_days = 14  # Bi-weekly

# Less frequent (lower transaction costs)
rebalance_days = 60  # Every 2 months

# Risk tolerance
target_volatility = 0.15  # Default: 15% annualized vol

# More conservative
target_volatility = 0.10  # 10% vol target

# More aggressive
target_volatility = 0.20  # 20% vol target
```

---

## 🎯 **Risk Management Parameters**

### **Portfolio-Level Settings**
```python
# Maximum total portfolio risk
max_total_risk = 0.20  # Default: 20% of account

# Conservative portfolio
max_total_risk = 0.10  # 10% max risk

# Aggressive portfolio (if proven profitable)
max_total_risk = 0.30  # 30% max risk

# Maximum single position size
max_position_size = 0.05  # Default: 5% per position

# Conservative
max_position_size = 0.02  # 2% per position

# Aggressive
max_position_size = 0.08  # 8% per position
```

### **Stop Loss Settings**
```python
# Portfolio-level stop loss
portfolio_stop_loss = 0.15  # Stop all trading if down 15%

# Conservative (protect capital)
portfolio_stop_loss = 0.10  # Stop at 10% drawdown

# Aggressive (ride out volatility)
portfolio_stop_loss = 0.25  # Stop at 25% drawdown
```

---

## 📈 **Optimization Process**

### **Step 1: Baseline Testing (Month 1)**
1. Start with **default parameters**
2. Run **paper trading for 30 days**
3. Track **all performance metrics**:
   - Win rate
   - Average gain/loss
   - Maximum drawdown
   - Profit factor
   - Sharpe ratio

### **Step 2: Parameter Optimization (Month 2)**
1. **Identify worst-performing trades**
2. **Adjust entry conditions** to filter out bad trades
3. **Test one parameter change at a time**
4. **Compare 2-week performance** before/after changes

### **Step 3: Market Adaptation (Month 3+)**
1. **Monitor market regime** (bull/bear/sideways)
2. **Adjust parameters based on current conditions**
3. **Test adaptations in paper trading first**
4. **Gradually implement in live trading**

---

## 🔧 **How to Implement Changes**

### **Method 1: Configuration Files**
Create custom configuration files:
```python
# config/aggressive_settings.py
WSB_DIP_BOT_CONFIG = {
    'min_run_percentage': 0.08,
    'min_dip_percentage': 0.03,
    'profit_target_multiplier': 2.5,
    'max_position_risk': 0.06
}

# config/conservative_settings.py
WSB_DIP_BOT_CONFIG = {
    'min_run_percentage': 0.15,
    'min_dip_percentage': 0.08,
    'profit_target_multiplier': 4.0,
    'max_position_risk': 0.03
}
```

### **Method 2: Strategy Profiles**
Use built-in strategy profiles:
```python
# Conservative profile
config = ProductionStrategyManagerConfig(
    profile=StrategyProfile.research_2024,  # Conservative
    max_total_risk=0.10,
    max_position_size=0.02
)

# Aggressive profile (only after proving profitability)
config = ProductionStrategyManagerConfig(
    profile=StrategyProfile.wsb_2025,  # Aggressive
    max_total_risk=0.25,
    max_position_size=0.06
)
```

### **Method 3: Dynamic Adaptation**
Implement market regime detection:
```python
def get_market_regime():
    """Detect current market regime."""
    spy_returns = get_spy_returns(lookback_days=20)

    if spy_returns.mean() > 0.001:  # 0.1% daily average
        return "bull"
    elif spy_returns.mean() < -0.001:
        return "bear"
    else:
        return "sideways"

def adapt_parameters(current_regime):
    """Adapt strategy parameters to market regime."""
    if current_regime == "bull":
        return BULL_MARKET_CONFIG
    elif current_regime == "bear":
        return BEAR_MARKET_CONFIG
    else:
        return SIDEWAYS_MARKET_CONFIG
```

---

## 📊 **Performance Tracking**

### **Key Metrics to Monitor**
```python
# Track these metrics weekly
performance_metrics = {
    'total_return': 0.15,          # 15% return
    'win_rate': 0.65,              # 65% of trades profitable
    'profit_factor': 1.8,          # Total gains / Total losses
    'max_drawdown': 0.08,          # 8% maximum drawdown
    'sharpe_ratio': 1.2,           # Risk-adjusted returns
    'avg_gain': 0.25,              # 25% average gain
    'avg_loss': -0.12,             # 12% average loss
    'avg_hold_days': 8,            # 8 days average hold
}
```

### **When to Adjust Parameters**
- **Win rate < 40%**: Tighten entry conditions
- **Profit factor < 1.2**: Improve exit timing
- **Max drawdown > 15%**: Reduce position sizes
- **Sharpe ratio < 0.8**: Improve risk-adjusted returns

---

## ⚠️ **Important Warnings**

### **Don't Over-Optimize**
- ❌ **Curve fitting**: Parameters that work perfectly on past data often fail forward
- ❌ **Too many changes**: Change one parameter at a time
- ❌ **Impatience**: Test each change for at least 2 weeks

### **Market Conditions Change**
- ✅ **Bull market 2021**: Very different from **bear market 2022**
- ✅ **What works now** may not work in 6 months
- ✅ **Stay adaptive** and keep testing

### **Position Sizing is Critical**
- ✅ **Even great strategies** can lose money with poor position sizing
- ✅ **Start small** and scale up gradually
- ✅ **Never risk more** than you can afford to lose

---

## 🎯 **Summary**

**Success Formula**:
1. **Start with default parameters**
2. **Paper trade for 30+ days**
3. **Identify what's not working**
4. **Make ONE change at a time**
5. **Test for 2+ weeks**
6. **Repeat until profitable**
7. **Scale up gradually with real money**

**Remember**: The best parameters are the ones that work consistently in YOUR market conditions with YOUR risk tolerance. There's no universal "best" settings - you must find what works for you through systematic testing.

**🚀 Good luck, and trade safely!**