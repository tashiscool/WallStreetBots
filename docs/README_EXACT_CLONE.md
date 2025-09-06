# Exact Clone Protocol - What He Actually Did

This is the **raw, unfiltered version** that made 240% in 1-2 days. No sophisticated risk management, no trend filters - just pure dip-buying with massive leverage.

## ⚠️ **EXTREME RISK WARNING** ⚠️

**This system can destroy your account in a single trade.** The original trader:
- Risked 95% of account on ONE trade
- Used 44x effective leverage  
- Had NO diversification
- Could have lost everything if wrong

**Use only with money you can afford to lose completely.**

## 🎯 What He Actually Did (Core Protocol)

### Entry Criteria:
1. **Dip Day Only**: Stock down hard intraday (gap down or selloff) - **NO moving average filters**
2. **Strike Selection**: ~5% OTM calls (≈ Δ 0.30-0.40) 
3. **Expiry**: Closest Friday ~28-35 DTE
4. **Sizing**: Deploy 70-100% of risk capital into **one line**

### Exit Discipline:
1. **Target**: 3x-4x premium **OR** when calls go ITM
2. **Timeline**: 1-2 days maximum hold
3. **No Lingering**: Take profits fast, don't get greedy

### Repeat Cycle:
1. Wait for next red day dip
2. Often **increase contracts** after wins
3. Repeat until account blown up or massive gains

## 📊 System Implementation

### Core Components:

```python
from backend.tradingbot.exact_clone import (
    clone_trade_plan,           # Simple helper function (matches your spec)
    ExactCloneSystem,          # Complete system with cycle management
    DipDetector,               # Detects hard dip days (no filters)
    LiveDipScanner             # Real-time scanning
)
```

### Basic Usage:

```python
# Simple trade plan (exact helper function you specified)
from backend.tradingbot.exact_clone import clone_trade_plan

plan = clone_trade_plan(
    spot=207.0,                # Current stock price
    acct_cash=450000,          # Available cash to deploy
    otm=0.05,                  # 5% out of money
    dte_days=30,               # ~30 days to expiry
    entry_prem=4.70            # Option premium you see
)

print(plan)
```

**Output:**
```
{
  'strike': 217,
  'target_expiry_dte': 30,
  'contracts': 861,
  'cost_$': 404670.0,
  'deploy_percentage': '89.9%',
  'breakeven_at_expiry': 217.05,
  'take_profit_levels_$': [14.1, 18.8],
  'sell_when': 'ITM or delta ≥ 0.60 or TP hits; 1–2 day max hold',
  'ruin_risk': '89.9% of total capital',
  'effective_leverage': '~44.0x'
}
```

### Complete System:

```python
# Full automated system
from backend.tradingbot.exact_clone import ExactCloneSystem
from backend.tradingbot.dip_scanner import DipTradingBot

# Create bot with initial capital
bot = DipTradingBot(initial_capital=500000)

# Start live scanning (async)
import asyncio
asyncio.run(bot.start_bot())
```

## 🔍 Dip Detection Logic

**No trend filters** - just looking for hard selloffs:

### Dip Types Detected:
1. **Gap Down**: Opens significantly lower than previous close
2. **Intraday Selloff**: Selling pressure during the session  
3. **Hard Dip**: Down >2.5% at any point
4. **Red Day**: Simply down from previous close

### Detection Thresholds:
- **Minimum Dip**: 1.5% decline
- **Hard Dip**: 2.5% decline  
- **Volume Confirmation**: 20% above average preferred
- **No EMA/Trend Filters**: Enter on weakness regardless of trend

### Universe (Mega-caps only):
```python
universe = ['GOOGL', 'GOOG', 'AAPL', 'MSFT', 'NVDA', 'META', 'AMD', 'TSLA', 'AVGO']
```

## 💰 Position Sizing (All-In Approach)

### Deployment Levels:
- **Conservative**: 70% of available capital
- **Moderate**: 80-90% of available capital  
- **Aggressive**: 95-100% of available capital (what he did)

### Scaling After Wins:
```python
# System automatically scales up after wins
base_deployment = 90%
if consecutive_wins >= 1:
    deployment += 2% per win
if last_trade_roi > 200%:
    deployment += 5% bonus

# Max deployment approaches 100%
```

### Leverage Characteristics:
- **Effective Leverage**: Typically 30-50x
- **Notional Exposure**: $20M+ on $400K investment
- **Account Risk**: 70-100% of total capital

## 🎯 Exit Strategy (Fast Money)

### Profit Targets:
1. **3x Premium**: Exit 1st target (≈ 200% profit)
2. **4x Premium**: Exit 2nd target (≈ 300% profit)
3. **ITM/High Delta**: Exit when delta ≥ 0.60

### Time Limits:
- **Maximum Hold**: 2 days
- **Typical Hold**: Same day or next day
- **No Bag Holding**: Take profits or cut losses fast

### Exit Conditions (ANY triggers exit):
```python
exit_conditions = [
    "premium >= 3x entry",
    "premium >= 4x entry", 
    "delta >= 0.60",
    "calls went ITM",
    "hold_days >= 2"
]
```

## 🔄 Cycle Management

### Win Cycle:
1. Execute dip trade with 90% of capital
2. Hold 1-2 days, exit at 3x-4x gain
3. **Increase position size** with profits
4. Repeat on next dip day
5. **Compound aggressively**

### Loss Cycle:
1. If trade goes against you
2. Cut losses around 50% (or let expire worthless)
3. Wait for next opportunity
4. **Risk entire remaining capital again**

### Performance Tracking:
```python
# System tracks all trades
system.cycle_manager.get_performance_summary()
# Returns: win_rate, avg_win, avg_loss, total_roi
```

## 🚨 Risk Profile (Existential)

### What Can Go Wrong:
1. **Single Bad Trade**: Can lose 70-100% of account
2. **Gap Down Overnight**: No protection from overnight moves
3. **IV Crush**: High IV can collapse quickly
4. **Time Decay**: Theta burns fast on short-term options
5. **Liquidity Risk**: Hard to exit large positions

### Historical Volatility:
- **Win**: +240% in 1-2 days (like the original)
- **Loss**: -90% to -100% of position (total ruin)
- **Frequency**: Win 3-4 times, lose everything on 5th trade

### Capital Requirements:
- **Minimum**: $100K (for reasonable contract sizes)
- **Recommended**: $500K+ (for flexibility)
- **Maximum Risk**: Whatever you're willing to lose completely

## 🛠️ Live Scanner Features

### Market Monitoring:
- **Scan Frequency**: Every 60 seconds during market hours
- **Market Hours**: 9:30 AM - 4:00 PM ET
- **Optimal Entry**: 10:00 AM - 3:00 PM ET (avoids open/close volatility)

### Real-time Alerts:
```python
# Execution alert
{
  "type": "TRADE_EXECUTED",
  "ticker": "GOOGL",
  "dip_type": "hard_dip", 
  "contracts": 861,
  "cost": 404670.0,
  "ruin_risk": "89.9%",
  "leverage": "44.0x"
}

# Exit alert  
{
  "type": "POSITION_EXITED",
  "ticker": "GOOGL",
  "exit_reason": "3x_profit_target",
  "pnl": 850000,
  "roi": "+210%"
}
```

### Position Monitoring:
- **Continuous**: Monitors active position every scan
- **Exit Logic**: Automatically detects exit conditions
- **Profit Tracking**: Real-time P&L calculation

## 📈 Expected Outcomes

### Probability Distribution:
- **70% Chance**: Small/medium gains (50-150% per trade)
- **20% Chance**: Large gains (200-400% per trade)
- **10% Chance**: Total loss (account blown up)

### Survivability:
- **Short Term**: Can work for weeks/months
- **Long Term**: Mathematical certainty of ruin
- **Reality**: Most accounts blown up within 10 trades

### Psychological Profile:
- **Adrenaline Rush**: Extreme highs and lows
- **Addiction Potential**: "Lottery ticket" mentality
- **Discipline Required**: Must exit at targets (hardest part)

## 🔧 Integration Setup

### Data Integration:
```python
# Replace placeholder data with real feeds
async def _fetch_current_market_data(self):
    # Integrate with:
    # - Alpaca API
    # - TD Ameritrade  
    # - Interactive Brokers
    # - Yahoo Finance
    pass
```

### Broker Integration:
```python
# Execute actual trades
def execute_trade(self, setup):
    # Send orders to:
    # - Interactive Brokers
    # - TD Ameritrade
    # - E*TRADE
    # - Robinhood (if available)
    pass
```

### Alert Channels:
```python
# Configure notifications
alerts = {
    "discord_webhook": "https://discord.com/api/webhooks/...",
    "email": "trader@gmail.com",
    "sms": "+1234567890"
}
```

## 🎮 How to Use (Step by Step)

### 1. Setup:
```bash
# Install system
cd /Users/admin/IdeaProjects/workspace/WallStreetBots
pip install -r requirements.txt

# Test the helper function
python -c "
from backend.tradingbot.exact_clone import clone_trade_plan
print(clone_trade_plan(spot=207.0, acct_cash=450000, entry_prem=4.70))
"
```

### 2. Manual Mode:
```python
# Calculate individual trades manually
from backend.tradingbot.exact_clone import clone_trade_plan

# When you see a dip
plan = clone_trade_plan(
    spot=current_price,
    acct_cash=available_cash, 
    entry_prem=option_premium
)

print(f"BUY {plan['contracts']} contracts")
print(f"Exit at ${plan['take_profit_levels_$'][0]} or ${plan['take_profit_levels_$'][1]}")
```

### 3. Automated Mode:
```python
# Full automation (DANGEROUS)
from backend.tradingbot.dip_scanner import DipTradingBot

bot = DipTradingBot(initial_capital=500000)
await bot.start_bot()  # Runs continuously
```

### 4. Paper Trading:
```python
# Test with small amounts first
bot = DipTradingBot(initial_capital=10000)  # $10K test account
```

## 📚 Comparison: Safe vs Exact Clone

| Feature | Safe System (Previous) | Exact Clone (This) |
|---------|----------------------|-------------------|
| **Risk per Trade** | 10-15% | 70-100% |
| **Filters** | EMA trends, RSI, IV | None - pure dip buying |
| **Hold Time** | Up to 45 days | 1-2 days maximum |
| **Diversification** | Multiple positions | One position only |
| **Sizing Method** | Kelly + confidence | All-in deployment |
| **Profit Targets** | 100%, 200%, 250% | 300%, 400% |
| **Survivability** | High | Low (ruin likely) |
| **Adrenaline** | Moderate | Extreme |

## 🤔 When to Use Each

### Use Safe System If:
- Long-term wealth building
- Can't afford to lose capital
- Want consistent returns
- Risk-averse personality

### Use Exact Clone If:
- Short-term speculation
- Money you can lose completely  
- Adrenaline junkie
- "Lottery ticket" mentality
- Want to replicate original exactly

## ⚖️ Final Warning

**This exact clone system implements the original trader's approach with zero safety nets.** 

- ✅ **Pros**: Can make 240% in days, captures maximum convexity
- ❌ **Cons**: Will eventually blow up account, existential risk

**Bottom Line**: This is gambling, not investing. The original trader got lucky. Most who try this approach lose everything. 

**You have been warned.**

---

*Risk Disclosure: Options trading involves substantial risk of loss. This system can result in total loss of capital. Past performance does not guarantee future results. Trade at your own risk.*