# 🚀 WallStreetBots Quick Start (5 Minutes!)

> **Perfect for beginners!** Get WSB-style algorithmic trading running with paper money in under 5 minutes.

## 🎯 What You'll Get
- **10 WSB Trading Strategies** running automatically
- **Paper Trading** (fake money - 100% safe!)
- **Two Risk Profiles**: Conservative vs WSB Aggressive
- **Real-time Options Trading** with proper risk management

---

## 📋 Step 1: Get Free Alpaca Account (30 seconds)

1. **Go to**: [alpaca.markets](https://alpaca.markets)
2. **Click**: "Sign Up" (100% free, no credit card needed)
3. **Navigate to**: "Paper Trading" → "API Keys"  
4. **Copy**: Your `API Key` and `Secret Key` (keep these private!)

---

## 💻 Step 2: One-Command Setup

Copy and paste this into your terminal/command prompt:

```bash
git clone https://github.com/yourusername/WallStreetBots.git
cd WallStreetBots
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install alpaca-py>=0.42.0
python manage.py migrate
```

**Windows Users**: Replace `source venv/bin/activate` with `venv\Scripts\activate`

---

## 🔑 Step 3: Set Up Your API Keys

```bash
# Copy the environment template
cp .env.example .env

# Edit the .env file (use any text editor)
nano .env
# OR
code .env   # If you have VS Code
# OR
notepad .env   # Windows
```

**Edit these lines in `.env` file:**
```bash
ALPACA_API_KEY=paste_your_actual_api_key_here
ALPACA_SECRET_KEY=paste_your_actual_secret_key_here
```

---

## 🎮 Step 4: Start Trading!

### Option A: Use the Quick Start Script (Easiest)

1. **Run**: `python quickstart.py`
2. The script will automatically load your API keys from the `.env` file

### Option B: Manual Setup (More Control)

Create a file called `my_trading.py`:

```python
import os
from backend.tradingbot.production.core.production_strategy_manager import (
    ProductionStrategyManagerConfig, ProductionStrategyManager, StrategyProfile
)

# 🛡️ CONSERVATIVE PROFILE (Recommended for beginners)
config = ProductionStrategyManagerConfig(
    alpaca_api_key=os.getenv('ALPACA_API_KEY'),      # Loads from .env file
    alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY'), # Loads from .env file
    paper_trading=True,                              # ✅ SAFE: Fake money only
    profile=StrategyProfile.research_2024,           # Conservative settings
)

# Start the system  
manager = ProductionStrategyManager(config)
print(f"🚀 Ready! Loaded {len(manager.strategies)}/10 strategies")
print(f"📊 Profile: {config.profile} (Max Risk: {config.max_total_risk:.0%})")
```

Then run: `python my_trading.py`

---

## 🔥 Step 4: Upgrade to WSB Mode (Optional)

**For aggressive WSB-style trading**, change one line:

```python
# Change this line:
profile=StrategyProfile.research_2024,       # Conservative

# To this:
profile=StrategyProfile.wsb_2025,           # 🔥 WSB Aggressive!
```

**WSB Profile includes:**
- ✅ **0DTE Options** (same-day expiry) 
- ✅ **Meme Stocks** (TSLA, NVDA, GME, etc.)
- ✅ **65% Max Risk** (vs 50% conservative)
- ✅ **10s Refresh Rate** (vs 30s conservative)
- ✅ **Higher Position Sizes** (30% vs 20%)

---

## 🎉 What Happens Next?

Once running, you'll see:

```
🚀 WallStreetBots Quick Start
==================================================
🔧 Initializing with research_2024 profile...
✅ Success! Loaded 10/10 strategies
📊 Trading Profile: research_2024  
🛡️ Max Portfolio Risk: 50%
💰 Max Position Size: 20%
⏱️ Data Refresh: 30s
🔒 Paper Trading: ✅ ENABLED

📋 Active Strategies:
    1. wsb_dip_bot
    2. earnings_protection  
    3. index_baseline
    4. wheel_strategy
    5. momentum_weeklies
    6. debit_spreads
    7. leaps_tracker
    8. swing_trading
    9. spx_credit_spreads
   10. lotto_scanner

🎯 System Status: 🟢 Ready
==================================================
🎉 READY TO TRADE!  
⚠️  Remember: This is PAPER TRADING (fake money)
```

---

## 🛡️ Safety First!

### ✅ What's Safe:
- **Paper Trading**: Uses fake money, zero risk
- **Conservative Profile**: Longer-dated options, lower risk
- **Built-in Stops**: Automatic loss protection  
- **Position Limits**: Can't risk more than configured %

### ⚠️ Before Real Money:
- **Test extensively** with paper trading first
- **Understand each strategy** before going live
- **Start small** when switching to real money
- **Monitor actively** - algorithmic ≠ set-and-forget

---

## 🤔 Troubleshooting

### "Import Error" or "Module not found"
```bash
pip install alpaca-py>=0.42.0
pip install django>=4.0
pip install yfinance pandas numpy
```

### "API Key Invalid"
- Double-check your keys from Alpaca dashboard
- Make sure you're using **Paper Trading** keys
- Verify your account is activated

### "Database Error"  
```bash
python manage.py migrate
python manage.py makemigrations tradingbot
python manage.py migrate
```

### Still Stuck?
- Check the full **README.md** for detailed setup
- Review the **FULL SETUP GUIDE** section
- All strategies are unit tested and working

---

## 📈 What Each Strategy Does

| Strategy | What It Does | Risk Level |
|----------|-------------|------------|
| **WSB Dip Bot** | Buys calls on dips after big runs | 🔥 High |
| **Earnings Protection** | Plays earnings with straddles/calendars | 🔥 High |  
| **Index Baseline** | Boring SPY/QQQ baseline (beats most WSB plays) | 🛡️ Low |
| **Wheel Strategy** | Sells puts → covered calls rotation | 📊 Medium |
| **Momentum Weeklies** | Quick momentum plays with weeklies | 🔥 High |
| **Debit Spreads** | Call spreads with reduced theta risk | 📊 Medium |
| **LEAPS Tracker** | Long-term growth plays with profit taking | 📊 Medium |
| **Swing Trading** | Fast 1-2 day breakout plays | 🔥 High |
| **SPX Credit Spreads** | 0DTE credit spreads on SPX | 🔥🔥 Extreme |
| **Lotto Scanner** | 0DTE lottery ticket scanner | 🔥🔥🔥 YOLO |

---

## 🎯 Next Steps

### **Immediate (First Day):**
1. ✅ Run paper trading for at least 1 week
2. ✅ Watch how strategies perform
3. ✅ Understand the risk levels  
4. ✅ Try both Conservative and WSB profiles

### **Before Real Money (1-2 Weeks):**
1. ✅ Paper trade successfully for 1-2 weeks  
2. ✅ Read strategy documentation
3. ✅ Understand risk management
4. ✅ Start with small real money amounts

### **Advanced (1+ Month):**
1. ✅ Customize strategy parameters
2. ✅ Add your own strategies
3. ✅ Implement advanced monitoring
4. ✅ Scale up position sizes gradually

---

<div align="center">

## 🎉 Ready to Start!

**Remember: Always start with paper trading!**

**🚧 This is PAPER MONEY - completely safe to experiment! 🚧**

[Back to Main README](README.md) | [Full Documentation](docs/)

</div>