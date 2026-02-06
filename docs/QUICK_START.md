# 🚀 WallStreetBots - 5 Minute Quick Start

## ⚡ Get Trading in 5 Minutes!

This guide gets you up and running with paper trading (fake money) in just 5 minutes. No risk, no commitment - just learning!

---

## ✅ Step 1: Get Free Alpaca Account (2 minutes)

1. **Go to** [alpaca.markets](https://alpaca.markets)
2. **Click** "Sign Up" (it's free, no credit card needed!)
3. **Create** your account
4. **Navigate to** "Paper Trading" → "API Keys"
5. **Copy** your API Key and Secret Key (save these somewhere safe!)

**💡 Tip:** Paper trading gives you $100,000 in fake money to practice with!

---

## ✅ Step 2: Install the System (2 minutes)

**Open your terminal/command prompt** and run:

```bash
# Clone the repository
git clone https://github.com/yourusername/WallStreetBots.git
cd WallStreetBots

# One-command setup (installs deps, creates .env, sets up database)
bash scripts/setup.sh
```

**💡 Don't have Python?** Download from [python.org](https://python.org/downloads) (get version 3.12+)

---

## ✅ Step 3: Add Your API Keys (1 minute)

1. **Open** the `.env` file (created by setup.sh) in any text editor (Notepad, TextEdit, VS Code, etc.)
2. **Replace** these lines with your actual keys:

```
ALPACA_API_KEY=paste_your_api_key_here
ALPACA_SECRET_KEY=paste_your_secret_key_here
```

3. **Save** the file

**💡 Example:**
```
ALPACA_API_KEY=AKIAIOSFODNN7EXAMPLE
ALPACA_SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

---

## ✅ Step 4: Test It Works (30 seconds)

```bash
# Start the development server
bash scripts/run.sh

# Open http://127.0.0.1:8000/health/ in your browser
# You should see: {"status": "healthy"}
```

**If you see errors:** Check that your API keys are correct in the `.env` file.

---

## ✅ Step 5: Start Trading! (Ready!)

```bash
# Start the platform
bash scripts/run.sh
```

**Open your browser to** `http://127.0.0.1:8000/` **and you'll see the Django web interface:**

- **Dashboard** - Overview of account status, open positions, and performance
- **Strategies** - Enable/disable trading strategies and configure parameters
- **Backtesting** - Test strategies against historical data
- **Admin Panel** - Full system configuration at `/admin/`

**The system will:**
- ✅ Connect to your Alpaca account
- ✅ Load all trading strategies
- ✅ Start scanning for opportunities
- ✅ Place trades automatically (with fake money!)
- ✅ Monitor positions and exit when targets are hit

---

## 🎉 You're Trading!

**What happens now:**
- System runs in the background
- Scans markets every few minutes
- Finds trading opportunities
- Places trades automatically
- Tracks performance

**To see what's happening:**
- Check the terminal output for trade notifications
- Run option 8 (System Status Check) to see current status
- Check your Alpaca dashboard to see positions

---

## 📊 Understanding What You See

### **When a Trade Happens:**
```
📈 Signal Generated: AAPL - WSB Dip Bot
✅ Risk Check: PASSED
💰 Position Size: $500 (5% of account)
📝 Order Placed: Buy 10 AAPL calls @ $2.50
✅ Order Filled: 10 contracts @ $2.50 = $2,500
📊 Position Opened: AAPL calls, Target: $7.50, Stop: $1.25
```

### **When a Trade Exits:**
```
🎯 Profit Target Hit: AAPL calls
💰 Current Value: $7.50 (3x profit!)
📝 Order Placed: Sell 10 AAPL calls
✅ Order Filled: 10 contracts @ $7.50 = $7,500
💵 Profit: $5,000 (200% return)
📊 Position Closed: AAPL calls
```

---

## 🛑 How to Stop the System

**To pause trading:**
- Press `Ctrl+C` in the terminal
- Or close the terminal window

**To stop completely:**
- Select option 9 (Exit) from the menu
- Or close the terminal

**Your positions will remain open** (the system just stops looking for new trades)

---

## ❓ Troubleshooting

### **"Python not found"**
- Install Python from [python.org](https://python.org/downloads)
- Make sure to check "Add Python to PATH" during installation

### **"Module not found" errors**
- Run `bash scripts/setup.sh` again

### **"API key invalid"**
- Double-check your keys in the `.env` file
- Make sure there are no extra spaces
- Try regenerating keys in Alpaca dashboard

### **"Database error"**
- Run `bash scripts/setup.sh` again

### **System won't start**
- Run option 8 (System Status Check) to see what's wrong
- Check the logs in the `logs/` folder
- Make sure market is open (9:30 AM - 4:00 PM ET on weekdays)

---

## 🎓 Next Steps

### **Week 1: Watch and Learn**
- ✅ Let the system run for a few days
- ✅ Watch how it finds opportunities
- ✅ See which strategies work
- ✅ Understand the trade flow

### **Week 2: Understand Strategies**
- ✅ Read about each strategy in the docs
- ✅ See which ones are making money
- ✅ Learn why trades are placed
- ✅ Understand exit conditions

### **Week 3-4: Customize**
- ✅ Adjust position sizes (conservatively!)
- ✅ Enable/disable specific strategies
- ✅ Change risk parameters
- ✅ Track performance metrics

### **Month 2-3: Master Paper Trading**
- ✅ Run for 30+ days
- ✅ Track detailed performance
- ✅ Optimize parameters
- ✅ Build confidence

### **Month 4+: Consider Live Trading** (Only if profitable!)
- ⚠️ Start with tiny positions (1-2%)
- ⚠️ Use only strategies that worked in paper trading
- ⚠️ Scale up very gradually
- ⚠️ Never risk more than you can afford to lose

---

## 💡 Pro Tips

1. **Start Conservative:** Use default settings first, then adjust
2. **Paper Trade First:** Always test with fake money before real money
3. **Monitor Daily:** Check performance every day
4. **Learn from Losses:** Review losing trades to understand why
5. **Be Patient:** Good trading takes time and practice
6. **Use Stop Losses:** Always protect your capital
7. **Start Small:** Even with real money, start with tiny positions

---

## 🆘 Need Help?

- **📖 Read the docs:** [How It Works](HOW_IT_WORKS.md) explains everything simply
- **🔍 Check status:** Run option 8 to see system health
- **📝 Check logs:** Look in the `logs/` folder for error messages
- **💬 Ask questions:** Check GitHub issues or discussions

---

## ⚠️ Important Reminders

- ✅ **Always start with paper trading** (fake money)
- ✅ **Never risk more than you can afford to lose**
- ✅ **Use stop losses** to limit losses
- ✅ **Monitor your account daily**
- ✅ **Learn continuously** - markets change
- ✅ **Be patient** - good trading takes time

---

<div align="center">

**🎉 Congratulations! You're ready to start trading!**

**📚 Want to understand how it works? Read [How It Works](HOW_IT_WORKS.md)!**

**🚀 Ready for more? Check [Getting Started Guide](user-guides/GETTING_STARTED_REAL.md)!**

**⚠️ Remember: Always start with paper trading and never risk money you can't afford to lose!**

</div>

