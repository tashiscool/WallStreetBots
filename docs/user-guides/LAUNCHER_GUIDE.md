# WallStreetBots Launcher Guide

## 🚀 Easy Launch Options

WallStreetBots now includes multiple ways to launch the system, similar to .exe/.bat files on Windows:

### 📱 Quick Start (Recommended)

**Windows:**
- Double-click `run_wallstreetbots.bat`
- Or run: `python run_wallstreetbots.py`

**macOS/Linux:**
- Double-click `run_wallstreetbots.sh` 
- Or run: `./run_wallstreetbots.sh`
- Or run: `python3 run_wallstreetbots.py`

### 🎯 What You Get

The launcher provides a menu-driven interface with these options:

1. **🚀 Start Simple Trading Bot (Paper Trading)** - Safe trading with fake money
2. **💰 Start Simple Trading Bot (Real Money) [DANGER]** - Live trading with real money
3. **🧪 Run Risk Model Tests** - Test the risk management system
4. **📊 Run Advanced Feature Tests** - Test Month 5-6 advanced features
5. **🔧 Django Admin Panel** - Web interface for system management
6. **📈 Demo Risk Models** - Interactive risk model demonstration
7. **🛠️ Setup/Install Dependencies** - Automatic dependency installation
8. **🔍 System Status Check** - Detailed system health check
9. **❌ Exit** - Quit the launcher

### 🔧 Create Desktop Shortcuts

Run this command to create desktop shortcuts:
```bash
python3 create_executable.py
```

This will create:
- **Windows**: Desktop shortcut (.lnk file)
- **macOS**: Desktop command file (.command)
- **Linux**: Desktop entry (.desktop file)

### 📋 System Requirements

- **Python 3.8+** (Python 3.12+ recommended)
- **Virtual environment** (optional but recommended)
- **API Keys** configured in `.env` file

### 🛡️ Safety Features

- **Environment checks** - Validates setup before running
- **Paper trading default** - Starts in safe mode
- **Real money confirmation** - Requires explicit confirmation for live trading
- **Dependency management** - Automatic installation of required packages

### 📂 File Structure

```
WallStreetBots/
├── run_wallstreetbots.py     # Main Python launcher (cross-platform)
├── run_wallstreetbots.bat    # Windows batch file
├── run_wallstreetbots.sh     # macOS/Linux shell script
├── create_executable.py      # Creates desktop shortcuts/executables
├── simple_bot.py            # Core trading bot
├── manage.py                # Django management
└── requirements.txt         # Dependencies
```

### 🚀 Advanced Usage

**Direct Python execution:**
```bash
# Start paper trading bot directly
python3 simple_bot.py

# Run Django admin
python3 manage.py runserver

# Run specific tests
python3 test_month_5_6_advanced_features.py
```

**Environment setup:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🔍 Troubleshooting

**"Python not found" error:**
- Install Python 3.8+ from python.org
- Add Python to your system PATH

**"Missing requirements" error:**
- Run option 7 in the launcher menu
- Or manually: `pip install -r requirements.txt`

**"Environment check failed" error:**
- Make sure you're in the WallStreetBots root directory
- Check that required files exist (manage.py, simple_bot.py, etc.)

**API key issues:**
- Copy `.env.example` to `.env`
- Add your Alpaca API credentials
- Make sure paper trading is enabled initially

### 💡 Tips

1. **Start with paper trading** - Always test with fake money first
2. **Check system status** - Use option 8 to verify everything is working
3. **Run tests first** - Use options 3-4 to verify system integrity
4. **Monitor carefully** - Watch the bot's performance closely
5. **Set appropriate limits** - Configure risk limits in your .env file

### 📞 Support

- Check logs in the `logs/` directory
- Review `README.md` for detailed documentation
- Test with the demo modes before live trading
- Use paper trading to validate strategies

## 🎯 Quick Demo

To quickly see the system in action:

1. Run the launcher: `python3 run_wallstreetbots.py`
2. Select option 8 (System Status Check)
3. Select option 6 (Demo Risk Models)
4. Select option 3 (Run Risk Model Tests)
5. Select option 1 (Start Paper Trading Bot)

This will give you a complete overview of the system capabilities without risking real money.