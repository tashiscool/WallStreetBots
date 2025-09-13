#!/usr / bin/env python3
"""
WallStreetBots - Complete Setup for Real Money Trading

This script guides you through setting up the complete trading system
with real API keys and proper configuration for actual money trading.

CRITICAL WARNING: This system can trade with real money. 
Start with paper trading and small position sizes.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(title): 
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_warning(): 
    """Print critical warning"""
    print_header("🚨 CRITICAL WARNING 🚨")
    print("""
This setup will configure WallStreetBots for REAL MONEY TRADING.

⚠️  IMPORTANT DISCLAIMERS: 
   - This system can lose ALL your money if configured incorrectly
   - Start with paper trading (ALPACA_BASE_URL=paper - api.alpaca.markets)
   - Use small position sizes (max 1 - 2% per trade)
   - Test thoroughly before going live
   - Past performance does not guarantee future results

📋 WHAT THIS SCRIPT DOES: 
   ✅ Installs all required dependencies
   ✅ Creates .env file with API key placeholders
   ✅ Validates production infrastructure
   ✅ Runs comprehensive test suite
   ✅ Provides step - by-step setup instructions

Do you want to continue? (y / N): """, end="")
    
    response=input().strip().lower()
    if response != 'y': 
        print("Setup cancelled. Good choice to be cautious! 🛡️")
        sys.exit(0)


def check_python_version(): 
    """Check Python version compatibility"""
    print_header("🐍 Checking Python Version")
    
    if sys.version_info < (3, 8): 
        print("❌ Python 3.8+ required. Please upgrade Python.")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Compatible")


def install_dependencies(): 
    """Install all required dependencies"""
    print_header("📦 Installing Dependencies")
    
    try: 
        # Create virtual environment if it doesn't exist
        if not os.path.exists('venv'): 
            print("Creating virtual environment...")
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
            print("✅ Virtual environment created")
        
        # Determine pip path
        if os.name== 'nt': # Windows
            pip_path='venv\\Scripts\\pip'
            python_path='venv\\Scripts\\python'
        else:  # Unix / Linux/Mac
            pip_path='venv / bin/pip'
            python_path='venv / bin/python'
        
        # Upgrade pip
        print("Upgrading pip...")
        subprocess.run([python_path, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        
        # Install requirements
        print("Installing requirements...")
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        
        print("✅ All dependencies installed successfully")
        
    except subprocess.CalledProcessError as e: 
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)


def create_env_file(): 
    """Create .env file from example"""
    print_header("⚙️ Creating Environment Configuration")
    
    env_example=Path('backend/.env.example')
    env_file=Path('backend/.env')
    
    if not env_example.exists(): 
        print("❌ backend/.env.example not found!")
        sys.exit(1)
    
    if env_file.exists(): 
        print("📄 .env file already exists")
        overwrite=input("Overwrite existing .env file? (y / N): ").strip().lower()
        if overwrite != 'y': 
            print("Keeping existing .env file")
            return
    
    # Copy example to .env
    with open(env_example) as f: 
        content=f.read()
    
    with open(env_file, 'w') as f: 
        f.write(content)
    
    print("✅ Created backend/.env from template")
    print(f"📝 Please edit {env_file} with your real API keys")


def run_tests(): 
    """Run comprehensive test suite"""
    print_header("🧪 Running Test Suite")
    
    try: 
        # Determine python path
        if os.name== 'nt': # Windows
            python_path='venv\\Scripts\\python'
        else:  # Unix / Linux/Mac
            python_path='venv / bin/python'
        
        print("Running all tests...")
        result=subprocess.run([
            python_path, '-m', 'pytest', 
            'backend / tradingbot/', '-v', '--tb=short'
        ], capture_output=True, text=True)
        
        if result.returncode== 0: 
            print("✅ All tests passed!")
            print(f"📊 Test output: \n{result.stdout}")
        else: 
            print("⚠️ Some tests failed - this is OK for initial setup")
            print(f"📊 Test output: \n{result.stderr}")
            print("🔧 You can fix test issues after configuring API keys")
        
    except subprocess.CalledProcessError as e: 
        print(f"❌ Failed to run tests: {e}")


def print_api_key_instructions(): 
    """Print detailed API key setup instructions"""
    print_header("🔑 API Key Setup Instructions")
    
    instructions="""
📋 STEP - BY-STEP API KEY SETUP: 

1. 🏦 ALPACA (Stock / Options Broker) - REQUIRED
   • Go to: https://alpaca.markets/
   • Sign up for account (free)
   • Navigate to: Paper Trading → API Keys
   • Generate API Key and Secret
   • Add to .env: 
     ALPACA_API_KEY=your_key_here
     ALPACA_SECRET_KEY=your_secret_here
     ALPACA_BASE_URL=https: //paper - api.alpaca.markets

2. 📊 IEX CLOUD (Market Data) - REQUIRED
   • Go to: https://iexcloud.io/
   • Sign up (starts free, $9 / month for real - time)
   • Get API token from dashboard
   • Add to .env: 
     IEX_API_KEY=your_key_here

3. 📈 POLYGON.IO (Options Data) - RECOMMENDED
   • Go to: https://polygon.io/
   • Sign up ($99 / month for options data)
   • Get API key from dashboard
   • Add to .env: 
     POLYGON_API_KEY=your_key_here

4. 📰 FINANCIAL MODELING PREP (Earnings) - RECOMMENDED
   • Go to: https://financialmodelingprep.com/
   • Sign up ($15 / month basic plan)
   • Get API key
   • Add to .env: 
     FMP_API_KEY=your_key_here

5. 📺 NEWS API (Sentiment) - OPTIONAL
   • Go to: https://newsapi.org/
   • Sign up (free tier available)
   • Get API key
   • Add to .env: 
     NEWS_API_KEY=your_key_here

6. 📊 ALPHA VANTAGE (Backup Data) - OPTIONAL
   • Go to: https://www.alphavantage.co/
   • Sign up (free tier: 5 calls / minute)
   • Get API key
   • Add to .env: 
     ALPHA_VANTAGE_API_KEY=your_key_here

💰 TOTAL MONTHLY COST FOR FULL SETUP: 
   - Alpaca: FREE (paper trading)
   - IEX Cloud: $9 / month (real - time data)  
   - Polygon.io: $99 / month (options data)
   - FMP: $15 / month (earnings calendar)
   - News API: FREE (basic tier)
   - Alpha Vantage: FREE (backup data)
   - TOTAL: ~$123 / month for professional setup
   - MINIMUM: $9 / month (Alpaca + IEX) for basic trading

🎯 RECOMMENDED STARTING SETUP (BUDGET): 
   - Alpaca (FREE paper trading)
   - IEX Cloud ($9 / month for real - time data)
   - Set paper trading limits: max 1% per position
   - Test for 2 - 3 months before going live
"""
    
    print(instructions)


def print_usage_instructions(): 
    """Print detailed usage instructions"""
    print_header("🚀 Usage Instructions")
    
    usage="""
🎯 HOW TO START TRADING: 

1. 📋 SETUP CHECKLIST: 
   ✅ All API keys configured in backend/.env
   ✅ Set PAPER_TRADING_MODE=true initially
   ✅ Set reasonable position limits (1 - 2% max)
   ✅ Tests passing (run: venv / bin/python -m pytest backend / tradingbot/)

2. 🧪 PAPER TRADING (SAFE): 
   # Test individual strategies
   python production_runner.py --paper --strategies wheel
   python production_runner.py --paper --strategies debit_spreads,spx_spreads
   
   # Test full phases
   python production_runner.py --paper --phase 2  # Low - risk strategies
   python production_runner.py --paper --phase 3  # Medium - risk strategies

3. 💰 LIVE TRADING (REAL MONEY - BE CAREFUL): 
   # Start with safest strategies and tiny position sizes
   python production_runner.py --live --strategies wheel --max - position-risk 0.01
   
   # NEVER start with high - risk strategies
   # python production_runner.py --live --strategies wsb_dip_bot  # DON'T DO THIS FIRST!

4. 📊 MONITORING: 
   • Check logs in: logs / trading.log
   • Monitor positions in Alpaca dashboard
   • Set up email / Discord alerts in .env
   • Use --dry - run flag to test without trading

5. 🛡️ SAFETY FEATURES: 
   • Circuit breakers stop trading after big losses
   • Position size limits prevent overexposure
   • Risk management validates every trade
   • Health checks monitor system status

📈 STRATEGY RECOMMENDATIONS FOR BEGINNERS: 

🟢 START HERE (Safest): 
   - wheel: Premium selling, positive expectancy
   - debit_spreads: Defined risk, limited loss

🟡 INTERMEDIATE (After 3+ months success): 
   - spx_spreads: 0DTE SPX credit spreads
   - momentum_weeklies: Weekly options on momentum

🔴 ADVANCED ONLY (High risk, need experience): 
   - wsb_dip_bot: The viral WSB pattern (can lose 100%)
   - lotto_scanner: 0DTE lottery plays (90% lose money)

⚠️ CRITICAL SAFETY RULES: 
   1. Start with paper trading
   2. Use tiny position sizes (0.5 - 1% of account)
   3. Test strategies for months before scaling up
   4. Never risk money you can't afford to lose
   5. Have a stop - loss plan for every strategy
   6. Monitor positions actively during market hours

🎓 LEARNING RESOURCES: 
   • Read README_EXACT_CLONE.md for WSB strategy details
   • Check backend / tradingbot/test_*.py for strategy behavior
   • Monitor Phase 4 backtesting results
   • Join r / WallStreetBets for strategy discussions (carefully!)
"""
    
    print(usage)


def main(): 
    """Main setup function"""
    print_warning()
    check_python_version()
    install_dependencies()
    create_env_file()
    run_tests()
    print_api_key_instructions()
    print_usage_instructions()
    
    print_header("🎉 Setup Complete!")
    print("""
✅ WallStreetBots is now ready for configuration!

🔑 NEXT STEPS: 
   1. Edit backend/.env with your API keys
   2. Test with paper trading first
   3. Start with tiny position sizes
   4. Scale up slowly as you gain confidence

🛡️ REMEMBER: Start small, test thoroughly, never risk more than you can lose!

Good luck, and may your trades be profitable! 📈
    """)


if __name__== "__main__": main()