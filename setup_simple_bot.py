#!/usr/bin/env python3
"""
Simple Bot Setup Script
Run this to set up your trading bot quickly
"""

import os
import subprocess
import sys

def setup_simple_bot():
    """Set up the simple trading bot"""
    print("🤖 Setting up Simple Trading Bot...")
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        env_content = """# Alpaca API Configuration
ALPACA_API_KEY=your_paper_trading_key_here
ALPACA_SECRET_KEY=your_paper_trading_secret_here

# Django Configuration
DJANGO_SETTINGS_MODULE=backend.settings
SECRET_KEY=your_secret_key_here

# Database Configuration
DATABASE_URL=sqlite:///db.sqlite3

# Trading Configuration
PAPER_TRADING=True
MAX_DAILY_LOSS=100
MAX_POSITION_SIZE=50
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file - please edit it with your API keys")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'alpaca-py'], check=True)
        print("✅ Installed alpaca-py")
    except subprocess.CalledProcessError:
        print("❌ Failed to install alpaca-py")
        return False
    
    # Initialize database
    print("🗄️ Initializing database...")
    try:
        subprocess.run([sys.executable, 'manage.py', 'migrate'], check=True)
        print("✅ Database initialized")
    except subprocess.CalledProcessError:
        print("❌ Failed to initialize database")
        return False
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your Alpaca API keys")
    print("2. Run: python simple_bot.py")
    print("3. Watch it trade with paper money!")
    
    return True

if __name__ == "__main__":
    setup_simple_bot()

