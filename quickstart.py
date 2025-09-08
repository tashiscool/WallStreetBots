#!/usr/bin/env python3
"""
WallStreetBots - Quick Start Script
==================================

🎯 Goal: Get up and running with paper trading in under 5 minutes!

Instructions:
1. Get your Alpaca API keys from https://alpaca.markets (free account)
2. Replace YOUR_API_KEY and YOUR_SECRET_KEY below with your actual keys
3. Run this script: python quickstart.py

⚠️ SAFETY: This script uses paper trading (fake money) by default!
"""

import asyncio
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from backend.tradingbot.production.core.production_strategy_manager import (
    ProductionStrategyManagerConfig, 
    ProductionStrategyManager, 
    StrategyProfile
)

def main():
    """Quick start for laymen - simple and safe paper trading setup."""
    
    print("🚀 WallStreetBots Quick Start")
    print("=" * 50)
    
    # ===========================================
    # 🔧 CONFIGURATION - USING .ENV FILE
    # ===========================================
    
    # 📋 API Keys loaded from .env file
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
    
    # 📋 Step 2: Choose your trading profile
    TRADING_PROFILE = StrategyProfile.research_2024  # 🛡️ Conservative (recommended)
    # TRADING_PROFILE = StrategyProfile.wsb_2025     # 🔥 WSB Aggressive (advanced)
    
    # ===========================================
    # ⚠️ SAFETY CHECK
    # ===========================================
    
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("❌ ERROR: Please set your Alpaca API keys in .env file first!")
        print("   1. Go to https://alpaca.markets")
        print("   2. Sign up for free paper trading account")
        print("   3. Get your API keys")
        print("   4. Copy .env.example to .env")
        print("   5. Edit .env file with your API keys")
        print("   6. Make sure .env file is in the same directory as this script")
        return
    
    # ===========================================
    # 🚀 SYSTEM STARTUP
    # ===========================================
    
    try:
        # Create configuration
        config = ProductionStrategyManagerConfig(
            alpaca_api_key=ALPACA_API_KEY,
            alpaca_secret_key=ALPACA_SECRET_KEY,
            paper_trading=True,  # ✅ SAFE: Using fake money only!
            profile=TRADING_PROFILE,
            user_id=1,
            enable_alerts=True
        )
        
        # Initialize the system
        print(f"🔧 Initializing with {config.profile} profile...")
        manager = ProductionStrategyManager(config)
        
        # Show startup info
        print(f"✅ Success! Loaded {len(manager.strategies)}/10 strategies")
        print(f"📊 Trading Profile: {config.profile}")
        print(f"🛡️ Max Portfolio Risk: {config.max_total_risk:.0%}")
        print(f"💰 Max Position Size: {config.max_position_size:.0%}")
        print(f"⏱️ Data Refresh: {config.data_refresh_interval}s")
        print(f"🔒 Paper Trading: {'✅ ENABLED' if config.paper_trading else '❌ DISABLED'}")
        
        # Show loaded strategies
        print("\n📋 Active Strategies:")
        for i, strategy_name in enumerate(manager.strategies.keys(), 1):
            print(f"   {i:2d}. {strategy_name}")
        
        # Get system status
        status = manager.get_system_status()
        print(f"\n🎯 System Status: {'🟢 Ready' if not status['is_running'] else '🔵 Running'}")
        
        print("\n" + "=" * 50)
        print("🎉 READY TO TRADE!")
        print("=" * 50)
        print("⚠️  Remember: This is PAPER TRADING (fake money)")
        print("💡 To start live trading, call: await manager.start_all_strategies()")
        print("📊 Monitor performance with: manager.get_system_status()")
        print("🛑 Always test thoroughly before using real money!")
        
        return manager
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Try: pip install alpaca-py>=0.42.0")
        return None
        
    except Exception as e:
        print(f"❌ Setup Error: {e}")
        print("💡 Check your API keys and internet connection")
        return None

async def run_live_demo():
    """
    Optional: Run a live demo with all strategies active.
    ⚠️ Only call this after testing thoroughly!
    """
    manager = main()
    if not manager:
        return
    
    print("\n🚀 Starting live trading demo...")
    print("⚠️  Press Ctrl+C to stop safely")
    
    try:
        # Start all strategies
        success = await manager.start_all_strategies()
        
        if success:
            print("✅ All strategies are now running!")
            
            # Monitor loop
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                status = manager.get_system_status()
                print(f"📊 Status: {len(status['strategy_status'])} strategies active")
                
                # Show brief performance summary
                if status.get('performance_metrics'):
                    metrics = status['performance_metrics']
                    print(f"⏱️  Uptime: {metrics.get('system_uptime', 0)/3600:.1f}h")
        else:
            print("❌ Failed to start strategies")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping safely...")
        await manager.stop_all_strategies()
        print("✅ All strategies stopped")
    except Exception as e:
        print(f"❌ Runtime error: {e}")

if __name__ == "__main__":
    print(__doc__)
    
    # For beginners: just show setup and configuration
    manager = main()
    
    # Uncomment the line below to run live demo (advanced users only)
    # asyncio.run(run_live_demo())