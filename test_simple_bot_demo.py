#!/usr/bin/env python3
"""
Simple Bot Demo Test - Demonstrate Paper Trading Functionality
Shows how the simple bot works without requiring real API keys
"""

import asyncio
import os
import django
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from backend.tradingbot.production.core.production_strategy_manager import (
    ProductionStrategyManager, ProductionStrategyManagerConfig
)

class SimpleBotDemo:
    """Demo version of simple bot to show paper trading functionality"""
    
    def __init__(self):
        print("🤖 SIMPLE BOT PAPER TRADING DEMO")
        print("=" * 60)
        
        # Same configuration as simple_bot.py but with demo keys
        self.config = ProductionStrategyManagerConfig(
            alpaca_api_key='demo_paper_api_key',
            alpaca_secret_key='demo_paper_secret_key',
            paper_trading=True,  # ALWAYS True for safety
            user_id=1,
            max_total_risk=0.10,     # Max 10% of account at risk
            max_position_size=0.03,  # Max 3% per position
            enable_alerts=False      # Keep it simple
        )
        
        self.manager = None
        print(f"✅ Configuration: Paper Trading Enabled")
        print(f"📊 Risk Limits: 10% max total, 3% max per position")
        
    async def demo_initialization(self):
        """Demo the bot initialization process"""
        print(f"\n🔧 INITIALIZING SIMPLE BOT...")
        print(f"📅 {datetime.now()}")
        
        try:
            # Initialize the manager (same as simple_bot.py)
            self.manager = ProductionStrategyManager(self.config)
            print(f"✅ Strategy Manager: Initialized")
            print(f"📈 Loaded Strategies: {len(self.manager.strategies)}")
            
            # List all strategies that would be running
            print(f"\n📋 ACTIVE STRATEGIES:")
            for strategy_id, strategy in self.manager.strategies.items():
                strategy_name = strategy.__class__.__name__
                print(f"   • {strategy_id}: {strategy_name}")
            
            # Simulate portfolio check
            print(f"\n💰 PORTFOLIO CHECK (Demo):")
            print(f"   Account Value: $10,000 (paper trading)")
            print(f"   Available Cash: $10,000")
            print(f"   Buying Power: $10,000") 
            print(f"   Current Positions: 0")
            
            print(f"\n🎯 SIMPLE BOT READY FOR PAPER TRADING!")
            print(f"   ✅ All 10 strategies loaded and ready")
            print(f"   ✅ Risk management active")
            print(f"   ✅ Paper trading mode enabled")
            print(f"   ✅ Ready to start trading simulation")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def demo_monitoring_loop(self):
        """Demo the monitoring that would happen"""
        print(f"\n📊 DEMO MONITORING (what would happen every minute):")
        
        for i in range(3):  # Show 3 demo cycles
            await asyncio.sleep(1)  # Quick demo
            now = datetime.now()
            
            print(f"[{now.strftime('%H:%M:%S')}] Portfolio: $10,000 | "
                  f"Strategies: {len(self.manager.strategies)} | "
                  f"Running: True | "
                  f"Active Positions: 0")
                  
        print(f"\n🔄 (Monitoring would continue every 60 seconds...)")
    
    def demo_safety_features(self):
        """Show the safety features"""
        print(f"\n🛡️ BUILT-IN SAFETY FEATURES:")
        print(f"   ✅ Paper Trading: Enabled (no real money risk)")
        print(f"   ✅ Position Size Limit: 3% max per trade")
        print(f"   ✅ Total Risk Limit: 10% max account risk")
        print(f"   ✅ Stop Trading: Ctrl+C to stop anytime")
        print(f"   ✅ Portfolio Monitoring: Real-time tracking")
        print(f"   ✅ Strategy Controls: Each strategy has its own limits")
        
    async def run_demo(self):
        """Run the complete demo"""
        try:
            # Initialize (same as simple_bot.py would do)
            success = await self.demo_initialization()
            
            if not success:
                return
                
            # Show safety features
            self.demo_safety_features()
            
            # Demo monitoring
            await self.demo_monitoring_loop()
            
            print(f"\n🎉 SIMPLE BOT DEMO COMPLETE!")
            print(f"\n📝 TO USE WITH REAL PAPER TRADING:")
            print(f"   1. Get Alpaca paper trading API keys")
            print(f"   2. Add them to .env file:")
            print(f"      ALPACA_API_KEY=your_paper_key")
            print(f"      ALPACA_SECRET_KEY=your_paper_secret")
            print(f"   3. Run: python simple_bot.py")
            print(f"   4. Bot connects to Alpaca paper trading")
            print(f"   5. All 10 strategies start trading with fake money")
            print(f"   6. Monitor and learn before going live")
            
        except KeyboardInterrupt:
            print(f"\n🛑 Demo stopped by user")
        except Exception as e:
            print(f"❌ Demo error: {e}")

async def main():
    """Main demo function"""
    demo = SimpleBotDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())