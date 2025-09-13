#!/usr / bin / env python3
"""Simple Trading Bot - Personal Use
Run this to start trading.
"""

import asyncio
import os
from datetime import datetime

import django
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup Django before imports
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from backend.tradingbot.production.core.production_strategy_manager import (
    ProductionStrategyManager,
    ProductionStrategyManagerConfig,
)


class SimpleTradingBot:
    def __init__(self):
        self.config = ProductionStrategyManagerConfig(
            alpaca_api_key=os.getenv("ALPACA_API_KEY"),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
            paper_trading=True,  # Change to False when ready for real money
            user_id=1,
            max_total_risk=0.10,  # Max 10% of account at risk
            max_position_size=0.03,  # Max 3% per position
            enable_alerts=False,  # Keep it simple
        )

        self.manager = None
        self.running = False

    async def start_trading(self):
        """Start the trading bot."""
        print("🤖 Starting Simple Trading Bot...")
        print(f"📅 {datetime.now()}")
        print(f"📊 Paper Trading: {self.config.paper_trading}")

        try:
            # Initialize the manager
            self.manager = ProductionStrategyManager(self.config)
            print(f"✅ Loaded {len(self.manager.strategies)} strategies")

            # Simple safety check
            try:
                portfolio_value = (
                    self.manager.integration_manager.alpaca_manager.get_account_value()
                )
                if portfolio_value and portfolio_value < 1000:
                    print("⚠️ Account too small - need at least $1000")
                    return

                print(f"💰 Account value: ${portfolio_value:,.2f}")
            except Exception as e:
                print(f"⚠️ Could not get account value: {e}")
                print("💰 Continuing with trading...")
                portfolio_value = 100000  # Assume paper account has $100k

            # Start trading
            self.running = True
            await self.manager.start_all_strategies()

            # Simple monitoring loop
            while self.running:
                await self.simple_status_check()
                await asyncio.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            print("\n🛑 Stopping bot...")
            await self.stop_trading()
        except Exception as e:
            print(f"❌ Error: {e}")
            await self.stop_trading()

    async def simple_status_check(self):
        """Simple status monitoring."""
        try:
            now = datetime.now()
            try:
                portfolio_value = (
                    self.manager.integration_manager.alpaca_manager.get_account_value()
                )
                portfolio_str = (
                    f"${portfolio_value: ,.2f}" if portfolio_value else "Unknown"
                )
            except Exception:
                portfolio_str = "Unknown"

            print(
                f"[{now.strftime('%H: %M:%S')}] Portfolio: {portfolio_str} | "
                f"Strategies: {len(self.manager.strategies)} | "
                f"Running: {self.manager.is_running}"
            )

            # Simple safety check - if we're down more than 5%, consider stopping
            # (You'd implement this based on your risk tolerance)

        except Exception as e:
            print(f"⚠️ Status check error: {e}")

    async def stop_trading(self):
        """Stop the trading bot."""
        self.running = False
        if self.manager:
            await self.manager.stop_all_strategies()
        print("🛑 Trading bot stopped")


# Simple command line interface
if __name__ == "__main__":
    bot = SimpleTradingBot()
    asyncio.run(bot.start_trading())
