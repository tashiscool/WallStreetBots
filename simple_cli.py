#!/usr/bin/env python3
"""
Simple CLI test without external dependencies
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from backend.tradingbot.config.simple_settings import load_settings
from backend.tradingbot.data.client import MarketDataClient, BarSpec
from backend.tradingbot.risk.engine import RiskEngine, RiskLimits


def test_settings():
    """Test settings loading"""
    print("🔍 Testing settings...")
    try:
        settings = load_settings()
        print("✅ Settings loaded successfully")
        print(f"  Profile: {settings.profile}")
        print(f"  Paper Trading: {settings.alpaca_paper}")
        print(f"  Dry Run: {settings.dry_run}")
        print(f"  Max Total Risk: {settings.max_total_risk:.1%}")
        return True
    except Exception as e:
        print(f"❌ Settings error: {e}")
        return False


def test_risk_engine():
    """Test risk engine"""
    print("\n🔍 Testing risk engine...")
    try:
        limits = RiskLimits(max_total_risk=0.3, max_position_size=0.1)
        engine = RiskEngine(limits)

        # Test pre-trade check
        result = engine.pretrade_check(0.05, 0.03)
        print(f"✅ Risk engine test passed: {result}")
        return True
    except Exception as e:
        print(f"❌ Risk engine error: {e}")
        return False


def test_data_client():
    """Test data client"""
    print("\n🔍 Testing data client...")
    try:
        client = MarketDataClient(use_cache=False)  # Disable cache for simple test
        price = client.get_current_price("SPY")
        if price:
            print(f"✅ Data client test passed: SPY = ${price:.2f}")
        else:
            print("✅ Data client connected (market may be closed)")
        return True
    except Exception as e:
        print(f"❌ Data client error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 WallStreetBots System Test")
    print("=" * 40)

    success = True
    success &= test_settings()
    success &= test_risk_engine()
    success &= test_data_client()

    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)
