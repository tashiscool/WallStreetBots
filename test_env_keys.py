#!/usr/bin/env python3
"""Test script to verify .env file loading and API keys."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_env_loading():
    """Test that .env file is loaded properly."""
    print("🔍 Testing .env file loading...")

    # Check for Alpaca API keys
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    alpaca_url = os.getenv("ALPACA_BASE_URL")

    print(f"ALPACA_API_KEY: {'✅ Set' if alpaca_key else '❌ Missing'}")
    print(f"ALPACA_SECRET_KEY: {'✅ Set' if alpaca_secret else '❌ Missing'}")
    print(f"ALPACA_BASE_URL: {alpaca_url or '❌ Missing'}")

    # Test Django settings
    django_secret = os.getenv("DJANGO_SECRET_KEY")
    django_settings = os.getenv("DJANGO_SETTINGS_MODULE")

    print(f"DJANGO_SECRET_KEY: {'✅ Set' if django_secret else '❌ Missing'}")
    print(f"DJANGO_SETTINGS_MODULE: {django_settings or '❌ Missing'}")

    # Check if we have the minimum required for trading
    if alpaca_key and alpaca_secret:
        print("✅ Minimum API keys available for trading")
        return True
    else:
        print("❌ Missing required API keys")
        return False

if __name__ == "__main__":
    success = test_env_loading()
    sys.exit(0 if success else 1)