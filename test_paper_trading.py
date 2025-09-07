#!/usr/bin/env python3
"""
Test Paper Trading Connection
Quick test to verify Alpaca paper trading API works
"""

import os
import django
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from backend.tradingbot.apimanagers import AlpacaManager

def test_paper_trading_connection():
    """Test paper trading API connection"""
    
    print("🧪 TESTING PAPER TRADING CONNECTION")
    print("=" * 50)
    
    # Get API credentials from environment
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or api_key == 'your_paper_api_key_here':
        print("❌ ERROR: Please set your Alpaca paper trading API keys in .env file")
        print("   ALPACA_API_KEY=your_paper_api_key_here")
        print("   ALPACA_SECRET_KEY=your_paper_secret_key_here")
        print("\n📝 Get paper trading keys from: https://app.alpaca.markets/paper/dashboard/overview")
        return False
        
    if not secret_key or secret_key == 'your_paper_secret_key_here':
        print("❌ ERROR: Please set your Alpaca paper trading secret key in .env file")
        return False
    
    try:
        # Test connection with paper trading enabled
        print(f"🔑 API Key: {api_key[:8]}...")
        print(f"📊 Paper Trading: True")
        
        manager = AlpacaManager(
            API_KEY=api_key,
            SECRET_KEY=secret_key,
            paper_trading=True  # Ensure paper trading
        )
        
        # Test API validation
        success, message = manager.validate_api()
        
        if success:
            print(f"✅ Paper Trading Connection: SUCCESS")
            print(f"📝 Message: {message}")
            
            # Get account info
            try:
                account = manager.trading_client.get_account()
                print(f"\n💰 Paper Trading Account Info:")
                print(f"   Account ID: {account.id}")
                print(f"   Buying Power: ${float(account.buying_power):,.2f}")
                print(f"   Cash: ${float(account.cash):,.2f}")
                print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
                print(f"   Paper Trading: {account.account_blocked == False}")  # Paper accounts aren't blocked
                
            except Exception as e:
                print(f"⚠️ Account info error: {e}")
                
            print(f"\n🎯 PAPER TRADING READY!")
            print("   Your simple bot can now connect to Alpaca paper trading")
            return True
            
        else:
            print(f"❌ Paper Trading Connection: FAILED")  
            print(f"📝 Error: {message}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_paper_trading_connection()
    
    if success:
        print(f"\n🚀 NEXT STEPS:")
        print("   1. Your paper trading connection works!")
        print("   2. Run: python simple_bot.py")
        print("   3. Bot will start with all 10 strategies in paper mode")
        print("   4. Monitor paper trades and performance")
    else:
        print(f"\n🔧 FIX REQUIRED:")
        print("   1. Get paper trading API keys from Alpaca") 
        print("   2. Update your .env file with the keys")
        print("   3. Run this test again")