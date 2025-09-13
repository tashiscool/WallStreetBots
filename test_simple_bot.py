#!/usr / bin/env python3
"""
Simple Bot Test - Test the basic functionality
"""

import asyncio
import os
from datetime import datetime
from backend.tradingbot.apimanagers import AlpacaManager

async def test_simple_bot(): 
    """Test the simple bot functionality"""
    print("🤖 Testing Simple Trading Bot...")
    print(f"📅 {datetime.now()}")
    
    # Test Alpaca connection
    manager = AlpacaManager(
        os.getenv('ALPACA_API_KEY', 'PKFUYPUACYYICLF36RE3'),
        os.getenv('ALPACA_SECRET_KEY', 'AiV6GeLGENOsL4wG93CCp123wVmkaHbg93dn2ws2'),
        paper_trading = True
    )
    
    # Test connection
    success, msg = manager.validate_api()
    print(f"✅ Connection: {success} - {msg}")
    
    if success: 
        # Get account info
        account_value = manager.get_account_value()
        balance = manager.get_balance()
        positions = manager.get_positions()
        
        print(f"💰 Account value: ${account_value:,.2f}")
        print(f"💵 Balance: ${balance:,.2f}")
        print(f"📊 Open positions: {len(positions)}")
        
        # Test a simple price check
        success, price = manager.get_price('AAPL')
        if success: 
            print(f"🍎 AAPL price: ${price:.2f}")
        else: 
            print(f"❌ Could not get AAPL price: {price}")
    
    print("\n🎉 Simple bot test complete!")
    print("✅ Ready for paper trading!")

if __name__ ==  "__main__": 
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    django.setup()
    
    asyncio.run(test_simple_bot())



