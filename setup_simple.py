#!/usr/bin/env python3
"""
Simple Setup Script for Personal Trading Bot
Run this once to get everything working
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and show the result"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description} - Found")
        return True
    else:
        print(f"❌ {description} - Missing")
        return False

def main():
    print("🤖 SIMPLE TRADING BOT SETUP")
    print("=" * 40)
    print()
    
    success_count = 0
    total_checks = 0
    
    # Check 1: Python version
    total_checks += 1
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python version: {python_version.major}.{python_version.minor}")
        success_count += 1
    else:
        print(f"❌ Python version: {python_version.major}.{python_version.minor} (need 3.8+)")
    
    # Check 2: Install alpaca-py
    total_checks += 1
    if run_command("pip install alpaca-py", "Installing alpaca-py"):
        success_count += 1
    
    # Check 3: Database setup
    total_checks += 1
    if run_command("python manage.py migrate", "Setting up database"):
        success_count += 1
    
    # Check 4: Check .env file
    total_checks += 1
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            run_command("cp .env.example .env", "Creating .env file")
            print("📝 Edit .env file with your Alpaca API keys:")
            print("   ALPACA_API_KEY=your_paper_key_here")
            print("   ALPACA_SECRET_KEY=your_paper_secret_here")
        else:
            print("❌ No .env.example file found")
            print("📝 Create .env file with:")
            print("   ALPACA_API_KEY=your_paper_key_here")
            print("   ALPACA_SECRET_KEY=your_paper_secret_here")
    else:
        print("✅ .env file exists")
        success_count += 1
    
    # Check 5: Test Django setup
    total_checks += 1
    test_cmd = 'python -c "import django; import os; os.environ.setdefault(\\"DJANGO_SETTINGS_MODULE\\", \\"backend.settings\\"); django.setup(); print(\\"Django OK\\")"'
    if run_command(test_cmd, "Testing Django setup"):
        success_count += 1
    
    # Summary
    print()
    print("=" * 40)
    print(f"📊 SETUP SUMMARY: {success_count}/{total_checks} checks passed")
    
    if success_count == total_checks:
        print("🎉 SETUP COMPLETE!")
        print()
        print("Next steps:")
        print("1. Get Alpaca API keys from https://alpaca.markets")
        print("2. Add them to your .env file")
        print("3. Run: python simple_bot.py")
    else:
        print("⚠️  Setup incomplete - fix the issues above")
        print()
        print("Common issues:")
        print("- Make sure you're in the WallStreetBots directory")
        print("- Make sure you have internet connection")
        print("- Make sure Python 3.8+ is installed")
    
    print()
    print("🔗 Get Alpaca API keys (free paper trading):")
    print("   https://alpaca.markets")
    print()

if __name__ == "__main__":
    main()