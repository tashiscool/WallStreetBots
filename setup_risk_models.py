#!/usr/bin/env python3
"""
Setup Advanced Risk Models - 2025 Implementation
Installation and configuration script for sophisticated risk management
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3.8, 0):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        # Install core requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "numpy", "pandas", "scipy", "scikit-learn"
        ])
        print("✅ Core packages installed")
        
        # Install optional ML packages
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "tensorflow", "torch"
            ])
            print("✅ ML packages installed")
        except subprocess.CalledProcessError:
            print("⚠️  Warning: Some ML packages failed to install (optional)")
        
        # Install financial data packages
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "yfinance", "alpha-vantage", "matplotlib", "seaborn"
        ])
        print("✅ Financial data packages installed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "backend/tradingbot/risk",
        "data/risk_models",
        "logs/risk",
        "reports/risk",
        "config/risk"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_config_files():
    """Create configuration files"""
    print("\n⚙️  Creating configuration files...")
    
    # Risk configuration
    risk_config = """
# Risk Management Configuration
[RISK_LIMITS]
max_var_1d = 0.05
max_var_5d = 0.10
max_cvar_99 = 0.08
max_concentration = 0.20
max_correlation = 0.80
min_liquidity = 0.10

[STRESS_TESTING]
scenarios = 2008_crisis,2010_flash_crash,2020_covid_pandemic,interest_rate_shock,geopolitical_crisis,ai_bubble_burst
max_drawdown_limit = 0.25
max_recovery_time = 30

[ML_MODELS]
enable_ml = true
model_retrain_frequency = 30  # days
confidence_threshold = 0.75
ensemble_weights = 0.4,0.4,0.2

[ALERTS]
enable_alerts = true
alert_email = your-email@example.com
alert_webhook = https://hooks.slack.com/your-webhook
"""
    
    with open("config/risk/risk_config.ini", "w") as f:
        f.write(risk_config)
    print("✅ Created: config/risk/risk_config.ini")
    
    # Environment variables template
    env_template = """
# Risk Management Environment Variables
RISK_MANAGEMENT_ENABLED=true
VAR_CONFIDENCE_LEVEL=0.95
STRESS_TEST_FREQUENCY=daily
ML_MODEL_PATH=data/risk_models/
ALERT_WEBHOOK_URL=
EMAIL_ALERTS_ENABLED=false
"""
    
    with open(".env.risk", "w") as f:
        f.write(env_template)
    print("✅ Created: .env.risk")

def run_tests():
    """Run basic tests to verify installation"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        import numpy as np
        import pandas as pd
        import scipy
        print("✅ Core packages imported successfully")
        
        # Test risk modules
        sys.path.append("backend")
        from tradingbot.risk import AdvancedVaREngine
        print("✅ Risk modules imported successfully")
        
        # Run basic VaR test
        engine = AdvancedVaREngine(100000)
        test_returns = np.random.normal(0.001, 0.02, 100)
        var_suite = engine.calculate_var_suite(test_returns)
        print("✅ VaR calculation test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_documentation():
    """Create basic documentation"""
    print("\n📚 Creating documentation...")
    
    readme_content = """
# Advanced Risk Models - 2025 Implementation

## Overview
Sophisticated risk management system for algorithmic trading with VaR, CVaR, stress testing, and machine learning capabilities.

## Quick Start

### 1. Run the test suite
```bash
python test_advanced_risk_models.py
```

### 2. Basic usage
```python
from tradingbot.risk import AdvancedVaREngine, RiskDashboard2025

# Initialize VaR engine
var_engine = AdvancedVaREngine(portfolio_value=100000)

# Calculate VaR
import numpy as np
returns = np.random.normal(0.001, 0.02, 252)
var_suite = var_engine.calculate_var_suite(returns)

# Initialize risk dashboard
dashboard = RiskDashboard2025(portfolio_value=100000)
dashboard_data = dashboard.get_risk_dashboard_data(portfolio)
```

## Features
- Multi-method VaR calculation (Parametric, Historical, Monte Carlo, EVT)
- FCA-compliant stress testing
- Machine learning risk prediction
- Real-time risk monitoring dashboard
- Alternative data integration
- Factor risk attribution

## Configuration
Edit `config/risk/risk_config.ini` to customize risk limits and parameters.

## Documentation
See `SOPHISTICATED_RISK_MODELS_2025.md` for detailed implementation guide.
"""
    
    with open("README_RISK_MODELS.md", "w") as f:
        f.write(readme_content)
    print("✅ Created: README_RISK_MODELS.md")

def main():
    """Main setup function"""
    print("🚀 WallStreetBots Advanced Risk Models Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Create config files
    create_config_files()
    
    # Run tests
    if not run_tests():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    # Create documentation
    create_documentation()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python test_advanced_risk_models.py")
    print("2. Review: README_RISK_MODELS.md")
    print("3. Configure: config/risk/risk_config.ini")
    print("4. Start using: from tradingbot.risk import AdvancedVaREngine")

if __name__ == "__main__":
    main()


