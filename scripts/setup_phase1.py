#!/usr / bin/env python3
"""
Phase 1 Setup Script
Setup production environment for WallStreetBots Phase 1 implementation
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def run_command(command: str, description: str):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try: 
        result = subprocess.run(command, shell = True, check = True, capture_output = True, text = True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e: 
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def create_directories(): 
    """Create necessary directories"""
    directories = [
        "config",
        "logs",
        "data",
        "backups",
        "migrations"
    ]
    
    for directory in directories: 
        Path(directory).mkdir(exist_ok = True)
        print(f"📁 Created directory: {directory}")


def create_config_files(): 
    """Create configuration files"""
    
    # Create production configuration template
    config_template = {
        "data_providers": {
            "iex_api_key": "",
            "polygon_api_key": "",
            "fmp_api_key": "",
            "news_api_key": "",
            "alpha_vantage_api_key": ""
        },
        "broker": {
            "alpaca_api_key": "",
            "alpaca_secret_key": "",
            "alpaca_base_url": "https://paper - api.alpaca.markets",
            "ibkr_host": "",
            "ibkr_port": 7497,
            "ibkr_client_id": 1
        },
        "risk": {
            "max_position_risk": 0.10,
            "max_total_risk": 0.30,
            "max_drawdown": 0.20,
            "max_correlation": 0.25,
            "account_size": 100000.0,
            "default_commission": 1.0,
            "default_slippage": 0.002
        },
        "trading": {
            "universe": ["AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AVGO", "AMD", "TSLA"],
            "scan_interval": 300,
            "max_concurrent_trades": 10,
            "enable_paper_trading": True,
            "enable_live_trading": False
        },
        "alerts": {
            "enable_slack": False,
            "slack_webhook_url": "",
            "enable_email": False,
            "email_smtp_server": "",
            "email_smtp_port": 587,
            "email_username": "",
            "email_password": "",
            "email_recipients": []
        },
        "database": {
            "engine": "postgresql",
            "host": "localhost",
            "port": 5432,
            "name": "wallstreetbots",
            "username": "postgres",
            "password": "",
            "ssl_mode": "prefer"
        }
    }
    
    with open("config / production.json", "w") as f: 
        json.dump(config_template, f, indent = 2)
    print("📄 Created config / production.json")
    
    # Create environment template
    env_template = """# WallStreetBots Phase 1 Configuration
# Copy this file to .env and fill in your actual values

# Data Provider API Keys
IEX_API_KEY = your_iex_api_key_here
POLYGON_API_KEY = your_polygon_api_key_here
FMP_API_KEY = your_fmp_api_key_here
NEWS_API_KEY = your_news_api_key_here
ALPHA_VANTAGE_API_KEY = your_alpha_vantage_api_key_here

# Broker Configuration
ALPACA_API_KEY = your_alpaca_api_key_here
ALPACA_SECRET_KEY = your_alpaca_secret_key_here
ALPACA_BASE_URL = https: //paper - api.alpaca.markets

# Risk Management
MAX_POSITION_RISK = 0.10
MAX_TOTAL_RISK = 0.30
MAX_DRAWDOWN = 0.20
MAX_CORRELATION = 0.25
ACCOUNT_SIZE = 100000.0
DEFAULT_COMMISSION = 1.0
DEFAULT_SLIPPAGE = 0.002

# Trading Configuration
TRADING_UNIVERSE = AAPL,MSFT,GOOGL,GOOG,META,NVDA,AVGO,AMD,TSLA
SCAN_INTERVAL = 300
MAX_CONCURRENT_TRADES = 10
ENABLE_PAPER_TRADING = true
ENABLE_LIVE_TRADING = false

# Alert Configuration
ENABLE_SLACK = false
SLACK_WEBHOOK_URL = your_slack_webhook_url_here
ENABLE_EMAIL = false
EMAIL_SMTP_SERVER = smtp.gmail.com
EMAIL_SMTP_PORT = 587
EMAIL_USERNAME = your_email_here
EMAIL_PASSWORD = your_email_password_here
EMAIL_RECIPIENTS = recipient1@example.com,recipient2@example.com

# Database Configuration
DB_ENGINE = postgresql
DB_HOST = localhost
DB_PORT = 5432
DB_NAME = wallstreetbots
DB_USERNAME = postgres
DB_PASSWORD = your_db_password_here
DB_SSL_MODE = prefer
"""
    
    with open(".env.template", "w") as f: 
        f.write(env_template)
    print("📄 Created .env.template")
    
    # Create Django settings
    django_settings = """# Django settings for Phase 1
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'your - secret-key - here'
DEBUG = False
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'backend.tradingbot',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'wallstreetbots'),
        'USER': os.getenv('DB_USERNAME', 'postgres'),
        'PASSWORD': os.getenv('DB_PASSWORD', ''),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

LANGUAGE_CODE = 'en - us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process: d} {thread: d} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs / django.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}
"""
    
    with open("backend / settings_phase1.py", "w") as f: 
        f.write(django_settings)
    print("📄 Created backend / settings_phase1.py")


def install_dependencies(): 
    """Install Phase 1 dependencies"""
    print("📦 Installing Phase 1 dependencies...")
    
    # Install requirements
    result = run_command(
        "pip install -r requirements_phase1.txt",
        "Installing Python dependencies"
    )
    
    if result is None: 
        print("❌ Failed to install dependencies")
        return False
    
    return True


def setup_database(): 
    """Setup database"""
    print("🗄️ Setting up database...")
    
    # Create database migrations
    result = run_command(
        "python manage.py makemigrations tradingbot",
        "Creating database migrations"
    )
    
    if result is None: 
        print("❌ Failed to create migrations")
        return False
    
    # Apply migrations
    result = run_command(
        "python manage.py migrate",
        "Applying database migrations"
    )
    
    if result is None: 
        print("❌ Failed to apply migrations")
        return False
    
    return True


def run_tests(): 
    """Run Phase 1 tests"""
    print("🧪 Running Phase 1 tests...")
    
    result = run_command(
        "python -m pytest backend / tradingbot/test_phase1_integration.py -v",
        "Running integration tests"
    )
    
    if result is None: 
        print("❌ Tests failed")
        return False
    
    print("✅ All tests passed")
    return True


def create_startup_script(): 
    """Create startup script"""
    startup_script = """#!/bin / bash
# WallStreetBots Phase 1 Startup Script

echo "🚀 Starting WallStreetBots Phase 1..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV"  ==  "" ]]; then
    echo "⚠️  Virtual environment not activated. Please activate it first."
    echo "   source venv / bin/activate"
    exit 1
fi

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "⚠️  .env file not found. Please copy .env.template to .env and configure it."
    echo "   cp .env.template .env"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Start Django development server
echo "🌐 Starting Django development server..."
python manage.py runserver 0.0.0.0: 8000
"""
    
    with open("start_phase1.sh", "w") as f: 
        f.write(startup_script)
    
    # Make executable
    os.chmod("start_phase1.sh", 0o755)
    print("📄 Created start_phase1.sh")


def main(): 
    """Main setup function"""
    print("🏗️  WallStreetBots Phase 1 Setup")
    print(" = " * 50)
    
    # Check Python version
    if sys.version_info  <  (3, 8): 
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create configuration files
    print("\n📄 Creating configuration files...")
    create_config_files()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies(): 
        print("❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Setup database
    print("\n🗄️  Setting up database...")
    if not setup_database(): 
        print("❌ Setup failed at database setup")
        sys.exit(1)
    
    # Run tests
    print("\n🧪 Running tests...")
    if not run_tests(): 
        print("❌ Setup failed at testing")
        sys.exit(1)
    
    # Create startup script
    print("\n🚀 Creating startup script...")
    create_startup_script()
    
    print("\n" + " = " * 50)
    print("✅ Phase 1 setup completed successfully!")
    print("\nNext steps: ")
    print("1. Copy .env.template to .env and configure your API keys")
    print("2. Configure your database settings in .env")
    print("3. Run: ./start_phase1.sh")
    print("4. Test the demo: python backend / tradingbot/phase1_demo.py")
    print("\n⚠️  Remember: This is for educational / testing purposes only!")
    print("   Do not use real money with this implementation.")


if __name__ ==  "__main__": main()
