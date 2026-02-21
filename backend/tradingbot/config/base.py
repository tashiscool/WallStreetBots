"""Base configuration for WallStreetBots trading system.

This module provides the core configuration settings that are shared
across all environments (development, testing, production).
"""

import os
from pathlib import Path
import dj_database_url

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Core Django settings
SECRET_KEY = os.getenv("SECRET_KEY", "development-secret-key-change-in-production")
DEBUG = os.getenv('DJANGO_DEBUG', 'False').lower() in ('true', '1', 'yes')
ALLOWED_HOSTS = ["*"]

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "social_django",
    "admin_interface",
    "colorfield",
    "backend.tradingbot",
    "backend.auth0login",
    "backend.home",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "backend.urls"

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DATABASES = {
        "default": dj_database_url.parse(DATABASE_URL)
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Trading system configuration
TRADING_CONFIG = {
    "DEFAULT_PAPER_TRADING": True,
    # Fraction of account value (0.0 to 1.0); matches Pydantic AppSettings default.
    "MAX_POSITION_SIZE": float(os.getenv("MAX_POSITION_SIZE", "0.10")),
    "MAX_TOTAL_RISK": float(os.getenv("MAX_TOTAL_RISK", "50000")),
    "STOP_LOSS_PCT": float(os.getenv("STOP_LOSS_PCT", "0.05")),
    "TAKE_PROFIT_MULTIPLIER": float(os.getenv("TAKE_PROFIT_MULTIPLIER", "2.0")),
}

# API Configuration
API_CONFIG = {
    "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY", ""),
    "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    "ALPACA_BASE_URL": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        },
        "simple": {
            "format": "{levelname} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "logs" / "trading_system.log",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}