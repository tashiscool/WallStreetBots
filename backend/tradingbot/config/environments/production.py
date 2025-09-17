"""Production environment configuration."""

import os
import dj_database_url

from ..base import (
    BASE_DIR,
    TRADING_CONFIG,
    LOGGING_CONFIG,
    API_CONFIG,
    INSTALLED_APPS,
    MIDDLEWARE,
    ROOT_URLCONF,
    LANGUAGE_CODE,
    TIME_ZONE,
    USE_I18N,
    USE_TZ,
    STATIC_URL,
    STATIC_ROOT,
    DEFAULT_AUTO_FIELD,
)

# Production-specific settings
DEBUG = False
SECRET_KEY = os.getenv("SECRET_KEY")  # Must be set in production

if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable must be set in production")

# Production database (PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable must be set in production")

DATABASES = {
    "default": dj_database_url.parse(DATABASE_URL)
}

# Production trading config
TRADING_CONFIG.update({
    "DEFAULT_PAPER_TRADING": os.getenv("PAPER_TRADING", "true").lower() == "true",
    "MAX_POSITION_SIZE": float(os.getenv("MAX_POSITION_SIZE", "50000")),
    "MAX_TOTAL_RISK": float(os.getenv("MAX_TOTAL_RISK", "250000")),
})

# Production API config
API_CONFIG.update({
    "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY"),
    "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY"),
    "ALPACA_BASE_URL": os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets"),
})

if not API_CONFIG["ALPACA_API_KEY"] or not API_CONFIG["ALPACA_SECRET_KEY"]:
    raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in production")

# Production logging
LOGGING_CONFIG["root"]["level"] = "INFO"
LOGGING_CONFIG["handlers"]["file"]["filename"] = "/var/log/wallstreetbots/trading_system.log"

# Security settings
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# CORS settings for production
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
CORS_ALLOW_CREDENTIALS = True