"""Development environment configuration."""

from ..base import *

# Development-specific settings
DEBUG = True
SECRET_KEY = "development-secret-key-not-for-production"

# Development database
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Development trading config
TRADING_CONFIG.update({
    "DEFAULT_PAPER_TRADING": True,
    "MAX_POSITION_SIZE": 1000,  # Smaller for development
    "MAX_TOTAL_RISK": 5000,
})

# Development logging
LOGGING_CONFIG["root"]["level"] = "DEBUG"

# CORS settings for development
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True