"""Development environment configuration."""

from ..base import (
    BASE_DIR,
    TRADING_CONFIG,
    LOGGING_CONFIG,
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
    API_CONFIG,
)

# Development-specific settings
DEBUG = True
SECRET_KEY = "development-secret-key-not-for-production"  # noqa: S105

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