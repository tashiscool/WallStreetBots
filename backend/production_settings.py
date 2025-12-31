"""Production Django settings for WallStreetBots.

This module provides production-ready settings with:
- Pydantic validation for all required environment variables
- Proper security headers
- Database connection pooling
- Structured logging
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import dj_database_url


class ProductionConfig(BaseSettings):
    """Production configuration with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required settings
    secret_key: str = Field(..., min_length=32, description="Django SECRET_KEY")
    database_url: str = Field(..., description="Database connection URL")

    # Security settings
    debug: bool = Field(default=False, description="Debug mode (MUST be False in prod)")
    allowed_hosts: str = Field(default="localhost", description="Comma-separated hosts")

    # Optional integrations
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")

    # Trading API settings
    alpaca_api_key: Optional[str] = Field(default=None, description="Alpaca API Key")
    alpaca_secret_key: Optional[str] = Field(default=None, description="Alpaca Secret Key")
    alpaca_paper: bool = Field(default=True, description="Use Alpaca paper trading")

    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    rate_limit_burst: int = Field(default=10, ge=1, le=100)

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if v.startswith("test") or v == "django-insecure":
            raise ValueError("Production SECRET_KEY must not be a test key")
        return v

    @field_validator("debug")
    @classmethod
    def validate_debug(cls, v: bool) -> bool:
        if v and os.getenv("ENVIRONMENT", "development") == "production":
            raise ValueError("DEBUG must be False in production environment")
        return v

    @field_validator("allowed_hosts")
    @classmethod
    def validate_allowed_hosts(cls, v: str) -> str:
        if v == "*":
            raise ValueError("ALLOWED_HOSTS cannot be '*' in production")
        return v

    @property
    def allowed_hosts_list(self) -> List[str]:
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]


def get_production_config() -> ProductionConfig:
    """Get and validate production configuration."""
    try:
        return ProductionConfig()
    except Exception as e:
        print(f"CRITICAL: Configuration validation failed: {e}", file=sys.stderr)
        sys.exit(1)


# Only load config if this is the active settings module
if os.environ.get("DJANGO_SETTINGS_MODULE") == "backend.production_settings":
    config = get_production_config()
else:
    # Provide defaults for non-production environments
    config = None


# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Core Django settings
if config:
    SECRET_KEY = config.secret_key
    DEBUG = config.debug
    ALLOWED_HOSTS = config.allowed_hosts_list
else:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-not-for-production")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "backend.tradingbot",
    "backend.auth0login",
    "backend.home",
]

MIDDLEWARE = [
    "backend.middleware.security.SecurityHeadersMiddleware",
    "backend.middleware.correlation.CorrelationIdMiddleware",
    "backend.middleware.rate_limit.RateLimitMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [
            BASE_DIR / "backend" / "auth0login" / "templates",
        ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# Database with connection pooling
DATABASE_URL = config.database_url if config else os.getenv("DATABASE_URL")
if DATABASE_URL:
    DATABASES = {
        "default": dj_database_url.parse(
            DATABASE_URL,
            conn_max_age=600,  # Keep connections open for 10 minutes
            conn_health_checks=True,  # Validate connections before use
        )
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Security settings for production
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = "DENY"
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_SSL_REDIRECT = os.getenv("SECURE_SSL_REDIRECT", "True").lower() == "true"

# CORS settings
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
CORS_ALLOWED_ORIGINS = [o for o in CORS_ALLOWED_ORIGINS if o]

# REST Framework settings
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/hour",
        "user": "1000/hour",
    },
}

# Logging configuration
LOG_LEVEL = config.log_level if config else os.getenv("LOG_LEVEL", "INFO")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json" if not DEBUG else "standard",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "backend.tradingbot": {
            "handlers": ["console"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    },
}

# Sentry integration
if config and config.sentry_dsn:
    import sentry_sdk
    from sentry_sdk.integrations.django import DjangoIntegration

    sentry_sdk.init(
        dsn=config.sentry_dsn,
        integrations=[DjangoIntegration()],
        traces_sample_rate=0.1,
        send_default_pii=False,
        environment=os.getenv("ENVIRONMENT", "production"),
    )

# Cache configuration
if config and config.redis_url:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": config.redis_url,
            "OPTIONS": {
                "CLIENT_CLASS": "django_redis.client.DefaultClient",
            },
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        }
    }

# Rate limiting configuration (used by middleware)
RATE_LIMIT_PER_MINUTE = config.rate_limit_per_minute if config else 60
RATE_LIMIT_BURST = config.rate_limit_burst if config else 10
