"""Django settings for WallStreetBots.

Default profile is safe for local development. Production runtime should use
``backend.production_settings``.
"""

import os
from pathlib import Path
import dj_database_url

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
IS_PRODUCTION = ENVIRONMENT in {"prod", "production"}

_secret_key = os.getenv("SECRET_KEY", "")
if not _secret_key and IS_PRODUCTION:
    raise RuntimeError("SECRET_KEY must be set when ENVIRONMENT=production")
SECRET_KEY = _secret_key or "development-only-secret-key-change-me"

# SECURITY WARNING: don't run with debug turned on in production.
DEBUG = os.getenv("DJANGO_DEBUG", os.getenv("DEBUG", "False")).lower() in (
    "true",
    "1",
    "yes",
)
if DEBUG and IS_PRODUCTION:
    raise RuntimeError("DEBUG must be disabled when ENVIRONMENT=production")

_allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1")
ALLOWED_HOSTS = [h.strip() for h in _allowed_hosts.split(",") if h.strip()]
if not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
if "*" in ALLOWED_HOSTS and not DEBUG:
    raise RuntimeError("ALLOWED_HOSTS cannot contain '*' unless DEBUG=true")

_csrf_trusted_origins = os.getenv("CSRF_TRUSTED_ORIGINS", "")
CSRF_TRUSTED_ORIGINS = [
    origin.strip() for origin in _csrf_trusted_origins.split(",") if origin.strip()
]

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party apps
    "social_django",
    # Application apps
    "backend.home",
    "backend.tradingbot",
    "backend.auth0login",
    "backend.api_docs",
]

MIDDLEWARE = [
    # Custom middleware (order matters!)
    "backend.middleware.correlation.CorrelationIdMiddleware",  # First: adds correlation ID
    "backend.middleware.rate_limit.RateLimitMiddleware",  # Early: rate limiting
    # Django security
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # Static file serving
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    # Audit trail (after auth: needs request.user)
    "backend.auth0login.audit.AuditMiddleware",
    # Custom security headers (last: adds security headers to response)
    "backend.middleware.security.SecurityHeadersMiddleware",
]

# Rate limiting configuration
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))

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

# Database
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

# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Email backend for testing
EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"

# Credential Encryption Settings.
_salt_value = os.getenv("CREDENTIAL_ENCRYPTION_SALT")
if _salt_value:
    CREDENTIAL_ENCRYPTION_SALT = _salt_value.encode("utf-8")
elif IS_PRODUCTION:
    raise RuntimeError(
        "CREDENTIAL_ENCRYPTION_SALT must be set when ENVIRONMENT=production"
    )
else:
    CREDENTIAL_ENCRYPTION_SALT = b"development_credential_salt_v1"

# Alpaca API settings - accept both APCA_* (Alpaca SDK convention) and ALPACA_* names
BACKEND_ALPACA_ID = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY", "")
BACKEND_ALPACA_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY", "")

# Cache configuration
REDIS_URL = os.getenv("REDIS_URL")
if REDIS_URL:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "OPTIONS": {
                "CLIENT_CLASS": "django_redis.client.DefaultClient",
            },
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "unique-snowflake",
        }
    }

# Logging
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}
