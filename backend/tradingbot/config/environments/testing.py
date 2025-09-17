"""Testing environment configuration."""

from ..base import *

# Testing-specific settings
DEBUG = False
SECRET_KEY = "test-secret-key-for-testing-only"

# Testing database (in-memory for speed)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Testing trading config
TRADING_CONFIG.update({
    "DEFAULT_PAPER_TRADING": True,
    "MAX_POSITION_SIZE": 100,  # Minimal for testing
    "MAX_TOTAL_RISK": 500,
})

# Testing logging (minimal)
LOGGING_CONFIG["root"]["level"] = "WARNING"

# Disable migrations for testing
class DisableMigrations:
    def __contains__(self, item):
        return True
    
    def __getitem__(self, item):
        return None

MIGRATION_MODULES = DisableMigrations()

# Fast password hashing for tests
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.MD5PasswordHasher',
]