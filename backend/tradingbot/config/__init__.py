"""Configuration management for WallStreetBots.

This module provides a unified configuration system that automatically
loads the appropriate environment-specific settings.
"""

import os
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """Get configuration based on environment.

    Returns:
        Dictionary containing all configuration settings
    """
    environment = os.getenv("DJANGO_ENVIRONMENT", "development")

    config = {}
    if environment == "production":
        from .environments.production import ProductionConfig
        config.update(ProductionConfig().dict())
    elif environment == "testing":
        from .environments.testing import TestingConfig
        config.update(TestingConfig().dict())
    else:
        from .environments.development import DevelopmentConfig
        config.update(DevelopmentConfig().dict())

    return config

def get_trading_config() -> Dict[str, Any]:
    """Get trading-specific configuration."""
    config = get_config()
    return config.get("TRADING_CONFIG", {})

def get_api_config() -> Dict[str, Any]:
    """Get API-specific configuration."""
    config = get_config()
    return config.get("API_CONFIG", {})

def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    config = get_config()
    return config.get("DATABASES", {})

# Export commonly used configurations
__all__ = [
    "get_api_config",
    "get_config",
    "get_database_config",
    "get_trading_config",
]