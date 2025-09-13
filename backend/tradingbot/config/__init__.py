"""Configuration module for WallStreetBots."""

try:
    from .settings import load_settings, AppSettings, StrategyProfile
except ImportError:
    # Fallback to simple settings
    from .simple_settings import load_settings, AppSettings, StrategyProfile

__all__ = ["AppSettings", "StrategyProfile", "load_settings"]
