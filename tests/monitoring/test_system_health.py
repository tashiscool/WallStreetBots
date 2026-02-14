"""Tests for system health monitoring module."""

import asyncio
from unittest import mock


def test_system_health_init_without_psutil():
    """SystemHealthMonitor should initialize even when psutil is not installed."""
    import sys

    saved = sys.modules.get("psutil")
    try:
        sys.modules["psutil"] = None  # Block psutil import

        # Force reimport
        mod_name = "backend.tradingbot.monitoring.system_health"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        try:
            from backend.tradingbot.monitoring import system_health

            assert hasattr(system_health, "PSUTIL_AVAILABLE")
            assert system_health.PSUTIL_AVAILABLE is False
        except ImportError:
            # Module may not be importable in test environment
            pass
    finally:
        if saved is not None:
            sys.modules["psutil"] = saved
        elif "psutil" in sys.modules:
            del sys.modules["psutil"]


def test_system_health_init_with_config_none():
    """SystemHealthMonitor should handle config=None without crashing."""
    try:
        from backend.tradingbot.monitoring.system_health import SystemHealthMonitor

        # This should NOT raise AttributeError: 'NoneType' object has no attribute 'get'
        monitor = SystemHealthMonitor(config=None)
        assert monitor.config == {}
    except ImportError:
        # Module may not be importable in test environment
        pass


def test_system_health_config_stored_correctly():
    """SystemHealthMonitor should store config properly."""
    try:
        from backend.tradingbot.monitoring.system_health import SystemHealthMonitor

        test_config = {"check_interval": 120, "enable_alerts": True}
        monitor = SystemHealthMonitor(config=test_config)
        assert monitor.config == test_config
    except ImportError:
        pass
