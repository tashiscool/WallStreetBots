"""Tests for system health monitoring module."""

import asyncio
import importlib
import sys
from unittest import mock

import pytest

from backend.tradingbot.monitoring.system_health import (
    HealthStatus,
    SystemHealthMonitor,
)


def test_system_health_init_without_psutil():
    """SystemHealthMonitor should initialize even when psutil is not installed."""
    from backend.tradingbot.monitoring import system_health

    saved_psutil = sys.modules.get("psutil")
    saved_avail = system_health.PSUTIL_AVAILABLE
    saved_obj = system_health.psutil
    try:
        # Simulate psutil not being available by blocking its import and
        # reloading the module so the top-level try/except re-executes.
        sys.modules["psutil"] = None
        importlib.reload(system_health)

        assert hasattr(system_health, "PSUTIL_AVAILABLE")
        assert system_health.PSUTIL_AVAILABLE is False
    finally:
        # Restore original state so other tests aren't affected
        if saved_psutil is not None:
            sys.modules["psutil"] = saved_psutil
        elif "psutil" in sys.modules:
            del sys.modules["psutil"]
        system_health.PSUTIL_AVAILABLE = saved_avail
        system_health.psutil = saved_obj
        importlib.reload(system_health)


def test_system_health_init_with_config_none():
    """SystemHealthMonitor should handle config=None without crashing."""
    monitor = SystemHealthMonitor(config=None)
    assert monitor.config == {}


def test_system_health_config_stored_correctly():
    """SystemHealthMonitor should store config properly."""
    test_config = {"check_interval": 120, "enable_alerts": True}
    monitor = SystemHealthMonitor(config=test_config)
    assert monitor.config == test_config


@pytest.mark.asyncio
async def test_system_health_monitor_accepts_none_config():
    monitor = SystemHealthMonitor(config=None)
    assert monitor.alert_thresholds["data_feed_latency"] == 5.0
    assert monitor.alert_thresholds["memory_usage"] == 0.80


@pytest.mark.asyncio
async def test_system_resource_check_degrades_gracefully_without_psutil(monkeypatch):
    monitor = SystemHealthMonitor(config={})
    monkeypatch.setattr(
        "backend.tradingbot.monitoring.system_health.PSUTIL_AVAILABLE", False
    )

    resources = await monitor._check_system_resources()

    assert resources.status.value == HealthStatus.UNKNOWN.value
    assert resources.details["error"] == "psutil is not installed"
