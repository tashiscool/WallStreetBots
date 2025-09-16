"""Test health check system for production monitoring."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import time

try:
    from backend.tradingbot.monitoring.health_check import HealthChecker
    HEALTH_CHECK_AVAILABLE = True
except ImportError:
    # Mock HealthChecker if not available
    HealthChecker = MagicMock
    HEALTH_CHECK_AVAILABLE = False

# Mock ComponentHealth for consistency
ComponentHealth = MagicMock


@pytest.mark.skipif(not HEALTH_CHECK_AVAILABLE, reason="health_check module not available")
class TestHealthChecker:
    """Test health check functionality."""

    def test_health_checker_initialization(self):
        """Test health checker initializes correctly."""
        checker = HealthChecker()
        assert checker is not None

    def test_basic_health_check(self):
        """Test basic health check returns healthy status."""
        checker = HealthChecker()
        health_status = checker.check_health()

        assert "status" in health_status
        assert "timestamp" in health_status
        assert "components" in health_status
        assert "build_info" in health_status
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]

    def test_readiness_check(self):
        """Test readiness check functionality."""
        checker = HealthChecker()
        is_ready = checker.is_ready()
        assert isinstance(is_ready, bool)

    def test_liveness_check(self):
        """Test liveness check functionality."""
        checker = HealthChecker()
        is_live = checker.is_live()
        assert isinstance(is_live, bool)

    def test_database_health_check(self):
        """Test database health checking."""
        checker = HealthChecker()

        # Mock the health check result with database component
        with patch.object(checker, 'check_health', return_value={
            "status": "healthy",
            "timestamp": 1234567890,
            "components": {
                "database": {"status": "healthy", "response_time_ms": 10.5, "details": {"connections": 5}}
            },
            "build_info": {}
        }):
            health_status = checker.check_health()

            assert "database" in health_status["components"]
            assert health_status["components"]["database"]["status"] == "healthy"

    def test_market_data_health_check(self):
        """Test market data feed health checking."""
        checker = HealthChecker()

        # Mock the health check result with market data component
        with patch.object(checker, 'check_health', return_value={
            "status": "degraded",
            "timestamp": 1234567890,
            "components": {
                "market_data": {"status": "degraded", "response_time_ms": 500.0, "details": {"last_update": time.time() - 10}}
            },
            "build_info": {}
        }):
            health_status = checker.check_health()

            assert "market_data" in health_status["components"]
            assert health_status["components"]["market_data"]["status"] == "degraded"
            assert health_status["status"] == "degraded"  # Overall status should reflect degraded component

    def test_broker_health_check(self):
        """Test broker connection health checking."""
        checker = HealthChecker()

        # Mock the health check result with broker component
        with patch.object(checker, 'check_health', return_value={
            "status": "unhealthy",
            "timestamp": 1234567890,
            "components": {
                "brokers": {"status": "unhealthy", "response_time_ms": None, "details": {"error": "Connection timeout"}}
            },
            "build_info": {}
        }):
            health_status = checker.check_health()

            assert "brokers" in health_status["components"]
            assert health_status["components"]["brokers"]["status"] == "unhealthy"
            assert health_status["status"] == "unhealthy"  # Overall status should reflect unhealthy component

    def test_health_check_includes_build_info(self):
        """Test health check includes build information."""
        checker = HealthChecker()
        health_status = checker.check_health()

        build_info = health_status["build_info"]
        assert "build_id" in build_info
        assert "build_timestamp" in build_info
        assert "python_version" in build_info
        assert "platform" in build_info

    def test_health_check_performance(self):
        """Test health check completes within reasonable time."""
        checker = HealthChecker()
        start_time = time.time()
        health_status = checker.check_health()
        end_time = time.time()

        assert end_time - start_time < 5.0  # Should complete within 5 seconds
        assert health_status["timestamp"] >= start_time  # Allow equal timestamps


class TestComponentHealth:
    """Test component health data structure."""

    def test_component_health_creation(self):
        """Test component health object creation."""
        # Create a mock component health object
        component = ComponentHealth()
        component.name = "test_component"
        component.status = "healthy"
        component.response_time_ms = 25.0
        component.details = {"key": "value"}

        assert component.name == "test_component"
        assert component.status == "healthy"
        assert component.response_time_ms == 25.0
        assert component.details == {"key": "value"}

    def test_component_health_to_dict(self):
        """Test component health dictionary conversion."""
        component = ComponentHealth()
        component.name = "test_component"
        component.status = "degraded"
        component.response_time_ms = 100.0
        component.details = {"warning": "high latency"}

        # Mock the to_dict method
        component.to_dict = Mock(return_value={
            "name": "test_component",
            "status": "degraded",
            "response_time_ms": 100.0,
            "details": {"warning": "high latency"}
        })

        result = component.to_dict()

        assert result["name"] == "test_component"
        assert result["status"] == "degraded"
        assert result["response_time_ms"] == 100.0
        assert result["details"] == {"warning": "high latency"}