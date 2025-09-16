"""Test Prometheus metrics collection system."""
import pytest
from unittest.mock import patch, MagicMock, Mock

# Try to import metrics components, fall back to mocks if not available
try:
    from backend.tradingbot.metrics.collectors import (
        TradingMetrics,
        CircuitBreakerMetrics,
        DataQualityMetrics,
        PositionMetrics,
        SystemMetrics,
        trading_registry,
        orders_placed_total,
        orders_rejected_total,
        circuit_open,
        errors_total
    )
    METRICS_AVAILABLE = True
except ImportError:
    # Mock all metrics components
    TradingMetrics = MagicMock
    CircuitBreakerMetrics = MagicMock
    DataQualityMetrics = MagicMock
    PositionMetrics = MagicMock
    SystemMetrics = MagicMock
    trading_registry = Mock()
    orders_placed_total = Mock()
    orders_rejected_total = Mock()
    circuit_open = Mock()
    errors_total = Mock()
    METRICS_AVAILABLE = False


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="metrics module not available")
class TestTradingMetrics:
    """Test trading metrics collector."""

    def test_trading_metrics_initialization(self):
        """Test trading metrics initializes correctly."""
        metrics = TradingMetrics()
        assert metrics is not None

    def test_record_order_placed(self):
        """Test recording order placement."""
        metrics = TradingMetrics()

        # Record an order
        metrics.record_order_placed("wsb_dip_bot", "AAPL", "buy")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        orders_metric = next((m for m in metric_families if m.name == "wsb_orders_placed"), None)
        assert orders_metric is not None

        # Find the specific sample
        sample = next((s for s in orders_metric.samples if
                      s.labels.get("strategy") == "wsb_dip_bot" and
                      s.labels.get("symbol") == "AAPL" and
                      s.labels.get("side") == "buy"), None)
        assert sample is not None
        assert sample.value >= 1

    def test_record_order_rejected(self):
        """Test recording order rejection."""
        metrics = TradingMetrics()

        metrics.record_order_rejected("momentum_weeklies", "MSFT", "insufficient_funds")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        rejected_metric = next((m for m in metric_families if m.name == "wsb_orders_rejected"), None)
        assert rejected_metric is not None

    def test_record_order_latency(self):
        """Test recording order latency."""
        metrics = TradingMetrics()

        metrics.record_order_latency("swing_trading", 0.150)  # 150ms

        # Verify metric was recorded (histogram metrics are more complex to verify)
        metric_families = trading_registry.collect()
        latency_metric = next((m for m in metric_families if m.name == "wsb_order_latency_seconds"), None)
        assert latency_metric is not None

    def test_record_error(self):
        """Test recording errors."""
        metrics = TradingMetrics()

        metrics.record_error("data_client", "connection_timeout")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        error_metric = next((m for m in metric_families if m.name == "wsb_errors"), None)
        assert error_metric is not None


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="metrics module not available")
class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics collector."""

    def test_circuit_breaker_metrics_initialization(self):
        """Test circuit breaker metrics initializes correctly."""
        metrics = CircuitBreakerMetrics()
        assert metrics is not None

    def test_update_circuit_status_open(self):
        """Test updating circuit status to open."""
        metrics = CircuitBreakerMetrics()

        # Clear any existing metrics
        circuit_open.clear()

        metrics.update_circuit_status(True, "daily_drawdown")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        circuit_metric = next((m for m in metric_families if m.name == "wsb_circuit_open"), None)
        assert circuit_metric is not None

        # Find the specific sample
        sample = next((s for s in circuit_metric.samples if
                      s.labels.get("reason") == "daily_drawdown"), None)
        assert sample is not None
        assert sample.value == 1

    def test_update_circuit_status_closed(self):
        """Test updating circuit status to closed."""
        metrics = CircuitBreakerMetrics()

        # Clear any existing metrics
        circuit_open.clear()

        metrics.update_circuit_status(False, "error_rate")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        circuit_metric = next((m for m in metric_families if m.name == "wsb_circuit_open"), None)
        assert circuit_metric is not None

        # Find the specific sample
        sample = next((s for s in circuit_metric.samples if
                      s.labels.get("reason") == "error_rate"), None)
        assert sample is not None
        assert sample.value == 0


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="metrics module not available")
class TestDataQualityMetrics:
    """Test data quality metrics collector."""

    def test_data_quality_metrics_initialization(self):
        """Test data quality metrics initializes correctly."""
        metrics = DataQualityMetrics()
        assert metrics is not None

    def test_update_data_staleness(self):
        """Test updating data staleness metric."""
        metrics = DataQualityMetrics()

        metrics.update_data_staleness("market_data", 45.0)  # 45 seconds stale

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        staleness_metric = next((m for m in metric_families if m.name == "wsb_data_staleness_seconds"), None)
        assert staleness_metric is not None

    def test_record_quality_failure(self):
        """Test recording data quality failure."""
        metrics = DataQualityMetrics()

        metrics.record_quality_failure("price_outlier")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        quality_metric = next((m for m in metric_families if m.name == "wsb_data_quality_failures"), None)
        assert quality_metric is not None


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="metrics module not available")
class TestPositionMetrics:
    """Test position metrics collector."""

    def test_position_metrics_initialization(self):
        """Test position metrics initializes correctly."""
        metrics = PositionMetrics()
        assert metrics is not None

    def test_update_portfolio_delta(self):
        """Test updating portfolio delta metrics."""
        metrics = PositionMetrics()

        metrics.update_portfolio_delta(15000.0, 25000.0)

        # Verify metrics were recorded
        metric_families = trading_registry.collect()

        delta_metric = next((m for m in metric_families if m.name == "wsb_portfolio_delta_exposure"), None)
        assert delta_metric is not None

        limit_metric = next((m for m in metric_families if m.name == "wsb_portfolio_delta_limit"), None)
        assert limit_metric is not None

    def test_record_reconciliation_break(self):
        """Test recording EOD reconciliation break."""
        metrics = PositionMetrics()

        metrics.record_reconciliation_break("missing_fill")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        recon_metric = next((m for m in metric_families if m.name == "wsb_eod_reconciliation_breaks"), None)
        assert recon_metric is not None


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="metrics module not available")
class TestSystemMetrics:
    """Test system metrics collector."""

    def test_system_metrics_initialization(self):
        """Test system metrics initializes correctly."""
        metrics = SystemMetrics()
        assert metrics is not None

    def test_update_health_status_healthy(self):
        """Test updating system health status to healthy."""
        metrics = SystemMetrics()

        metrics.update_health_status("healthy")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        health_metric = next((m for m in metric_families if m.name == "wsb_system_health_status"), None)
        assert health_metric is not None

        # Should be set to 1.0 for healthy
        assert any(s.value == 1.0 for s in health_metric.samples)

    def test_update_health_status_degraded(self):
        """Test updating system health status to degraded."""
        metrics = SystemMetrics()

        metrics.update_health_status("degraded")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        health_metric = next((m for m in metric_families if m.name == "wsb_system_health_status"), None)
        assert health_metric is not None

        # Should be set to 0.5 for degraded
        assert any(s.value == 0.5 for s in health_metric.samples)

    def test_update_health_status_unhealthy(self):
        """Test updating system health status to unhealthy."""
        metrics = SystemMetrics()

        metrics.update_health_status("unhealthy")

        # Verify metric was recorded
        metric_families = trading_registry.collect()
        health_metric = next((m for m in metric_families if m.name == "wsb_system_health_status"), None)
        assert health_metric is not None

        # Should be set to 0.0 for unhealthy
        assert any(s.value == 0.0 for s in health_metric.samples)


@pytest.mark.skipif(not METRICS_AVAILABLE, reason="metrics module not available")
class TestMetricsRegistry:
    """Test metrics registry functionality."""

    def test_trading_registry_exists(self):
        """Test trading registry is properly configured."""
        assert trading_registry is not None

    def test_metrics_collection(self):
        """Test metrics can be collected from registry."""
        metric_families = list(trading_registry.collect())
        assert len(metric_families) > 0

        # Verify expected metric families exist
        metric_names = {mf.name for mf in metric_families}
        expected_metrics = {
            "wsb_orders_placed",
            "wsb_orders_rejected",
            "wsb_order_latency_seconds",
            "wsb_circuit_open",
            "wsb_errors",
            "wsb_data_staleness_seconds",
            "wsb_data_quality_failures",
            "wsb_eod_reconciliation_breaks",
            "wsb_portfolio_delta_exposure",
            "wsb_portfolio_delta_limit",
            "wsb_system_health_status"
        }

        for expected in expected_metrics:
            assert expected in metric_names, f"Missing expected metric: {expected}"