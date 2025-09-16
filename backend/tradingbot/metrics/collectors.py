"""Prometheus metrics collectors for WallStreetBots."""
from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from typing import Dict, Any
import time

# Create a custom registry for trading metrics
trading_registry = CollectorRegistry()

# Order execution metrics
orders_placed_total = Counter(
    'wsb_orders_placed_total',
    'Total number of orders placed',
    ['strategy', 'symbol', 'side'],
    registry=trading_registry
)

orders_rejected_total = Counter(
    'wsb_orders_rejected_total',
    'Total number of orders rejected',
    ['strategy', 'symbol', 'reason'],
    registry=trading_registry
)

order_latency_seconds = Histogram(
    'wsb_order_latency_seconds',
    'Order placement latency in seconds',
    ['strategy'],
    registry=trading_registry
)

# Circuit breaker metrics
circuit_open = Gauge(
    'wsb_circuit_open',
    'Circuit breaker status (1=open, 0=closed)',
    ['reason'],
    registry=trading_registry
)

errors_total = Counter(
    'wsb_errors_total',
    'Total number of errors',
    ['component', 'error_type'],
    registry=trading_registry
)

# Data quality metrics
data_staleness_seconds = Gauge(
    'wsb_data_staleness_seconds',
    'Data staleness in seconds',
    ['source'],
    registry=trading_registry
)

data_quality_failures_total = Counter(
    'wsb_data_quality_failures_total',
    'Total data quality failures',
    ['check_type'],
    registry=trading_registry
)

# Portfolio metrics
portfolio_delta_exposure = Gauge(
    'wsb_portfolio_delta_exposure',
    'Current portfolio delta exposure',
    registry=trading_registry
)

portfolio_delta_limit = Gauge(
    'wsb_portfolio_delta_limit',
    'Portfolio delta limit',
    registry=trading_registry
)

# EOD reconciliation metrics
eod_reconciliation_breaks_total = Counter(
    'wsb_eod_reconciliation_breaks_total',
    'Total EOD reconciliation breaks',
    ['break_type'],
    registry=trading_registry
)

# System health metrics
system_health_status = Gauge(
    'wsb_system_health_status',
    'System health status (1=healthy, 0.5=degraded, 0=unhealthy)',
    registry=trading_registry
)


class TradingMetrics:
    """Trading metrics collector."""

    def record_order_placed(self, strategy: str, symbol: str, side: str) -> None:
        """Record an order placement."""
        orders_placed_total.labels(strategy=strategy, symbol=symbol, side=side).inc()

    def record_order_rejected(self, strategy: str, symbol: str, reason: str) -> None:
        """Record an order rejection."""
        orders_rejected_total.labels(strategy=strategy, symbol=symbol, reason=reason).inc()

    def record_order_latency(self, strategy: str, latency_seconds: float) -> None:
        """Record order placement latency."""
        order_latency_seconds.labels(strategy=strategy).observe(latency_seconds)

    def record_error(self, component: str, error_type: str) -> None:
        """Record an error occurrence."""
        errors_total.labels(component=component, error_type=error_type).inc()


class CircuitBreakerMetrics:
    """Circuit breaker metrics collector."""

    def update_circuit_status(self, is_open: bool, reason: str = "") -> None:
        """Update circuit breaker status."""
        circuit_open.labels(reason=reason).set(1 if is_open else 0)


class DataQualityMetrics:
    """Data quality metrics collector."""

    def update_data_staleness(self, source: str, staleness_seconds: float) -> None:
        """Update data staleness metric."""
        data_staleness_seconds.labels(source=source).set(staleness_seconds)

    def record_quality_failure(self, check_type: str) -> None:
        """Record a data quality failure."""
        data_quality_failures_total.labels(check_type=check_type).inc()


class PositionMetrics:
    """Position and portfolio metrics collector."""

    def update_portfolio_delta(self, delta: float, limit: float) -> None:
        """Update portfolio delta metrics."""
        portfolio_delta_exposure.set(delta)
        portfolio_delta_limit.set(limit)

    def record_reconciliation_break(self, break_type: str) -> None:
        """Record an EOD reconciliation break."""
        eod_reconciliation_breaks_total.labels(break_type=break_type).inc()


class SystemMetrics:
    """System health metrics collector."""

    def update_health_status(self, status: str) -> None:
        """Update system health status."""
        status_value = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0}.get(status, 0.0)
        system_health_status.set(status_value)


# Global metric collectors
trading_metrics = TradingMetrics()
circuit_breaker_metrics = CircuitBreakerMetrics()
data_quality_metrics = DataQualityMetrics()
order_metrics = TradingMetrics()  # Alias for backward compatibility
position_metrics = PositionMetrics()
system_metrics = SystemMetrics()