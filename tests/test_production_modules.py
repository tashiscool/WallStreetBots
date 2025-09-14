"""Test production modules for basic functionality."""
import pytest
import time
from decimal import Decimal

from backend.tradingbot.risk.circuit_breaker import CircuitBreaker, BreakerLimits
from backend.tradingbot.execution.replay_guard import ReplayGuard
from backend.tradingbot.ops.eod_recon import LocalOrder, BrokerFill, reconcile
from backend.tradingbot.data.quality import DataQualityMonitor
from backend.tradingbot.infra.build_info import build_id, version_info
from backend.tradingbot.risk.greek_exposure_limits import GreekExposureLimiter, GreekLimits, PositionGreeks


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes correctly."""
        breaker = CircuitBreaker(start_equity=100000.0)
        assert breaker.start_equity == 100000.0
        assert breaker.can_trade() is True
        assert breaker.tripped_at is None
    
    def test_circuit_breaker_error_tracking(self):
        """Test error tracking functionality."""
        breaker = CircuitBreaker(start_equity=100000.0, limits=BreakerLimits(max_error_rate_per_min=2))
        
        # Should not trip with 1 error
        breaker.mark_error()
        assert breaker.can_trade() is True
        
        # Should trip with 2 errors
        breaker.mark_error()
        assert breaker.can_trade() is False
        assert breaker.reason == "error-rate"
    
    def test_circuit_breaker_drawdown_check(self):
        """Test drawdown checking."""
        breaker = CircuitBreaker(start_equity=100000.0, limits=BreakerLimits(max_daily_drawdown=0.05))
        
        # Should trip with 6% drawdown
        breaker.check_mtm(94000.0)  # 6% drawdown
        assert breaker.can_trade() is False
        assert "daily-dd" in breaker.reason
    
    def test_circuit_breaker_data_staleness(self):
        """Test data staleness checking."""
        breaker = CircuitBreaker(start_equity=100000.0, limits=BreakerLimits(max_data_staleness_sec=1))
        
        # Should trip after staleness timeout
        time.sleep(1.1)
        breaker.poll()
        assert breaker.can_trade() is False
        assert breaker.reason == "stale-data"


class TestReplayGuard:
    """Test replay guard functionality."""
    
    def test_replay_guard_initialization(self):
        """Test replay guard initializes correctly."""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            os.unlink(temp_path)  # Remove the file so it starts fresh
            guard = ReplayGuard(path=temp_path)
            assert guard.seen("test_order_123") is False
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_replay_guard_record_and_seen(self):
        """Test recording and checking orders."""
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            os.unlink(temp_path)  # Remove the file so it starts fresh
            guard = ReplayGuard(path=temp_path)

            # Record an order
            guard.record("test_order_123", "acknowledged")
            assert guard.seen("test_order_123") is True
            assert guard.get_state("test_order_123") == "acknowledged"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestEODReconciliation:
    """Test EOD reconciliation functionality."""
    
    def test_reconciliation_no_breaks(self):
        """Test reconciliation with no breaks."""
        local_orders = [
            LocalOrder("order1", "broker1", "AAPL", "buy", 100, "filled"),
            LocalOrder("order2", "broker2", "MSFT", "sell", 50, "filled"),
        ]
        
        broker_fills = [
            BrokerFill("broker1", "AAPL", 100, 150.0, "filled"),
            BrokerFill("broker2", "MSFT", 50, 300.0, "filled"),
        ]
        
        breaks = reconcile(local_orders, broker_fills)
        assert len(breaks.missing_fill) == 0
        assert len(breaks.unknown_broker_fill) == 0
        assert len(breaks.state_mismatch) == 0
    
    def test_reconciliation_missing_fill(self):
        """Test reconciliation with missing fills."""
        local_orders = [
            LocalOrder("order1", "broker1", "AAPL", "buy", 100, "filled"),
            LocalOrder("order2", None, "MSFT", "sell", 50, "pending"),
        ]
        
        broker_fills = [
            BrokerFill("broker1", "AAPL", 100, 150.0, "filled"),
        ]
        
        breaks = reconcile(local_orders, broker_fills)
        assert len(breaks.missing_fill) == 1
        assert "order2" in breaks.missing_fill


class TestDataQualityMonitor:
    """Test data quality monitoring."""
    
    def test_data_quality_monitor_initialization(self):
        """Test data quality monitor initializes correctly."""
        monitor = DataQualityMonitor(max_staleness_sec=30, max_return_z=5.0)
        assert monitor.max_staleness_sec == 30
        assert monitor.max_return_z == 5.0
    
    def test_data_freshness_tracking(self):
        """Test data freshness tracking."""
        monitor = DataQualityMonitor(max_staleness_sec=1)
        
        # Should be fresh initially
        monitor.assert_fresh()
        
        # Should fail after staleness timeout
        time.sleep(1.1)
        with pytest.raises(RuntimeError, match="DATA_STALE"):
            monitor.assert_fresh()


class TestBuildInfo:
    """Test build info functionality."""
    
    def test_build_id(self):
        """Test build ID generation."""
        build_id_str = build_id()
        assert isinstance(build_id_str, str)
        assert len(build_id_str) > 0
    
    def test_version_info(self):
        """Test version info generation."""
        info = version_info()
        assert "build_id" in info
        assert "build_timestamp" in info
        assert "python_version" in info
        assert "platform" in info


class TestGreekExposureLimiter:
    """Test Greek exposure limiting."""
    
    def test_greek_limiter_initialization(self):
        """Test Greek limiter initializes correctly."""
        limiter = GreekExposureLimiter()
        assert len(limiter.positions) == 0
    
    def test_position_management(self):
        """Test adding and removing positions."""
        limiter = GreekExposureLimiter()
        
        position = PositionGreeks(
            symbol="AAPL",
            delta=1000.0,
            gamma=50.0,
            theta=-10.0,
            vega=200.0,
            beta=1.2,
            notional=100000.0
        )
        
        limiter.add_position(position)
        assert "AAPL" in limiter.positions
        
        limiter.remove_position("AAPL")
        assert "AAPL" not in limiter.positions
    
    def test_portfolio_greeks_calculation(self):
        """Test portfolio Greeks calculation."""
        limiter = GreekExposureLimiter()
        
        position1 = PositionGreeks("AAPL", 1000.0, 50.0, -10.0, 200.0, 1.2, 100000.0)
        position2 = PositionGreeks("MSFT", 500.0, 25.0, -5.0, 100.0, 0.8, 50000.0)
        
        limiter.add_position(position1)
        limiter.add_position(position2)
        
        greeks = limiter.get_portfolio_greeks()
        assert greeks["delta"] == 1500.0
        assert greeks["gamma"] == 75.0
        assert greeks["theta"] == -15.0
        assert greeks["vega"] == 300.0
        assert greeks["beta_adjusted_exposure"] == 160000.0  # 100k*1.2 + 50k*0.8
    
    def test_limit_violations(self):
        """Test limit violation detection."""
        limiter = GreekExposureLimiter(limits=GreekLimits(max_portfolio_delta=1000.0))
        
        position = PositionGreeks("AAPL", 1500.0, 50.0, -10.0, 200.0, 1.2, 100000.0)
        limiter.add_position(position)
        
        violations = limiter.check_portfolio_limits()
        assert len(violations) > 0
        assert any("delta" in v for v in violations)

