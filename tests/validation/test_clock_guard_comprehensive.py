#!/usr/bin/env python3
"""
Comprehensive tests for ClockGuard to improve coverage.
Tests clock skew and latency validation scenarios.
"""

import pytest
import time
from unittest.mock import patch, Mock
from backend.validation.operations.clock_guard import ClockGuard, ClockGuardConfig, ClockGuardMonitor


class TestClockGuardComprehensive:
    """Comprehensive tests for ClockGuard."""

    def test_default_config(self):
        """Test default configuration."""
        guard = ClockGuard()
        
        assert guard.cfg.max_clock_skew_ms == 250
        assert guard.cfg.max_decision_to_ack_ms == 250

    def test_custom_config(self):
        """Test custom configuration."""
        config = ClockGuardConfig(
            max_clock_skew_ms=100,
            max_decision_to_ack_ms=150
        )
        guard = ClockGuard(config)
        
        assert guard.cfg.max_clock_skew_ms == 100
        assert guard.cfg.max_decision_to_ack_ms == 150

    def test_clock_skew_within_limits(self):
        """Test clock skew within acceptable limits."""
        guard = ClockGuard()
        
        # Test with skew within limits
        feed_ts_ms = 1000000
        local_recv_ms = 1000200  # 200ms later
        
        # Should not raise exception
        guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_clock_skew_exceeds_limits(self):
        """Test clock skew exceeding limits."""
        guard = ClockGuard()
        
        # Test with skew exceeding limits
        feed_ts_ms = 1000000
        local_recv_ms = 1000300  # 300ms later (exceeds 250ms limit)
        
        with pytest.raises(RuntimeError, match="Clock skew 300 ms exceeds 250 ms"):
            guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_clock_skew_negative_difference(self):
        """Test clock skew with negative difference."""
        guard = ClockGuard()
        
        # Test with local time ahead of feed time
        feed_ts_ms = 1000000
        local_recv_ms = 999800  # 200ms earlier
        
        # Should not raise exception (absolute difference is 200ms)
        guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_clock_skew_negative_exceeds_limits(self):
        """Test clock skew with negative difference exceeding limits."""
        guard = ClockGuard()
        
        # Test with local time ahead of feed time by more than limit
        feed_ts_ms = 1000000
        local_recv_ms = 999600  # 400ms earlier (exceeds 250ms limit)
        
        with pytest.raises(RuntimeError, match="Clock skew 400 ms exceeds 250 ms"):
            guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_clock_skew_exact_limit(self):
        """Test clock skew at exact limit."""
        guard = ClockGuard()
        
        # Test with skew at exact limit
        feed_ts_ms = 1000000
        local_recv_ms = 1000250  # Exactly 250ms later
        
        # Should not raise exception (skew == limit, not > limit)
        guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_clock_skew_zero_difference(self):
        """Test clock skew with zero difference."""
        guard = ClockGuard()
        
        # Test with identical timestamps
        feed_ts_ms = 1000000
        local_recv_ms = 1000000
        
        # Should not raise exception
        guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_latency_within_limits(self):
        """Test latency within acceptable limits."""
        guard = ClockGuard()
        
        # Test with latency within limits
        decision_ms = 1000000
        broker_ack_ms = 1000200  # 200ms later
        
        # Should not raise exception
        guard.check_latency(decision_ms, broker_ack_ms)

    def test_latency_exceeds_limits(self):
        """Test latency exceeding limits."""
        guard = ClockGuard()
        
        # Test with latency exceeding limits
        decision_ms = 1000000
        broker_ack_ms = 1000300  # 300ms later (exceeds 250ms limit)
        
        with pytest.raises(RuntimeError, match="Decision→Ack 300 ms exceeds 250 ms"):
            guard.check_latency(decision_ms, broker_ack_ms)

    def test_latency_exact_limit(self):
        """Test latency at exact limit."""
        guard = ClockGuard()
        
        # Test with latency at exact limit
        decision_ms = 1000000
        broker_ack_ms = 1000250  # Exactly 250ms later
        
        # Should not raise exception (latency == limit, not > limit)
        guard.check_latency(decision_ms, broker_ack_ms)

    def test_latency_zero_difference(self):
        """Test latency with zero difference."""
        guard = ClockGuard()
        
        # Test with identical timestamps
        decision_ms = 1000000
        broker_ack_ms = 1000000
        
        # Should not raise exception
        guard.check_latency(decision_ms, broker_ack_ms)

    def test_latency_negative_difference(self):
        """Test latency with negative difference (broker ack before decision)."""
        guard = ClockGuard()
        
        # Test with broker ack before decision (should not happen in practice)
        decision_ms = 1000000
        broker_ack_ms = 999800  # 200ms earlier
        
        # Should not raise exception (latency is 0, not negative)
        guard.check_latency(decision_ms, broker_ack_ms)

    def test_custom_limits_clock_skew(self):
        """Test clock skew with custom limits."""
        config = ClockGuardConfig(max_clock_skew_ms=100)
        guard = ClockGuard(config)
        
        # Test with skew within custom limits
        feed_ts_ms = 1000000
        local_recv_ms = 1000050  # 50ms later
        
        # Should not raise exception
        guard.check_clock_skew(feed_ts_ms, local_recv_ms)
        
        # Test with skew exceeding custom limits
        local_recv_ms = 1000150  # 150ms later (exceeds 100ms limit)
        
        with pytest.raises(RuntimeError, match="Clock skew 150 ms exceeds 100 ms"):
            guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_custom_limits_latency(self):
        """Test latency with custom limits."""
        config = ClockGuardConfig(max_decision_to_ack_ms=100)
        guard = ClockGuard(config)
        
        # Test with latency within custom limits
        decision_ms = 1000000
        broker_ack_ms = 1000050  # 50ms later
        
        # Should not raise exception
        guard.check_latency(decision_ms, broker_ack_ms)
        
        # Test with latency exceeding custom limits
        broker_ack_ms = 1000150  # 150ms later (exceeds 100ms limit)
        
        with pytest.raises(RuntimeError, match="Decision→Ack 150 ms exceeds 100 ms"):
            guard.check_latency(decision_ms, broker_ack_ms)

    def test_very_large_timestamps(self):
        """Test with very large timestamp values."""
        guard = ClockGuard()
        
        # Test with very large timestamps
        feed_ts_ms = 1640995200000  # Large timestamp
        local_recv_ms = 1640995200200  # 200ms later
        
        # Should not raise exception
        guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_very_small_timestamps(self):
        """Test with very small timestamp values."""
        guard = ClockGuard()
        
        # Test with small timestamps
        feed_ts_ms = 1000
        local_recv_ms = 1200  # 200ms later
        
        # Should not raise exception
        guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_edge_case_maximum_skew(self):
        """Test edge case with maximum allowed skew."""
        guard = ClockGuard()
        
        # Test with maximum allowed skew (249ms)
        feed_ts_ms = 1000000
        local_recv_ms = 1000249  # 249ms later (just under 250ms limit)
        
        # Should not raise exception
        guard.check_clock_skew(feed_ts_ms, local_recv_ms)

    def test_edge_case_maximum_latency(self):
        """Test edge case with maximum allowed latency."""
        guard = ClockGuard()
        
        # Test with maximum allowed latency (249ms)
        decision_ms = 1000000
        broker_ack_ms = 1000249  # 249ms later (just under 250ms limit)
        
        # Should not raise exception
        guard.check_latency(decision_ms, broker_ack_ms)

    def test_multiple_checks_same_instance(self):
        """Test multiple checks with same instance."""
        guard = ClockGuard()
        
        # Perform multiple checks
        guard.check_clock_skew(1000000, 1000200)
        guard.check_latency(1000000, 1000200)
        guard.check_clock_skew(2000000, 2000200)
        guard.check_latency(2000000, 2000200)
        
        # Should not raise exceptions
        assert True  # If we get here, all checks passed

    def test_error_message_formatting(self):
        """Test error message formatting."""
        guard = ClockGuard()
        
        # Test clock skew error message
        with pytest.raises(RuntimeError) as exc_info:
            guard.check_clock_skew(1000000, 1000300)
        
        assert "Clock skew 300 ms exceeds 250 ms" in str(exc_info.value)
        
        # Test latency error message
        with pytest.raises(RuntimeError) as exc_info:
            guard.check_latency(1000000, 1000300)
        
        assert "Decision→Ack 300 ms exceeds 250 ms" in str(exc_info.value)

    def test_config_modification(self):
        """Test that config can be modified after creation."""
        config = ClockGuardConfig()
        guard = ClockGuard(config)
        
        # Config should be accessible
        assert guard.cfg.max_clock_skew_ms == 250
        
        # Modifying the config should affect the guard's behavior
        config.max_clock_skew_ms = 100
        
        # Guard should use modified config
        assert guard.cfg.max_clock_skew_ms == 100

    def test_none_config_handling(self):
        """Test handling of None config."""
        guard = ClockGuard(None)
        
        # Should use default config
        assert guard.cfg.max_clock_skew_ms == 250
        assert guard.cfg.max_decision_to_ack_ms == 250
        
        # Should work normally
        guard.check_clock_skew(1000000, 1000200)
        guard.check_latency(1000000, 1000200)

    def test_boundary_value_testing(self):
        """Test boundary values systematically."""
        guard = ClockGuard()
        
        # Test clock skew boundaries
        base_time = 1000000
        
        # Just under limit
        guard.check_clock_skew(base_time, base_time + 249)
        
        # At limit (should not fail, since it's == not >)
        guard.check_clock_skew(base_time, base_time + 250)
        
        # Just over limit
        with pytest.raises(RuntimeError):
            guard.check_clock_skew(base_time, base_time + 251)
        
        # Test latency boundaries
        # Just under limit
        guard.check_latency(base_time, base_time + 249)
        
        # At limit (should not fail, since it's == not >)
        guard.check_latency(base_time, base_time + 250)
        
        # Just over limit
        with pytest.raises(RuntimeError):
            guard.check_latency(base_time, base_time + 251)

    def test_negative_timestamps(self):
        """Test with negative timestamp values."""
        guard = ClockGuard()

        # Test with negative timestamps
        feed_ts_ms = -1000000
        local_recv_ms = -999800  # 200ms later

        # Should not raise exception
        guard.check_clock_skew(feed_ts_ms, local_recv_ms)

        # Test latency with negative timestamps
        decision_ms = -1000000
        broker_ack_ms = -999800  # 200ms later

        # Should not raise exception
        guard.check_latency(decision_ms, broker_ack_ms)


class TestClockGuardFeedLatency:
    """Test feed latency checking functionality."""

    def test_feed_latency_within_limits(self):
        """Test feed latency within acceptable limits."""
        guard = ClockGuard()

        # Test with feed latency within limits (500ms)
        feed_ts_ms = 1000000
        current_ms = 1000500  # 500ms later (within 1000ms limit)

        # Should not raise exception
        guard.check_feed_latency(feed_ts_ms, current_ms)

    def test_feed_latency_exceeds_limits(self):
        """Test feed latency exceeding limits."""
        guard = ClockGuard()

        # Test with feed latency exceeding limits
        feed_ts_ms = 1000000
        current_ms = 1001500  # 1500ms later (exceeds 1000ms limit)

        with pytest.raises(RuntimeError, match="Feed latency 1500 ms exceeds 1000 ms"):
            guard.check_feed_latency(feed_ts_ms, current_ms)

    def test_feed_latency_exact_limit(self):
        """Test feed latency at exact limit."""
        guard = ClockGuard()

        # Test with feed latency at exact limit
        feed_ts_ms = 1000000
        current_ms = 1001000  # Exactly 1000ms later

        # Should not raise exception (latency == limit, not > limit)
        guard.check_feed_latency(feed_ts_ms, current_ms)

    def test_feed_latency_custom_config(self):
        """Test feed latency with custom configuration."""
        config = ClockGuardConfig(max_feed_latency_ms=500)
        guard = ClockGuard(config)

        # Test within custom limit
        feed_ts_ms = 1000000
        current_ms = 1000400  # 400ms later (within 500ms limit)
        guard.check_feed_latency(feed_ts_ms, current_ms)

        # Test exceeding custom limit
        current_ms = 1000600  # 600ms later (exceeds 500ms limit)
        with pytest.raises(RuntimeError, match="Feed latency 600 ms exceeds 500 ms"):
            guard.check_feed_latency(feed_ts_ms, current_ms)

    def test_feed_latency_negative_time(self):
        """Test feed latency with future feed timestamp."""
        guard = ClockGuard()

        # Test with feed timestamp in the future
        feed_ts_ms = 1000000
        current_ms = 999500  # 500ms earlier (negative latency)

        # Should not raise exception (latency is negative, so < max)
        guard.check_feed_latency(feed_ts_ms, current_ms)


class TestClockGuardViolations:
    """Test violation tracking and accumulation."""

    def test_violations_initial_state(self):
        """Test initial violations state."""
        guard = ClockGuard()

        assert guard.violations == 0
        assert guard.max_violations == 3
        assert not guard.should_halt()

    def test_violation_accumulation_clock_skew(self):
        """Test violation accumulation for clock skew."""
        guard = ClockGuard()

        # First violation
        with pytest.raises(RuntimeError, match="violation 1/3"):
            guard.check_clock_skew(1000000, 1000300)
        assert guard.violations == 1
        assert not guard.should_halt()

        # Second violation
        with pytest.raises(RuntimeError, match="violation 2/3"):
            guard.check_clock_skew(2000000, 2000300)
        assert guard.violations == 2
        assert not guard.should_halt()

        # Third violation (should trigger halt)
        with pytest.raises(RuntimeError, match="violation 3/3"):
            guard.check_clock_skew(3000000, 3000300)
        assert guard.violations == 3
        assert guard.should_halt()

    def test_violation_accumulation_latency(self):
        """Test violation accumulation for latency."""
        guard = ClockGuard()

        # First violation
        with pytest.raises(RuntimeError, match="violation 1/3"):
            guard.check_latency(1000000, 1000300)
        assert guard.violations == 1

        # Second violation
        with pytest.raises(RuntimeError, match="violation 2/3"):
            guard.check_latency(2000000, 2000300)
        assert guard.violations == 2

    def test_violation_accumulation_feed_latency(self):
        """Test violation accumulation for feed latency."""
        guard = ClockGuard()

        # First violation
        with pytest.raises(RuntimeError, match="violation 1/3"):
            guard.check_feed_latency(1000000, 1002000)  # 2000ms exceeds 1000ms limit
        assert guard.violations == 1

        # Second violation
        with pytest.raises(RuntimeError, match="violation 2/3"):
            guard.check_feed_latency(2000000, 2002000)
        assert guard.violations == 2

    def test_mixed_violations(self):
        """Test violations across different check types."""
        guard = ClockGuard()

        # Clock skew violation
        with pytest.raises(RuntimeError):
            guard.check_clock_skew(1000000, 1000300)
        assert guard.violations == 1

        # Latency violation
        with pytest.raises(RuntimeError):
            guard.check_latency(2000000, 2000300)
        assert guard.violations == 2

        # Feed latency violation (should trigger halt)
        with pytest.raises(RuntimeError):
            guard.check_feed_latency(3000000, 3002000)
        assert guard.violations == 3
        assert guard.should_halt()

    def test_reset_violations(self):
        """Test violation reset functionality."""
        guard = ClockGuard()

        # Accumulate violations
        with pytest.raises(RuntimeError):
            guard.check_clock_skew(1000000, 1000300)
        with pytest.raises(RuntimeError):
            guard.check_latency(2000000, 2000300)
        assert guard.violations == 2

        # Reset violations
        guard.reset_violations()
        assert guard.violations == 0
        assert not guard.should_halt()

        # Violations should start counting from 0 again
        with pytest.raises(RuntimeError, match="violation 1/3"):
            guard.check_clock_skew(3000000, 3000300)
        assert guard.violations == 1

    def test_should_halt_boundary(self):
        """Test should_halt boundary conditions."""
        guard = ClockGuard()

        # Below threshold
        guard.violations = 2
        assert not guard.should_halt()

        # At threshold
        guard.violations = 3
        assert guard.should_halt()

        # Above threshold
        guard.violations = 4
        assert guard.should_halt()


class TestClockGuardStatus:
    """Test guard status reporting."""

    def test_get_status_initial(self):
        """Test initial status reporting."""
        guard = ClockGuard()

        status = guard.get_status()

        assert status['violations'] == 0
        assert status['max_violations'] == 3
        assert status['should_halt'] is False
        assert status['config']['max_clock_skew_ms'] == 250
        assert status['config']['max_decision_to_ack_ms'] == 250
        assert status['config']['max_feed_latency_ms'] == 1000

    def test_get_status_with_violations(self):
        """Test status reporting with violations."""
        guard = ClockGuard()

        # Add violations
        with pytest.raises(RuntimeError):
            guard.check_clock_skew(1000000, 1000300)
        with pytest.raises(RuntimeError):
            guard.check_latency(2000000, 2000300)

        status = guard.get_status()

        assert status['violations'] == 2
        assert status['max_violations'] == 3
        assert status['should_halt'] is False

    def test_get_status_at_halt_threshold(self):
        """Test status reporting at halt threshold."""
        guard = ClockGuard()

        # Trigger halt condition
        guard.violations = 3

        status = guard.get_status()

        assert status['violations'] == 3
        assert status['max_violations'] == 3
        assert status['should_halt'] is True

    def test_get_status_custom_config(self):
        """Test status reporting with custom configuration."""
        config = ClockGuardConfig(
            max_clock_skew_ms=100,
            max_decision_to_ack_ms=150,
            max_feed_latency_ms=500
        )
        guard = ClockGuard(config)

        status = guard.get_status()

        assert status['config']['max_clock_skew_ms'] == 100
        assert status['config']['max_decision_to_ack_ms'] == 150
        assert status['config']['max_feed_latency_ms'] == 500


class TestClockGuardMonitor:
    """Test ClockGuardMonitor functionality."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        assert monitor.clock_guard is guard
        assert monitor.last_check_ms == 0
        assert monitor.check_interval_ms == 1000

    def test_should_check_timing(self):
        """Test check timing logic."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        current_ms = 10000

        # First check should always be performed
        assert monitor.should_check(current_ms)

        # Update last check time
        monitor.last_check_ms = current_ms

        # Check within interval should not be needed
        assert not monitor.should_check(current_ms + 500)

        # Check after interval should be needed
        assert monitor.should_check(current_ms + 1000)
        assert monitor.should_check(current_ms + 1500)

    def test_perform_checks_success(self):
        """Test successful performance of all checks."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        current_ms = 1000000
        feed_ts_ms = 999800  # 200ms old feed (within 250ms clock skew limit)
        decision_ms = 999600
        broker_ack_ms = 999750  # 150ms latency (within limits)

        results = monitor.perform_checks(feed_ts_ms, current_ms, decision_ms, broker_ack_ms)

        assert 'feed_latency' in results['checks_performed']
        assert 'clock_skew' in results['checks_performed']
        assert 'decision_to_ack' in results['checks_performed']
        assert len(results['violations']) == 0
        assert not results['should_halt']
        assert results['recommendation'] == 'CONTINUE'
        assert monitor.last_check_ms == current_ms

    def test_perform_checks_with_violations(self):
        """Test performance of checks with violations."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        current_ms = 1000000
        feed_ts_ms = 998500  # 1500ms old feed (exceeds 1000ms limit) and clock skew exceeds 250ms
        decision_ms = 999000
        broker_ack_ms = 999400  # 400ms latency (exceeds 250ms limit)

        results = monitor.perform_checks(feed_ts_ms, current_ms, decision_ms, broker_ack_ms)

        # Checks that pass get added to checks_performed, failures don't
        assert len(results['checks_performed']) == 0  # All checks should fail
        assert len(results['violations']) == 3  # All three checks should fail
        # Since 3 violations occur in one call, it will hit halt threshold
        assert results['should_halt'] is True  # Hits halt threshold
        assert results['recommendation'] == 'HALT'

    def test_perform_checks_partial_parameters(self):
        """Test performance of checks with partial parameters."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        current_ms = 1000000
        feed_ts_ms = 999800  # 200ms old feed (within both feed latency and clock skew limits)

        # Only provide feed and current timestamps
        results = monitor.perform_checks(feed_ts_ms, current_ms)

        assert 'feed_latency' in results['checks_performed']
        assert 'clock_skew' in results['checks_performed']
        assert 'decision_to_ack' not in results['checks_performed']  # Should be skipped
        assert len(results['violations']) == 0
        assert results['recommendation'] == 'CONTINUE'

    def test_perform_checks_halt_recommendation(self):
        """Test halt recommendation when violations exceed threshold."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        # Pre-load guard with violations
        guard.violations = 3

        current_ms = 1000000
        feed_ts_ms = 999500

        results = monitor.perform_checks(feed_ts_ms, current_ms)

        assert results['should_halt'] is True
        assert results['recommendation'] == 'HALT'

    def test_perform_checks_only_decision_provided(self):
        """Test when only decision timestamp is provided (missing broker_ack)."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        current_ms = 1000000
        feed_ts_ms = 999800  # 200ms old feed (within limits)
        decision_ms = 999600

        results = monitor.perform_checks(feed_ts_ms, current_ms, decision_ms=decision_ms)

        assert 'feed_latency' in results['checks_performed']
        assert 'clock_skew' in results['checks_performed']
        assert 'decision_to_ack' not in results['checks_performed']  # Should be skipped

    def test_perform_checks_only_broker_ack_provided(self):
        """Test when only broker_ack timestamp is provided (missing decision)."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        current_ms = 1000000
        feed_ts_ms = 999800  # 200ms old feed (within limits)
        broker_ack_ms = 999750

        results = monitor.perform_checks(feed_ts_ms, current_ms, broker_ack_ms=broker_ack_ms)

        assert 'feed_latency' in results['checks_performed']
        assert 'clock_skew' in results['checks_performed']
        assert 'decision_to_ack' not in results['checks_performed']  # Should be skipped

    def test_monitor_check_interval_customization(self):
        """Test customization of monitor check interval."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        # Customize check interval
        monitor.check_interval_ms = 2000

        current_ms = 10000
        monitor.last_check_ms = current_ms

        # Check within custom interval should not be needed
        assert not monitor.should_check(current_ms + 1500)

        # Check after custom interval should be needed
        assert monitor.should_check(current_ms + 2000)


class TestClockGuardIntegration:
    """Integration tests for complete clock guard workflow."""

    def test_realistic_trading_scenario(self):
        """Test realistic trading scenario with various latencies."""
        config = ClockGuardConfig(
            max_clock_skew_ms=100,
            max_decision_to_ack_ms=200,
            max_feed_latency_ms=500
        )
        guard = ClockGuard(config)
        monitor = ClockGuardMonitor(guard)

        base_time = int(time.time() * 1000)

        # Simulate normal operation
        feed_ts = base_time - 50  # 50ms old feed
        current_time = base_time
        decision_time = base_time - 30
        broker_ack_time = base_time - 10  # 20ms decision-to-ack latency

        results = monitor.perform_checks(feed_ts, current_time, decision_time, broker_ack_time)

        assert results['recommendation'] == 'CONTINUE'
        assert len(results['violations']) == 0

    def test_degraded_performance_scenario(self):
        """Test scenario with degraded but acceptable performance."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        base_time = 1000000

        # Simulate degraded performance within limits
        feed_ts = base_time - 200  # 200ms old feed (within both 1000ms feed latency and 250ms clock skew limits)
        current_time = base_time
        decision_time = base_time - 300
        broker_ack_time = base_time - 100  # 200ms decision-to-ack (within 250ms limit)

        results = monitor.perform_checks(feed_ts, current_time, decision_time, broker_ack_time)

        assert results['recommendation'] == 'CONTINUE'
        assert len(results['violations']) == 0

    def test_violation_recovery_scenario(self):
        """Test scenario where violations occur but system recovers."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        base_time = 1000000

        # First check with violations
        feed_ts = base_time - 1500  # Exceeds feed latency limit
        current_time = base_time

        results1 = monitor.perform_checks(feed_ts, current_time)
        assert results1['recommendation'] == 'THROTTLE'
        assert len(results1['violations']) > 0

        # Second check - system recovered
        base_time += 10000
        feed_ts = base_time - 200  # Normal feed latency
        current_time = base_time

        results2 = monitor.perform_checks(feed_ts, current_time)
        assert results2['recommendation'] == 'CONTINUE'
        assert len(results2['violations']) == 0

    def test_progressive_degradation_to_halt(self):
        """Test progressive degradation leading to halt."""
        guard = ClockGuard()
        monitor = ClockGuardMonitor(guard)

        base_time = 1000000

        # Two successive violations, then halt on third
        for i in range(3):
            feed_ts = base_time - 1500  # Always exceeds feed latency limit (and clock skew)
            current_time = base_time + i * 1000

            results = monitor.perform_checks(feed_ts, current_time)

            # After first violation, recommend throttle. After 3rd violation (when violations = 3), recommend halt
            if guard.violations < 3:
                assert results['recommendation'] == 'THROTTLE'
                assert not results['should_halt']
            else:
                assert results['recommendation'] == 'HALT'
                assert results['should_halt']

    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        guard = ClockGuard()

        # Test with extreme timestamp values
        very_large_time = 9999999999999
        very_small_time = -9999999999999

        # Should handle extreme values without crashing
        guard.check_clock_skew(very_small_time, very_small_time + 100)
        guard.check_latency(very_large_time, very_large_time + 100)
        guard.check_feed_latency(very_small_time, very_small_time + 100)

        # Test status reporting with extreme violation count
        guard.violations = 999
        status = guard.get_status()
        assert status['violations'] == 999
        assert status['should_halt'] is True
