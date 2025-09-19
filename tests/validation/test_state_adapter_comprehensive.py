#!/usr/bin/env python3
"""
Comprehensive tests for ValidationStateAdapter to improve coverage.
Tests state transitions and adapter functionality.
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch
from datetime import datetime
from backend.validation.operations.state_adapter import (
    ValidationStateAdapter,
    TradingState,
    StateChange,
    StateAdapterMonitor
)


class TestValidationStateAdapterComprehensive:
    """Comprehensive tests for ValidationStateAdapter."""

    def test_initial_state(self):
        """Test initial state after creation."""
        adapter = ValidationStateAdapter()
        
        assert adapter.get_state() == TradingState.HEALTHY
        assert adapter.get_reason() == 'System initialized'

    def test_state_transitions(self):
        """Test all possible state transitions."""
        adapter = ValidationStateAdapter()
        
        # Test HEALTHY -> THROTTLE
        adapter.set_state(TradingState.THROTTLE, 'high_risk_detected')
        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_reason() == 'high_risk_detected'
        
        # Test THROTTLE -> HALT
        adapter.set_state(TradingState.HALT, 'critical_failure')
        assert adapter.get_state() == TradingState.HALT
        assert adapter.get_reason() == 'critical_failure'
        
        # Test HALT -> HEALTHY
        adapter.set_state(TradingState.HEALTHY, 'recovery_complete')
        assert adapter.get_state() == TradingState.HEALTHY
        assert adapter.get_reason() == 'recovery_complete'

    def test_all_trading_states(self):
        """Test all trading states."""
        adapter = ValidationStateAdapter()
        
        # Test HEALTHY state (already HEALTHY, so reason won't change)
        adapter.set_state(TradingState.HEALTHY, 'system_normal')
        assert adapter.get_state() == TradingState.HEALTHY
        assert adapter.get_reason() == 'System initialized'  # Reason doesn't change if state is same
        
        # Test THROTTLE state
        adapter.set_state(TradingState.THROTTLE, 'performance_degraded')
        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_reason() == 'performance_degraded'
        
        # Test HALT state
        adapter.set_state(TradingState.HALT, 'emergency_stop')
        assert adapter.get_state() == TradingState.HALT
        assert adapter.get_reason() == 'emergency_stop'

    def test_reason_updates(self):
        """Test reason updates with same state."""
        adapter = ValidationStateAdapter()
        
        # Set initial state (already HEALTHY, so reason won't change)
        adapter.set_state(TradingState.HEALTHY, 'initial_reason')
        assert adapter.get_reason() == 'System initialized'  # Reason doesn't change if state is same
        
        # Update reason for same state (still won't change)
        adapter.set_state(TradingState.HEALTHY, 'updated_reason')
        assert adapter.get_state() == TradingState.HEALTHY
        assert adapter.get_reason() == 'System initialized'  # Still won't change

    def test_multiple_state_changes(self):
        """Test multiple state changes."""
        adapter = ValidationStateAdapter()
        
        # Sequence of state changes
        states_and_reasons = [
            (TradingState.HEALTHY, 'startup'),
            (TradingState.THROTTLE, 'high_volatility'),
            (TradingState.HALT, 'market_crash'),
            (TradingState.THROTTLE, 'recovery_started'),
            (TradingState.HEALTHY, 'fully_recovered')
        ]
        
        for i, (state, reason) in enumerate(states_and_reasons):
            adapter.set_state(state, reason)
            assert adapter.get_state() == state
            if i == 0:  # First call, state is already HEALTHY
                assert adapter.get_reason() == 'System initialized'
            else:
                assert adapter.get_reason() == reason

    def test_empty_reason(self):
        """Test with empty reason string."""
        adapter = ValidationStateAdapter()
        
        adapter.set_state(TradingState.THROTTLE, '')
        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_reason() == ''

    def test_long_reason(self):
        """Test with long reason string."""
        adapter = ValidationStateAdapter()
        
        long_reason = 'This is a very long reason string that contains multiple words and describes a complex situation that led to the state change'
        adapter.set_state(TradingState.HALT, long_reason)
        assert adapter.get_state() == TradingState.HALT
        assert adapter.get_reason() == long_reason

    def test_special_characters_in_reason(self):
        """Test with special characters in reason."""
        adapter = ValidationStateAdapter()
        
        special_reason = 'Reason with special chars: !@#$%^&*()_+-=[]{}|;:,.<>?'
        adapter.set_state(TradingState.THROTTLE, special_reason)
        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_reason() == special_reason

    def test_unicode_reason(self):
        """Test with unicode characters in reason."""
        adapter = ValidationStateAdapter()
        
        unicode_reason = 'Reason with unicode: αβγδε 中文 🚀'
        adapter.set_state(TradingState.HEALTHY, unicode_reason)
        assert adapter.get_state() == TradingState.HEALTHY
        assert adapter.get_reason() == 'System initialized'  # Reason doesn't change if state is same

    def test_none_reason(self):
        """Test with None reason."""
        adapter = ValidationStateAdapter()
        
        adapter.set_state(TradingState.HALT, None)
        assert adapter.get_state() == TradingState.HALT
        assert adapter.get_reason() is None

    def test_state_enum_values(self):
        """Test TradingState enum values."""
        assert TradingState.HEALTHY.value == 'HEALTHY'
        assert TradingState.THROTTLE.value == 'THROTTLE'
        assert TradingState.HALT.value == 'HALT'

    def test_state_enum_comparison(self):
        """Test TradingState enum comparison."""
        adapter = ValidationStateAdapter()
        
        adapter.set_state(TradingState.HEALTHY, 'test')
        assert adapter.get_state() == TradingState.HEALTHY
        assert adapter.get_state() is TradingState.HEALTHY
        assert adapter.get_state() is not TradingState.THROTTLE
        assert adapter.get_state() is not TradingState.HALT

    def test_state_persistence(self):
        """Test that state persists between method calls."""
        adapter = ValidationStateAdapter()
        
        # Set state
        adapter.set_state(TradingState.THROTTLE, 'persistent_state')
        
        # Multiple calls should return same state
        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_state() == TradingState.THROTTLE
        
        # Reason should also persist
        assert adapter.get_reason() == 'persistent_state'
        assert adapter.get_reason() == 'persistent_state'

    def test_rapid_state_changes(self):
        """Test rapid state changes."""
        adapter = ValidationStateAdapter()
        
        # Rapid state changes
        for i in range(100):
            state = TradingState.HEALTHY if i % 3 == 0 else TradingState.THROTTLE if i % 3 == 1 else TradingState.HALT
            reason = f'rapid_change_{i}'
            adapter.set_state(state, reason)
            assert adapter.get_state() == state
            if i == 0:  # First call, state is already HEALTHY
                assert adapter.get_reason() == 'System initialized'
            else:
                assert adapter.get_reason() == reason

    def test_state_transition_scenarios(self):
        """Test realistic state transition scenarios."""
        adapter = ValidationStateAdapter()
        
        # Scenario 1: Normal operation
        adapter.set_state(TradingState.HEALTHY, 'system_startup')
        assert adapter.get_state() == TradingState.HEALTHY
        
        # Scenario 2: Performance degradation
        adapter.set_state(TradingState.THROTTLE, 'high_latency_detected')
        assert adapter.get_state() == TradingState.THROTTLE
        
        # Scenario 3: Critical failure
        adapter.set_state(TradingState.HALT, 'database_connection_lost')
        assert adapter.get_state() == TradingState.HALT
        
        # Scenario 4: Recovery
        adapter.set_state(TradingState.THROTTLE, 'database_reconnected')
        assert adapter.get_state() == TradingState.THROTTLE
        
        # Scenario 5: Full recovery
        adapter.set_state(TradingState.HEALTHY, 'all_systems_normal')
        assert adapter.get_state() == TradingState.HEALTHY

    def test_edge_case_initialization(self):
        """Test edge cases during initialization."""
        # Test multiple instances
        adapter1 = ValidationStateAdapter()
        adapter2 = ValidationStateAdapter()
        
        # Each should have independent state
        adapter1.set_state(TradingState.THROTTLE, 'adapter1_reason')
        adapter2.set_state(TradingState.HALT, 'adapter2_reason')
        
        assert adapter1.get_state() == TradingState.THROTTLE
        assert adapter1.get_reason() == 'adapter1_reason'
        assert adapter2.get_state() == TradingState.HALT
        assert adapter2.get_reason() == 'adapter2_reason'

    def test_state_method_consistency(self):
        """Test consistency between get_state and reason methods."""
        adapter = ValidationStateAdapter()
        
        # Test that both methods work consistently
        adapter.set_state(TradingState.THROTTLE, 'consistent_test')
        
        # Multiple calls should be consistent
        state1 = adapter.get_state()
        reason1 = adapter.get_reason()
        state2 = adapter.get_state()
        reason2 = adapter.get_reason()
        
        assert state1 == state2
        assert reason1 == reason2
        assert state1 == TradingState.THROTTLE
        assert reason1 == 'consistent_test'

    def test_trading_state_enum_functionality(self):
        """Test TradingState enum functionality."""
        # Test enum iteration
        states = list(TradingState)
        assert len(states) == 3
        assert TradingState.HEALTHY in states
        assert TradingState.THROTTLE in states
        assert TradingState.HALT in states
        
        # Test enum string conversion
        assert str(TradingState.HEALTHY) == 'TradingState.HEALTHY'
        assert str(TradingState.THROTTLE) == 'TradingState.THROTTLE'
        assert str(TradingState.HALT) == 'TradingState.HALT'

    def test_adapter_integration_scenario(self):
        """Test realistic integration scenario."""
        adapter = ValidationStateAdapter()
        
        # Simulate a trading day scenario
        # Morning: System starts healthy
        adapter.set_state(TradingState.HEALTHY, 'market_open')
        
        # Mid-morning: High volatility detected
        adapter.set_state(TradingState.THROTTLE, 'volatility_spike')
        
        # Lunch: Market stabilizes
        adapter.set_state(TradingState.HEALTHY, 'market_stabilized')
        
        # Afternoon: System issue
        adapter.set_state(TradingState.THROTTLE, 'connection_timeout')
        
        # Late afternoon: Critical failure
        adapter.set_state(TradingState.HALT, 'data_feed_lost')
        
        # End of day: Recovery
        adapter.set_state(TradingState.THROTTLE, 'backup_feed_active')
        adapter.set_state(TradingState.HEALTHY, 'end_of_day_recovery')
        
        # Final state should be healthy
        assert adapter.get_state() == TradingState.HEALTHY
        assert adapter.get_reason() == 'end_of_day_recovery'


class TestStateChangeDataClass:
    """Test StateChange dataclass functionality."""

    def test_state_change_creation(self):
        """Test StateChange dataclass creation."""
        timestamp = datetime.now()
        change = StateChange(
            timestamp=timestamp,
            from_state=TradingState.HEALTHY,
            to_state=TradingState.THROTTLE,
            reason="Performance degradation",
            source="validation"
        )

        assert change.timestamp == timestamp
        assert change.from_state == TradingState.HEALTHY
        assert change.to_state == TradingState.THROTTLE
        assert change.reason == "Performance degradation"
        assert change.source == "validation"

    def test_state_change_equality(self):
        """Test StateChange equality comparison."""
        timestamp = datetime.now()
        change1 = StateChange(
            timestamp=timestamp,
            from_state=TradingState.HEALTHY,
            to_state=TradingState.THROTTLE,
            reason="test",
            source="test"
        )
        change2 = StateChange(
            timestamp=timestamp,
            from_state=TradingState.HEALTHY,
            to_state=TradingState.THROTTLE,
            reason="test",
            source="test"
        )

        assert change1 == change2

    def test_state_change_repr(self):
        """Test StateChange string representation."""
        timestamp = datetime.now()
        change = StateChange(
            timestamp=timestamp,
            from_state=TradingState.HEALTHY,
            to_state=TradingState.THROTTLE,
            reason="test",
            source="test"
        )

        repr_str = repr(change)
        assert "StateChange" in repr_str
        assert "HEALTHY" in repr_str
        assert "THROTTLE" in repr_str


class TestStateHistoryFunctionality:
    """Test state history tracking functionality."""

    def test_state_history_empty_initially(self):
        """Test that state history is empty initially."""
        adapter = ValidationStateAdapter()

        history = adapter.get_state_history()
        assert len(history) == 0

    def test_state_history_records_changes(self):
        """Test that state history records state changes."""
        adapter = ValidationStateAdapter()

        # Change state and check history
        adapter.set_state(TradingState.THROTTLE, "first change", "test")
        history = adapter.get_state_history()

        assert len(history) == 1
        assert history[0].from_state == TradingState.HEALTHY
        assert history[0].to_state == TradingState.THROTTLE
        assert history[0].reason == "first change"
        assert history[0].source == "test"

    def test_state_history_multiple_changes(self):
        """Test state history with multiple changes."""
        adapter = ValidationStateAdapter()

        # Multiple state changes
        adapter.set_state(TradingState.THROTTLE, "change 1", "source1")
        adapter.set_state(TradingState.HALT, "change 2", "source2")
        adapter.set_state(TradingState.HEALTHY, "change 3", "source3")

        history = adapter.get_state_history()
        assert len(history) == 3

        # Check chronological order
        assert history[0].to_state == TradingState.THROTTLE
        assert history[1].to_state == TradingState.HALT
        assert history[2].to_state == TradingState.HEALTHY

    def test_state_history_limit_parameter(self):
        """Test state history limit parameter."""
        adapter = ValidationStateAdapter()

        # Add 5 state changes
        for i in range(5):
            state = TradingState.THROTTLE if i % 2 == 0 else TradingState.HALT
            adapter.set_state(state, f"change {i}", f"source{i}")

        # Test different limits
        assert len(adapter.get_state_history(limit=2)) == 2
        assert len(adapter.get_state_history(limit=3)) == 3
        assert len(adapter.get_state_history(limit=10)) == 5  # All available

        # Test that it returns most recent changes
        recent_history = adapter.get_state_history(limit=2)
        assert recent_history[0].reason == "change 3"
        assert recent_history[1].reason == "change 4"

    def test_state_history_max_limit_enforcement(self):
        """Test that state history enforces max limit."""
        adapter = ValidationStateAdapter()

        # Add more than max_history changes (default 100)
        for i in range(105):
            state = TradingState.THROTTLE if i % 2 == 0 else TradingState.HALT
            adapter.set_state(state, f"change {i}", f"source{i}")

        # Should only keep the most recent 100
        history = adapter.get_state_history(limit=200)  # Request more than available
        assert len(history) == 100

        # Should contain the most recent changes
        assert history[-1].reason == "change 104"
        assert history[0].reason == "change 5"  # First 5 should be dropped

    def test_state_history_no_duplicate_when_same_state(self):
        """Test that setting same state doesn't create history entry."""
        adapter = ValidationStateAdapter()

        # Change to different state (should create history)
        adapter.set_state(TradingState.THROTTLE, "first change", "test")
        assert len(adapter.get_state_history()) == 1

        # Set same state again (should not create history)
        adapter.set_state(TradingState.THROTTLE, "same state", "test")
        assert len(adapter.get_state_history()) == 1

        # Change to different state again (should create history)
        adapter.set_state(TradingState.HALT, "second change", "test")
        assert len(adapter.get_state_history()) == 2

    def test_state_history_timestamps(self):
        """Test that state history timestamps are realistic."""
        adapter = ValidationStateAdapter()

        start_time = datetime.now()
        adapter.set_state(TradingState.THROTTLE, "test change", "test")
        end_time = datetime.now()

        history = adapter.get_state_history()
        assert len(history) == 1

        # Timestamp should be between start and end
        assert start_time <= history[0].timestamp <= end_time


class TestConvenienceMethods:
    """Test convenience methods for state checking and setting."""

    def test_is_healthy(self):
        """Test is_healthy() method."""
        adapter = ValidationStateAdapter()

        # Initially healthy
        assert adapter.is_healthy() is True

        # Change to throttle
        adapter.set_state(TradingState.THROTTLE, "test", "test")
        assert adapter.is_healthy() is False

        # Change to halt
        adapter.set_state(TradingState.HALT, "test", "test")
        assert adapter.is_healthy() is False

        # Back to healthy
        adapter.set_state(TradingState.HEALTHY, "test", "test")
        assert adapter.is_healthy() is True

    def test_is_throttled(self):
        """Test is_throttled() method."""
        adapter = ValidationStateAdapter()

        # Initially not throttled
        assert adapter.is_throttled() is False

        # Change to throttle
        adapter.set_state(TradingState.THROTTLE, "test", "test")
        assert adapter.is_throttled() is True

        # Change to halt
        adapter.set_state(TradingState.HALT, "test", "test")
        assert adapter.is_throttled() is False

        # Back to healthy
        adapter.set_state(TradingState.HEALTHY, "test", "test")
        assert adapter.is_throttled() is False

    def test_is_halted(self):
        """Test is_halted() method."""
        adapter = ValidationStateAdapter()

        # Initially not halted
        assert adapter.is_halted() is False

        # Change to throttle
        adapter.set_state(TradingState.THROTTLE, "test", "test")
        assert adapter.is_halted() is False

        # Change to halt
        adapter.set_state(TradingState.HALT, "test", "test")
        assert adapter.is_halted() is True

        # Back to healthy
        adapter.set_state(TradingState.HEALTHY, "test", "test")
        assert adapter.is_halted() is False

    def test_can_trade(self):
        """Test can_trade() method."""
        adapter = ValidationStateAdapter()

        # Initially can trade (healthy)
        assert adapter.can_trade() is True

        # Throttled - can still trade
        adapter.set_state(TradingState.THROTTLE, "test", "test")
        assert adapter.can_trade() is True

        # Halted - cannot trade
        adapter.set_state(TradingState.HALT, "test", "test")
        assert adapter.can_trade() is False

        # Back to healthy - can trade
        adapter.set_state(TradingState.HEALTHY, "test", "test")
        assert adapter.can_trade() is True

    def test_reset_to_healthy(self):
        """Test reset_to_healthy() convenience method."""
        adapter = ValidationStateAdapter()

        # Change to non-healthy state
        adapter.set_state(TradingState.HALT, "critical error", "error_handler")

        # Reset to healthy
        adapter.reset_to_healthy("Manual intervention")

        assert adapter.get_state() == TradingState.HEALTHY
        assert adapter.get_reason() == "Manual intervention"
        assert adapter.get_source() == "manual"

        # Test with custom reason
        adapter.set_state(TradingState.THROTTLE, "test", "test")
        adapter.reset_to_healthy("System recovered")

        assert adapter.get_reason() == "System recovered"

    def test_throttle_convenience_method(self):
        """Test throttle() convenience method."""
        adapter = ValidationStateAdapter()

        # Test with default source
        adapter.throttle("High volatility detected")

        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_reason() == "High volatility detected"
        assert adapter.get_source() == "validation"

        # Reset to healthy first, then test with custom source
        adapter.reset_to_healthy("System recovered")
        adapter.throttle("Performance degradation", "performance_monitor")

        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_reason() == "Performance degradation"
        assert adapter.get_source() == "performance_monitor"

    def test_halt_convenience_method(self):
        """Test halt() convenience method."""
        adapter = ValidationStateAdapter()

        # Test with default source
        adapter.halt("Critical system failure")

        assert adapter.get_state() == TradingState.HALT
        assert adapter.get_reason() == "Critical system failure"
        assert adapter.get_source() == "validation"

        # Reset to healthy first, then test with custom source
        adapter.reset_to_healthy("System recovered")
        adapter.halt("Database connection lost", "database_monitor")

        assert adapter.get_state() == TradingState.HALT
        assert adapter.get_reason() == "Database connection lost"
        assert adapter.get_source() == "database_monitor"


class TestGetStatusMethod:
    """Test comprehensive status reporting."""

    def test_get_status_structure(self):
        """Test get_status() returns correct structure."""
        adapter = ValidationStateAdapter()

        status = adapter.get_status()

        # Check required keys
        required_keys = [
            'state', 'reason', 'source', 'can_trade',
            'is_healthy', 'is_throttled', 'is_halted',
            'recent_changes'
        ]
        for key in required_keys:
            assert key in status

    def test_get_status_healthy_state(self):
        """Test get_status() for healthy state."""
        adapter = ValidationStateAdapter()

        status = adapter.get_status()

        assert status['state'] == 'HEALTHY'
        assert status['reason'] == 'System initialized'
        assert status['source'] == 'startup'
        assert status['can_trade'] is True
        assert status['is_healthy'] is True
        assert status['is_throttled'] is False
        assert status['is_halted'] is False
        assert len(status['recent_changes']) == 0

    def test_get_status_throttled_state(self):
        """Test get_status() for throttled state."""
        adapter = ValidationStateAdapter()
        adapter.throttle("Performance issues", "monitor")

        status = adapter.get_status()

        assert status['state'] == 'THROTTLE'
        assert status['reason'] == 'Performance issues'
        assert status['source'] == 'monitor'
        assert status['can_trade'] is True
        assert status['is_healthy'] is False
        assert status['is_throttled'] is True
        assert status['is_halted'] is False

    def test_get_status_halted_state(self):
        """Test get_status() for halted state."""
        adapter = ValidationStateAdapter()
        adapter.halt("Critical failure", "emergency")

        status = adapter.get_status()

        assert status['state'] == 'HALT'
        assert status['reason'] == 'Critical failure'
        assert status['source'] == 'emergency'
        assert status['can_trade'] is False
        assert status['is_healthy'] is False
        assert status['is_throttled'] is False
        assert status['is_halted'] is True

    def test_get_status_recent_changes(self):
        """Test get_status() recent changes reporting."""
        adapter = ValidationStateAdapter()

        # Add some state changes
        adapter.throttle("Issue 1", "source1")
        adapter.halt("Issue 2", "source2")
        adapter.reset_to_healthy("Recovery")

        status = adapter.get_status()
        recent_changes = status['recent_changes']

        assert len(recent_changes) == 3

        # Check structure of recent changes
        for change in recent_changes:
            assert 'timestamp' in change
            assert 'from_state' in change
            assert 'to_state' in change
            assert 'reason' in change
            assert 'source' in change

        # Check that timestamps are ISO format strings
        for change in recent_changes:
            # Should be parseable as ISO format
            datetime.fromisoformat(change['timestamp'])

    def test_get_status_limits_recent_changes(self):
        """Test that get_status() limits recent changes to 5."""
        adapter = ValidationStateAdapter()

        # Add 10 state changes
        for i in range(10):
            state = TradingState.THROTTLE if i % 2 == 0 else TradingState.HALT
            adapter.set_state(state, f"change {i}", f"source{i}")

        status = adapter.get_status()
        recent_changes = status['recent_changes']

        # Should only include 5 most recent
        assert len(recent_changes) == 5

        # Should be the most recent changes (5-9)
        assert recent_changes[0]['reason'] == "change 5"
        assert recent_changes[4]['reason'] == "change 9"


class TestSourceParameterHandling:
    """Test source parameter handling in state changes."""

    def test_get_source_method(self):
        """Test get_source() method."""
        adapter = ValidationStateAdapter()

        # Initially 'startup'
        assert adapter.get_source() == 'startup'

        # Change with specific source
        adapter.set_state(TradingState.THROTTLE, "test", "custom_source")
        assert adapter.get_source() == "custom_source"

    def test_source_in_history(self):
        """Test that source is recorded in history."""
        adapter = ValidationStateAdapter()

        adapter.set_state(TradingState.THROTTLE, "test", "test_source")
        history = adapter.get_state_history()

        assert len(history) == 1
        assert history[0].source == "test_source"

    def test_default_source_parameter(self):
        """Test default source parameter in convenience methods."""
        adapter = ValidationStateAdapter()

        # Test throttle default source
        adapter.throttle("test reason")
        assert adapter.get_source() == "validation"

        # Test halt default source
        adapter.halt("test reason")
        assert adapter.get_source() == "validation"

        # Test reset default source
        adapter.reset_to_healthy()
        assert adapter.get_source() == "manual"

    def test_custom_source_parameter(self):
        """Test custom source parameter in convenience methods."""
        adapter = ValidationStateAdapter()

        # Test throttle with custom source
        adapter.throttle("test reason", "custom_throttle")
        assert adapter.get_source() == "custom_throttle"

        # Test halt with custom source
        adapter.halt("test reason", "custom_halt")
        assert adapter.get_source() == "custom_halt"


class TestThreadSafety:
    """Test thread safety of ValidationStateAdapter."""

    def test_concurrent_state_changes(self):
        """Test concurrent state changes from multiple threads."""
        adapter = ValidationStateAdapter()
        results = []
        errors = []

        def change_state(thread_id):
            try:
                for i in range(10):
                    state = TradingState.THROTTLE if i % 2 == 0 else TradingState.HALT
                    adapter.set_state(state, f"thread_{thread_id}_change_{i}", f"thread_{thread_id}")
                    results.append((thread_id, i, adapter.get_state()))
            except Exception as e:
                errors.append((thread_id, e))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=change_state, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that no errors occurred
        assert len(errors) == 0

        # Check that adapter is in a valid state
        final_state = adapter.get_state()
        assert final_state in [TradingState.HEALTHY, TradingState.THROTTLE, TradingState.HALT]

    def test_concurrent_reads_and_writes(self):
        """Test concurrent reads and writes."""
        adapter = ValidationStateAdapter()
        read_results = []
        write_results = []
        errors = []

        def read_state(thread_id):
            try:
                for i in range(20):
                    state = adapter.get_state()
                    reason = adapter.get_reason()
                    source = adapter.get_source()
                    read_results.append((thread_id, i, state, reason, source))
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append((f"read_{thread_id}", e))

        def write_state(thread_id):
            try:
                for i in range(10):
                    state = TradingState.THROTTLE if i % 2 == 0 else TradingState.HALT
                    adapter.set_state(state, f"write_{thread_id}_{i}", f"writer_{thread_id}")
                    write_results.append((thread_id, i, state))
                    time.sleep(0.002)  # Small delay
            except Exception as e:
                errors.append((f"write_{thread_id}", e))

        # Create reader and writer threads
        threads = []

        # Add reader threads
        for i in range(3):
            thread = threading.Thread(target=read_state, args=(i,))
            threads.append(thread)

        # Add writer threads
        for i in range(2):
            thread = threading.Thread(target=write_state, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that no errors occurred
        assert len(errors) == 0

        # Check that we got results from both readers and writers
        assert len(read_results) > 0
        assert len(write_results) > 0

    def test_history_thread_safety(self):
        """Test that state history is thread-safe."""
        adapter = ValidationStateAdapter()
        errors = []

        def access_history(thread_id):
            try:
                for i in range(10):
                    # Write some state changes
                    state = TradingState.THROTTLE if i % 2 == 0 else TradingState.HALT
                    adapter.set_state(state, f"history_test_{thread_id}_{i}", f"thread_{thread_id}")

                    # Read history
                    history = adapter.get_state_history()

                    # Verify history is valid
                    for change in history:
                        assert isinstance(change, StateChange)
                        assert change.from_state in [TradingState.HEALTHY, TradingState.THROTTLE, TradingState.HALT]
                        assert change.to_state in [TradingState.HEALTHY, TradingState.THROTTLE, TradingState.HALT]

            except Exception as e:
                errors.append((thread_id, e))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=access_history, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that no errors occurred
        assert len(errors) == 0


class TestStateAdapterMonitor:
    """Test StateAdapterMonitor functionality."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        assert monitor.adapter is adapter
        assert len(monitor.callbacks) == 0

    def test_add_callback(self):
        """Test adding callbacks to monitor."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        callback1 = Mock()
        callback2 = Mock()

        monitor.add_callback(callback1)
        assert len(monitor.callbacks) == 1

        monitor.add_callback(callback2)
        assert len(monitor.callbacks) == 2

        assert callback1 in monitor.callbacks
        assert callback2 in monitor.callbacks

    def test_notify_state_change(self):
        """Test state change notifications."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        callback1 = Mock()
        callback2 = Mock()
        monitor.add_callback(callback1)
        monitor.add_callback(callback2)

        # Notify state change
        monitor.notify_state_change(
            TradingState.HEALTHY,
            TradingState.THROTTLE,
            "test reason",
            "test source"
        )

        # Both callbacks should be called
        callback1.assert_called_once_with(
            TradingState.HEALTHY,
            TradingState.THROTTLE,
            "test reason",
            "test source"
        )
        callback2.assert_called_once_with(
            TradingState.HEALTHY,
            TradingState.THROTTLE,
            "test reason",
            "test source"
        )

    def test_notify_state_change_with_callback_error(self):
        """Test state change notification handles callback errors."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        # Create callback that raises exception
        failing_callback = Mock(side_effect=Exception("Callback error"))
        working_callback = Mock()

        monitor.add_callback(failing_callback)
        monitor.add_callback(working_callback)

        # Should not raise exception despite failing callback
        monitor.notify_state_change(
            TradingState.HEALTHY,
            TradingState.THROTTLE,
            "test reason",
            "test source"
        )

        # Working callback should still be called
        working_callback.assert_called_once()

    def test_check_and_respond_drift_detected(self):
        """Test check_and_respond with drift detected."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        validation_results = {'drift_detected': True}
        monitor.check_and_respond(validation_results)

        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_reason() == "Drift detected in live performance"
        assert adapter.get_source() == "drift_monitor"

    def test_check_and_respond_reconciliation_failed(self):
        """Test check_and_respond with reconciliation failure."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        validation_results = {'reconciliation_failed': True}
        monitor.check_and_respond(validation_results)

        assert adapter.get_state() == TradingState.HALT
        assert adapter.get_reason() == "Broker reconciliation failed"
        assert adapter.get_source() == "reconciliation"

    def test_check_and_respond_clock_skew_exceeded(self):
        """Test check_and_respond with clock skew exceeded."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        validation_results = {'clock_skew_exceeded': True}
        monitor.check_and_respond(validation_results)

        assert adapter.get_state() == TradingState.HALT
        assert adapter.get_reason() == "Clock skew exceeds threshold"
        assert adapter.get_source() == "clock_guard"

    def test_check_and_respond_slo_breach(self):
        """Test check_and_respond with SLO breach."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        validation_results = {'slo_breach': True}
        monitor.check_and_respond(validation_results)

        assert adapter.get_state() == TradingState.THROTTLE
        assert adapter.get_reason() == "SLO breach detected"
        assert adapter.get_source() == "slo_monitor"

    def test_check_and_respond_multiple_issues(self):
        """Test check_and_respond with multiple issues."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        # Multiple issues, halt should take precedence
        validation_results = {
            'drift_detected': True,
            'reconciliation_failed': True,
            'slo_breach': True
        }
        monitor.check_and_respond(validation_results)

        # Should be halted due to reconciliation failure
        assert adapter.get_state() == TradingState.HALT

    def test_check_and_respond_no_issues(self):
        """Test check_and_respond with no issues."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        validation_results = {}
        monitor.check_and_respond(validation_results)

        # Should remain healthy
        assert adapter.get_state() == TradingState.HEALTHY

    def test_check_and_respond_false_values(self):
        """Test check_and_respond with false values."""
        adapter = ValidationStateAdapter()
        monitor = StateAdapterMonitor(adapter)

        validation_results = {
            'drift_detected': False,
            'reconciliation_failed': False,
            'clock_skew_exceeded': False,
            'slo_breach': False
        }
        monitor.check_and_respond(validation_results)

        # Should remain healthy
        assert adapter.get_state() == TradingState.HEALTHY


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_large_max_history_setting(self):
        """Test with large max_history setting."""
        adapter = ValidationStateAdapter()
        adapter._max_history = 1000

        # Add many state changes
        for i in range(1500):
            state = TradingState.THROTTLE if i % 2 == 0 else TradingState.HALT
            adapter.set_state(state, f"change {i}", "test")

        # Should keep only max_history items
        history = adapter.get_state_history(limit=2000)
        assert len(history) == 1000

    def test_zero_max_history(self):
        """Test with zero max_history."""
        adapter = ValidationStateAdapter()
        adapter._max_history = 0

        adapter.set_state(TradingState.THROTTLE, "test", "test")

        # Should keep no history (but initial state change is still recorded)
        history = adapter.get_state_history()
        assert len(history) == 0

    def test_negative_history_limit(self):
        """Test get_state_history with negative limit."""
        adapter = ValidationStateAdapter()

        adapter.set_state(TradingState.THROTTLE, "test", "test")

        # Negative limit should return empty list
        history = adapter.get_state_history(limit=-1)
        assert len(history) == 0

    def test_zero_history_limit(self):
        """Test get_state_history with zero limit."""
        adapter = ValidationStateAdapter()

        adapter.set_state(TradingState.THROTTLE, "test", "test")

        # Zero limit should return empty list
        history = adapter.get_state_history(limit=0)
        assert len(history) == 0

    def test_very_long_reason_string(self):
        """Test with very long reason string."""
        adapter = ValidationStateAdapter()

        # Create very long reason (10KB)
        long_reason = "x" * 10000
        adapter.set_state(TradingState.THROTTLE, long_reason, "test")

        assert adapter.get_reason() == long_reason

    def test_binary_data_in_reason(self):
        """Test with binary data in reason."""
        adapter = ValidationStateAdapter()

        # Binary data as reason
        binary_reason = b"binary data \x00\x01\x02"
        adapter.set_state(TradingState.THROTTLE, binary_reason, "test")

        assert adapter.get_reason() == binary_reason

    def test_none_values_handling(self):
        """Test handling of None values."""
        adapter = ValidationStateAdapter()

        # None reason and source
        adapter.set_state(TradingState.THROTTLE, None, None)

        assert adapter.get_reason() is None
        assert adapter.get_source() is None

    def test_status_serialization_edge_cases(self):
        """Test status serialization with edge case values."""
        adapter = ValidationStateAdapter()

        # Set state with edge case values
        adapter.set_state(TradingState.THROTTLE, None, "")

        status = adapter.get_status()

        # Should handle None and empty string gracefully
        assert status['reason'] is None
        assert status['source'] == ""
        assert isinstance(status['state'], str)

    def test_concurrent_max_history_modification(self):
        """Test concurrent modification of max_history setting."""
        adapter = ValidationStateAdapter()
        errors = []

        def modify_history_and_state(thread_id):
            try:
                for i in range(10):
                    # Modify max_history setting
                    adapter._max_history = 50 + i

                    # Add state change
                    adapter.set_state(TradingState.THROTTLE, f"test_{thread_id}_{i}", "test")

                    # Read history
                    history = adapter.get_state_history()

                    # Verify history length is reasonable
                    assert len(history) <= adapter._max_history

            except Exception as e:
                errors.append((thread_id, e))

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=modify_history_and_state, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that no errors occurred
        assert len(errors) == 0
