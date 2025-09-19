"""Validation-to-Bot State Adapter for Trading System Control."""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import threading


class TradingState(Enum):
    """Trading system states."""
    HEALTHY = 'HEALTHY'
    THROTTLE = 'THROTTLE'
    HALT = 'HALT'


@dataclass
class StateChange:
    """Record of state change."""
    timestamp: datetime
    from_state: TradingState
    to_state: TradingState
    reason: str
    source: str


class ValidationStateAdapter:
    """
    Bridge that validation/SLO/drift modules call; bot listens to get_state().
    Thread-safe state management for trading system control.
    """
    
    def __init__(self):
        self._state = TradingState.HEALTHY
        self._reason = 'System initialized'
        self._source = 'startup'
        self._lock = threading.RLock()
        self._state_history: list[StateChange] = []
        self._max_history = 100

    def set_state(self, state: TradingState, reason: str, source: str = 'validation') -> None:
        """
        Set trading system state.
        
        Args:
            state: New trading state
            reason: Reason for state change
            source: Source of the state change
        """
        with self._lock:
            if state != self._state:
                # Record state change only if max_history > 0
                if self._max_history > 0:
                    change = StateChange(
                        timestamp=datetime.now(),
                        from_state=self._state,
                        to_state=state,
                        reason=reason,
                        source=source
                    )
                    
                    self._state_history.append(change)
                    
                    # Keep only recent history
                    if len(self._state_history) > self._max_history:
                        self._state_history = self._state_history[-self._max_history:]
                
                # Update state
                self._state = state
                self._reason = reason
                self._source = source

    def get_state(self) -> TradingState:
        """Get current trading state."""
        with self._lock:
            return self._state

    def get_reason(self) -> str:
        """Get reason for current state."""
        with self._lock:
            return self._reason

    def get_source(self) -> str:
        """Get source of current state."""
        with self._lock:
            return self._source

    def get_state_history(self, limit: int = 10) -> list[StateChange]:
        """Get recent state change history."""
        with self._lock:
            if limit <= 0:
                return []
            return self._state_history[-limit:]

    def is_healthy(self) -> bool:
        """Check if system is in healthy state."""
        return self.get_state() == TradingState.HEALTHY

    def is_throttled(self) -> bool:
        """Check if system is throttled."""
        return self.get_state() == TradingState.THROTTLE

    def is_halted(self) -> bool:
        """Check if system is halted."""
        return self.get_state() == TradingState.HALT

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return self.get_state() in [TradingState.HEALTHY, TradingState.THROTTLE]

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        with self._lock:
            return {
                'state': self._state.value,
                'reason': self._reason,
                'source': self._source,
                'can_trade': self.can_trade(),
                'is_healthy': self.is_healthy(),
                'is_throttled': self.is_throttled(),
                'is_halted': self.is_halted(),
                'recent_changes': [
                    {
                        'timestamp': change.timestamp.isoformat(),
                        'from_state': change.from_state.value,
                        'to_state': change.to_state.value,
                        'reason': change.reason,
                        'source': change.source
                    }
                    for change in self._state_history[-5:]
                ]
            }

    def reset_to_healthy(self, reason: str = 'Manual reset') -> None:
        """Reset system to healthy state."""
        self.set_state(TradingState.HEALTHY, reason, 'manual')

    def throttle(self, reason: str, source: str = 'validation') -> None:
        """Throttle trading system."""
        self.set_state(TradingState.THROTTLE, reason, source)

    def halt(self, reason: str, source: str = 'validation') -> None:
        """Halt trading system."""
        self.set_state(TradingState.HALT, reason, source)


class StateAdapterMonitor:
    """Monitors state adapter and provides integration hooks."""
    
    def __init__(self, adapter: ValidationStateAdapter):
        self.adapter = adapter
        self.callbacks: list[callable] = []

    def add_callback(self, callback: callable) -> None:
        """Add callback for state changes."""
        self.callbacks.append(callback)

    def notify_state_change(self, old_state: TradingState, new_state: TradingState, 
                           reason: str, source: str) -> None:
        """Notify callbacks of state change."""
        for callback in self.callbacks:
            try:
                callback(old_state, new_state, reason, source)
            except Exception as e:
                print(f"State change callback error: {e}")

    def check_and_respond(self, validation_results: Dict[str, Any]) -> None:
        """
        Check validation results and respond with state changes.
        
        Args:
            validation_results: Results from validation modules
        """
        # Check for critical failures that require halting first
        if validation_results.get('reconciliation_failed', False):
            self.adapter.halt(
                "Broker reconciliation failed", 
                'reconciliation'
            )
            return  # Halt takes precedence
        
        if validation_results.get('clock_skew_exceeded', False):
            self.adapter.halt(
                "Clock skew exceeds threshold", 
                'clock_guard'
            )
            return  # Halt takes precedence
        
        # Check for throttling conditions
        if validation_results.get('drift_detected', False):
            self.adapter.throttle(
                "Drift detected in live performance", 
                'drift_monitor'
            )
        
        if validation_results.get('slo_breach', False):
            self.adapter.throttle(
                "SLO breach detected", 
                'slo_monitor'
            )


# Example usage and testing
if __name__ == "__main__":
    def test_state_adapter():
        """Test the state adapter."""
        print("=== State Adapter Test ===")
        
        adapter = ValidationStateAdapter()
        
        # Test initial state
        if not adapter.is_healthy():
            raise ValueError("Expected initial state to be healthy")
        if not adapter.can_trade():
            raise ValueError("Expected initial state to allow trading")
        print("✓ Initial state is healthy")
        
        # Test throttling
        adapter.throttle("High volatility detected", "volatility_monitor")
        if not adapter.is_throttled():
            raise ValueError("Expected throttled state")
        if not adapter.can_trade():
            raise ValueError("Expected throttled state to allow trading")
        print("✓ System throttled correctly")
        
        # Test halting
        adapter.halt("Critical error detected", "error_handler")
        if not adapter.is_halted():
            raise ValueError("Expected halted state")
        if adapter.can_trade():
            raise ValueError("Expected halted state to prevent trading")
        print("✓ System halted correctly")
        
        # Test reset
        adapter.reset_to_healthy("Manual intervention")
        if not adapter.is_healthy():
            raise ValueError("Expected reset state to be healthy")
        if not adapter.can_trade():
            raise ValueError("Expected reset state to allow trading")
        print("✓ System reset to healthy")
        
        # Test status
        status = adapter.get_status()
        print(f"Status: {status}")
        
        # Test history
        history = adapter.get_state_history()
        print(f"State history: {len(history)} changes")
    
    test_state_adapter()
