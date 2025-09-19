"""Clock-Skew and Latency Guards for Trading Systems."""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClockGuardConfig:
    """Configuration for clock guard thresholds."""
    max_clock_skew_ms: int = 250
    max_decision_to_ack_ms: int = 250
    max_feed_latency_ms: int = 1000


class ClockGuard:
    """Guards against clock skew and excessive latency in trading systems."""
    
    def __init__(self, cfg: ClockGuardConfig | None = None):
        self.cfg = cfg or ClockGuardConfig()
        self.violations = 0
        self.max_violations = 3  # Halt after 3 violations

    def check_clock_skew(self, feed_ts_ms: int, local_recv_ms: int) -> None:
        """
        Check for clock skew between feed and local system.
        
        Args:
            feed_ts_ms: Feed timestamp in milliseconds
            local_recv_ms: Local receive timestamp in milliseconds
            
        Raises:
            RuntimeError: If clock skew exceeds threshold
        """
        skew = abs(local_recv_ms - feed_ts_ms)
        if skew > self.cfg.max_clock_skew_ms:
            self.violations += 1
            raise RuntimeError(
                f"Clock skew {skew} ms exceeds {self.cfg.max_clock_skew_ms} ms "
                f"(violation {self.violations}/{self.max_violations})"
            )

    def check_latency(self, decision_ms: int, broker_ack_ms: int) -> None:
        """
        Check decision-to-acknowledgment latency.
        
        Args:
            decision_ms: Decision timestamp in milliseconds
            broker_ack_ms: Broker acknowledgment timestamp in milliseconds
            
        Raises:
            RuntimeError: If latency exceeds threshold
        """
        lat = broker_ack_ms - decision_ms
        if lat > self.cfg.max_decision_to_ack_ms:
            self.violations += 1
            raise RuntimeError(
                f"Decision→Ack {lat} ms exceeds {self.cfg.max_decision_to_ack_ms} ms "
                f"(violation {self.violations}/{self.max_violations})"
            )

    def check_feed_latency(self, feed_ts_ms: int, current_ms: int) -> None:
        """
        Check feed latency (time since feed timestamp).
        
        Args:
            feed_ts_ms: Feed timestamp in milliseconds
            current_ms: Current timestamp in milliseconds
            
        Raises:
            RuntimeError: If feed latency exceeds threshold
        """
        latency = current_ms - feed_ts_ms
        if latency > self.cfg.max_feed_latency_ms:
            self.violations += 1
            raise RuntimeError(
                f"Feed latency {latency} ms exceeds {self.cfg.max_feed_latency_ms} ms "
                f"(violation {self.violations}/{self.max_violations})"
            )

    def should_halt(self) -> bool:
        """Check if system should halt due to violations."""
        return self.violations >= self.max_violations

    def reset_violations(self) -> None:
        """Reset violation counter."""
        self.violations = 0

    def get_status(self) -> dict:
        """Get current guard status."""
        return {
            'violations': self.violations,
            'max_violations': self.max_violations,
            'should_halt': self.should_halt(),
            'config': {
                'max_clock_skew_ms': self.cfg.max_clock_skew_ms,
                'max_decision_to_ack_ms': self.cfg.max_decision_to_ack_ms,
                'max_feed_latency_ms': self.cfg.max_feed_latency_ms
            }
        }


class ClockGuardMonitor:
    """Monitors clock guard status and integrates with trading state."""
    
    def __init__(self, clock_guard: ClockGuard):
        self.clock_guard = clock_guard
        self.last_check_ms = 0
        self.check_interval_ms = 1000  # Check every second

    def should_check(self, current_ms: int) -> bool:
        """Check if it's time for a clock guard check."""
        return current_ms - self.last_check_ms >= self.check_interval_ms

    def perform_checks(self, feed_ts_ms: int, current_ms: int, 
                      decision_ms: Optional[int] = None, 
                      broker_ack_ms: Optional[int] = None) -> dict:
        """
        Perform all applicable clock guard checks.
        
        Returns:
            Dictionary with check results and recommendations
        """
        results = {
            'checks_performed': [],
            'violations': [],
            'should_halt': False,
            'recommendation': 'CONTINUE'
        }
        
        try:
            # Check feed latency
            self.clock_guard.check_feed_latency(feed_ts_ms, current_ms)
            results['checks_performed'].append('feed_latency')
        except RuntimeError as e:
            results['violations'].append(str(e))
        
        try:
            # Check clock skew
            self.clock_guard.check_clock_skew(feed_ts_ms, current_ms)
            results['checks_performed'].append('clock_skew')
        except RuntimeError as e:
            results['violations'].append(str(e))
        
        # Check decision-to-ack latency if both timestamps provided
        if decision_ms is not None and broker_ack_ms is not None:
            try:
                self.clock_guard.check_latency(decision_ms, broker_ack_ms)
                results['checks_performed'].append('decision_to_ack')
            except RuntimeError as e:
                results['violations'].append(str(e))
        
        # Determine recommendation
        if self.clock_guard.should_halt():
            results['should_halt'] = True
            results['recommendation'] = 'HALT'
        elif results['violations']:
            results['recommendation'] = 'THROTTLE'
        
        self.last_check_ms = current_ms
        return results


# Example usage and testing
if __name__ == "__main__":
    def test_clock_guard():
        """Test the clock guard implementation."""
        print("=== Clock Guard Test ===")
        
        # Test normal operation
        guard = ClockGuard()
        current_ms = int(time.time() * 1000)
        feed_ms = current_ms - 50  # 50ms old feed
        
        try:
            guard.check_feed_latency(feed_ms, current_ms)
            print("✓ Feed latency check passed")
        except RuntimeError as e:
            print(f"✗ Feed latency check failed: {e}")
        
        try:
            guard.check_clock_skew(feed_ms, current_ms)
            print("✓ Clock skew check passed")
        except RuntimeError as e:
            print(f"✗ Clock skew check failed: {e}")
        
        # Test violation
        old_feed_ms = current_ms - 2000  # 2 second old feed
        try:
            guard.check_feed_latency(old_feed_ms, current_ms)
            print("✗ Should have failed")
        except RuntimeError as e:
            print(f"✓ Correctly caught violation: {e}")
        
        print(f"Guard status: {guard.get_status()}")
    
    test_clock_guard()



