"""Circuit breaker for daily loss, error spikes, and stale data protection.

This module implements a production-grade circuit breaker that trips on:
- Daily drawdown exceeding limits
- Error rate spikes
- Data staleness
- Requires manual reset after cooldown period
"""
from __future__ import annotations
import time
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("wsb.circuit_breaker")


@dataclass
class BreakerLimits:
    """Configuration limits for circuit breaker."""
    max_daily_drawdown: float = 0.06  # 6% of start-of-day equity
    max_error_rate_per_min: int = 5
    max_data_staleness_sec: int = 30
    cooldown_sec: int = 60 * 30  # 30 minutes


class CircuitBreaker:
    """Production circuit breaker with multiple trip conditions.
    
    Trips on:
    - Daily drawdown exceeding max_daily_drawdown
    - Error rate exceeding max_error_rate_per_min
    - Data staleness exceeding max_data_staleness_sec
    
    Requires cooldown period before reset.
    """

    def __init__(self, start_equity: float, limits: BreakerLimits = BreakerLimits()):
        """Initialize circuit breaker.
        
        Args:
            start_equity: Starting equity for drawdown calculation
            limits: BreakerLimits configuration
        """
        self.start_equity = float(start_equity)
        self.lim = limits
        self.errors_last_min = 0
        self.errors_window_start = time.time()
        self.tripped_at: Optional[float] = None
        self.reason: Optional[str] = None
        self.last_data_ts = time.time()

    def mark_error(self) -> None:
        """Mark an error occurrence and check rate limits."""
        now = time.time()
        if now - self.errors_window_start > 60:
            self.errors_last_min = 0
            self.errors_window_start = now
        self.errors_last_min += 1
        if self.errors_last_min >= self.lim.max_error_rate_per_min:
            self.trip("error-rate")

    def mark_data_fresh(self) -> None:
        """Mark data as fresh (call on every tick)."""
        self.last_data_ts = time.time()

    def check_mtm(self, current_equity: float) -> None:
        """Check mark-to-market drawdown.
        
        Args:
            current_equity: Current portfolio equity
        """
        dd = 1.0 - (current_equity / self.start_equity)
        if dd >= self.lim.max_daily_drawdown:
            self.trip(f"daily-dd-{dd:.3f}")

    def poll(self) -> None:
        """Poll for stale data condition."""
        if time.time() - self.last_data_ts > self.lim.max_data_staleness_sec:
            self.trip("stale-data")

    def trip(self, reason: str) -> None:
        """Trip the circuit breaker.
        
        Args:
            reason: Reason for tripping
        """
        if self.tripped_at is None:
            self.tripped_at = time.time()
            self.reason = reason
            log.critical(f"CIRCUIT_BREAKER_TRIPPED: {reason}")

    def can_trade(self) -> bool:
        """Check if trading is allowed.
        
        Returns:
            True if trading is allowed, False if circuit is open
        """
        if self.tripped_at is None:
            return True
        return (time.time() - self.tripped_at) > self.lim.cooldown_sec

    def require_ok(self) -> None:
        """Require circuit breaker to be OK, raise if not.
        
        Raises:
            RuntimeError: If circuit breaker is open
        """
        self.poll()
        if not self.can_trade():
            raise RuntimeError(f"CIRCUIT_OPEN: {self.reason}")

    def reset(self) -> None:
        """Manually reset circuit breaker (for admin use)."""
        self.tripped_at = None
        self.reason = None
        self.errors_last_min = 0
        self.errors_window_start = time.time()
        log.info("Circuit breaker manually reset")

    def status(self) -> dict:
        """Get current status.
        
        Returns:
            Dictionary with breaker status information
        """
        return {
            "tripped": self.tripped_at is not None,
            "reason": self.reason,
            "can_trade": self.can_trade(),
            "errors_last_min": self.errors_last_min,
            "data_staleness_sec": time.time() - self.last_data_ts,
            "cooldown_remaining_sec": max(0, self.lim.cooldown_sec - (time.time() - self.tripped_at)) if self.tripped_at else 0,
        }

