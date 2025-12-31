"""Circuit breaker for daily loss, error spikes, stale data, and VIX protection.

This module implements a production-grade circuit breaker that trips on:
- Daily drawdown exceeding limits
- Error rate spikes
- Data staleness
- VIX (volatility) exceeding critical threshold
- Requires manual reset after cooldown period

State Persistence:
- When `persist=True`, state is saved to database for restart persistence
- Use `restore_from_db()` on startup to restore previous state
- Use `sync_to_db()` periodically or after state changes
"""
from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from django.contrib.auth.models import User

log = logging.getLogger("wsb.circuit_breaker")


class VIXTripLevel(Enum):
    """VIX levels that affect circuit breaker behavior."""
    NONE = "none"           # No VIX impact
    ELEVATED = "elevated"   # VIX 25-35: reduce position sizes
    EXTREME = "extreme"     # VIX 35-45: severely reduce positions
    CRITICAL = "critical"   # VIX > 45: trip circuit breaker


@dataclass
class VIXBreakerConfig:
    """VIX-specific circuit breaker configuration."""
    enabled: bool = True
    elevated_threshold: float = 25.0   # Start reducing positions
    extreme_threshold: float = 35.0    # Severely reduce positions
    critical_threshold: float = 45.0   # Trip circuit breaker
    spike_threshold_pct: float = 20.0  # % increase = spike warning
    check_interval_sec: int = 300      # 5 minutes


@dataclass
class BreakerLimits:
    """Configuration limits for circuit breaker."""
    max_daily_drawdown: float = 0.06  # 6% of start-of-day equity
    max_error_rate_per_min: int = 5
    max_data_staleness_sec: int = 30
    cooldown_sec: int = 60 * 30  # 30 minutes
    # VIX configuration
    vix_config: VIXBreakerConfig = field(default_factory=VIXBreakerConfig)


class CircuitBreaker:
    """Production circuit breaker with multiple trip conditions.

    Trips on:
    - Daily drawdown exceeding max_daily_drawdown
    - Error rate exceeding max_error_rate_per_min
    - Data staleness exceeding max_data_staleness_sec
    - VIX exceeding critical threshold (> 45)

    Soft limits (position size reduction, not trip):
    - VIX elevated (25-35): reduce position sizes by 50%
    - VIX extreme (35-45): reduce position sizes by 75%

    Requires cooldown period before reset.

    State Persistence:
    - Set `persist=True` and provide a `user` to enable DB persistence
    - Call `restore_from_db()` on startup to restore state
    - State is automatically synced on trip/reset when persistence is enabled
    """

    def __init__(
        self,
        start_equity: float,
        limits: BreakerLimits = None,
        persist: bool = False,
        user: 'User' = None,
    ):
        """Initialize circuit breaker.

        Args:
            start_equity: Starting equity for drawdown calculation
            limits: BreakerLimits configuration
            persist: Enable database persistence
            user: Django User for persistence (optional)
        """
        self.start_equity = float(start_equity)
        self.lim = limits if limits is not None else BreakerLimits()
        self.errors_last_min = 0
        self.errors_window_start = time.time()
        self.tripped_at: Optional[float] = None
        self.reason: Optional[str] = None
        self.last_data_ts = time.time()
        # VIX tracking
        self.current_vix: Optional[float] = None
        self.vix_level: VIXTripLevel = VIXTripLevel.NONE
        self.last_vix_check: float = 0
        self._vix_monitor = None  # Lazy-loaded
        # Persistence
        self._persist = persist
        self._user = user
        self._persistence = None  # Lazy-loaded

    def _get_persistence(self):
        """Lazy-load persistence service."""
        if self._persistence is None and self._persist:
            try:
                from backend.auth0login.services.circuit_breaker_persistence import (
                    get_circuit_breaker_persistence
                )
                self._persistence = get_circuit_breaker_persistence(self._user)
            except ImportError:
                log.warning("Circuit breaker persistence not available")
                self._persistence = False  # Mark as unavailable
        return self._persistence if self._persistence else None

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

    def _get_vix_monitor(self):
        """Lazy-load VIX monitor to avoid circular imports."""
        if self._vix_monitor is None:
            try:
                from backend.auth0login.services.market_monitor import get_market_monitor
                self._vix_monitor = get_market_monitor()
            except ImportError:
                log.warning("Market monitor not available for VIX checking")
                self._vix_monitor = False  # Mark as unavailable
        return self._vix_monitor if self._vix_monitor else None

    def check_vix(self, force: bool = False) -> VIXTripLevel:
        """Check VIX level and potentially trip breaker.

        Args:
            force: Force refresh even if within check interval

        Returns:
            Current VIX trip level
        """
        if not self.lim.vix_config.enabled:
            return VIXTripLevel.NONE

        now = time.time()
        # Skip if checked recently (unless forced)
        if not force and (now - self.last_vix_check) < self.lim.vix_config.check_interval_sec:
            return self.vix_level

        monitor = self._get_vix_monitor()
        if monitor is None:
            return VIXTripLevel.NONE

        try:
            vix_value = monitor.get_current_vix()
            if vix_value is None:
                return self.vix_level  # Keep previous level if fetch fails

            self.current_vix = vix_value
            self.last_vix_check = now

            # Determine level
            cfg = self.lim.vix_config
            if vix_value >= cfg.critical_threshold:
                self.vix_level = VIXTripLevel.CRITICAL
                self.trip(f"vix-critical-{vix_value:.1f}")
            elif vix_value >= cfg.extreme_threshold:
                self.vix_level = VIXTripLevel.EXTREME
                log.warning(f"VIX extreme: {vix_value:.1f} - Position sizes reduced 75%")
            elif vix_value >= cfg.elevated_threshold:
                self.vix_level = VIXTripLevel.ELEVATED
                log.info(f"VIX elevated: {vix_value:.1f} - Position sizes reduced 50%")
            else:
                self.vix_level = VIXTripLevel.NONE

            return self.vix_level

        except Exception as e:
            log.error(f"VIX check failed: {e}")
            return self.vix_level

    def get_vix_position_multiplier(self) -> float:
        """Get position size multiplier based on VIX level.

        Returns:
            Multiplier (0.0 to 1.0) to apply to position sizes
        """
        if self.vix_level == VIXTripLevel.CRITICAL:
            return 0.0  # No trading
        elif self.vix_level == VIXTripLevel.EXTREME:
            return 0.25  # 75% reduction
        elif self.vix_level == VIXTripLevel.ELEVATED:
            return 0.50  # 50% reduction
        else:
            return 1.0  # Normal

    def update_vix(self, vix_value: float) -> VIXTripLevel:
        """Manually update VIX value (for testing or external feeds).

        Args:
            vix_value: Current VIX value

        Returns:
            Resulting VIX trip level
        """
        self.current_vix = vix_value
        self.last_vix_check = time.time()

        cfg = self.lim.vix_config
        if vix_value >= cfg.critical_threshold:
            self.vix_level = VIXTripLevel.CRITICAL
            self.trip(f"vix-critical-{vix_value:.1f}")
        elif vix_value >= cfg.extreme_threshold:
            self.vix_level = VIXTripLevel.EXTREME
        elif vix_value >= cfg.elevated_threshold:
            self.vix_level = VIXTripLevel.ELEVATED
        else:
            self.vix_level = VIXTripLevel.NONE

        return self.vix_level

    def poll(self) -> None:
        """Poll for stale data and VIX conditions."""
        if time.time() - self.last_data_ts > self.lim.max_data_staleness_sec:
            self.trip("stale-data")
        # Check VIX periodically
        if self.lim.vix_config.enabled:
            self.check_vix()

    def trip(self, reason: str, user=None, trigger_value: float = None) -> None:
        """Trip the circuit breaker.

        Args:
            reason: Reason for tripping
            user: Django user to create event for (optional)
            trigger_value: Value that caused the trigger (optional)
        """
        if self.tripped_at is None:
            self.tripped_at = time.time()
            self.reason = reason
            log.critical(f"CIRCUIT_BREAKER_TRIPPED: {reason}")

            # Create CircuitBreakerEvent for tracking and recovery
            trip_user = user or self._user
            if trip_user:
                try:
                    self._create_breaker_event(reason, trip_user, trigger_value)
                except Exception as e:
                    log.error(f"Failed to create breaker event: {e}")

            # Sync to database if persistence enabled
            self._sync_to_db()

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
        # Reset VIX state (but keep current value)
        self.vix_level = VIXTripLevel.NONE
        log.info("Circuit breaker manually reset")

        # Sync to database if persistence enabled
        self._sync_to_db()

    def status(self) -> Dict[str, Any]:
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
            # VIX status
            "vix": {
                "enabled": self.lim.vix_config.enabled,
                "current_value": self.current_vix,
                "level": self.vix_level.value,
                "position_multiplier": self.get_vix_position_multiplier(),
                "thresholds": {
                    "elevated": self.lim.vix_config.elevated_threshold,
                    "extreme": self.lim.vix_config.extreme_threshold,
                    "critical": self.lim.vix_config.critical_threshold,
                },
                "last_check": self.last_vix_check,
            }
        }

    def _create_breaker_event(self, reason: str, user, trigger_value: float = None):
        """Create a CircuitBreakerEvent for tracking and recovery.

        Args:
            reason: Reason string from trip()
            user: Django User instance
            trigger_value: Value that caused the trigger
        """
        from backend.auth0login.services.recovery_manager import get_recovery_manager

        # Parse breaker type from reason
        breaker_type = 'manual'
        threshold = 0.0

        if reason.startswith('daily-dd'):
            breaker_type = 'daily_loss'
            # Extract value from reason like "daily-dd-0.065"
            try:
                threshold = self.lim.max_daily_drawdown
                if trigger_value is None:
                    parts = reason.split('-')
                    if len(parts) >= 3:
                        trigger_value = float(parts[2])
            except (ValueError, IndexError):
                trigger_value = trigger_value or threshold

        elif reason.startswith('vix-critical'):
            breaker_type = 'vix_critical'
            threshold = self.lim.vix_config.critical_threshold
            try:
                if trigger_value is None:
                    parts = reason.split('-')
                    if len(parts) >= 3:
                        trigger_value = float(parts[2])
            except (ValueError, IndexError):
                trigger_value = trigger_value or self.current_vix or threshold

        elif reason == 'error-rate':
            breaker_type = 'error_rate'
            threshold = self.lim.max_error_rate_per_min
            trigger_value = trigger_value or self.errors_last_min

        elif reason == 'stale-data':
            breaker_type = 'stale_data'
            threshold = self.lim.max_data_staleness_sec
            trigger_value = trigger_value or (time.time() - self.last_data_ts)

        # Build context
        context = {
            'reason': reason,
            'start_equity': self.start_equity,
            'vix_level': self.vix_level.value if self.vix_level else None,
            'current_vix': self.current_vix,
            'errors_last_min': self.errors_last_min,
        }

        # Create event via recovery manager
        recovery_mgr = get_recovery_manager(user)
        recovery_mgr.create_breaker_event(
            breaker_type=breaker_type,
            trigger_value=trigger_value or 0,
            trigger_threshold=threshold,
            context=context,
            notes=f"Auto-created from circuit breaker trip: {reason}",
        )

    def _sync_to_db(self) -> None:
        """Sync current state to database if persistence is enabled."""
        persistence = self._get_persistence()
        if persistence:
            try:
                persistence.sync_from_memory(self)
                log.debug("Circuit breaker state synced to database")
            except Exception as e:
                log.error(f"Failed to sync circuit breaker to DB: {e}")

    def restore_from_db(self) -> bool:
        """Restore state from database.

        Should be called on application startup to restore any
        previously triggered breakers.

        Returns:
            True if a tripped state was restored
        """
        persistence = self._get_persistence()
        if persistence:
            try:
                restored = persistence.restore_to_memory(self)
                if restored:
                    log.warning("Restored tripped circuit breaker from database")
                return restored
            except Exception as e:
                log.error(f"Failed to restore circuit breaker from DB: {e}")
                return False
        return False

    def sync_to_db(self) -> None:
        """Manually sync state to database.

        Call this periodically or after significant state changes
        if you want more frequent persistence than the automatic
        sync on trip/reset.
        """
        self._sync_to_db()

    def enable_persistence(self, user: 'User' = None) -> None:
        """Enable database persistence.

        Args:
            user: Django User for persistence
        """
        self._persist = True
        if user:
            self._user = user
        self._persistence = None  # Reset to force reload

    def disable_persistence(self) -> None:
        """Disable database persistence."""
        self._persist = False
        self._persistence = None

