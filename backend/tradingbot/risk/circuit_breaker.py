"""
Circuit Breaker - Inspired by Polymarket-Kalshi Arbitrage Bot.

Advanced risk control system that automatically halts trading when:
- Daily loss limit exceeded
- Position limits breached
- Consecutive errors threshold reached
- Manual halt triggered

Concepts from: https://github.com/terauss/Polymarket-Kalshi-Arbitrage-bot
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Set, Callable, Any
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


class TripReason(Enum):
    """Reason the circuit breaker tripped."""
    NONE = "none"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_LIMIT = "position_limit"
    TOTAL_POSITION_LIMIT = "total_position_limit"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    MANUAL_HALT = "manual_halt"
    DRAWDOWN_LIMIT = "drawdown_limit"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for the Circuit Breaker.

    Inspired by the Polymarket-Kalshi arbitrage bot's risk controls.
    """
    enabled: bool = True

    # Loss limits
    max_daily_loss: float = 500.0  # Maximum daily loss in dollars
    max_drawdown_percent: float = 10.0  # Maximum drawdown from peak

    # Position limits
    max_position_per_symbol: int = 50000  # Max contracts per symbol
    max_total_position: int = 100000  # Max total contracts across all
    max_position_value: float = 100000.0  # Max position value in dollars

    # Error limits
    max_consecutive_errors: int = 5  # Errors before halt

    # Cooldown
    cooldown_seconds: int = 300  # Time before auto-reset (5 minutes)
    auto_reset: bool = True  # Whether to auto-reset after cooldown

    # Callbacks
    on_trip: Optional[Callable[[TripReason, str], None]] = None
    on_reset: Optional[Callable[[], None]] = None


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""
    is_halted: bool = False
    trip_reason: TripReason = TripReason.NONE
    trip_message: str = ""
    trip_time: Optional[datetime] = None

    # Tracking
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    consecutive_errors: int = 0
    last_reset_date: date = field(default_factory=date.today)

    # Position tracking
    positions: Dict[str, int] = field(default_factory=dict)
    position_values: Dict[str, float] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit Breaker for trading risk management.

    Automatically halts trading when risk thresholds are breached.
    Thread-safe implementation using locks.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._lock = threading.RLock()
        self._error_symbols: Set[str] = set()

    @property
    def is_halted(self) -> bool:
        """Check if circuit breaker is tripped."""
        with self._lock:
            return self._state.is_halted

    @property
    def is_enabled(self) -> bool:
        """Check if circuit breaker is enabled."""
        return self._config.enabled

    @property
    def trip_reason(self) -> TripReason:
        """Get the reason for the trip."""
        with self._lock:
            return self._state.trip_reason

    @property
    def daily_pnl(self) -> float:
        """Current daily P&L."""
        with self._lock:
            return self._state.daily_pnl

    @property
    def consecutive_errors(self) -> int:
        """Current consecutive error count."""
        with self._lock:
            return self._state.consecutive_errors

    def can_execute(self, symbol: str, quantity: int,
                   value: Optional[float] = None) -> tuple:
        """
        Check if an order can be executed.

        Args:
            symbol: Trading symbol
            quantity: Number of contracts/shares
            value: Dollar value of the order

        Returns:
            Tuple of (can_execute: bool, reason: str)
        """
        if not self._config.enabled:
            return True, ""

        with self._lock:
            # Check auto-reset
            self._check_auto_reset()

            # Check if halted
            if self._state.is_halted:
                return False, f"Circuit breaker halted: {self._state.trip_message}"

            # Check daily loss
            if self._state.daily_pnl <= -self._config.max_daily_loss:
                self._trip(TripReason.DAILY_LOSS_LIMIT,
                          f"Daily loss ${abs(self._state.daily_pnl):.2f} exceeds limit ${self._config.max_daily_loss:.2f}")
                return False, self._state.trip_message

            # Check drawdown
            if self._state.peak_equity > 0:
                drawdown = (self._state.peak_equity - self._state.current_equity) / self._state.peak_equity * 100
                if drawdown >= self._config.max_drawdown_percent:
                    self._trip(TripReason.DRAWDOWN_LIMIT,
                              f"Drawdown {drawdown:.1f}% exceeds limit {self._config.max_drawdown_percent:.1f}%")
                    return False, self._state.trip_message

            # Check position limit for symbol
            current_pos = self._state.positions.get(symbol, 0)
            new_pos = current_pos + quantity
            if abs(new_pos) > self._config.max_position_per_symbol:
                self._trip(TripReason.POSITION_LIMIT,
                          f"Position {new_pos} for {symbol} exceeds limit {self._config.max_position_per_symbol}")
                return False, self._state.trip_message

            # Check total position limit
            total_pos = sum(abs(p) for p in self._state.positions.values()) + abs(quantity)
            if total_pos > self._config.max_total_position:
                self._trip(TripReason.TOTAL_POSITION_LIMIT,
                          f"Total position {total_pos} exceeds limit {self._config.max_total_position}")
                return False, self._state.trip_message

            # Check position value limit
            if value is not None:
                current_value = sum(abs(v) for v in self._state.position_values.values())
                if current_value + abs(value) > self._config.max_position_value:
                    self._trip(TripReason.POSITION_LIMIT,
                              f"Position value ${current_value + abs(value):.2f} exceeds limit ${self._config.max_position_value:.2f}")
                    return False, self._state.trip_message

            # Check consecutive errors
            if self._state.consecutive_errors >= self._config.max_consecutive_errors:
                self._trip(TripReason.CONSECUTIVE_ERRORS,
                          f"Consecutive errors {self._state.consecutive_errors} reached limit")
                return False, self._state.trip_message

            return True, ""

    def record_fill(self, symbol: str, quantity: int, pnl: float,
                   value: Optional[float] = None) -> None:
        """
        Record a successful fill.

        Args:
            symbol: Trading symbol
            quantity: Filled quantity (positive for buy, negative for sell)
            pnl: P&L from this fill
            value: Dollar value of the position
        """
        with self._lock:
            self._check_daily_reset()

            # Update position
            current = self._state.positions.get(symbol, 0)
            self._state.positions[symbol] = current + quantity

            # Update position value
            if value is not None:
                current_value = self._state.position_values.get(symbol, 0)
                self._state.position_values[symbol] = current_value + value

            # Update P&L
            self._state.daily_pnl += pnl
            self._state.current_equity += pnl

            # Update peak equity
            if self._state.current_equity > self._state.peak_equity:
                self._state.peak_equity = self._state.current_equity

            # Reset error count on success
            self._state.consecutive_errors = 0
            self._error_symbols.discard(symbol)

    def record_error(self, symbol: str, error: str) -> None:
        """
        Record a trading error.

        Args:
            symbol: Trading symbol
            error: Error message
        """
        with self._lock:
            if symbol not in self._error_symbols:
                self._state.consecutive_errors += 1
                self._error_symbols.add(symbol)

            logger.warning(f"Circuit breaker recorded error #{self._state.consecutive_errors}: {error}")

            if self._state.consecutive_errors >= self._config.max_consecutive_errors:
                self._trip(TripReason.CONSECUTIVE_ERRORS,
                          f"Max consecutive errors ({self._config.max_consecutive_errors}) reached: {error}")

    def record_pnl(self, pnl: float) -> None:
        """Record P&L without a fill."""
        with self._lock:
            self._check_daily_reset()
            self._state.daily_pnl += pnl
            self._state.current_equity += pnl

            if self._state.current_equity > self._state.peak_equity:
                self._state.peak_equity = self._state.current_equity

    def halt(self, reason: str = "Manual halt") -> None:
        """Manually halt trading."""
        with self._lock:
            self._trip(TripReason.MANUAL_HALT, reason)

    def reset(self) -> None:
        """Reset the circuit breaker."""
        with self._lock:
            self._state.is_halted = False
            self._state.trip_reason = TripReason.NONE
            self._state.trip_message = ""
            self._state.trip_time = None
            self._state.consecutive_errors = 0
            self._error_symbols.clear()

            logger.info("Circuit breaker reset")

            if self._config.on_reset:
                self._config.on_reset()

    def reset_daily(self) -> None:
        """Reset daily counters."""
        with self._lock:
            self._state.daily_pnl = 0.0
            self._state.last_reset_date = date.today()
            logger.info("Circuit breaker daily counters reset")

    def set_equity(self, equity: float) -> None:
        """Set current equity for drawdown tracking."""
        with self._lock:
            self._state.current_equity = equity
            if equity > self._state.peak_equity:
                self._state.peak_equity = equity

    def _trip(self, reason: TripReason, message: str) -> None:
        """Trip the circuit breaker (internal)."""
        if self._state.is_halted:
            return  # Already tripped

        self._state.is_halted = True
        self._state.trip_reason = reason
        self._state.trip_message = message
        self._state.trip_time = datetime.now()

        logger.error(f"CIRCUIT BREAKER TRIPPED: {reason.value} - {message}")

        if self._config.on_trip:
            self._config.on_trip(reason, message)

    def _check_auto_reset(self) -> None:
        """Check if cooldown has passed for auto-reset."""
        if not self._config.auto_reset or not self._state.is_halted:
            return

        if self._state.trip_time is None:
            return

        elapsed = datetime.now() - self._state.trip_time
        if elapsed >= timedelta(seconds=self._config.cooldown_seconds):
            logger.info(f"Circuit breaker auto-reset after {self._config.cooldown_seconds}s cooldown")
            self.reset()

    def _check_daily_reset(self) -> None:
        """Check if daily counters need reset."""
        today = date.today()
        if self._state.last_reset_date < today:
            self.reset_daily()

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "enabled": self._config.enabled,
                "is_halted": self._state.is_halted,
                "trip_reason": self._state.trip_reason.value,
                "trip_message": self._state.trip_message,
                "trip_time": self._state.trip_time.isoformat() if self._state.trip_time else None,
                "daily_pnl": self._state.daily_pnl,
                "current_equity": self._state.current_equity,
                "peak_equity": self._state.peak_equity,
                "consecutive_errors": self._state.consecutive_errors,
                "total_position": sum(abs(p) for p in self._state.positions.values()),
                "positions": dict(self._state.positions),
                "limits": {
                    "max_daily_loss": self._config.max_daily_loss,
                    "max_drawdown_percent": self._config.max_drawdown_percent,
                    "max_position_per_symbol": self._config.max_position_per_symbol,
                    "max_total_position": self._config.max_total_position,
                    "max_consecutive_errors": self._config.max_consecutive_errors,
                    "cooldown_seconds": self._config.cooldown_seconds,
                },
            }


# Global circuit breaker instance
_global_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create the global circuit breaker."""
    global _global_breaker
    if _global_breaker is None:
        _global_breaker = CircuitBreaker()
    return _global_breaker


def configure_circuit_breaker(config: CircuitBreakerConfig) -> CircuitBreaker:
    """Configure the global circuit breaker."""
    global _global_breaker
    _global_breaker = CircuitBreaker(config)
    return _global_breaker
