"""
Enhanced Risk Engine - Inspired by Nautilus Trader.

Provides institutional-grade risk management features:
- Order rate limiting
- Notional value caps
- Trading state management
- Pre-trade risk checks

Concepts from: https://github.com/nautechsystems/nautilus_trader
License: LGPL-3.0 (concepts only, clean-room implementation)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
import re


class TradingState(Enum):
    """Trading state enumeration."""
    ACTIVE = "active"      # Normal trading
    REDUCING = "reducing"  # Close-only mode
    HALTED = "halted"      # All trading suspended


@dataclass
class RiskEngineConfig:
    """
    Configuration for the Enhanced Risk Engine.

    Inspired by Nautilus Trader's RiskEngineConfig.
    """
    bypass: bool = False  # Bypass all checks (dangerous!)
    max_order_submit_rate: str = "100/00:00:01"  # 100 orders per second
    max_order_modify_rate: str = "100/00:00:01"  # 100 modifications per second
    max_notional_per_order: Dict[str, float] = field(default_factory=dict)
    max_quantity_per_order: Dict[str, float] = field(default_factory=dict)
    max_orders_per_symbol: Dict[str, int] = field(default_factory=dict)
    max_position_value: Optional[float] = None
    max_total_exposure: Optional[float] = None
    debug: bool = False


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    passed: bool
    reason: str = ""
    blocked_by: str = ""

    @classmethod
    def allow(cls) -> "RiskCheckResult":
        return cls(passed=True)

    @classmethod
    def deny(cls, reason: str, blocked_by: str = "") -> "RiskCheckResult":
        return cls(passed=False, reason=reason, blocked_by=blocked_by)


class RateLimiter:
    """
    Rate limiter for order submissions.

    Parses rate strings like "100/00:00:01" (100 per second).
    """

    def __init__(self, rate_string: str):
        self._max_count, self._window = self._parse_rate(rate_string)
        self._timestamps: List[datetime] = []

    def _parse_rate(self, rate_string: str) -> tuple:
        """Parse rate string like '100/00:00:01'."""
        match = re.match(r"(\d+)/(\d{2}):(\d{2}):(\d{2})", rate_string)
        if not match:
            raise ValueError(f"Invalid rate format: {rate_string}")

        count = int(match.group(1))
        hours = int(match.group(2))
        minutes = int(match.group(3))
        seconds = int(match.group(4))

        window = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return count, window

    def check(self, timestamp: datetime) -> bool:
        """Check if rate limit allows another action."""
        # Clean old timestamps
        cutoff = timestamp - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        return len(self._timestamps) < self._max_count

    def record(self, timestamp: datetime) -> None:
        """Record an action."""
        self._timestamps.append(timestamp)

    @property
    def current_count(self) -> int:
        """Current count in window."""
        return len(self._timestamps)

    @property
    def max_count(self) -> int:
        """Maximum allowed count."""
        return self._max_count


class EnhancedRiskEngine:
    """
    Enhanced Risk Engine with institutional-grade features.

    Provides:
    - Pre-trade risk checks
    - Order rate limiting
    - Notional value caps per instrument
    - Trading state management
    - Duplicate order detection
    """

    def __init__(self, config: Optional[RiskEngineConfig] = None):
        self._config = config or RiskEngineConfig()
        self._trading_state = TradingState.ACTIVE
        self._submit_limiter = RateLimiter(self._config.max_order_submit_rate)
        self._modify_limiter = RateLimiter(self._config.max_order_modify_rate)

        # Tracking
        self._pending_orders: Dict[str, Set[str]] = {}  # symbol -> order_ids
        self._recent_order_ids: Set[str] = set()
        self._positions: Dict[str, float] = {}  # symbol -> quantity
        self._position_values: Dict[str, float] = {}  # symbol -> value

    @property
    def trading_state(self) -> TradingState:
        """Current trading state."""
        return self._trading_state

    def set_trading_state(self, state: TradingState) -> None:
        """Set trading state."""
        self._trading_state = state

    def halt_trading(self, reason: str = "") -> None:
        """Halt all trading."""
        self._trading_state = TradingState.HALTED

    def resume_trading(self) -> None:
        """Resume normal trading."""
        self._trading_state = TradingState.ACTIVE

    def set_reducing_only(self) -> None:
        """Set to close-only mode."""
        self._trading_state = TradingState.REDUCING

    def update_position(self, symbol: str, quantity: float, value: float) -> None:
        """Update position tracking."""
        self._positions[symbol] = quantity
        self._position_values[symbol] = value

    def check_order_submit(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> RiskCheckResult:
        """
        Pre-trade risk check for order submission.

        Args:
            order_id: Unique order identifier
            symbol: Instrument symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Order price (or estimated fill price)
            timestamp: Order timestamp

        Returns:
            RiskCheckResult indicating if order is allowed
        """
        timestamp = timestamp or datetime.now()

        if self._config.bypass:
            return RiskCheckResult.allow()

        # Check trading state
        if self._trading_state == TradingState.HALTED:
            return RiskCheckResult.deny(
                "Trading is halted",
                blocked_by="TradingState"
            )

        if self._trading_state == TradingState.REDUCING:
            # Only allow orders that reduce position
            current_pos = self._positions.get(symbol, 0)
            is_reducing = (
                (side == "sell" and current_pos > 0) or
                (side == "buy" and current_pos < 0)
            )
            if not is_reducing:
                return RiskCheckResult.deny(
                    "Only position-reducing orders allowed",
                    blocked_by="TradingState"
                )

        # Check duplicate order ID
        if order_id in self._recent_order_ids:
            return RiskCheckResult.deny(
                f"Duplicate order ID: {order_id}",
                blocked_by="DuplicateCheck"
            )

        # Check rate limit
        if not self._submit_limiter.check(timestamp):
            return RiskCheckResult.deny(
                f"Order submit rate exceeded: {self._submit_limiter.current_count}/{self._submit_limiter.max_count}",
                blocked_by="RateLimit"
            )

        # Check notional value
        notional = quantity * price
        max_notional = self._config.max_notional_per_order.get(symbol)
        if max_notional is not None and notional > max_notional:
            return RiskCheckResult.deny(
                f"Notional {notional:.2f} exceeds max {max_notional:.2f} for {symbol}",
                blocked_by="NotionalLimit"
            )

        # Check quantity limit
        max_qty = self._config.max_quantity_per_order.get(symbol)
        if max_qty is not None and quantity > max_qty:
            return RiskCheckResult.deny(
                f"Quantity {quantity} exceeds max {max_qty} for {symbol}",
                blocked_by="QuantityLimit"
            )

        # Check orders per symbol
        max_orders = self._config.max_orders_per_symbol.get(symbol)
        if max_orders is not None:
            current_orders = len(self._pending_orders.get(symbol, set()))
            if current_orders >= max_orders:
                return RiskCheckResult.deny(
                    f"Max orders ({max_orders}) reached for {symbol}",
                    blocked_by="OrderCountLimit"
                )

        # Check max position value
        if self._config.max_position_value is not None:
            new_position_value = self._position_values.get(symbol, 0) + notional
            if new_position_value > self._config.max_position_value:
                return RiskCheckResult.deny(
                    f"Position value {new_position_value:.2f} would exceed max {self._config.max_position_value:.2f}",
                    blocked_by="PositionValueLimit"
                )

        # Check max total exposure
        if self._config.max_total_exposure is not None:
            total_exposure = sum(abs(v) for v in self._position_values.values())
            new_total = total_exposure + notional
            if new_total > self._config.max_total_exposure:
                return RiskCheckResult.deny(
                    f"Total exposure {new_total:.2f} would exceed max {self._config.max_total_exposure:.2f}",
                    blocked_by="TotalExposureLimit"
                )

        # All checks passed - record the order
        self._submit_limiter.record(timestamp)
        self._recent_order_ids.add(order_id)
        if symbol not in self._pending_orders:
            self._pending_orders[symbol] = set()
        self._pending_orders[symbol].add(order_id)

        return RiskCheckResult.allow()

    def check_order_modify(
        self,
        order_id: str,
        symbol: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> RiskCheckResult:
        """
        Pre-trade risk check for order modification.
        """
        timestamp = timestamp or datetime.now()

        if self._config.bypass:
            return RiskCheckResult.allow()

        if self._trading_state == TradingState.HALTED:
            return RiskCheckResult.deny(
                "Trading is halted",
                blocked_by="TradingState"
            )

        # Check rate limit
        if not self._modify_limiter.check(timestamp):
            return RiskCheckResult.deny(
                f"Order modify rate exceeded: {self._modify_limiter.current_count}/{self._modify_limiter.max_count}",
                blocked_by="RateLimit"
            )

        # Check notional if both new values provided
        if new_quantity is not None and new_price is not None:
            notional = new_quantity * new_price
            max_notional = self._config.max_notional_per_order.get(symbol)
            if max_notional is not None and notional > max_notional:
                return RiskCheckResult.deny(
                    f"New notional {notional:.2f} exceeds max {max_notional:.2f}",
                    blocked_by="NotionalLimit"
                )

        self._modify_limiter.record(timestamp)
        return RiskCheckResult.allow()

    def on_order_filled(self, order_id: str, symbol: str) -> None:
        """Handle order fill - remove from pending."""
        if symbol in self._pending_orders:
            self._pending_orders[symbol].discard(order_id)

    def on_order_canceled(self, order_id: str, symbol: str) -> None:
        """Handle order cancel - remove from pending."""
        if symbol in self._pending_orders:
            self._pending_orders[symbol].discard(order_id)

    def reset(self) -> None:
        """Reset engine state."""
        self._trading_state = TradingState.ACTIVE
        self._pending_orders.clear()
        self._recent_order_ids.clear()
        self._positions.clear()
        self._position_values.clear()

    def get_status(self) -> Dict:
        """Get current risk engine status."""
        return {
            "trading_state": self._trading_state.value,
            "submit_rate": f"{self._submit_limiter.current_count}/{self._submit_limiter.max_count}",
            "modify_rate": f"{self._modify_limiter.current_count}/{self._modify_limiter.max_count}",
            "pending_orders": {s: len(ids) for s, ids in self._pending_orders.items()},
            "positions": self._positions.copy(),
            "position_values": self._position_values.copy(),
        }
