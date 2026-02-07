"""
Pre-Trade Compliance Service — Independent risk checks before order submission.

Runs outside strategy code as a standalone compliance gate.
Enforces position limits, notional limits, restricted lists, and rate limits.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from .interfaces import OrderRequest

logger = logging.getLogger(__name__)


@dataclass
class ComplianceRule:
    """A single compliance check rule."""

    name: str
    enabled: bool = True
    description: str = ""


@dataclass
class ComplianceResult:
    """Result of a pre-trade compliance check."""

    approved: bool
    order: Optional[OrderRequest] = None
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)
    checked_by: str = "PreTradeCompliance"

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


@dataclass
class ComplianceLimits:
    """Configurable compliance limits."""

    # Position limits
    max_position_pct: float = 0.20  # Max 20% of portfolio in single name
    max_position_shares: int = 100_000
    max_portfolio_exposure_pct: float = 1.0  # Max 100% net exposure

    # Notional limits
    max_order_notional: float = 500_000.0
    max_daily_notional: float = 5_000_000.0
    max_single_order_pct: float = 0.05  # Max 5% of portfolio per order

    # Rate limits
    max_orders_per_minute: int = 60
    max_orders_per_symbol_per_minute: int = 10

    # Restricted list
    restricted_symbols: Set[str] = field(default_factory=set)
    restricted_sectors: Set[str] = field(default_factory=set)

    # Order type restrictions
    allow_market_orders: bool = True
    allow_short_selling: bool = False
    max_leverage: float = 1.0


class PreTradeComplianceService:
    """
    Independent pre-trade compliance gate.

    Sits between strategy signal generation and broker order submission.
    All checks are synchronous and must pass before an order can proceed.
    """

    def __init__(
        self,
        limits: Optional[ComplianceLimits] = None,
        portfolio_value: float = 100_000.0,
    ) -> None:
        self.limits = limits or ComplianceLimits()
        self.portfolio_value = portfolio_value
        self._lock = threading.Lock()

        # Tracking state
        self._positions: Dict[str, float] = {}  # symbol -> qty
        self._daily_notional: float = 0.0
        self._daily_reset: datetime = datetime.utcnow()
        self._order_timestamps: List[datetime] = []
        self._symbol_order_timestamps: Dict[str, List[datetime]] = {}
        self._audit_log: List[ComplianceResult] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check(
        self,
        order: OrderRequest,
        current_price: Optional[float] = None,
    ) -> ComplianceResult:
        """Run all compliance checks on an order.

        Args:
            order: The order to validate.
            current_price: Current market price for notional calculations.

        Returns:
            ``ComplianceResult`` with approval status and any violations.
        """
        with self._lock:
            self._maybe_reset_daily()

            violations: List[str] = []
            warnings: List[str] = []
            price = current_price or 0.0
            notional = order.qty * price

            # 1. Restricted symbol check
            if order.symbol in self.limits.restricted_symbols:
                violations.append(
                    f"RESTRICTED: {order.symbol} is on restricted list"
                )

            # 2. Order type check
            if order.type == "market" and not self.limits.allow_market_orders:
                violations.append("MARKET_ORDER_BLOCKED: Market orders not allowed")

            # 3. Short selling check
            if order.side == "sell":
                current_pos = self._positions.get(order.symbol, 0)
                if order.qty > current_pos and not self.limits.allow_short_selling:
                    violations.append(
                        f"SHORT_SELL_BLOCKED: Would create short position "
                        f"(have {current_pos}, selling {order.qty})"
                    )

            # 4. Position concentration check
            if price > 0 and self.portfolio_value > 0:
                current_pos_value = self._positions.get(order.symbol, 0) * price
                if order.side == "buy":
                    new_pos_value = current_pos_value + notional
                else:
                    new_pos_value = current_pos_value - notional

                pos_pct = abs(new_pos_value) / self.portfolio_value
                if pos_pct > self.limits.max_position_pct:
                    violations.append(
                        f"POSITION_LIMIT: {order.symbol} would be "
                        f"{pos_pct:.1%} of portfolio (limit {self.limits.max_position_pct:.1%})"
                    )

            # 5. Max shares check
            if order.qty > self.limits.max_position_shares:
                violations.append(
                    f"SHARE_LIMIT: {order.qty} shares > max {self.limits.max_position_shares}"
                )

            # 6. Single order notional check
            if notional > self.limits.max_order_notional:
                violations.append(
                    f"ORDER_NOTIONAL: ${notional:,.0f} > max ${self.limits.max_order_notional:,.0f}"
                )

            # 7. Single order portfolio percentage check
            if self.portfolio_value > 0 and notional > 0:
                order_pct = notional / self.portfolio_value
                if order_pct > self.limits.max_single_order_pct:
                    violations.append(
                        f"ORDER_SIZE: {order_pct:.1%} of portfolio > "
                        f"max {self.limits.max_single_order_pct:.1%}"
                    )

            # 8. Daily notional check
            if self._daily_notional + notional > self.limits.max_daily_notional:
                violations.append(
                    f"DAILY_NOTIONAL: ${self._daily_notional + notional:,.0f} "
                    f"> max ${self.limits.max_daily_notional:,.0f}"
                )

            # 9. Rate limit check
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=1)
            recent = [t for t in self._order_timestamps if t > cutoff]
            if len(recent) >= self.limits.max_orders_per_minute:
                violations.append(
                    f"RATE_LIMIT: {len(recent)} orders/min >= max {self.limits.max_orders_per_minute}"
                )

            # 10. Per-symbol rate limit
            sym_recent = [
                t for t in self._symbol_order_timestamps.get(order.symbol, [])
                if t > cutoff
            ]
            if len(sym_recent) >= self.limits.max_orders_per_symbol_per_minute:
                violations.append(
                    f"SYMBOL_RATE: {len(sym_recent)} orders/min for {order.symbol} "
                    f">= max {self.limits.max_orders_per_symbol_per_minute}"
                )

            # Warnings (non-blocking)
            if self.portfolio_value > 0 and notional > 0:
                order_pct = notional / self.portfolio_value
                if order_pct > self.limits.max_single_order_pct * 0.8:
                    warnings.append(
                        f"Approaching single order limit ({order_pct:.1%})"
                    )

            approved = len(violations) == 0
            result = ComplianceResult(
                approved=approved,
                order=order,
                violations=violations,
                warnings=warnings,
            )

            # Record for audit
            self._audit_log.append(result)

            if approved:
                # Update tracking
                self._order_timestamps.append(now)
                self._symbol_order_timestamps.setdefault(order.symbol, []).append(now)
                self._daily_notional += notional
                logger.debug("Order approved: %s %s %s", order.side, order.qty, order.symbol)
            else:
                logger.warning(
                    "Order DENIED: %s %s %s — %s",
                    order.side, order.qty, order.symbol, "; ".join(violations),
                )

            return result

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def update_position(self, symbol: str, qty: float) -> None:
        """Update tracked position for a symbol."""
        with self._lock:
            self._positions[symbol] = qty

    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value for percentage calculations."""
        self.portfolio_value = value

    def add_restricted_symbol(self, symbol: str) -> None:
        """Add a symbol to the restricted list."""
        self.limits.restricted_symbols.add(symbol.upper())

    def remove_restricted_symbol(self, symbol: str) -> None:
        """Remove a symbol from the restricted list."""
        self.limits.restricted_symbols.discard(symbol.upper())

    def get_audit_log(self, last_n: int = 100) -> List[ComplianceResult]:
        """Get recent compliance check results."""
        return self._audit_log[-last_n:]

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily compliance statistics."""
        total = len(self._audit_log)
        approved = sum(1 for r in self._audit_log if r.approved)
        return {
            "total_checks": total,
            "approved": approved,
            "denied": total - approved,
            "daily_notional": self._daily_notional,
            "daily_notional_remaining": max(
                0, self.limits.max_daily_notional - self._daily_notional
            ),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self) -> None:
        now = datetime.utcnow()
        if now.date() > self._daily_reset.date():
            self._daily_notional = 0.0
            self._daily_reset = now
            self._audit_log.clear()
            logger.info("Daily compliance counters reset")
