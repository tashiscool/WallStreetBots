"""
Pre-trade Risk Validation Engine.

Ported from Nautilus Trader's risk engine.
Validates orders before submission to prevent risk limit breaches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class TradingState(Enum):
    """Trading state of the risk engine."""
    ACTIVE = "active"       # Normal trading
    REDUCING = "reducing"   # Only position-reducing trades allowed
    HALTED = "halted"       # No trading allowed


class RejectionReason(Enum):
    """Reason for order rejection."""
    TRADING_HALTED = "trading_halted"
    REDUCING_ONLY = "reducing_only_mode"
    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    EXCEEDS_POSITION_LIMIT = "exceeds_position_limit"
    EXCEEDS_ORDER_SIZE = "exceeds_order_size"
    EXCEEDS_NOTIONAL_LIMIT = "exceeds_notional_limit"
    EXCEEDS_CONCENTRATION_LIMIT = "exceeds_concentration_limit"
    PRICE_OUT_OF_RANGE = "price_out_of_range"
    SYMBOL_NOT_TRADABLE = "symbol_not_tradable"
    SYMBOL_LOCKED = "symbol_locked"
    TOO_MANY_ORDERS = "too_many_orders"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    VALIDATION_FAILED = "validation_failed"
    MARGIN_CALL_ACTIVE = "margin_call_active"


@dataclass
class ValidationResult:
    """Result of order validation."""
    is_valid: bool
    rejection_reason: Optional[RejectionReason] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # Position limits
    max_position_size: int = 10000              # Max shares per position
    max_position_value: Decimal = Decimal("100000")  # Max $ value per position
    max_positions: int = 50                     # Max open positions

    # Order limits
    max_order_size: int = 1000                  # Max shares per order
    max_order_value: Decimal = Decimal("50000")  # Max $ value per order
    max_orders_per_minute: int = 60             # Rate limit

    # Portfolio limits
    max_total_exposure: Decimal = Decimal("500000")  # Max total exposure
    max_sector_concentration: float = 0.25      # Max % in one sector
    max_single_stock_concentration: float = 0.10  # Max % in one stock
    max_daily_loss: Decimal = Decimal("10000")  # Max daily loss
    max_drawdown_pct: float = 0.20              # Max portfolio drawdown

    # Price limits
    min_price: Decimal = Decimal("0.01")        # Min stock price
    max_price: Decimal = Decimal("10000")       # Max stock price
    max_spread_pct: float = 0.05                # Max bid-ask spread %

    # Leverage
    max_leverage: Decimal = Decimal("4.0")      # Max leverage ratio


@dataclass
class Position:
    """Position for risk validation."""
    symbol: str
    quantity: int
    average_price: Decimal
    current_price: Decimal
    sector: Optional[str] = None

    @property
    def market_value(self) -> Decimal:
        """Current market value."""
        return abs(self.quantity) * self.current_price

    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.quantity > 0

    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized P&L."""
        return (self.current_price - self.average_price) * self.quantity


@dataclass
class Order:
    """Order for risk validation."""
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop', etc.
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    estimated_price: Optional[Decimal] = None

    @property
    def is_buy(self) -> bool:
        """True if buy order."""
        return self.side.lower() == "buy"

    @property
    def estimated_value(self) -> Decimal:
        """Estimated order value."""
        price = self.limit_price or self.estimated_price or Decimal("0")
        return abs(self.quantity) * price


class IOrderValidator(ABC):
    """Abstract interface for order validators."""

    @abstractmethod
    def validate(
        self,
        order: Order,
        positions: Dict[str, Position],
        account_value: Decimal,
        cash: Decimal,
    ) -> ValidationResult:
        """
        Validate an order.

        Args:
            order: Order to validate
            positions: Current positions
            account_value: Total account value
            cash: Available cash

        Returns:
            ValidationResult
        """
        pass


class PositionSizeValidator(IOrderValidator):
    """Validate position size limits."""

    def __init__(self, limits: RiskLimits):
        """Initialize with limits."""
        self.limits = limits

    def validate(
        self,
        order: Order,
        positions: Dict[str, Position],
        account_value: Decimal,
        cash: Decimal,
    ) -> ValidationResult:
        """Check position size limits."""
        # Get current position
        current_qty = 0
        if order.symbol in positions:
            current_qty = positions[order.symbol].quantity

        # Calculate resulting position
        if order.is_buy:
            new_qty = current_qty + order.quantity
        else:
            new_qty = current_qty - order.quantity

        # Check max position size
        if abs(new_qty) > self.limits.max_position_size:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.EXCEEDS_POSITION_LIMIT,
                message=f"Position would exceed {self.limits.max_position_size} shares",
                details={
                    "current_qty": current_qty,
                    "order_qty": order.quantity,
                    "resulting_qty": new_qty,
                    "limit": self.limits.max_position_size,
                },
            )

        # Check max position value
        price = order.limit_price or order.estimated_price or Decimal("0")
        position_value = abs(new_qty) * price

        if position_value > self.limits.max_position_value:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.EXCEEDS_POSITION_LIMIT,
                message=f"Position value would exceed ${self.limits.max_position_value}",
                details={
                    "position_value": position_value,
                    "limit": self.limits.max_position_value,
                },
            )

        return ValidationResult(is_valid=True)


class OrderSizeValidator(IOrderValidator):
    """Validate order size limits."""

    def __init__(self, limits: RiskLimits):
        """Initialize with limits."""
        self.limits = limits

    def validate(
        self,
        order: Order,
        positions: Dict[str, Position],
        account_value: Decimal,
        cash: Decimal,
    ) -> ValidationResult:
        """Check order size limits."""
        # Check max order size
        if abs(order.quantity) > self.limits.max_order_size:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.EXCEEDS_ORDER_SIZE,
                message=f"Order exceeds {self.limits.max_order_size} shares",
                details={
                    "order_qty": order.quantity,
                    "limit": self.limits.max_order_size,
                },
            )

        # Check max order value
        if order.estimated_value > self.limits.max_order_value:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.EXCEEDS_NOTIONAL_LIMIT,
                message=f"Order value exceeds ${self.limits.max_order_value}",
                details={
                    "order_value": order.estimated_value,
                    "limit": self.limits.max_order_value,
                },
            )

        return ValidationResult(is_valid=True)


class ConcentrationValidator(IOrderValidator):
    """Validate portfolio concentration limits."""

    def __init__(self, limits: RiskLimits):
        """Initialize with limits."""
        self.limits = limits

    def validate(
        self,
        order: Order,
        positions: Dict[str, Position],
        account_value: Decimal,
        cash: Decimal,
    ) -> ValidationResult:
        """Check concentration limits."""
        if account_value <= 0:
            return ValidationResult(is_valid=True)

        # Calculate total exposure and sector exposures
        sector_exposures: Dict[str, Decimal] = {}

        for pos in positions.values():
            if pos.sector:
                sector_exposures[pos.sector] = (
                    sector_exposures.get(pos.sector, Decimal("0")) +
                    pos.market_value
                )

        # Estimate resulting position value
        current_value = Decimal("0")
        if order.symbol in positions:
            current_value = positions[order.symbol].market_value

        order_impact = order.estimated_value
        if not order.is_buy:
            order_impact = -order_impact

        new_position_value = current_value + order_impact

        # Check single stock concentration
        concentration = new_position_value / account_value
        if concentration > Decimal(str(self.limits.max_single_stock_concentration)):
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.EXCEEDS_CONCENTRATION_LIMIT,
                message=f"Single stock concentration would exceed {self.limits.max_single_stock_concentration:.0%}",
                details={
                    "concentration": float(concentration),
                    "limit": self.limits.max_single_stock_concentration,
                },
            )

        return ValidationResult(is_valid=True)


class PriceValidator(IOrderValidator):
    """Validate price limits."""

    def __init__(self, limits: RiskLimits):
        """Initialize with limits."""
        self.limits = limits

    def validate(
        self,
        order: Order,
        positions: Dict[str, Position],
        account_value: Decimal,
        cash: Decimal,
    ) -> ValidationResult:
        """Check price limits."""
        price = order.limit_price or order.estimated_price

        if price is None:
            return ValidationResult(is_valid=True)

        # Check min price
        if price < self.limits.min_price:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.PRICE_OUT_OF_RANGE,
                message=f"Price ${price} below minimum ${self.limits.min_price}",
                details={
                    "price": price,
                    "min_price": self.limits.min_price,
                },
            )

        # Check max price
        if price > self.limits.max_price:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.PRICE_OUT_OF_RANGE,
                message=f"Price ${price} above maximum ${self.limits.max_price}",
                details={
                    "price": price,
                    "max_price": self.limits.max_price,
                },
            )

        return ValidationResult(is_valid=True)


class RiskValidationEngine:
    """
    Central risk validation engine.

    Validates all orders before submission.
    Manages trading state and locked symbols.
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize risk engine.

        Args:
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()
        self._state = TradingState.ACTIVE
        self._locked_symbols: Set[str] = set()
        self._validators: List[IOrderValidator] = []
        self._order_timestamps: List[datetime] = []
        self._daily_pnl: Decimal = Decimal("0")
        self._callbacks: List[Callable[[Order, ValidationResult], None]] = []

        # Register default validators
        self._register_default_validators()

    def _register_default_validators(self) -> None:
        """Register default validators."""
        self._validators.append(PositionSizeValidator(self.limits))
        self._validators.append(OrderSizeValidator(self.limits))
        self._validators.append(ConcentrationValidator(self.limits))
        self._validators.append(PriceValidator(self.limits))

    @property
    def state(self) -> TradingState:
        """Current trading state."""
        return self._state

    def set_state(self, state: TradingState) -> None:
        """
        Set trading state.

        Args:
            state: New trading state
        """
        old_state = self._state
        self._state = state
        logger.info(f"Trading state changed: {old_state.value} -> {state.value}")

    def activate(self) -> None:
        """Set state to ACTIVE."""
        self.set_state(TradingState.ACTIVE)

    def reduce_only(self) -> None:
        """Set state to REDUCING (only allow position-reducing trades)."""
        self.set_state(TradingState.REDUCING)

    def halt(self) -> None:
        """Set state to HALTED (no trading)."""
        self.set_state(TradingState.HALTED)

    def lock_symbol(self, symbol: str) -> None:
        """Lock a symbol from trading."""
        self._locked_symbols.add(symbol)
        logger.info(f"Symbol locked: {symbol}")

    def unlock_symbol(self, symbol: str) -> None:
        """Unlock a symbol for trading."""
        self._locked_symbols.discard(symbol)
        logger.info(f"Symbol unlocked: {symbol}")

    def is_symbol_locked(self, symbol: str) -> bool:
        """Check if symbol is locked."""
        return symbol in self._locked_symbols

    def add_validator(self, validator: IOrderValidator) -> None:
        """Add a custom validator."""
        self._validators.append(validator)

    def on_validation(
        self,
        callback: Callable[[Order, ValidationResult], None],
    ) -> None:
        """Register callback for validation results."""
        self._callbacks.append(callback)

    def _check_rate_limit(self) -> ValidationResult:
        """Check order rate limit."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Clean old timestamps
        self._order_timestamps = [
            t for t in self._order_timestamps if t > cutoff
        ]

        if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.RATE_LIMIT_EXCEEDED,
                message=f"Rate limit exceeded ({self.limits.max_orders_per_minute}/min)",
            )

        return ValidationResult(is_valid=True)

    def _check_state(
        self,
        order: Order,
        positions: Dict[str, Position],
    ) -> ValidationResult:
        """Check trading state."""
        if self._state == TradingState.HALTED:
            return ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.TRADING_HALTED,
                message="Trading is halted",
            )

        if self._state == TradingState.REDUCING:
            # Only allow position-reducing trades
            position = positions.get(order.symbol)

            if position is None:
                # No position - reject opening trade
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=RejectionReason.REDUCING_ONLY,
                    message="Only position-reducing trades allowed",
                )

            is_reducing = (
                (position.is_long and not order.is_buy) or
                (not position.is_long and order.is_buy)
            )

            if not is_reducing:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=RejectionReason.REDUCING_ONLY,
                    message="Only position-reducing trades allowed",
                )

        return ValidationResult(is_valid=True)

    def validate(
        self,
        order: Order,
        positions: Dict[str, Position],
        account_value: Decimal,
        cash: Decimal,
    ) -> ValidationResult:
        """
        Validate an order against all risk rules.

        Args:
            order: Order to validate
            positions: Current positions
            account_value: Total account value
            cash: Available cash

        Returns:
            ValidationResult indicating if order is valid
        """
        # Check locked symbol
        if self.is_symbol_locked(order.symbol):
            result = ValidationResult(
                is_valid=False,
                rejection_reason=RejectionReason.SYMBOL_LOCKED,
                message=f"Symbol {order.symbol} is locked",
            )
            self._notify_callbacks(order, result)
            return result

        # Check trading state
        state_result = self._check_state(order, positions)
        if not state_result.is_valid:
            self._notify_callbacks(order, state_result)
            return state_result

        # Check rate limit
        rate_result = self._check_rate_limit()
        if not rate_result.is_valid:
            self._notify_callbacks(order, rate_result)
            return rate_result

        # Run all validators
        for validator in self._validators:
            result = validator.validate(order, positions, account_value, cash)
            if not result.is_valid:
                self._notify_callbacks(order, result)
                return result

        # All validations passed
        self._order_timestamps.append(datetime.now())
        result = ValidationResult(is_valid=True)
        self._notify_callbacks(order, result)
        return result

    def _notify_callbacks(
        self,
        order: Order,
        result: ValidationResult,
    ) -> None:
        """Notify all callbacks of validation result."""
        for callback in self._callbacks:
            try:
                callback(order, result)
            except Exception as e:
                logger.error(f"Validation callback error: {e}")

    def update_daily_pnl(self, pnl: Decimal) -> None:
        """Update daily P&L tracking."""
        self._daily_pnl = pnl

        # Check daily loss limit
        if pnl < -self.limits.max_daily_loss:
            logger.warning(
                f"Daily loss limit exceeded: ${pnl} < -${self.limits.max_daily_loss}"
            )
            self.reduce_only()

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L (call at start of day)."""
        self._daily_pnl = Decimal("0")
        if self._state == TradingState.REDUCING:
            self.activate()

    def get_status(self) -> Dict[str, Any]:
        """Get risk engine status."""
        return {
            "state": self._state.value,
            "locked_symbols": list(self._locked_symbols),
            "daily_pnl": str(self._daily_pnl),
            "orders_last_minute": len(self._order_timestamps),
            "validators": len(self._validators),
        }

