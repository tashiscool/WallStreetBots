"""
Margin Call Simulation.

Ported from QuantConnect/LEAN's margin call model.
Simulates broker margin calls and automatic liquidation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MarginCallType(Enum):
    """Type of margin call."""
    MAINTENANCE_CALL = "maintenance_call"  # Below maintenance margin
    REGULATION_T_CALL = "regulation_t_call"  # Initial margin violation
    DAY_TRADE_CALL = "day_trade_call"  # PDT margin call
    EXCHANGE_CALL = "exchange_call"  # Exchange-specific call


class MarginCallStatus(Enum):
    """Status of margin call."""
    ACTIVE = "active"
    MET = "met"  # Deposit made
    LIQUIDATED = "liquidated"  # Position liquidated
    EXPIRED = "expired"


class LiquidationPriority(Enum):
    """Priority for liquidation order."""
    HIGHEST_LOSS = "highest_loss"  # Liquidate biggest losers first
    HIGHEST_MARGIN = "highest_margin"  # Liquidate highest margin first
    MOST_LIQUID = "most_liquid"  # Liquidate most liquid first
    FIFO = "fifo"  # First in, first out


@dataclass
class Position:
    """Position for margin call calculation."""
    symbol: str
    quantity: int
    average_price: Decimal
    current_price: Decimal
    margin_requirement: Decimal
    sector: Optional[str] = None
    open_date: Optional[datetime] = None
    is_option: bool = False

    @property
    def market_value(self) -> Decimal:
        """Current market value."""
        return abs(self.quantity) * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized P&L."""
        return (self.current_price - self.average_price) * self.quantity

    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.quantity > 0


@dataclass
class MarginCall:
    """Represents a margin call."""
    call_type: MarginCallType
    call_amount: Decimal
    call_date: datetime
    due_date: datetime
    status: MarginCallStatus = MarginCallStatus.ACTIVE
    positions_at_risk: List[str] = field(default_factory=list)
    amount_met: Decimal = Decimal("0")
    liquidated_positions: List[str] = field(default_factory=list)

    @property
    def remaining_amount(self) -> Decimal:
        """Amount still needed."""
        return max(Decimal("0"), self.call_amount - self.amount_met)

    @property
    def is_overdue(self) -> bool:
        """True if past due date."""
        return datetime.now() > self.due_date and self.status == MarginCallStatus.ACTIVE


@dataclass
class LiquidationOrder:
    """Order to liquidate a position."""
    symbol: str
    quantity: int
    reason: str
    expected_proceeds: Decimal
    priority: int = 0


@dataclass
class MarginCallResult:
    """Result of margin call processing."""
    margin_call: Optional[MarginCall]
    is_margin_call: bool
    current_margin: Decimal
    required_margin: Decimal
    excess_margin: Decimal
    margin_ratio: Decimal
    liquidation_orders: List[LiquidationOrder] = field(default_factory=list)


class MarginCallModel:
    """
    Simulates margin calls and automatic liquidation.

    Monitors margin requirements and initiates liquidation
    when maintenance margin is breached.
    """

    # Standard margin thresholds
    INITIAL_MARGIN_RATIO = Decimal("0.50")      # 50% initial (Reg T)
    MAINTENANCE_MARGIN_RATIO = Decimal("0.25")  # 25% maintenance
    MARGIN_CALL_BUFFER = Decimal("0.05")        # 5% buffer before call
    LIQUIDATION_THRESHOLD = Decimal("0.20")     # 20% triggers liquidation

    # Timing
    MARGIN_CALL_DAYS = 5  # Days to meet margin call

    def __init__(
        self,
        initial_margin_ratio: Optional[Decimal] = None,
        maintenance_margin_ratio: Optional[Decimal] = None,
        auto_liquidate: bool = True,
        liquidation_priority: LiquidationPriority = LiquidationPriority.HIGHEST_LOSS,
    ):
        """
        Initialize margin call model.

        Args:
            initial_margin_ratio: Initial margin requirement
            maintenance_margin_ratio: Maintenance margin requirement
            auto_liquidate: Automatically liquidate positions
            liquidation_priority: Order for liquidation
        """
        self.initial_margin_ratio = (
            initial_margin_ratio or self.INITIAL_MARGIN_RATIO
        )
        self.maintenance_margin_ratio = (
            maintenance_margin_ratio or self.MAINTENANCE_MARGIN_RATIO
        )
        self.auto_liquidate = auto_liquidate
        self.liquidation_priority = liquidation_priority

        self._active_calls: List[MarginCall] = []
        self._call_history: List[MarginCall] = []
        self._callbacks: List[Callable[[MarginCall], None]] = []

    def on_margin_call(
        self,
        callback: Callable[[MarginCall], None],
    ) -> None:
        """Register callback for margin calls."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, margin_call: MarginCall) -> None:
        """Notify all callbacks of margin call."""
        for callback in self._callbacks:
            try:
                callback(margin_call)
            except Exception as e:
                logger.error(f"Margin call callback error: {e}")

    def calculate_margin_status(
        self,
        positions: Dict[str, Position],
        cash: Decimal,
        account_value: Decimal,
    ) -> MarginCallResult:
        """
        Calculate current margin status.

        Args:
            positions: Current positions
            cash: Available cash
            account_value: Total account value

        Returns:
            MarginCallResult with margin status
        """
        # Calculate total margin requirement
        total_margin_required = sum(
            pos.margin_requirement for pos in positions.values()
        )

        # Calculate total exposure
        total_exposure = sum(
            pos.market_value for pos in positions.values()
        )

        # Current margin = equity / exposure
        if total_exposure > 0:
            current_margin = account_value / total_exposure
        else:
            current_margin = Decimal("1.0")  # No positions

        # Excess margin
        excess_margin = account_value - total_margin_required

        # Check for margin call
        is_margin_call = excess_margin < 0

        margin_call = None
        liquidation_orders = []

        if is_margin_call:
            call_amount = abs(excess_margin)

            # Create margin call
            margin_call = MarginCall(
                call_type=MarginCallType.MAINTENANCE_CALL,
                call_amount=call_amount,
                call_date=datetime.now(),
                due_date=datetime.now() + timedelta(days=self.MARGIN_CALL_DAYS),
                positions_at_risk=[
                    symbol for symbol, pos in positions.items()
                    if pos.margin_requirement > 0
                ],
            )

            self._active_calls.append(margin_call)
            self._notify_callbacks(margin_call)

            logger.warning(
                f"Margin call initiated: ${call_amount} required by "
                f"{margin_call.due_date}"
            )

            # Check if liquidation needed
            if current_margin < self.LIQUIDATION_THRESHOLD and self.auto_liquidate:
                liquidation_orders = self._generate_liquidation_orders(
                    positions, call_amount
                )

        return MarginCallResult(
            margin_call=margin_call,
            is_margin_call=is_margin_call,
            current_margin=current_margin,
            required_margin=total_margin_required,
            excess_margin=excess_margin,
            margin_ratio=current_margin,
            liquidation_orders=liquidation_orders,
        )

    def _generate_liquidation_orders(
        self,
        positions: Dict[str, Position],
        amount_needed: Decimal,
    ) -> List[LiquidationOrder]:
        """
        Generate liquidation orders to meet margin call.

        Args:
            positions: Current positions
            amount_needed: Amount to raise

        Returns:
            List of liquidation orders
        """
        orders = []
        amount_raised = Decimal("0")

        # Sort positions by liquidation priority
        sorted_positions = self._sort_for_liquidation(positions)

        for symbol, pos in sorted_positions:
            if amount_raised >= amount_needed:
                break

            # Calculate how much this position can contribute
            position_value = pos.market_value

            # Determine quantity to liquidate
            if position_value <= amount_needed - amount_raised:
                # Liquidate entire position
                liquidate_qty = abs(pos.quantity)
            else:
                # Partial liquidation
                needed = amount_needed - amount_raised
                liquidate_qty = int(
                    (needed / pos.current_price) + 1  # Round up
                )
                liquidate_qty = min(liquidate_qty, abs(pos.quantity))

            # Adjust for position direction
            if pos.is_long:
                liquidate_qty = -liquidate_qty  # Sell

            expected_proceeds = abs(liquidate_qty) * pos.current_price

            orders.append(LiquidationOrder(
                symbol=symbol,
                quantity=liquidate_qty,
                reason="margin_call",
                expected_proceeds=expected_proceeds,
                priority=len(orders),
            ))

            amount_raised += expected_proceeds
            logger.info(
                f"Liquidation order: {symbol} x{liquidate_qty} "
                f"(expected ${expected_proceeds})"
            )

        return orders

    def _sort_for_liquidation(
        self,
        positions: Dict[str, Position],
    ) -> List[Tuple[str, Position]]:
        """Sort positions for liquidation order."""
        items = list(positions.items())

        if self.liquidation_priority == LiquidationPriority.HIGHEST_LOSS:
            # Liquidate biggest losers first
            items.sort(key=lambda x: x[1].unrealized_pnl)

        elif self.liquidation_priority == LiquidationPriority.HIGHEST_MARGIN:
            # Liquidate highest margin first
            items.sort(
                key=lambda x: x[1].margin_requirement,
                reverse=True
            )

        elif self.liquidation_priority == LiquidationPriority.MOST_LIQUID:
            # Liquidate most liquid (highest value) first
            items.sort(
                key=lambda x: x[1].market_value,
                reverse=True
            )

        elif self.liquidation_priority == LiquidationPriority.FIFO:
            # Liquidate oldest first
            items.sort(
                key=lambda x: x[1].open_date or datetime.max
            )

        return items

    def deposit_funds(
        self,
        amount: Decimal,
    ) -> List[MarginCall]:
        """
        Record deposit to meet margin calls.

        Args:
            amount: Deposit amount

        Returns:
            List of margin calls that were met
        """
        met_calls = []
        remaining = amount

        for call in self._active_calls:
            if remaining <= 0:
                break

            if call.status != MarginCallStatus.ACTIVE:
                continue

            apply_amount = min(remaining, call.remaining_amount)
            call.amount_met += apply_amount
            remaining -= apply_amount

            if call.remaining_amount <= 0:
                call.status = MarginCallStatus.MET
                met_calls.append(call)
                logger.info(f"Margin call met: {call.call_type.value}")

        return met_calls

    def record_liquidation(
        self,
        symbol: str,
        proceeds: Decimal,
    ) -> None:
        """
        Record that a position was liquidated.

        Args:
            symbol: Liquidated symbol
            proceeds: Proceeds from liquidation
        """
        for call in self._active_calls:
            if call.status != MarginCallStatus.ACTIVE:
                continue

            if symbol in call.positions_at_risk:
                call.liquidated_positions.append(symbol)
                call.amount_met += proceeds

                if call.remaining_amount <= 0:
                    call.status = MarginCallStatus.LIQUIDATED
                    logger.info(
                        f"Margin call met via liquidation: {call.call_type.value}"
                    )

    def process_expired_calls(self) -> List[MarginCall]:
        """
        Process any overdue margin calls.

        Returns:
            List of expired calls
        """
        expired = []

        for call in self._active_calls:
            if call.is_overdue:
                call.status = MarginCallStatus.EXPIRED
                expired.append(call)
                logger.warning(
                    f"Margin call expired: {call.call_type.value} "
                    f"(${call.remaining_amount} remaining)"
                )

        return expired

    def get_active_calls(self) -> List[MarginCall]:
        """Get all active margin calls."""
        return [c for c in self._active_calls if c.status == MarginCallStatus.ACTIVE]

    def get_total_call_amount(self) -> Decimal:
        """Get total amount needed for all active calls."""
        return sum(c.remaining_amount for c in self.get_active_calls())

    def clear_met_calls(self) -> None:
        """Move met calls to history."""
        met = [c for c in self._active_calls if c.status != MarginCallStatus.ACTIVE]
        self._call_history.extend(met)
        self._active_calls = [
            c for c in self._active_calls if c.status == MarginCallStatus.ACTIVE
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get margin call status summary."""
        active = self.get_active_calls()

        return {
            "active_calls": len(active),
            "total_call_amount": str(self.get_total_call_amount()),
            "calls": [
                {
                    "type": c.call_type.value,
                    "amount": str(c.remaining_amount),
                    "due_date": c.due_date.isoformat(),
                    "positions_at_risk": c.positions_at_risk,
                }
                for c in active
            ],
        }

