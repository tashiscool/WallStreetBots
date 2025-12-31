"""
Order State Machine

Explicit state machine for order lifecycle management.
Inspired by nautilus_trader's robust order state handling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status states."""
    # Initial states
    INITIALIZED = "initialized"     # Order created but not submitted
    DENIED = "denied"               # Denied by risk/pre-trade check
    EMULATED = "emulated"           # Client-side emulated (stop/limit)

    # Submitted states
    SUBMITTED = "submitted"         # Sent to broker
    PENDING_UPDATE = "pending_update"  # Update request pending
    PENDING_CANCEL = "pending_cancel"  # Cancel request pending

    # Broker acknowledged
    ACCEPTED = "accepted"           # Broker acknowledged
    REJECTED = "rejected"           # Broker rejected
    TRIGGERED = "triggered"         # Stop/limit triggered

    # Fill states
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"

    # Terminal states
    CANCELED = "canceled"
    EXPIRED = "expired"
    ERROR = "error"


class OrderTransitionError(Exception):
    """Invalid order state transition."""
    pass


# Explicit state transition table
# (current_status, target_status) -> allowed_status
ORDER_STATE_TABLE: Dict[Tuple[OrderStatus, OrderStatus], OrderStatus] = {
    # From INITIALIZED
    (OrderStatus.INITIALIZED, OrderStatus.DENIED): OrderStatus.DENIED,
    (OrderStatus.INITIALIZED, OrderStatus.EMULATED): OrderStatus.EMULATED,
    (OrderStatus.INITIALIZED, OrderStatus.SUBMITTED): OrderStatus.SUBMITTED,
    (OrderStatus.INITIALIZED, OrderStatus.REJECTED): OrderStatus.REJECTED,
    (OrderStatus.INITIALIZED, OrderStatus.ACCEPTED): OrderStatus.ACCEPTED,
    (OrderStatus.INITIALIZED, OrderStatus.CANCELED): OrderStatus.CANCELED,
    (OrderStatus.INITIALIZED, OrderStatus.EXPIRED): OrderStatus.EXPIRED,
    (OrderStatus.INITIALIZED, OrderStatus.TRIGGERED): OrderStatus.TRIGGERED,

    # From EMULATED (client-side orders)
    (OrderStatus.EMULATED, OrderStatus.CANCELED): OrderStatus.CANCELED,
    (OrderStatus.EMULATED, OrderStatus.EXPIRED): OrderStatus.EXPIRED,
    (OrderStatus.EMULATED, OrderStatus.SUBMITTED): OrderStatus.SUBMITTED,  # Released
    (OrderStatus.EMULATED, OrderStatus.TRIGGERED): OrderStatus.TRIGGERED,

    # From SUBMITTED
    (OrderStatus.SUBMITTED, OrderStatus.PENDING_UPDATE): OrderStatus.PENDING_UPDATE,
    (OrderStatus.SUBMITTED, OrderStatus.PENDING_CANCEL): OrderStatus.PENDING_CANCEL,
    (OrderStatus.SUBMITTED, OrderStatus.REJECTED): OrderStatus.REJECTED,
    (OrderStatus.SUBMITTED, OrderStatus.CANCELED): OrderStatus.CANCELED,
    (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED): OrderStatus.ACCEPTED,
    (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED): OrderStatus.PARTIALLY_FILLED,
    (OrderStatus.SUBMITTED, OrderStatus.FILLED): OrderStatus.FILLED,
    (OrderStatus.SUBMITTED, OrderStatus.EXPIRED): OrderStatus.EXPIRED,

    # From ACCEPTED
    (OrderStatus.ACCEPTED, OrderStatus.PENDING_UPDATE): OrderStatus.PENDING_UPDATE,
    (OrderStatus.ACCEPTED, OrderStatus.PENDING_CANCEL): OrderStatus.PENDING_CANCEL,
    (OrderStatus.ACCEPTED, OrderStatus.CANCELED): OrderStatus.CANCELED,
    (OrderStatus.ACCEPTED, OrderStatus.TRIGGERED): OrderStatus.TRIGGERED,
    (OrderStatus.ACCEPTED, OrderStatus.EXPIRED): OrderStatus.EXPIRED,
    (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED): OrderStatus.PARTIALLY_FILLED,
    (OrderStatus.ACCEPTED, OrderStatus.FILLED): OrderStatus.FILLED,

    # From PENDING_UPDATE
    (OrderStatus.PENDING_UPDATE, OrderStatus.ACCEPTED): OrderStatus.ACCEPTED,
    (OrderStatus.PENDING_UPDATE, OrderStatus.PENDING_CANCEL): OrderStatus.PENDING_CANCEL,
    (OrderStatus.PENDING_UPDATE, OrderStatus.CANCELED): OrderStatus.CANCELED,
    (OrderStatus.PENDING_UPDATE, OrderStatus.TRIGGERED): OrderStatus.TRIGGERED,
    (OrderStatus.PENDING_UPDATE, OrderStatus.EXPIRED): OrderStatus.EXPIRED,
    (OrderStatus.PENDING_UPDATE, OrderStatus.PARTIALLY_FILLED): OrderStatus.PARTIALLY_FILLED,
    (OrderStatus.PENDING_UPDATE, OrderStatus.FILLED): OrderStatus.FILLED,

    # From PENDING_CANCEL - Race conditions can cause fills during cancel!
    (OrderStatus.PENDING_CANCEL, OrderStatus.CANCELED): OrderStatus.CANCELED,
    (OrderStatus.PENDING_CANCEL, OrderStatus.ACCEPTED): OrderStatus.ACCEPTED,
    (OrderStatus.PENDING_CANCEL, OrderStatus.FILLED): OrderStatus.FILLED,  # Race condition
    (OrderStatus.PENDING_CANCEL, OrderStatus.PARTIALLY_FILLED): OrderStatus.PARTIALLY_FILLED,
    (OrderStatus.PENDING_CANCEL, OrderStatus.EXPIRED): OrderStatus.EXPIRED,

    # From TRIGGERED (stop orders)
    (OrderStatus.TRIGGERED, OrderStatus.PENDING_CANCEL): OrderStatus.PENDING_CANCEL,
    (OrderStatus.TRIGGERED, OrderStatus.REJECTED): OrderStatus.REJECTED,
    (OrderStatus.TRIGGERED, OrderStatus.CANCELED): OrderStatus.CANCELED,
    (OrderStatus.TRIGGERED, OrderStatus.EXPIRED): OrderStatus.EXPIRED,
    (OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED): OrderStatus.PARTIALLY_FILLED,
    (OrderStatus.TRIGGERED, OrderStatus.FILLED): OrderStatus.FILLED,

    # From PARTIALLY_FILLED
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_CANCEL): OrderStatus.PENDING_CANCEL,
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELED): OrderStatus.CANCELED,
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.PARTIALLY_FILLED): OrderStatus.PARTIALLY_FILLED,
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED): OrderStatus.FILLED,
    (OrderStatus.PARTIALLY_FILLED, OrderStatus.EXPIRED): OrderStatus.EXPIRED,
}


@dataclass
class OrderEvent:
    """Event in order lifecycle."""
    timestamp: datetime
    status: OrderStatus
    message: str = ""
    filled_qty: Optional[Decimal] = None
    fill_price: Optional[Decimal] = None
    broker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderStateMachine:
    """
    State machine for order lifecycle management.

    Ensures valid state transitions and maintains event history.

    Example:
        order = OrderStateMachine(order_id="12345")
        order.transition(OrderStatus.SUBMITTED)
        order.transition(OrderStatus.ACCEPTED)
        order.add_fill(Decimal("50"), Decimal("100.00"))
        order.transition(OrderStatus.PARTIALLY_FILLED)
    """
    order_id: str
    initial_status: OrderStatus = OrderStatus.INITIALIZED

    # Filled after __post_init__
    _status: OrderStatus = field(init=False)
    _history: List[OrderEvent] = field(default_factory=list)
    _filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    _total_quantity: Optional[Decimal] = None
    _avg_fill_price: Optional[Decimal] = None
    _broker_order_id: Optional[str] = None

    def __post_init__(self):
        self._status = self.initial_status
        self._history = [OrderEvent(
            timestamp=datetime.now(),
            status=self.initial_status,
            message="Order initialized",
        )]

    @property
    def status(self) -> OrderStatus:
        """Current order status."""
        return self._status

    @property
    def filled_quantity(self) -> Decimal:
        """Total filled quantity."""
        return self._filled_quantity

    @property
    def avg_fill_price(self) -> Optional[Decimal]:
        """Volume-weighted average fill price."""
        return self._avg_fill_price

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self._status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.DENIED,
            OrderStatus.ERROR,
        )

    @property
    def is_active(self) -> bool:
        """Check if order is active (can receive fills)."""
        return self._status in (
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PENDING_UPDATE,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.TRIGGERED,
            OrderStatus.PARTIALLY_FILLED,
        )

    @property
    def is_pending_action(self) -> bool:
        """Check if order has pending action."""
        return self._status in (
            OrderStatus.PENDING_UPDATE,
            OrderStatus.PENDING_CANCEL,
        )

    def can_transition(self, new_status: OrderStatus) -> bool:
        """Check if transition is valid without performing it."""
        return (self._status, new_status) in ORDER_STATE_TABLE

    def transition(
        self,
        new_status: OrderStatus,
        message: str = "",
        broker_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrderStatus:
        """
        Transition to a new status.

        Args:
            new_status: Target status
            message: Event message
            broker_id: Broker-assigned order ID
            metadata: Additional event data

        Returns:
            New status

        Raises:
            OrderTransitionError: If transition is invalid
        """
        key = (self._status, new_status)

        if key not in ORDER_STATE_TABLE:
            raise OrderTransitionError(
                f"Invalid transition: {self._status.value} -> {new_status.value} "
                f"for order {self.order_id}"
            )

        old_status = self._status
        self._status = ORDER_STATE_TABLE[key]

        if broker_id:
            self._broker_order_id = broker_id

        event = OrderEvent(
            timestamp=datetime.now(),
            status=self._status,
            message=message or f"Transition: {old_status.value} -> {self._status.value}",
            broker_id=broker_id,
            metadata=metadata or {},
        )
        self._history.append(event)

        logger.debug(
            f"Order {self.order_id}: {old_status.value} -> {self._status.value}"
        )

        return self._status

    def add_fill(
        self,
        quantity: Decimal,
        price: Decimal,
        message: str = "",
    ) -> None:
        """
        Record a fill event.

        Args:
            quantity: Filled quantity
            price: Fill price
            message: Fill message
        """
        # Update average fill price
        if self._avg_fill_price is None:
            self._avg_fill_price = price
        else:
            total_value = (
                self._avg_fill_price * self._filled_quantity +
                price * quantity
            )
            self._avg_fill_price = total_value / (self._filled_quantity + quantity)

        self._filled_quantity += quantity

        event = OrderEvent(
            timestamp=datetime.now(),
            status=self._status,
            message=message or f"Filled {quantity} @ {price}",
            filled_qty=quantity,
            fill_price=price,
        )
        self._history.append(event)

        # Auto-transition to FILLED if complete
        if self._total_quantity and self._filled_quantity >= self._total_quantity:
            if self.can_transition(OrderStatus.FILLED):
                self.transition(OrderStatus.FILLED, "Order fully filled")

    def set_total_quantity(self, quantity: Decimal) -> None:
        """Set total order quantity for fill tracking."""
        self._total_quantity = quantity

    def get_history(self) -> List[OrderEvent]:
        """Get full event history."""
        return self._history.copy()

    def get_last_event(self) -> Optional[OrderEvent]:
        """Get most recent event."""
        return self._history[-1] if self._history else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'order_id': self.order_id,
            'status': self._status.value,
            'filled_quantity': str(self._filled_quantity),
            'avg_fill_price': str(self._avg_fill_price) if self._avg_fill_price else None,
            'broker_order_id': self._broker_order_id,
            'is_terminal': self.is_terminal,
            'history': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'status': e.status.value,
                    'message': e.message,
                    'filled_qty': str(e.filled_qty) if e.filled_qty else None,
                    'fill_price': str(e.fill_price) if e.fill_price else None,
                }
                for e in self._history
            ],
        }

    def __repr__(self) -> str:
        return (
            f"OrderStateMachine(id={self.order_id}, "
            f"status={self._status.value}, "
            f"filled={self._filled_quantity})"
        )


class OrderManager:
    """
    Manages multiple orders with state machines.

    Provides order tracking, lookup, and lifecycle management.
    """

    def __init__(self):
        self._orders: Dict[str, OrderStateMachine] = {}
        self._by_broker_id: Dict[str, str] = {}  # broker_id -> order_id

    def create_order(
        self,
        order_id: str,
        quantity: Decimal,
        initial_status: OrderStatus = OrderStatus.INITIALIZED,
    ) -> OrderStateMachine:
        """Create and track a new order."""
        order = OrderStateMachine(
            order_id=order_id,
            initial_status=initial_status,
        )
        order.set_total_quantity(quantity)
        self._orders[order_id] = order
        return order

    def get_order(self, order_id: str) -> Optional[OrderStateMachine]:
        """Get order by internal ID."""
        return self._orders.get(order_id)

    def get_by_broker_id(self, broker_id: str) -> Optional[OrderStateMachine]:
        """Get order by broker-assigned ID."""
        order_id = self._by_broker_id.get(broker_id)
        if order_id:
            return self._orders.get(order_id)
        return None

    def link_broker_id(self, order_id: str, broker_id: str) -> None:
        """Link broker ID to internal order ID."""
        self._by_broker_id[broker_id] = order_id
        order = self._orders.get(order_id)
        if order:
            order._broker_order_id = broker_id

    def get_active_orders(self) -> List[OrderStateMachine]:
        """Get all active (non-terminal) orders."""
        return [o for o in self._orders.values() if o.is_active]

    def get_pending_orders(self) -> List[OrderStateMachine]:
        """Get orders with pending actions."""
        return [o for o in self._orders.values() if o.is_pending_action]

    def cleanup_terminal(self, max_age_hours: int = 24) -> int:
        """Remove old terminal orders. Returns count removed."""
        cutoff = datetime.now()
        to_remove = []

        for order_id, order in self._orders.items():
            if order.is_terminal:
                last_event = order.get_last_event()
                if last_event:
                    age = (cutoff - last_event.timestamp).total_seconds() / 3600
                    if age > max_age_hours:
                        to_remove.append(order_id)

        for order_id in to_remove:
            order = self._orders.pop(order_id)
            if order._broker_order_id:
                self._by_broker_id.pop(order._broker_order_id, None)

        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get order manager statistics."""
        status_counts = {}
        for order in self._orders.values():
            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'total_orders': len(self._orders),
            'active_orders': len(self.get_active_orders()),
            'pending_orders': len(self.get_pending_orders()),
            'status_breakdown': status_counts,
        }
