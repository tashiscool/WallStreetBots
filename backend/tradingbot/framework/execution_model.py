"""
Execution Model Base Class

ExecutionModels determine how portfolio targets are executed as orders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

from .portfolio_target import PortfolioTarget

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good til cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    OPG = "opg"  # At the open
    CLS = "cls"  # At the close


@dataclass
class Order:
    """
    Order to be submitted to broker.

    Output from ExecutionModel.
    """
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY

    # Source tracking
    source_target_id: Optional[str] = None

    # Metadata
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    tag: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for broker submission."""
        result = {
            'symbol': self.symbol,
            'side': self.side.value,
            'qty': str(self.quantity),
            'type': self.order_type.value,
            'time_in_force': self.time_in_force.value,
        }

        if self.limit_price:
            result['limit_price'] = str(self.limit_price)
        if self.stop_price:
            result['stop_price'] = str(self.stop_price)

        return result

    def __repr__(self) -> str:
        price_info = ""
        if self.limit_price:
            price_info = f" @ {self.limit_price}"
        if self.stop_price:
            price_info += f" stop={self.stop_price}"
        return f"Order({self.side.value} {self.quantity} {self.symbol}{price_info})"


class ExecutionModel(ABC):
    """
    Base class for order execution logic.

    ExecutionModels take PortfolioTargets and determine how to execute
    them as orders (timing, order types, slicing, etc.).

    Override execute() to implement your execution logic.

    Example:
        class VWAPExecutionModel(ExecutionModel):
            def __init__(self, duration_minutes=30):
                super().__init__("VWAP")
                self.duration_minutes = duration_minutes

            def execute(self, targets, market_data):
                orders = []
                for target in targets:
                    # Split order into slices over time
                    slices = self._calculate_vwap_slices(
                        target, market_data, self.duration_minutes
                    )
                    orders.extend(slices)
                return orders
    """

    def __init__(self, name: str = "ExecutionModel"):
        """
        Initialize ExecutionModel.

        Args:
            name: Name of this model (for tracking)
        """
        self.name = name
        self._pending_orders: List[Order] = []
        self._executed_orders: List[Order] = []

    @abstractmethod
    def execute(
        self,
        targets: List[PortfolioTarget],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Order]:
        """
        Generate orders from portfolio targets.

        This is the main method to override in subclasses.

        Args:
            targets: List of portfolio targets to execute
            market_data: Optional market data for intelligent execution

        Returns:
            List of Order objects to submit
        """
        pass

    def on_order_filled(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal,
    ) -> None:
        """
        Called when an order is filled.

        Override to handle fill events.

        Args:
            order: The filled order
            fill_price: Execution price
            fill_quantity: Quantity filled
        """
        self._executed_orders.append(order)
        logger.debug(f"{self.name}: Order filled - {order} @ {fill_price}")

    def on_order_cancelled(self, order: Order) -> None:
        """
        Called when an order is cancelled.

        Args:
            order: The cancelled order
        """
        logger.debug(f"{self.name}: Order cancelled - {order}")

    def get_pending_orders(self) -> List[Order]:
        """Get list of pending orders."""
        return self._pending_orders.copy()

    def calculate_order_side(self, target: PortfolioTarget, current_quantity: Decimal) -> OrderSide:
        """
        Determine order side based on target and current position.

        Args:
            target: Portfolio target
            current_quantity: Current position quantity

        Returns:
            OrderSide.BUY or OrderSide.SELL
        """
        delta = target.quantity - current_quantity
        return OrderSide.BUY if delta > 0 else OrderSide.SELL

    def calculate_order_quantity(
        self,
        target: PortfolioTarget,
        current_quantity: Decimal
    ) -> Decimal:
        """
        Calculate order quantity to reach target.

        Args:
            target: Portfolio target
            current_quantity: Current position quantity

        Returns:
            Absolute quantity to order
        """
        return abs(target.quantity - current_quantity)

    def create_market_order(
        self,
        target: PortfolioTarget,
        current_quantity: Decimal = Decimal("0"),
    ) -> Optional[Order]:
        """
        Create a market order from target.

        Args:
            target: Portfolio target
            current_quantity: Current position quantity

        Returns:
            Order object or None if no order needed
        """
        order_quantity = self.calculate_order_quantity(target, current_quantity)
        if order_quantity == 0:
            return None

        return Order(
            symbol=target.symbol,
            side=self.calculate_order_side(target, current_quantity),
            quantity=order_quantity,
            order_type=OrderType.MARKET,
            source_target_id=target.id,
        )

    def create_limit_order(
        self,
        target: PortfolioTarget,
        limit_price: Decimal,
        current_quantity: Decimal = Decimal("0"),
    ) -> Optional[Order]:
        """
        Create a limit order from target.

        Args:
            target: Portfolio target
            limit_price: Limit price
            current_quantity: Current position quantity

        Returns:
            Order object or None if no order needed
        """
        order_quantity = self.calculate_order_quantity(target, current_quantity)
        if order_quantity == 0:
            return None

        return Order(
            symbol=target.symbol,
            side=self.calculate_order_side(target, current_quantity),
            quantity=order_quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            source_target_id=target.id,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'name': self.name,
            'pending_orders': len(self._pending_orders),
            'executed_orders': len(self._executed_orders),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
