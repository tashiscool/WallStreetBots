"""
Position Adjustment - Averaging and Scaling into Positions.

Ported from freqtrade's position adjustment feature.
Allows scaling into positions (averaging down/up) based on
price movement and configurable rules.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AdjustmentType(Enum):
    """Type of position adjustment."""
    SCALE_IN = "scale_in"       # Add to position
    SCALE_OUT = "scale_out"     # Reduce position
    AVERAGE_DOWN = "average_down"  # Buy more at lower price
    AVERAGE_UP = "average_up"   # Buy more at higher price (momentum)
    PYRAMID = "pyramid"         # Add as position profits


class AdjustmentTrigger(Enum):
    """What triggers an adjustment."""
    PRICE_DROP = "price_drop"       # Price dropped X%
    PRICE_RISE = "price_rise"       # Price rose X%
    TIME_BASED = "time_based"       # After X time
    INDICATOR = "indicator"         # Based on indicator
    PROFIT_TARGET = "profit_target" # Based on unrealized profit
    LOSS_LIMIT = "loss_limit"       # Based on unrealized loss


@dataclass
class AdjustmentConfig:
    """Configuration for position adjustments."""
    enabled: bool = True

    # Scaling rules
    max_adjustments: int = 3  # Maximum additional entries
    adjustment_size_pct: float = 1.0  # Size relative to original (1.0 = same size)

    # Trigger thresholds
    price_drop_threshold: float = 0.05  # Trigger at 5% drop
    price_rise_threshold: float = 0.05  # Trigger at 5% rise
    time_between_adjustments: timedelta = timedelta(hours=4)

    # Position limits
    max_position_size: Optional[Decimal] = None  # Max total position value
    max_position_pct: float = 0.10  # Max 10% of portfolio

    # Risk management
    stop_loss_after_adjustment: Optional[float] = None  # New stop after adjustment
    reduce_on_profit: bool = False  # Take partial profits
    profit_reduction_pct: float = 0.50  # Reduce 50% at profit target

    # DCA settings
    dca_enabled: bool = False
    dca_interval: timedelta = timedelta(days=7)
    dca_amount: Optional[Decimal] = None


@dataclass
class AdjustmentOrder:
    """An order for adjusting a position."""
    symbol: str
    action: str  # "buy" or "sell"
    quantity: int
    adjustment_type: AdjustmentType
    trigger: AdjustmentTrigger
    entry_price: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    estimated_value: Optional[Decimal] = None
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PositionState:
    """Tracks state of a position for adjustment decisions."""
    symbol: str
    original_quantity: int
    current_quantity: int
    avg_entry_price: Decimal
    current_price: Decimal
    adjustments_made: int = 0
    last_adjustment_time: Optional[datetime] = None
    adjustment_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        return (self.current_price - self.avg_entry_price) * self.current_quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.avg_entry_price == 0:
            return 0.0
        return float((self.current_price - self.avg_entry_price) / self.avg_entry_price)

    @property
    def current_value(self) -> Decimal:
        """Calculate current position value."""
        return self.current_price * self.current_quantity


class PositionAdjuster:
    """
    Manages position adjustments (averaging, scaling, pyramiding).

    Usage:
        adjuster = PositionAdjuster(
            config=AdjustmentConfig(
                max_adjustments=3,
                price_drop_threshold=0.05,
            )
        )

        # Check if adjustment is needed
        order = adjuster.check_adjustment(
            position_state,
            portfolio_value=100000
        )

        if order:
            # Execute the adjustment
            await broker.submit_order(...)
    """

    def __init__(
        self,
        config: Optional[AdjustmentConfig] = None,
        broker=None,
    ):
        """
        Initialize position adjuster.

        Args:
            config: Adjustment configuration
            broker: Broker client for executing orders
        """
        self.config = config or AdjustmentConfig()
        self.broker = broker
        self._positions: Dict[str, PositionState] = {}

        # Custom adjustment rules
        self._custom_rules: List[Callable] = []

    def track_position(self, position: PositionState) -> None:
        """Start tracking a position for adjustments."""
        self._positions[position.symbol] = position

    def stop_tracking(self, symbol: str) -> None:
        """Stop tracking a position."""
        self._positions.pop(symbol, None)

    def update_price(self, symbol: str, current_price: Decimal) -> None:
        """Update current price for a tracked position."""
        if symbol in self._positions:
            self._positions[symbol].current_price = current_price

    def add_custom_rule(
        self,
        rule: Callable[[PositionState, Decimal], Optional[AdjustmentOrder]],
    ) -> None:
        """
        Add a custom adjustment rule.

        Rule receives: (position_state, portfolio_value) -> Optional[AdjustmentOrder]
        """
        self._custom_rules.append(rule)

    def check_adjustment(
        self,
        position: PositionState,
        portfolio_value: Decimal,
    ) -> Optional[AdjustmentOrder]:
        """
        Check if a position adjustment is needed.

        Args:
            position: Current position state
            portfolio_value: Total portfolio value

        Returns:
            AdjustmentOrder if adjustment needed, None otherwise
        """
        if not self.config.enabled:
            return None

        # Check if we've hit max adjustments
        if position.adjustments_made >= self.config.max_adjustments:
            return None

        # Check time between adjustments
        if position.last_adjustment_time:
            time_since = datetime.now() - position.last_adjustment_time
            if time_since < self.config.time_between_adjustments:
                return None

        # Check various triggers
        order = None

        # 1. Check for price drop (averaging down)
        order = self._check_average_down(position, portfolio_value)
        if order:
            return order

        # 2. Check for price rise (pyramiding/momentum)
        order = self._check_pyramid(position, portfolio_value)
        if order:
            return order

        # 3. Check for profit taking (scale out)
        order = self._check_profit_taking(position, portfolio_value)
        if order:
            return order

        # 4. Check DCA schedule
        order = self._check_dca(position, portfolio_value)
        if order:
            return order

        # 5. Apply custom rules
        for rule in self._custom_rules:
            try:
                order = rule(position, portfolio_value)
                if order:
                    return order
            except Exception as e:
                logger.error(f"Custom rule error: {e}")

        return None

    def _check_average_down(
        self,
        position: PositionState,
        portfolio_value: Decimal,
    ) -> Optional[AdjustmentOrder]:
        """Check if we should average down."""
        pnl_pct = position.unrealized_pnl_pct

        if pnl_pct <= -self.config.price_drop_threshold:
            # Price has dropped enough to average down
            order_size = self._calculate_adjustment_size(position, portfolio_value)

            if order_size > 0:
                return AdjustmentOrder(
                    symbol=position.symbol,
                    action="buy",
                    quantity=order_size,
                    adjustment_type=AdjustmentType.AVERAGE_DOWN,
                    trigger=AdjustmentTrigger.PRICE_DROP,
                    entry_price=position.avg_entry_price,
                    current_price=position.current_price,
                    estimated_value=position.current_price * order_size,
                    reason=f"Price dropped {abs(pnl_pct):.1%} from entry",
                )

        return None

    def _check_pyramid(
        self,
        position: PositionState,
        portfolio_value: Decimal,
    ) -> Optional[AdjustmentOrder]:
        """Check if we should pyramid (add on strength)."""
        pnl_pct = position.unrealized_pnl_pct

        if pnl_pct >= self.config.price_rise_threshold:
            # Position is profitable, consider adding
            order_size = self._calculate_adjustment_size(position, portfolio_value)

            # Pyramid with smaller sizes as price rises
            pyramid_factor = max(0.5, 1.0 - (position.adjustments_made * 0.25))
            order_size = int(order_size * pyramid_factor)

            if order_size > 0:
                return AdjustmentOrder(
                    symbol=position.symbol,
                    action="buy",
                    quantity=order_size,
                    adjustment_type=AdjustmentType.PYRAMID,
                    trigger=AdjustmentTrigger.PRICE_RISE,
                    entry_price=position.avg_entry_price,
                    current_price=position.current_price,
                    estimated_value=position.current_price * order_size,
                    reason=f"Price up {pnl_pct:.1%}, pyramiding",
                )

        return None

    def _check_profit_taking(
        self,
        position: PositionState,
        portfolio_value: Decimal,
    ) -> Optional[AdjustmentOrder]:
        """Check if we should take partial profits."""
        if not self.config.reduce_on_profit:
            return None

        pnl_pct = position.unrealized_pnl_pct

        # Take profits at 2x the rise threshold
        profit_threshold = self.config.price_rise_threshold * 2

        if pnl_pct >= profit_threshold:
            # Take partial profits
            reduction_qty = int(position.current_quantity * self.config.profit_reduction_pct)

            if reduction_qty > 0:
                return AdjustmentOrder(
                    symbol=position.symbol,
                    action="sell",
                    quantity=reduction_qty,
                    adjustment_type=AdjustmentType.SCALE_OUT,
                    trigger=AdjustmentTrigger.PROFIT_TARGET,
                    entry_price=position.avg_entry_price,
                    current_price=position.current_price,
                    estimated_value=position.current_price * reduction_qty,
                    reason=f"Taking {self.config.profit_reduction_pct:.0%} profits at {pnl_pct:.1%} gain",
                )

        return None

    def _check_dca(
        self,
        position: PositionState,
        portfolio_value: Decimal,
    ) -> Optional[AdjustmentOrder]:
        """Check if DCA buy is due."""
        if not self.config.dca_enabled:
            return None

        # Check if it's time for DCA
        last_adj = position.last_adjustment_time or datetime.min
        time_since = datetime.now() - last_adj

        if time_since >= self.config.dca_interval:
            # Calculate DCA amount
            if self.config.dca_amount:
                dca_value = self.config.dca_amount
            else:
                # Default to same as adjustment size
                dca_value = position.current_price * self._calculate_adjustment_size(
                    position, portfolio_value
                )

            if position.current_price > 0:
                order_size = int(dca_value / position.current_price)

                if order_size > 0:
                    return AdjustmentOrder(
                        symbol=position.symbol,
                        action="buy",
                        quantity=order_size,
                        adjustment_type=AdjustmentType.SCALE_IN,
                        trigger=AdjustmentTrigger.TIME_BASED,
                        current_price=position.current_price,
                        estimated_value=position.current_price * order_size,
                        reason="Scheduled DCA purchase",
                    )

        return None

    def _calculate_adjustment_size(
        self,
        position: PositionState,
        portfolio_value: Decimal,
    ) -> int:
        """Calculate the size of an adjustment order."""
        # Base size on original position * multiplier
        base_qty = int(position.original_quantity * self.config.adjustment_size_pct)

        # Check position limits
        if self.config.max_position_size:
            current_value = position.current_value
            remaining = self.config.max_position_size - current_value
            if remaining <= 0:
                return 0
            max_qty = int(remaining / position.current_price)
            base_qty = min(base_qty, max_qty)

        # Check portfolio percentage limit
        max_position_value = portfolio_value * Decimal(str(self.config.max_position_pct))
        current_value = position.current_value
        remaining = max_position_value - current_value

        if remaining <= 0:
            return 0

        max_qty_by_pct = int(remaining / position.current_price)
        base_qty = min(base_qty, max_qty_by_pct)

        return max(0, base_qty)

    def record_adjustment(
        self,
        symbol: str,
        adjustment: AdjustmentOrder,
        fill_price: Decimal,
        fill_quantity: int,
    ) -> None:
        """Record that an adjustment was made."""
        if symbol not in self._positions:
            return

        position = self._positions[symbol]
        position.adjustments_made += 1
        position.last_adjustment_time = datetime.now()

        # Update average entry price if buying
        if adjustment.action == "buy":
            total_cost = (
                position.avg_entry_price * position.current_quantity +
                fill_price * fill_quantity
            )
            new_total_qty = position.current_quantity + fill_quantity
            if new_total_qty > 0:
                position.avg_entry_price = total_cost / new_total_qty
            position.current_quantity = new_total_qty
        else:  # selling
            position.current_quantity -= fill_quantity

        # Record in history
        position.adjustment_history.append({
            "type": adjustment.adjustment_type.value,
            "trigger": adjustment.trigger.value,
            "action": adjustment.action,
            "quantity": fill_quantity,
            "price": float(fill_price),
            "timestamp": datetime.now().isoformat(),
        })

    async def execute_adjustment(
        self,
        adjustment: AdjustmentOrder,
        broker=None,
    ) -> bool:
        """
        Execute an adjustment order.

        Args:
            adjustment: The adjustment order to execute
            broker: Broker to use (defaults to self.broker)

        Returns:
            True if successful
        """
        broker = broker or self.broker
        if not broker:
            raise ValueError("No broker provided")

        try:
            result = await broker.submit_order(
                symbol=adjustment.symbol,
                qty=adjustment.quantity,
                side=adjustment.action,
                type="market",
            )

            if result:
                self.record_adjustment(
                    adjustment.symbol,
                    adjustment,
                    adjustment.current_price or Decimal("0"),
                    adjustment.quantity,
                )
                logger.info(
                    f"Executed {adjustment.adjustment_type.value}: "
                    f"{adjustment.action} {adjustment.quantity} {adjustment.symbol}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to execute adjustment: {e}")

        return False

    def get_adjustment_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get summary of adjustments for a position."""
        if symbol not in self._positions:
            return None

        position = self._positions[symbol]
        return {
            "symbol": symbol,
            "original_quantity": position.original_quantity,
            "current_quantity": position.current_quantity,
            "avg_entry_price": float(position.avg_entry_price),
            "current_price": float(position.current_price),
            "unrealized_pnl": float(position.unrealized_pnl),
            "unrealized_pnl_pct": position.unrealized_pnl_pct,
            "adjustments_made": position.adjustments_made,
            "max_adjustments": self.config.max_adjustments,
            "remaining_adjustments": self.config.max_adjustments - position.adjustments_made,
            "last_adjustment": position.last_adjustment_time,
            "history": position.adjustment_history,
        }


# Common adjustment strategies
class AdjustmentStrategies:
    """Pre-built adjustment strategy configurations."""

    @staticmethod
    def aggressive_averaging() -> AdjustmentConfig:
        """Aggressive averaging down strategy."""
        return AdjustmentConfig(
            max_adjustments=5,
            adjustment_size_pct=1.0,
            price_drop_threshold=0.03,  # Average at 3% drops
            time_between_adjustments=timedelta(hours=2),
        )

    @staticmethod
    def conservative_dca() -> AdjustmentConfig:
        """Conservative DCA strategy."""
        return AdjustmentConfig(
            max_adjustments=10,
            adjustment_size_pct=0.5,
            dca_enabled=True,
            dca_interval=timedelta(days=14),
            price_drop_threshold=0.10,  # Only average at 10% drops
        )

    @staticmethod
    def momentum_pyramid() -> AdjustmentConfig:
        """Momentum-based pyramiding strategy."""
        return AdjustmentConfig(
            max_adjustments=3,
            adjustment_size_pct=0.5,  # Decreasing sizes
            price_rise_threshold=0.05,  # Add at 5% gains
            price_drop_threshold=0.20,  # Only average at 20% drops
            reduce_on_profit=True,
            profit_reduction_pct=0.25,
        )

    @staticmethod
    def mean_reversion() -> AdjustmentConfig:
        """Mean reversion averaging strategy."""
        return AdjustmentConfig(
            max_adjustments=4,
            adjustment_size_pct=1.5,  # Larger as price drops
            price_drop_threshold=0.05,
            time_between_adjustments=timedelta(days=1),
            stop_loss_after_adjustment=-0.15,  # Tight stop after averaging
        )
