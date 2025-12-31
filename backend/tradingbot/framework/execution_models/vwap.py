"""VWAP Execution Model"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
import numpy as np

from ..execution_model import ExecutionModel, Order, OrderType, OrderSide, TimeInForce
from ..portfolio_target import PortfolioTarget


class VWAPExecutionModel(ExecutionModel):
    """
    Volume-Weighted Average Price execution.

    Splits orders over time weighted by typical volume profile.
    Aims to achieve execution at or better than VWAP.
    """

    def __init__(
        self,
        duration_minutes: int = 30,
        num_slices: int = 6,
        volume_participation_rate: float = 0.10,  # Max 10% of volume
        use_limit_orders: bool = True,
        name: str = "VWAPExecution",
    ):
        super().__init__(name)
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
        self.volume_participation_rate = volume_participation_rate
        self.use_limit_orders = use_limit_orders
        self._current_positions: Dict[str, Decimal] = {}
        self._scheduled_slices: List[Dict] = []

    def set_current_positions(self, positions: Dict[str, Decimal]) -> None:
        """Update current positions."""
        self._current_positions = positions

    def execute(
        self,
        targets: List[PortfolioTarget],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Order]:
        """Generate VWAP-sliced orders."""
        orders = []
        now = datetime.now()

        for target in targets:
            current_qty = self._current_positions.get(target.symbol, Decimal("0"))
            total_qty = self.calculate_order_quantity(target, current_qty)

            if total_qty == 0:
                continue

            side = self.calculate_order_side(target, current_qty)

            # Get volume profile
            volume_weights = self._get_volume_weights(
                target.symbol,
                market_data
            )

            # Calculate slice quantities
            slice_interval = timedelta(minutes=self.duration_minutes / self.num_slices)

            for i, weight in enumerate(volume_weights):
                slice_qty = (total_qty * Decimal(str(weight))).quantize(Decimal("1"))

                if slice_qty <= 0:
                    continue

                # Determine order type
                if self.use_limit_orders and market_data:
                    symbol_data = market_data.get(target.symbol, {})
                    current_price = Decimal(str(symbol_data.get('close', [100])[-1]))

                    # Set limit price slightly worse than current
                    # (buy slightly higher, sell slightly lower)
                    if side == OrderSide.BUY:
                        limit_price = current_price * Decimal("1.001")  # 0.1% above
                    else:
                        limit_price = current_price * Decimal("0.999")  # 0.1% below

                    order = Order(
                        symbol=target.symbol,
                        side=side,
                        quantity=slice_qty,
                        order_type=OrderType.LIMIT,
                        limit_price=limit_price.quantize(Decimal("0.01")),
                        time_in_force=TimeInForce.IOC,  # Immediate or cancel
                        source_target_id=target.id,
                        metadata={
                            'slice_index': i,
                            'total_slices': self.num_slices,
                            'algorithm': 'VWAP',
                        },
                    )
                else:
                    order = Order(
                        symbol=target.symbol,
                        side=side,
                        quantity=slice_qty,
                        order_type=OrderType.MARKET,
                        source_target_id=target.id,
                        metadata={
                            'slice_index': i,
                            'total_slices': self.num_slices,
                            'algorithm': 'VWAP',
                        },
                    )

                orders.append(order)

                # Schedule future slices
                if i > 0:
                    scheduled_time = now + (slice_interval * i)
                    self._scheduled_slices.append({
                        'order': order,
                        'scheduled_time': scheduled_time,
                    })

        # Return only first slice immediately
        # (scheduled slices would be handled by a scheduler)
        immediate_orders = [o for o in orders if o.metadata.get('slice_index', 0) == 0]
        return immediate_orders

    def _get_volume_weights(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]],
    ) -> List[float]:
        """Get volume-based weights for order slicing."""
        # Default: equal slices
        weights = [1.0 / self.num_slices] * self.num_slices

        if market_data and symbol in market_data:
            volumes = market_data[symbol].get('volume', [])
            if len(volumes) >= self.num_slices:
                # Use recent volume pattern
                recent_volumes = np.array(volumes[-self.num_slices:])
                total = recent_volumes.sum()
                if total > 0:
                    weights = (recent_volumes / total).tolist()

        return weights

    def get_scheduled_slices(self) -> List[Dict]:
        """Get pending scheduled slices."""
        return self._scheduled_slices.copy()
