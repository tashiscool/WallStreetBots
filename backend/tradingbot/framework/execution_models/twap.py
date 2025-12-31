"""TWAP Execution Model"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..execution_model import ExecutionModel, Order, OrderType, OrderSide, TimeInForce
from ..portfolio_target import PortfolioTarget


class TWAPExecutionModel(ExecutionModel):
    """
    Time-Weighted Average Price execution.

    Splits orders into equal slices over time.
    Simpler than VWAP, good for less liquid securities.
    """

    def __init__(
        self,
        duration_minutes: int = 30,
        num_slices: int = 10,
        randomize_timing: bool = True,  # Add slight randomness to avoid detection
        name: str = "TWAPExecution",
    ):
        super().__init__(name)
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
        self.randomize_timing = randomize_timing
        self._current_positions: Dict[str, Decimal] = {}

    def set_current_positions(self, positions: Dict[str, Decimal]) -> None:
        """Update current positions."""
        self._current_positions = positions

    def execute(
        self,
        targets: List[PortfolioTarget],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Order]:
        """Generate TWAP-sliced orders."""
        orders = []

        for target in targets:
            current_qty = self._current_positions.get(target.symbol, Decimal("0"))
            total_qty = self.calculate_order_quantity(target, current_qty)

            if total_qty == 0:
                continue

            side = self.calculate_order_side(target, current_qty)

            # Equal slice sizes
            slice_qty = (total_qty / Decimal(str(self.num_slices))).quantize(Decimal("1"))
            remainder = total_qty - (slice_qty * self.num_slices)

            for i in range(self.num_slices):
                qty = slice_qty
                # Add remainder to last slice
                if i == self.num_slices - 1:
                    qty += remainder

                if qty <= 0:
                    continue

                order = Order(
                    symbol=target.symbol,
                    side=side,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    source_target_id=target.id,
                    metadata={
                        'slice_index': i,
                        'total_slices': self.num_slices,
                        'algorithm': 'TWAP',
                    },
                )
                orders.append(order)

        # Return only first slice immediately
        return [o for o in orders if o.metadata.get('slice_index', 0) == 0]
