"""Immediate Execution Model"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..execution_model import ExecutionModel, Order, OrderType, OrderSide
from ..portfolio_target import PortfolioTarget


class ImmediateExecutionModel(ExecutionModel):
    """
    Execute all targets immediately with market orders.

    Simplest execution - submit market orders for all targets.
    """

    def __init__(self, name: str = "ImmediateExecution"):
        super().__init__(name)
        self._current_positions: Dict[str, Decimal] = {}

    def set_current_positions(self, positions: Dict[str, Decimal]) -> None:
        """Update current positions for delta calculation."""
        self._current_positions = positions

    def execute(
        self,
        targets: List[PortfolioTarget],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Order]:
        """Generate immediate market orders."""
        orders = []

        for target in targets:
            current_qty = self._current_positions.get(target.symbol, Decimal("0"))
            order = self.create_market_order(target, current_qty)

            if order and order.quantity > 0:
                orders.append(order)

        return orders
