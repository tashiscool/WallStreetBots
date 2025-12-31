"""Limit Order Execution Model"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..execution_model import ExecutionModel, Order, OrderType, OrderSide, TimeInForce
from ..portfolio_target import PortfolioTarget


class LimitOrderExecutionModel(ExecutionModel):
    """
    Execute with limit orders for price improvement.

    Uses spread analysis to place limits inside the spread.
    """

    def __init__(
        self,
        price_improvement_pct: float = 0.001,  # Try to improve by 0.1%
        time_in_force: TimeInForce = TimeInForce.DAY,
        fallback_to_market: bool = True,  # Use market if no price data
        name: str = "LimitOrderExecution",
    ):
        super().__init__(name)
        self.price_improvement_pct = price_improvement_pct
        self.time_in_force = time_in_force
        self.fallback_to_market = fallback_to_market
        self._current_positions: Dict[str, Decimal] = {}

    def set_current_positions(self, positions: Dict[str, Decimal]) -> None:
        """Update current positions."""
        self._current_positions = positions

    def execute(
        self,
        targets: List[PortfolioTarget],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Order]:
        """Generate limit orders with price improvement."""
        orders = []

        for target in targets:
            current_qty = self._current_positions.get(target.symbol, Decimal("0"))
            order_qty = self.calculate_order_quantity(target, current_qty)

            if order_qty == 0:
                continue

            side = self.calculate_order_side(target, current_qty)

            # Get pricing data
            symbol_data = market_data.get(target.symbol, {}) if market_data else {}
            bid = symbol_data.get('bid')
            ask = symbol_data.get('ask')
            last_price = symbol_data.get('close', [None])[-1] if 'close' in symbol_data else None

            # Calculate limit price
            limit_price = None

            if bid and ask:
                bid = Decimal(str(bid))
                ask = Decimal(str(ask))
                spread = ask - bid
                mid = (bid + ask) / 2

                # Place inside spread for better fill
                if side == OrderSide.BUY:
                    # Buy slightly above bid
                    limit_price = bid + (spread * Decimal("0.3"))
                else:
                    # Sell slightly below ask
                    limit_price = ask - (spread * Decimal("0.3"))

            elif last_price:
                last_price = Decimal(str(last_price))
                improvement = last_price * Decimal(str(self.price_improvement_pct))

                if side == OrderSide.BUY:
                    limit_price = last_price - improvement
                else:
                    limit_price = last_price + improvement

            # Create order
            if limit_price:
                order = Order(
                    symbol=target.symbol,
                    side=side,
                    quantity=order_qty,
                    order_type=OrderType.LIMIT,
                    limit_price=limit_price.quantize(Decimal("0.01")),
                    time_in_force=self.time_in_force,
                    source_target_id=target.id,
                )
            elif self.fallback_to_market:
                order = Order(
                    symbol=target.symbol,
                    side=side,
                    quantity=order_qty,
                    order_type=OrderType.MARKET,
                    source_target_id=target.id,
                    metadata={'fallback': 'no_price_data'},
                )
            else:
                continue

            orders.append(order)

        return orders
