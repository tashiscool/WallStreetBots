"""
Multi-Leg Execution Engine - Inspired by Polymarket-Kalshi Arbitrage Bot.

Executes multi-leg orders concurrently with:
- Concurrent leg submission
- Partial fill handling
- Automatic position reconciliation
- Exposure management for mismatched fills

Concepts from: https://github.com/terauss/Polymarket-Kalshi-Arbitrage-bot
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import asyncio
import threading
import logging
import uuid

logger = logging.getLogger(__name__)


class LegSide(Enum):
    """Side of a leg."""
    BUY = "buy"
    SELL = "sell"


class LegStatus(Enum):
    """Status of a leg."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OrderLeg:
    """Represents one leg of a multi-leg order."""
    leg_id: str
    symbol: str
    side: LegSide
    quantity: int
    price: Optional[float] = None  # Limit price, None for market
    platform: str = "default"

    # Execution state
    status: LegStatus = LegStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    commission: float = 0.0
    order_id: Optional[str] = None
    error: Optional[str] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    @property
    def is_complete(self) -> bool:
        return self.status in (LegStatus.FILLED, LegStatus.FAILED, LegStatus.CANCELLED)

    @property
    def unfilled_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def fill_rate(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity


@dataclass
class MultiLegOrder:
    """A multi-leg order (spread, arbitrage, etc.)."""
    order_id: str
    legs: List[OrderLeg]
    strategy: str = ""  # e.g., "arbitrage", "spread", "pairs"
    created_at: datetime = field(default_factory=datetime.now)

    # Execution state
    is_complete: bool = False
    matched_quantity: int = 0
    total_cost: float = 0.0
    total_pnl: float = 0.0
    reconciled: bool = False

    def get_leg(self, leg_id: str) -> Optional[OrderLeg]:
        for leg in self.legs:
            if leg.leg_id == leg_id:
                return leg
        return None

    @property
    def all_legs_complete(self) -> bool:
        return all(leg.is_complete for leg in self.legs)

    @property
    def any_leg_failed(self) -> bool:
        return any(leg.status == LegStatus.FAILED for leg in self.legs)

    @property
    def min_filled(self) -> int:
        """Minimum filled quantity across all legs (matched quantity)."""
        if not self.legs:
            return 0
        return min(leg.filled_quantity for leg in self.legs)


@dataclass
class ExecutionResult:
    """Result of multi-leg execution."""
    order_id: str
    success: bool
    matched_quantity: int
    legs: List[OrderLeg]
    total_cost: float
    estimated_pnl: float
    needs_reconciliation: bool
    excess_legs: List[Tuple[OrderLeg, int]]  # Legs with excess fills
    error: Optional[str] = None


class MultiLegExecutor:
    """
    Executes multi-leg orders concurrently.

    Handles:
    - Concurrent leg submission
    - Partial fill tracking
    - Automatic reconciliation of mismatched fills
    - Exposure management
    """

    def __init__(
        self,
        execute_func: Callable[[OrderLeg], asyncio.Future],
        close_func: Optional[Callable[[str, int, float], asyncio.Future]] = None,
        max_concurrent: int = 10,
        reconciliation_delay: float = 2.0,
        auto_close_discount: float = 0.10,  # 10% discount for auto-close
    ):
        """
        Args:
            execute_func: Async function to execute a single leg
            close_func: Async function to close excess position
            max_concurrent: Maximum concurrent executions
            reconciliation_delay: Seconds to wait before reconciliation
            auto_close_discount: Price discount for closing excess
        """
        self._execute = execute_func
        self._close = close_func
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._reconciliation_delay = reconciliation_delay
        self._auto_close_discount = auto_close_discount

        # Tracking
        self._in_flight: Dict[str, MultiLegOrder] = {}
        self._lock = threading.Lock()

    async def execute(self, order: MultiLegOrder) -> ExecutionResult:
        """
        Execute a multi-leg order concurrently.

        All legs are submitted simultaneously, and the result
        reconciles any mismatched fills.
        """
        with self._lock:
            self._in_flight[order.order_id] = order

        try:
            # Execute all legs concurrently
            tasks = [self._execute_leg(leg) for leg in order.legs]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate matched quantity
            matched = order.min_filled
            order.matched_quantity = matched

            # Calculate total cost
            total_cost = sum(
                leg.filled_quantity * leg.average_price + leg.commission
                for leg in order.legs
            )
            order.total_cost = total_cost

            # Find legs with excess fills
            excess_legs = []
            for leg in order.legs:
                excess = leg.filled_quantity - matched
                if excess > 0:
                    excess_legs.append((leg, excess))

            # Determine if reconciliation needed
            needs_reconciliation = len(excess_legs) > 0

            # Auto-reconcile if configured
            if needs_reconciliation and self._close:
                await self._reconcile_excess(excess_legs)

            # Calculate estimated P&L (for arbitrage: matched * $1 - cost)
            if order.strategy == "arbitrage":
                estimated_pnl = matched * 100 - total_cost  # In cents
            else:
                estimated_pnl = 0.0

            order.total_pnl = estimated_pnl
            order.is_complete = order.all_legs_complete
            order.reconciled = not needs_reconciliation or self._close is not None

            return ExecutionResult(
                order_id=order.order_id,
                success=not order.any_leg_failed and matched > 0,
                matched_quantity=matched,
                legs=order.legs,
                total_cost=total_cost,
                estimated_pnl=estimated_pnl,
                needs_reconciliation=needs_reconciliation,
                excess_legs=excess_legs,
            )

        finally:
            with self._lock:
                self._in_flight.pop(order.order_id, None)

    async def _execute_leg(self, leg: OrderLeg) -> None:
        """Execute a single leg with semaphore limiting."""
        async with self._semaphore:
            leg.status = LegStatus.SUBMITTED
            leg.submitted_at = datetime.now()

            try:
                await self._execute(leg)

                if leg.filled_quantity > 0:
                    if leg.filled_quantity >= leg.quantity:
                        leg.status = LegStatus.FILLED
                    else:
                        leg.status = LegStatus.PARTIAL
                    leg.filled_at = datetime.now()
                else:
                    leg.status = LegStatus.FAILED

            except Exception as e:
                leg.status = LegStatus.FAILED
                leg.error = str(e)
                logger.error(f"Leg {leg.leg_id} failed: {e}")

    async def _reconcile_excess(self, excess_legs: List[Tuple[OrderLeg, int]]) -> None:
        """Close excess positions from mismatched fills."""
        if not excess_legs or not self._close:
            return

        # Wait for settlement
        await asyncio.sleep(self._reconciliation_delay)

        for leg, excess in excess_legs:
            try:
                # Calculate close price with discount
                if leg.average_price > 0:
                    close_price = leg.average_price * (1 - self._auto_close_discount)
                else:
                    close_price = 0.0

                logger.info(
                    f"Auto-closing {excess} excess on {leg.symbol} at ${close_price:.2f}"
                )

                await self._close(leg.symbol, excess, close_price)

            except Exception as e:
                logger.error(f"Failed to close excess for {leg.symbol}: {e}")

    def create_order(
        self,
        legs_config: List[Dict[str, Any]],
        strategy: str = "",
    ) -> MultiLegOrder:
        """
        Create a multi-leg order from configuration.

        Args:
            legs_config: List of leg configurations
            strategy: Strategy name (e.g., "arbitrage", "spread")

        Returns:
            MultiLegOrder ready for execution
        """
        order_id = str(uuid.uuid4())[:8]
        legs = []

        for i, config in enumerate(legs_config):
            leg = OrderLeg(
                leg_id=f"{order_id}-{i}",
                symbol=config["symbol"],
                side=LegSide(config["side"]),
                quantity=config["quantity"],
                price=config.get("price"),
                platform=config.get("platform", "default"),
            )
            legs.append(leg)

        return MultiLegOrder(
            order_id=order_id,
            legs=legs,
            strategy=strategy,
        )


class ArbitrageExecutor(MultiLegExecutor):
    """
    Specialized executor for arbitrage strategies.

    Executes YES + NO pairs and calculates arbitrage profit.
    """

    def __init__(
        self,
        execute_func: Callable[[OrderLeg], asyncio.Future],
        close_func: Optional[Callable[[str, int, float], asyncio.Future]] = None,
    ):
        super().__init__(execute_func, close_func)

    def create_arbitrage_order(
        self,
        yes_symbol: str,
        no_symbol: str,
        yes_price: float,
        no_price: float,
        quantity: int,
        yes_platform: str = "platform_a",
        no_platform: str = "platform_b",
    ) -> MultiLegOrder:
        """
        Create an arbitrage order (YES + NO for guaranteed payout).

        Args:
            yes_symbol: Symbol for YES position
            no_symbol: Symbol for NO position
            yes_price: Price for YES (0-1)
            no_price: Price for NO (0-1)
            quantity: Number of contracts
            yes_platform: Platform for YES leg
            no_platform: Platform for NO leg

        Returns:
            MultiLegOrder configured for arbitrage
        """
        total_cost = (yes_price + no_price) * quantity
        potential_profit = quantity - total_cost  # $1 payout - cost

        if potential_profit <= 0:
            raise ValueError(
                f"No arbitrage opportunity: cost ${total_cost:.2f} >= payout ${quantity:.2f}"
            )

        return self.create_order(
            legs_config=[
                {
                    "symbol": yes_symbol,
                    "side": "buy",
                    "quantity": quantity,
                    "price": yes_price,
                    "platform": yes_platform,
                },
                {
                    "symbol": no_symbol,
                    "side": "buy",
                    "quantity": quantity,
                    "price": no_price,
                    "platform": no_platform,
                },
            ],
            strategy="arbitrage",
        )

    @staticmethod
    def calculate_arbitrage_profit(
        yes_price: float,
        no_price: float,
        quantity: int,
        yes_fee: float = 0.0,
        no_fee: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Calculate arbitrage profit.

        Args:
            yes_price: YES price (0-1)
            no_price: NO price (0-1)
            quantity: Number of contracts
            yes_fee: Fee for YES trade
            no_fee: Fee for NO trade

        Returns:
            Tuple of (profit, profit_percent)
        """
        total_cost = (yes_price + no_price) * quantity + yes_fee + no_fee
        payout = quantity  # $1 per contract
        profit = payout - total_cost
        profit_percent = (profit / total_cost) * 100 if total_cost > 0 else 0

        return profit, profit_percent


class SpreadExecutor(MultiLegExecutor):
    """
    Specialized executor for spread trades.

    Handles calendar spreads, vertical spreads, etc.
    """

    def create_spread_order(
        self,
        buy_symbol: str,
        sell_symbol: str,
        buy_price: float,
        sell_price: float,
        quantity: int,
    ) -> MultiLegOrder:
        """Create a spread order (buy one, sell another)."""
        return self.create_order(
            legs_config=[
                {
                    "symbol": buy_symbol,
                    "side": "buy",
                    "quantity": quantity,
                    "price": buy_price,
                },
                {
                    "symbol": sell_symbol,
                    "side": "sell",
                    "quantity": quantity,
                    "price": sell_price,
                },
            ],
            strategy="spread",
        )
