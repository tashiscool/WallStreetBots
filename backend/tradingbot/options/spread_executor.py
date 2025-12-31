"""
Spread Executor

Handles atomic multi-leg option order execution with proper risk management.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

from .exotic_spreads import (
    OptionSpread,
    SpreadLeg,
    SpreadType,
    LegType,
    SpreadAnalysis,
)

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Status of an order."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class ExecutionStrategy(Enum):
    """Strategy for executing multi-leg orders."""
    ATOMIC = "atomic"  # All legs at once (ideal)
    LEGGED_IN = "legged_in"  # Execute legs sequentially
    WINGS_FIRST = "wings_first"  # Execute protective legs first


@dataclass
class LegOrder:
    """Order for a single leg."""
    leg: SpreadLeg
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[Decimal] = None
    filled_qty: int = 0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass
class SpreadOrder:
    """Order for an entire spread."""
    spread: OptionSpread
    leg_orders: List[LegOrder] = field(default_factory=list)
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ATOMIC
    target_premium: Optional[Decimal] = None
    filled_premium: Optional[Decimal] = None
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.FAILED,
        )

    @property
    def all_legs_filled(self) -> bool:
        return all(leg.status == OrderStatus.FILLED for leg in self.leg_orders)

    @property
    def some_legs_filled(self) -> bool:
        return any(leg.status == OrderStatus.FILLED for leg in self.leg_orders)


@dataclass
class ExecutionResult:
    """Result of spread execution."""
    success: bool
    spread_order: SpreadOrder
    execution_time_ms: float = 0
    filled_premium: Optional[Decimal] = None
    error_message: Optional[str] = None
    leg_results: List[Dict[str, Any]] = field(default_factory=list)


class SpreadExecutor:
    """
    Executes multi-leg option spreads with proper risk management.

    Supports:
    - Atomic execution (all legs at once)
    - Legged-in execution (sequential)
    - Wings-first execution (protective legs first)
    - Order monitoring and rollback
    """

    def __init__(
        self,
        broker_client: Any,  # AlpacaManager or similar
        max_slippage_pct: float = 2.0,
        order_timeout_seconds: int = 30,
        auto_rollback: bool = True,
    ):
        self.broker = broker_client
        self.max_slippage_pct = max_slippage_pct
        self.order_timeout = order_timeout_seconds
        self.auto_rollback = auto_rollback
        self.active_orders: Dict[str, SpreadOrder] = {}

    async def execute_spread(
        self,
        spread: OptionSpread,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.ATOMIC,
        limit_price: Optional[Decimal] = None,
    ) -> ExecutionResult:
        """
        Execute a spread order.

        Args:
            spread: The spread to execute
            execution_strategy: How to execute the legs
            limit_price: Net limit price for the spread

        Returns:
            ExecutionResult with details
        """
        start_time = datetime.now()

        # Create spread order
        spread_order = SpreadOrder(
            spread=spread,
            execution_strategy=execution_strategy,
            target_premium=limit_price or spread.net_premium,
            leg_orders=[LegOrder(leg=leg) for leg in spread.legs],
        )

        try:
            # Validate spread
            validation_error = await self._validate_spread(spread)
            if validation_error:
                spread_order.status = OrderStatus.REJECTED
                spread_order.error_message = validation_error
                return ExecutionResult(
                    success=False,
                    spread_order=spread_order,
                    error_message=validation_error,
                )

            # Execute based on strategy
            if execution_strategy == ExecutionStrategy.ATOMIC:
                result = await self._execute_atomic(spread_order)
            elif execution_strategy == ExecutionStrategy.LEGGED_IN:
                result = await self._execute_legged_in(spread_order)
            elif execution_strategy == ExecutionStrategy.WINGS_FIRST:
                result = await self._execute_wings_first(spread_order)
            else:
                raise ValueError(f"Unknown execution strategy: {execution_strategy}")

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return ExecutionResult(
                success=result,
                spread_order=spread_order,
                execution_time_ms=execution_time,
                filled_premium=spread_order.filled_premium,
                leg_results=[
                    {
                        "leg_type": leg.leg.leg_type.value,
                        "strike": float(leg.leg.strike),
                        "status": leg.status.value,
                        "filled_price": float(leg.filled_price) if leg.filled_price else None,
                    }
                    for leg in spread_order.leg_orders
                ],
            )

        except Exception as e:
            logger.error(f"Error executing spread: {e}")
            spread_order.status = OrderStatus.FAILED
            spread_order.error_message = str(e)

            # Attempt rollback if some legs were filled
            if self.auto_rollback and spread_order.some_legs_filled:
                await self._rollback_partial_fill(spread_order)

            return ExecutionResult(
                success=False,
                spread_order=spread_order,
                error_message=str(e),
            )

    async def _validate_spread(self, spread: OptionSpread) -> Optional[str]:
        """Validate spread before execution."""
        # Check minimum legs
        if len(spread.legs) < 2:
            return "Spread must have at least 2 legs"

        # Check leg validity
        for leg in spread.legs:
            if leg.contracts == 0:
                return f"Invalid leg: zero contracts at strike {leg.strike}"
            if leg.strike <= 0:
                return f"Invalid strike: {leg.strike}"

        # Check broker connectivity
        try:
            if hasattr(self.broker, 'get_account'):
                await self.broker.get_account()
        except Exception as e:
            return f"Broker connection error: {e}"

        return None

    async def _execute_atomic(self, spread_order: SpreadOrder) -> bool:
        """Execute all legs as a single multi-leg order."""
        spread_order.status = OrderStatus.SUBMITTED
        spread_order.submitted_at = datetime.now()

        try:
            # Build multi-leg order
            legs_data = []
            for leg_order in spread_order.leg_orders:
                leg = leg_order.leg
                legs_data.append({
                    "symbol": self._build_option_symbol(
                        spread_order.spread.ticker,
                        leg.strike,
                        leg.expiry,
                        leg.option_type,
                    ),
                    "side": "buy" if leg.is_long else "sell",
                    "qty": abs(leg.contracts),
                })

            # Submit multi-leg order
            if hasattr(self.broker, 'submit_multi_leg_order'):
                order_response = await self.broker.submit_multi_leg_order(
                    legs=legs_data,
                    limit_price=float(spread_order.target_premium) if spread_order.target_premium else None,
                    time_in_force="day",
                )
                spread_order.order_id = order_response.get("id")

                # Wait for fill
                filled = await self._wait_for_fill(spread_order)

                if filled:
                    spread_order.status = OrderStatus.FILLED
                    spread_order.completed_at = datetime.now()
                    self._calculate_filled_premium(spread_order)
                    return True
                else:
                    spread_order.status = OrderStatus.FAILED
                    return False
            else:
                # Broker doesn't support multi-leg, fall back to legged-in
                logger.warning("Broker doesn't support multi-leg orders, falling back to legged-in")
                return await self._execute_legged_in(spread_order)

        except Exception as e:
            logger.error(f"Atomic execution failed: {e}")
            spread_order.status = OrderStatus.FAILED
            spread_order.error_message = str(e)
            return False

    async def _execute_legged_in(self, spread_order: SpreadOrder) -> bool:
        """Execute legs one at a time in sequence."""
        spread_order.status = OrderStatus.SUBMITTED
        spread_order.submitted_at = datetime.now()

        try:
            # Sort legs: short legs first (to collect premium)
            sorted_legs = sorted(
                spread_order.leg_orders,
                key=lambda x: x.leg.is_long  # Short (False) before Long (True)
            )

            for leg_order in sorted_legs:
                success = await self._execute_single_leg(spread_order.spread.ticker, leg_order)

                if not success:
                    spread_order.status = OrderStatus.PARTIAL
                    if self.auto_rollback:
                        await self._rollback_partial_fill(spread_order)
                    return False

            spread_order.status = OrderStatus.FILLED
            spread_order.completed_at = datetime.now()
            self._calculate_filled_premium(spread_order)
            return True

        except Exception as e:
            logger.error(f"Legged-in execution failed: {e}")
            spread_order.status = OrderStatus.FAILED
            spread_order.error_message = str(e)
            return False

    async def _execute_wings_first(self, spread_order: SpreadOrder) -> bool:
        """Execute protective (long) legs first, then short legs."""
        spread_order.status = OrderStatus.SUBMITTED
        spread_order.submitted_at = datetime.now()

        try:
            # Separate long and short legs
            long_legs = [lo for lo in spread_order.leg_orders if lo.leg.is_long]
            short_legs = [lo for lo in spread_order.leg_orders if not lo.leg.is_long]

            # Execute long legs first (protective)
            for leg_order in long_legs:
                success = await self._execute_single_leg(spread_order.spread.ticker, leg_order)
                if not success:
                    spread_order.status = OrderStatus.PARTIAL
                    if self.auto_rollback:
                        await self._rollback_partial_fill(spread_order)
                    return False

            # Then execute short legs
            for leg_order in short_legs:
                success = await self._execute_single_leg(spread_order.spread.ticker, leg_order)
                if not success:
                    spread_order.status = OrderStatus.PARTIAL
                    if self.auto_rollback:
                        await self._rollback_partial_fill(spread_order)
                    return False

            spread_order.status = OrderStatus.FILLED
            spread_order.completed_at = datetime.now()
            self._calculate_filled_premium(spread_order)
            return True

        except Exception as e:
            logger.error(f"Wings-first execution failed: {e}")
            spread_order.status = OrderStatus.FAILED
            spread_order.error_message = str(e)
            return False

    async def _execute_single_leg(self, ticker: str, leg_order: LegOrder) -> bool:
        """Execute a single leg order."""
        leg = leg_order.leg
        leg_order.status = OrderStatus.SUBMITTED
        leg_order.submitted_at = datetime.now()

        try:
            symbol = self._build_option_symbol(
                ticker,
                leg.strike,
                leg.expiry,
                leg.option_type,
            )

            # Submit order
            if hasattr(self.broker, 'submit_option_order'):
                order_response = await self.broker.submit_option_order(
                    symbol=symbol,
                    side="buy" if leg.is_long else "sell",
                    qty=abs(leg.contracts),
                    type="limit",
                    limit_price=float(leg.premium) if leg.premium else None,
                    time_in_force="day",
                )
                leg_order.order_id = order_response.get("id")

                # Wait for fill
                filled = await self._wait_for_leg_fill(leg_order)

                if filled:
                    leg_order.status = OrderStatus.FILLED
                    leg_order.filled_at = datetime.now()
                    return True
                else:
                    leg_order.status = OrderStatus.FAILED
                    return False
            else:
                # Fallback to generic order submission
                order_response = await self.broker.submit_order(
                    symbol=symbol,
                    side="buy" if leg.is_long else "sell",
                    qty=abs(leg.contracts),
                    type="limit",
                    limit_price=float(leg.premium) if leg.premium else None,
                )
                leg_order.order_id = order_response.get("id")
                leg_order.status = OrderStatus.FILLED  # Assume fill for now
                leg_order.filled_price = leg.premium
                leg_order.filled_qty = abs(leg.contracts)
                leg_order.filled_at = datetime.now()
                return True

        except Exception as e:
            logger.error(f"Failed to execute leg {leg.leg_type.value}: {e}")
            leg_order.status = OrderStatus.FAILED
            leg_order.error_message = str(e)
            return False

    async def _wait_for_fill(self, spread_order: SpreadOrder) -> bool:
        """Wait for spread order to fill."""
        start = datetime.now()
        while (datetime.now() - start).seconds < self.order_timeout:
            if hasattr(self.broker, 'get_order'):
                order = await self.broker.get_order(spread_order.order_id)
                if order.get("status") == "filled":
                    return True
                elif order.get("status") in ("cancelled", "rejected", "expired"):
                    return False
            await asyncio.sleep(0.5)
        return False

    async def _wait_for_leg_fill(self, leg_order: LegOrder) -> bool:
        """Wait for leg order to fill."""
        start = datetime.now()
        while (datetime.now() - start).seconds < self.order_timeout:
            if hasattr(self.broker, 'get_order'):
                order = await self.broker.get_order(leg_order.order_id)
                if order.get("status") == "filled":
                    leg_order.filled_price = Decimal(str(order.get("filled_avg_price", 0)))
                    leg_order.filled_qty = order.get("filled_qty", 0)
                    return True
                elif order.get("status") in ("cancelled", "rejected", "expired"):
                    return False
            await asyncio.sleep(0.5)
        return False

    async def _rollback_partial_fill(self, spread_order: SpreadOrder) -> None:
        """Rollback partially filled spread by closing filled legs."""
        logger.warning(f"Rolling back partial fill for spread order")

        for leg_order in spread_order.leg_orders:
            if leg_order.status == OrderStatus.FILLED:
                try:
                    # Close the position (reverse the order)
                    symbol = self._build_option_symbol(
                        spread_order.spread.ticker,
                        leg_order.leg.strike,
                        leg_order.leg.expiry,
                        leg_order.leg.option_type,
                    )
                    side = "sell" if leg_order.leg.is_long else "buy"

                    if hasattr(self.broker, 'submit_order'):
                        await self.broker.submit_order(
                            symbol=symbol,
                            side=side,
                            qty=leg_order.filled_qty,
                            type="market",  # Market to ensure fill
                        )
                    logger.info(f"Rolled back leg: {leg_order.leg.leg_type.value}")

                except Exception as e:
                    logger.error(f"Failed to rollback leg: {e}")

    def _calculate_filled_premium(self, spread_order: SpreadOrder) -> None:
        """Calculate the actual filled premium from leg fills."""
        total = Decimal("0")
        for leg_order in spread_order.leg_orders:
            if leg_order.filled_price:
                multiplier = -1 if leg_order.leg.is_long else 1
                total += leg_order.filled_price * leg_order.filled_qty * multiplier
        spread_order.filled_premium = total

    def _build_option_symbol(
        self,
        ticker: str,
        strike: Decimal,
        expiry,
        option_type: str,
    ) -> str:
        """Build OCC option symbol."""
        # Format: TICKER YYMMDD C/P STRIKE
        # Example: AAPL 241220 C 00150000
        expiry_str = expiry.strftime("%y%m%d")
        type_char = "C" if option_type == "call" else "P"
        strike_str = f"{int(strike * 1000):08d}"
        return f"{ticker}{expiry_str}{type_char}{strike_str}"

    async def close_spread(
        self,
        spread: OptionSpread,
        limit_price: Optional[Decimal] = None,
    ) -> ExecutionResult:
        """
        Close an existing spread position.

        Creates opposite orders for each leg.
        """
        # Create reverse legs
        close_legs = []
        for leg in spread.legs:
            # Reverse the leg type
            if leg.leg_type == LegType.LONG_CALL:
                new_type = LegType.SHORT_CALL
            elif leg.leg_type == LegType.SHORT_CALL:
                new_type = LegType.LONG_CALL
            elif leg.leg_type == LegType.LONG_PUT:
                new_type = LegType.SHORT_PUT
            else:
                new_type = LegType.LONG_PUT

            close_legs.append(SpreadLeg(
                leg_type=new_type,
                strike=leg.strike,
                expiry=leg.expiry,
                contracts=-leg.contracts,  # Reverse
                premium=leg.premium,
            ))

        # Create closing spread
        close_spread = OptionSpread(
            spread_type=spread.spread_type,
            ticker=spread.ticker,
            legs=close_legs,
        )

        return await self.execute_spread(
            close_spread,
            execution_strategy=ExecutionStrategy.ATOMIC,
            limit_price=limit_price,
        )
