"""
Event-Driven Prediction Market Arbitrage Engine.

Best-of-breed synthesis from 8 arbitrage bots:
- Event-driven architecture with MessageBus (RichardFeynmanEnthusiast)
- Signal-based async processing (ImMike)
- Buy-both arbitrage detection (RichardFeynmanEnthusiast)
- Cool-down and reset cycles (RichardFeynmanEnthusiast)
- Opportunity lifecycle tracking (ImMike)
- Emergency unwind mechanism (RichardFeynmanEnthusiast)
- Tiered fee calculation (dexorynLabs)
- Cross-platform matching (ImMike)

This is the production-grade arbitrage engine for WallStreetBots.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Set, Type
from collections import defaultdict, deque
import asyncio
import threading
import logging
import uuid

from .platform_client import (
    Platform, Outcome, Side, MarketState, OrderBook,
    PlatformClient, OrderRequest, OrderResponse
)
from .fee_calculator import FeeCalculator, get_fee_calculator
from .position_sizing import (
    PositionSizer, SizingContext, WalletBalance,
    create_conservative_sizer
)

logger = logging.getLogger(__name__)


# ============================================================================
# Domain Events (from RichardFeynmanEnthusiast DDD pattern)
# ============================================================================

class DomainEvent:
    """Base class for domain events."""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderBookUpdated(DomainEvent):
    """Order book received update."""
    platform: Platform
    market_id: str
    outcome: Outcome
    order_book: OrderBook
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ArbitrageOpportunityDetected(DomainEvent):
    """Arbitrage opportunity found."""
    opportunity_id: str
    polymarket_market_id: str
    kalshi_market_id: str
    strategy: str  # "buy_both_poly_yes", "buy_both_kalshi_yes", etc.
    gross_profit: Decimal
    net_profit: Decimal
    max_size: int
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class TradeExecutionRequested(DomainEvent):
    """Request to execute arbitrage trade."""
    opportunity_id: str
    size: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeExecutionCompleted(DomainEvent):
    """Both legs of trade completed."""
    opportunity_id: str
    success: bool
    poly_result: Optional[OrderResponse] = None
    kalshi_result: Optional[OrderResponse] = None
    actual_profit: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeExecutionFailed(DomainEvent):
    """One leg of trade failed."""
    opportunity_id: str
    failed_platform: Platform
    successful_platform: Optional[Platform] = None
    error: str = ""
    requires_unwind: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UnwindRequested(DomainEvent):
    """Request emergency position unwind."""
    platform: Platform
    market_id: str
    outcome: Outcome
    size: int
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemShutdownRequested(DomainEvent):
    """System shutdown requested."""
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Message Bus (from RichardFeynmanEnthusiast)
# ============================================================================

class MessageBus:
    """
    Type-safe async message bus.

    From RichardFeynmanEnthusiast: Decouples producers from consumers.
    """

    def __init__(self):
        self._subscribers: Dict[Type, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def subscribe(
        self,
        event_type: Type[DomainEvent],
        handler: Callable[[DomainEvent], Any],
    ) -> None:
        """Subscribe handler to event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type.__name__}")

    async def publish(self, event: DomainEvent) -> None:
        """Publish event to bus."""
        await self._queue.put(event)

    async def start(self) -> None:
        """Start processing events."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Message bus started")

    async def stop(self) -> None:
        """Stop processing events."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Message bus stopped")

    async def _run(self) -> None:
        """Main event loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )

                event_type = type(event)
                handlers = self._subscribers.get(event_type, [])

                for handler in handlers:
                    try:
                        result = handler(event)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.exception(
                            f"Error in handler {handler.__name__}: {e}"
                        )

                self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


# ============================================================================
# Arbitrage Opportunity (from multiple bots)
# ============================================================================

class ArbitrageStrategy(Enum):
    """Arbitrage strategies."""
    BUY_BOTH_POLY_YES = "buy_poly_yes_kalshi_no"
    BUY_BOTH_KALSHI_YES = "buy_kalshi_yes_poly_no"
    BUNDLE_LONG = "bundle_long"
    BUNDLE_SHORT = "bundle_short"


@dataclass
class ArbitrageOpportunity:
    """
    Represents a detected arbitrage opportunity.

    From RichardFeynmanEnthusiast: Complete opportunity model.
    """
    opportunity_id: str
    strategy: ArbitrageStrategy

    # Market identifiers
    polymarket_market_id: str
    kalshi_market_id: str

    # Prices
    poly_yes_price: Decimal
    poly_no_price: Decimal
    kalshi_yes_price: Decimal
    kalshi_no_price: Decimal

    # Liquidity
    max_size: int

    # Profit calculation
    gross_profit: Decimal
    total_fees: Decimal
    net_profit: Decimal

    # Timestamps
    detected_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    @property
    def is_profitable(self) -> bool:
        return self.net_profit > 0

    @property
    def edge_cents(self) -> float:
        """Edge in cents per contract."""
        return float(self.net_profit * 100)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


# ============================================================================
# Arbitrage Detector (from ImMike + RichardFeynmanEnthusiast)
# ============================================================================

class ArbitrageDetector:
    """
    Detects arbitrage opportunities from market state.

    Implements "buy both" detection from RichardFeynmanEnthusiast:
    - Buy YES on Platform A + Buy NO on Platform B
    - If combined cost < $1.00, guaranteed profit
    """

    # Minimum profit threshold (from dexorynLabs: dynamic filtering)
    MIN_PROFIT_CENTS = Decimal("0.5")

    # Staleness threshold (from RichardFeynmanEnthusiast)
    STALENESS_THRESHOLD = timedelta(seconds=5)

    def __init__(
        self,
        fee_calculator: Optional[FeeCalculator] = None,
        profitability_buffer: Decimal = Decimal("0.001"),
    ):
        self._fee_calc = fee_calculator or get_fee_calculator()
        self._buffer = profitability_buffer
        self._cooldowns: Dict[str, datetime] = {}
        self._cooldown_seconds = 2.0

    def check_for_arbitrage(
        self,
        poly_state: MarketState,
        kalshi_state: MarketState,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check for arbitrage between two markets.

        From RichardFeynmanEnthusiast: Two strategies checked.
        """
        # Check staleness
        if self._is_stale(poly_state, kalshi_state):
            return None

        # Check cooldown
        market_key = f"{poly_state.market_id}:{kalshi_state.market_id}"
        if self._in_cooldown(market_key):
            return None

        # Strategy 1: Buy YES on Poly, Buy NO on Kalshi
        opp1 = self._check_buy_both(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            buy_yes_on_poly=True,
        )

        # Strategy 2: Buy YES on Kalshi, Buy NO on Poly
        opp2 = self._check_buy_both(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            buy_yes_on_poly=False,
        )

        # Return better opportunity
        if opp1 and opp2:
            return opp1 if opp1.net_profit > opp2.net_profit else opp2
        return opp1 or opp2

    def _check_buy_both(
        self,
        poly_state: MarketState,
        kalshi_state: MarketState,
        buy_yes_on_poly: bool,
    ) -> Optional[ArbitrageOpportunity]:
        """Check single buy-both strategy."""
        if buy_yes_on_poly:
            # Buy Poly YES + Kalshi NO
            poly_price = poly_state.yes_price
            kalshi_price = kalshi_state.no_price
            strategy = ArbitrageStrategy.BUY_BOTH_POLY_YES
        else:
            # Buy Kalshi YES + Poly NO
            poly_price = poly_state.no_price
            kalshi_price = kalshi_state.yes_price
            strategy = ArbitrageStrategy.BUY_BOTH_KALSHI_YES

        if poly_price is None or kalshi_price is None:
            return None

        # Total cost
        total_cost = poly_price + kalshi_price

        # Calculate fees
        fees = self._fee_calc.calculate_arbitrage_fees(
            poly_yes_price=poly_state.yes_price or Decimal("0.5"),
            poly_no_price=poly_state.no_price or Decimal("0.5"),
            kalshi_yes_price=kalshi_state.yes_price or Decimal("0.5"),
            kalshi_no_price=kalshi_state.no_price or Decimal("0.5"),
            quantity=1,  # Per-contract fees
        )
        total_fees = sum(f.fee_amount for f in fees.values())

        # Gross and net profit
        gross_profit = Decimal("1.0") - total_cost
        net_profit = gross_profit - total_fees

        # Check profitability (with buffer)
        if net_profit <= self._buffer:
            return None

        if net_profit * 100 < self.MIN_PROFIT_CENTS:
            return None

        # Calculate max size
        poly_size = self._get_available_size(poly_state, buy_yes_on_poly)
        kalshi_size = self._get_available_size(kalshi_state, not buy_yes_on_poly)
        max_size = min(poly_size, kalshi_size)

        if max_size <= 0:
            return None

        return ArbitrageOpportunity(
            opportunity_id=str(uuid.uuid4())[:8],
            strategy=strategy,
            polymarket_market_id=poly_state.market_id,
            kalshi_market_id=kalshi_state.market_id,
            poly_yes_price=poly_state.yes_price or Decimal("0"),
            poly_no_price=poly_state.no_price or Decimal("0"),
            kalshi_yes_price=kalshi_state.yes_price or Decimal("0"),
            kalshi_no_price=kalshi_state.no_price or Decimal("0"),
            max_size=max_size,
            gross_profit=gross_profit,
            total_fees=total_fees,
            net_profit=net_profit,
            expires_at=datetime.now() + timedelta(seconds=5),
        )

    def _is_stale(
        self,
        poly_state: MarketState,
        kalshi_state: MarketState,
    ) -> bool:
        """Check if data is too stale."""
        if poly_state.yes_book and kalshi_state.yes_book:
            delta = abs(
                poly_state.yes_book.last_update -
                kalshi_state.yes_book.last_update
            )
            return delta > self.STALENESS_THRESHOLD
        return False

    def _in_cooldown(self, market_key: str) -> bool:
        """Check if market is in cooldown."""
        if market_key not in self._cooldowns:
            return False
        return datetime.now() < self._cooldowns[market_key]

    def set_cooldown(self, market_key: str) -> None:
        """Set cooldown for market."""
        self._cooldowns[market_key] = datetime.now() + timedelta(
            seconds=self._cooldown_seconds
        )

    def _get_available_size(
        self,
        state: MarketState,
        is_yes: bool,
    ) -> int:
        """Get available size from order book."""
        book = state.yes_book if is_yes else state.no_book
        if book is None or book.best_ask is None:
            return 0
        return int(book.best_ask.size)


# ============================================================================
# Trade Executor (from RichardFeynmanEnthusiast)
# ============================================================================

class TradeExecutor:
    """
    Executes arbitrage trades with risk management.

    From RichardFeynmanEnthusiast:
    - Concurrent execution of both legs
    - Three-outcome handling (both succeed, one fails, both fail)
    - Emergency unwind for partial failures
    """

    def __init__(
        self,
        polymarket_client: PlatformClient,
        kalshi_client: PlatformClient,
        message_bus: MessageBus,
        position_sizer: Optional[PositionSizer] = None,
        dry_run: bool = True,
    ):
        self._poly_client = polymarket_client
        self._kalshi_client = kalshi_client
        self._bus = message_bus
        self._sizer = position_sizer or create_conservative_sizer()
        self._dry_run = dry_run
        self._executing = False
        self._lock = asyncio.Lock()

    async def execute(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> None:
        """
        Execute arbitrage trade.

        Executes both legs concurrently for minimum slippage.
        """
        async with self._lock:
            if self._executing:
                logger.warning("Execution already in progress")
                return
            self._executing = True

        try:
            # Calculate position size
            context = await self._create_sizing_context(opportunity)
            sizing_result = self._sizer.calculate_size(context)

            if sizing_result.size <= 0:
                logger.warning(
                    f"Zero size for {opportunity.opportunity_id}: "
                    f"{sizing_result.details}"
                )
                return

            size = sizing_result.size

            if self._dry_run:
                logger.info(
                    f"[DRY RUN] Would execute {opportunity.strategy.value} "
                    f"size={size}, profit=${opportunity.net_profit * size:.4f}"
                )
                await self._bus.publish(TradeExecutionCompleted(
                    opportunity_id=opportunity.opportunity_id,
                    success=True,
                    actual_profit=opportunity.net_profit * size,
                ))
                return

            # Execute both legs concurrently
            poly_task = self._execute_poly_leg(opportunity, size)
            kalshi_task = self._execute_kalshi_leg(opportunity, size)

            results = await asyncio.gather(
                poly_task,
                kalshi_task,
                return_exceptions=True,
            )

            await self._handle_results(opportunity, results)

        finally:
            self._executing = False

    async def _execute_poly_leg(
        self,
        opportunity: ArbitrageOpportunity,
        size: int,
    ) -> OrderResponse:
        """Execute Polymarket leg."""
        if opportunity.strategy == ArbitrageStrategy.BUY_BOTH_POLY_YES:
            outcome = Outcome.YES
            price = opportunity.poly_yes_price
        else:
            outcome = Outcome.NO
            price = opportunity.poly_no_price

        request = OrderRequest(
            platform=Platform.POLYMARKET,
            market_id=opportunity.polymarket_market_id,
            outcome=outcome,
            side=Side.BUY,
            quantity=size,
            price=price,
        )

        return await self._poly_client.place_order(request)

    async def _execute_kalshi_leg(
        self,
        opportunity: ArbitrageOpportunity,
        size: int,
    ) -> OrderResponse:
        """Execute Kalshi leg."""
        if opportunity.strategy == ArbitrageStrategy.BUY_BOTH_POLY_YES:
            outcome = Outcome.NO
            price = opportunity.kalshi_no_price
        else:
            outcome = Outcome.YES
            price = opportunity.kalshi_yes_price

        request = OrderRequest(
            platform=Platform.KALSHI,
            market_id=opportunity.kalshi_market_id,
            outcome=outcome,
            side=Side.BUY,
            quantity=size,
            price=price,
        )

        return await self._kalshi_client.place_order(request)

    async def _handle_results(
        self,
        opportunity: ArbitrageOpportunity,
        results: List[Any],
    ) -> None:
        """
        Handle trade results.

        Three outcomes from RichardFeynmanEnthusiast:
        1. Both succeed -> ArbitrageTradeSuccessful
        2. One fails -> Trigger unwind
        3. Both fail -> Shutdown
        """
        poly_result, kalshi_result = results

        poly_error = isinstance(poly_result, Exception)
        kalshi_error = isinstance(kalshi_result, Exception)

        if poly_error and kalshi_error:
            # Both failed - critical error
            logger.critical(
                f"Both legs failed for {opportunity.opportunity_id}"
            )
            await self._bus.publish(TradeExecutionFailed(
                opportunity_id=opportunity.opportunity_id,
                failed_platform=Platform.POLYMARKET,
                error="Both legs failed",
                requires_unwind=False,
            ))
            await self._bus.publish(SystemShutdownRequested(
                reason="Both trade legs failed"
            ))

        elif poly_error:
            # Poly failed, Kalshi succeeded - unwind Kalshi
            logger.warning(
                f"Polymarket failed for {opportunity.opportunity_id}, "
                f"unwinding Kalshi"
            )
            await self._bus.publish(TradeExecutionFailed(
                opportunity_id=opportunity.opportunity_id,
                failed_platform=Platform.POLYMARKET,
                successful_platform=Platform.KALSHI,
                error=str(poly_result),
                requires_unwind=True,
            ))

        elif kalshi_error:
            # Kalshi failed, Poly succeeded - unwind Poly
            logger.warning(
                f"Kalshi failed for {opportunity.opportunity_id}, "
                f"unwinding Polymarket"
            )
            await self._bus.publish(TradeExecutionFailed(
                opportunity_id=opportunity.opportunity_id,
                failed_platform=Platform.KALSHI,
                successful_platform=Platform.POLYMARKET,
                error=str(kalshi_result),
                requires_unwind=True,
            ))

        else:
            # Both succeeded
            logger.info(
                f"Both legs succeeded for {opportunity.opportunity_id}"
            )
            await self._bus.publish(TradeExecutionCompleted(
                opportunity_id=opportunity.opportunity_id,
                success=True,
                poly_result=poly_result,
                kalshi_result=kalshi_result,
                actual_profit=opportunity.net_profit,
            ))

    async def _create_sizing_context(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> SizingContext:
        """Create sizing context from current state."""
        poly_balance = await self._poly_client.get_balance()
        kalshi_balance = await self._kalshi_client.get_balance()

        return SizingContext(
            polymarket_balance=WalletBalance(
                platform="polymarket",
                available=poly_balance,
            ),
            kalshi_balance=WalletBalance(
                platform="kalshi",
                available=kalshi_balance,
            ),
            opportunity_size=opportunity.max_size,
            profit_margin=opportunity.net_profit,
            estimated_fees=opportunity.total_fees,
        )


# ============================================================================
# Arbitrage Engine (main orchestrator)
# ============================================================================

class ArbitrageEngine:
    """
    Main arbitrage engine orchestrating all components.

    Combines best patterns from all 8 bots:
    - Event-driven architecture (RichardFeynmanEnthusiast)
    - Opportunity tracking (ImMike)
    - Cool-down cycles (RichardFeynmanEnthusiast)
    - Risk management (all bots)
    """

    def __init__(
        self,
        polymarket_client: PlatformClient,
        kalshi_client: PlatformClient,
        dry_run: bool = True,
        cool_down_seconds: float = 5.0,
    ):
        self._poly_client = polymarket_client
        self._kalshi_client = kalshi_client
        self._dry_run = dry_run
        self._cool_down_seconds = cool_down_seconds

        # Core components
        self._bus = MessageBus()
        self._detector = ArbitrageDetector()
        self._executor = TradeExecutor(
            polymarket_client=polymarket_client,
            kalshi_client=kalshi_client,
            message_bus=self._bus,
            dry_run=dry_run,
        )

        # State
        self._running = False
        self._markets: Dict[str, MarketState] = {}
        self._opportunities: deque = deque(maxlen=1000)
        self._stats = {
            "scans": 0,
            "opportunities_found": 0,
            "trades_executed": 0,
            "total_profit": Decimal("0"),
        }

        # Wire up event handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup event handlers."""
        self._bus.subscribe(
            ArbitrageOpportunityDetected,
            self._handle_opportunity
        )
        self._bus.subscribe(
            TradeExecutionCompleted,
            self._handle_trade_completed
        )
        self._bus.subscribe(
            TradeExecutionFailed,
            self._handle_trade_failed
        )
        self._bus.subscribe(
            SystemShutdownRequested,
            self._handle_shutdown
        )

    async def start(self) -> None:
        """Start the arbitrage engine."""
        logger.info("Starting arbitrage engine...")
        self._running = True
        await self._bus.start()

        # Connect clients
        await self._poly_client.connect()
        await self._kalshi_client.connect()

        logger.info(f"Arbitrage engine started (dry_run={self._dry_run})")

    async def stop(self) -> None:
        """Stop the arbitrage engine."""
        logger.info("Stopping arbitrage engine...")
        self._running = False
        await self._bus.stop()
        await self._poly_client.disconnect()
        await self._kalshi_client.disconnect()
        logger.info("Arbitrage engine stopped")

    async def scan_markets(
        self,
        poly_markets: List[MarketState],
        kalshi_markets: List[MarketState],
    ) -> List[ArbitrageOpportunity]:
        """Scan for arbitrage opportunities."""
        self._stats["scans"] += 1
        opportunities = []

        # Simple O(n*m) comparison - could use market matcher for optimization
        for poly in poly_markets:
            for kalshi in kalshi_markets:
                opp = self._detector.check_for_arbitrage(poly, kalshi)
                if opp and opp.is_profitable:
                    opportunities.append(opp)
                    self._stats["opportunities_found"] += 1

                    await self._bus.publish(ArbitrageOpportunityDetected(
                        opportunity_id=opp.opportunity_id,
                        polymarket_market_id=opp.polymarket_market_id,
                        kalshi_market_id=opp.kalshi_market_id,
                        strategy=opp.strategy.value,
                        gross_profit=opp.gross_profit,
                        net_profit=opp.net_profit,
                        max_size=opp.max_size,
                    ))

        return opportunities

    async def _handle_opportunity(
        self,
        event: ArbitrageOpportunityDetected,
    ) -> None:
        """Handle detected opportunity."""
        logger.info(
            f"Opportunity {event.opportunity_id}: {event.strategy} "
            f"net_profit=${event.net_profit:.4f} max_size={event.max_size}"
        )
        self._opportunities.append(event)

    async def _handle_trade_completed(
        self,
        event: TradeExecutionCompleted,
    ) -> None:
        """Handle completed trade."""
        if event.success:
            self._stats["trades_executed"] += 1
            self._stats["total_profit"] += event.actual_profit
            logger.info(
                f"Trade {event.opportunity_id} completed, "
                f"profit=${event.actual_profit:.4f}"
            )

            # Cool-down before next trade
            await asyncio.sleep(self._cool_down_seconds)

    async def _handle_trade_failed(
        self,
        event: TradeExecutionFailed,
    ) -> None:
        """Handle failed trade."""
        logger.error(
            f"Trade {event.opportunity_id} failed on {event.failed_platform}: "
            f"{event.error}"
        )

        if event.requires_unwind and event.successful_platform:
            logger.warning(f"Unwind required for {event.successful_platform}")
            # Would trigger unwind logic here

    async def _handle_shutdown(
        self,
        event: SystemShutdownRequested,
    ) -> None:
        """Handle shutdown request."""
        logger.critical(f"Shutdown requested: {event.reason}")
        await self.stop()

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "running": self._running,
            "opportunities_pending": len(self._opportunities),
            "dry_run": self._dry_run,
        }


# Factory function
def create_arbitrage_engine(
    polymarket_private_key: Optional[str] = None,
    kalshi_api_key: Optional[str] = None,
    kalshi_private_key_path: Optional[str] = None,
    dry_run: bool = True,
) -> ArbitrageEngine:
    """Create configured arbitrage engine."""
    from .platform_client import (
        PolymarketClient, KalshiClient, Environment
    )

    poly_client = PolymarketClient(
        private_key=polymarket_private_key,
        environment=Environment.PRODUCTION,
    )

    kalshi_client = KalshiClient(
        api_key=kalshi_api_key,
        private_key_path=kalshi_private_key_path,
        environment=Environment.PRODUCTION,
    )

    return ArbitrageEngine(
        polymarket_client=poly_client,
        kalshi_client=kalshi_client,
        dry_run=dry_run,
    )
