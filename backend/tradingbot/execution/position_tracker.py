"""
Real-Time Position Tracker - Inspired by Polymarket-Kalshi Arbitrage Bot.

Channel-based position tracking with:
- Async fill recording
- Real-time P&L calculation
- Daily/all-time P&L tracking
- Position settlement handling

Concepts from: https://github.com/terauss/Polymarket-Kalshi-Arbitrage-bot
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio
import threading
import logging
import queue

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class FillRecord:
    """Records a single fill."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    price: float
    commission: float = 0.0
    platform: str = "default"
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: Optional[str] = None
    trade_id: Optional[str] = None

    @property
    def cost(self) -> float:
        """Total cost including commission."""
        return self.quantity * self.price + self.commission

    @property
    def notional(self) -> float:
        """Notional value."""
        return self.quantity * self.price


@dataclass
class Position:
    """Represents a position in a single instrument."""
    symbol: str
    quantity: int = 0
    average_cost: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    platform: str = "default"
    opened_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # For prediction markets
    yes_quantity: int = 0
    no_quantity: int = 0
    yes_cost: float = 0.0
    no_cost: float = 0.0

    @property
    def side(self) -> PositionSide:
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    @property
    def is_open(self) -> bool:
        return self.quantity != 0

    @property
    def total_cost(self) -> float:
        return abs(self.quantity) * self.average_cost

    @property
    def market_value(self) -> float:
        """Market value (needs current price to be accurate)."""
        return self.total_cost  # Placeholder

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.quantity == 0:
            return 0.0

        if self.quantity > 0:  # Long
            return self.quantity * (current_price - self.average_cost)
        else:  # Short
            return abs(self.quantity) * (self.average_cost - current_price)


@dataclass
class PositionTrackerState:
    """State of the position tracker."""
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_realized_pnl: float = 0.0
    all_time_realized_pnl: float = 0.0
    daily_commission: float = 0.0
    all_time_commission: float = 0.0
    trade_count: int = 0
    last_reset_date: date = field(default_factory=date.today)


class PositionChannel:
    """
    Async channel for non-blocking fill submissions.

    Allows trading operations to record fills without blocking on I/O.
    """

    def __init__(self, tracker: "PositionTracker"):
        self._tracker = tracker
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background fill processor."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("Position channel started")

    def stop(self) -> None:
        """Stop the background fill processor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Position channel stopped")

    def record_fill(self, fill: FillRecord) -> None:
        """Submit a fill for processing (non-blocking)."""
        self._queue.put(fill)

    def _process_loop(self) -> None:
        """Background loop that processes fills."""
        batch: List[FillRecord] = []
        batch_size = 16
        flush_interval = 0.1  # 100ms

        while self._running:
            try:
                # Try to get a fill with timeout
                fill = self._queue.get(timeout=flush_interval)
                batch.append(fill)

                # Drain additional fills up to batch size
                while len(batch) < batch_size:
                    try:
                        fill = self._queue.get_nowait()
                        batch.append(fill)
                    except queue.Empty:
                        break

                # Process batch
                if batch:
                    self._tracker._process_fills(batch)
                    batch = []

            except queue.Empty:
                # Flush any pending fills on timeout
                if batch:
                    self._tracker._process_fills(batch)
                    batch = []


class PositionTracker:
    """
    Real-time position and P&L tracker.

    Tracks:
    - Open positions by symbol
    - Realized and unrealized P&L
    - Daily and all-time statistics
    - Trade history
    """

    def __init__(self):
        self._state = PositionTrackerState()
        self._lock = threading.RLock()
        self._channel: Optional[PositionChannel] = None
        self._fill_history: List[FillRecord] = []
        self._max_history = 10000

    def start_channel(self) -> PositionChannel:
        """Start the async fill channel."""
        if self._channel is None:
            self._channel = PositionChannel(self)
        self._channel.start()
        return self._channel

    def stop_channel(self) -> None:
        """Stop the async fill channel."""
        if self._channel:
            self._channel.stop()

    def record_fill(self, fill: FillRecord) -> None:
        """
        Record a fill (synchronous).

        For async recording, use the channel.
        """
        self._process_fills([fill])

    def _process_fills(self, fills: List[FillRecord]) -> None:
        """Process a batch of fills."""
        with self._lock:
            self._check_daily_reset()

            for fill in fills:
                self._apply_fill(fill)
                self._fill_history.append(fill)

                # Trim history
                if len(self._fill_history) > self._max_history:
                    self._fill_history = self._fill_history[-self._max_history:]

    def _apply_fill(self, fill: FillRecord) -> None:
        """Apply a single fill to positions."""
        symbol = fill.symbol

        # Get or create position
        if symbol not in self._state.positions:
            self._state.positions[symbol] = Position(
                symbol=symbol,
                platform=fill.platform,
                opened_at=fill.timestamp,
            )

        pos = self._state.positions[symbol]
        pos.last_updated = fill.timestamp

        # Update commission
        pos.commission_paid += fill.commission
        self._state.daily_commission += fill.commission
        self._state.all_time_commission += fill.commission

        # Calculate P&L and update position
        if fill.side == "buy":
            if pos.quantity >= 0:
                # Adding to long or opening long
                total_cost = pos.quantity * pos.average_cost + fill.cost
                pos.quantity += fill.quantity
                if pos.quantity > 0:
                    pos.average_cost = total_cost / pos.quantity
            else:
                # Closing short
                closed_qty = min(fill.quantity, abs(pos.quantity))
                pnl = closed_qty * (pos.average_cost - fill.price)
                pos.realized_pnl += pnl
                self._state.daily_realized_pnl += pnl
                self._state.all_time_realized_pnl += pnl

                pos.quantity += fill.quantity
                if pos.quantity > 0:
                    # Flipped to long
                    pos.average_cost = fill.price

        else:  # sell
            if pos.quantity <= 0:
                # Adding to short or opening short
                total_cost = abs(pos.quantity) * pos.average_cost + fill.cost
                pos.quantity -= fill.quantity
                if pos.quantity < 0:
                    pos.average_cost = total_cost / abs(pos.quantity)
            else:
                # Closing long
                closed_qty = min(fill.quantity, pos.quantity)
                pnl = closed_qty * (fill.price - pos.average_cost)
                pos.realized_pnl += pnl
                self._state.daily_realized_pnl += pnl
                self._state.all_time_realized_pnl += pnl

                pos.quantity -= fill.quantity
                if pos.quantity < 0:
                    # Flipped to short
                    pos.average_cost = fill.price

        self._state.trade_count += 1

    def record_prediction_fill(
        self,
        symbol: str,
        is_yes: bool,
        quantity: int,
        price: float,
        commission: float = 0.0,
        platform: str = "default",
    ) -> None:
        """
        Record a fill for prediction market (YES/NO).

        Args:
            symbol: Market symbol
            is_yes: True for YES, False for NO
            quantity: Number of contracts
            price: Price per contract (0-1)
            commission: Trading commission
            platform: Trading platform
        """
        with self._lock:
            if symbol not in self._state.positions:
                self._state.positions[symbol] = Position(
                    symbol=symbol,
                    platform=platform,
                    opened_at=datetime.now(),
                )

            pos = self._state.positions[symbol]
            cost = quantity * price + commission

            if is_yes:
                pos.yes_quantity += quantity
                pos.yes_cost += cost
            else:
                pos.no_quantity += quantity
                pos.no_cost += cost

            pos.last_updated = datetime.now()
            self._state.all_time_commission += commission

    def resolve_prediction(
        self,
        symbol: str,
        yes_won: bool,
    ) -> float:
        """
        Resolve a prediction market position.

        Args:
            symbol: Market symbol
            yes_won: True if YES outcome, False if NO outcome

        Returns:
            Realized P&L from settlement
        """
        with self._lock:
            if symbol not in self._state.positions:
                return 0.0

            pos = self._state.positions[symbol]

            # Calculate payout
            if yes_won:
                payout = pos.yes_quantity  # $1 per YES contract
            else:
                payout = pos.no_quantity  # $1 per NO contract

            # Calculate P&L
            total_cost = pos.yes_cost + pos.no_cost
            pnl = payout - total_cost

            pos.realized_pnl += pnl
            self._state.daily_realized_pnl += pnl
            self._state.all_time_realized_pnl += pnl

            # Clear position
            pos.yes_quantity = 0
            pos.no_quantity = 0
            pos.yes_cost = 0
            pos.no_cost = 0

            logger.info(
                f"Resolved {symbol}: {'YES' if yes_won else 'NO'} won, "
                f"P&L: ${pnl:.2f}"
            )

            return pnl

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        with self._lock:
            return self._state.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        with self._lock:
            return dict(self._state.positions)

    def get_open_positions(self) -> Dict[str, Position]:
        """Get only open positions."""
        with self._lock:
            return {
                s: p for s, p in self._state.positions.items()
                if p.is_open
            }

    def get_daily_pnl(self) -> float:
        """Get daily realized P&L."""
        with self._lock:
            return self._state.daily_realized_pnl

    def get_all_time_pnl(self) -> float:
        """Get all-time realized P&L."""
        with self._lock:
            return self._state.all_time_realized_pnl

    def get_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """
        Calculate total unrealized P&L.

        Args:
            prices: Current prices by symbol

        Returns:
            Total unrealized P&L
        """
        with self._lock:
            total = 0.0
            for symbol, pos in self._state.positions.items():
                if symbol in prices and pos.is_open:
                    total += pos.unrealized_pnl(prices[symbol])
            return total

    def get_total_exposure(self) -> float:
        """Get total position exposure (absolute value)."""
        with self._lock:
            return sum(
                abs(pos.quantity * pos.average_cost)
                for pos in self._state.positions.values()
            )

    def reset_daily(self) -> None:
        """Reset daily counters."""
        with self._lock:
            self._state.daily_realized_pnl = 0.0
            self._state.daily_commission = 0.0
            self._state.last_reset_date = date.today()
            logger.info("Position tracker daily counters reset")

    def _check_daily_reset(self) -> None:
        """Check if daily reset needed."""
        today = date.today()
        if self._state.last_reset_date < today:
            self.reset_daily()

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status."""
        with self._lock:
            return {
                "daily_realized_pnl": self._state.daily_realized_pnl,
                "all_time_realized_pnl": self._state.all_time_realized_pnl,
                "daily_commission": self._state.daily_commission,
                "all_time_commission": self._state.all_time_commission,
                "trade_count": self._state.trade_count,
                "open_positions": len(self.get_open_positions()),
                "total_exposure": self.get_total_exposure(),
                "positions": {
                    s: {
                        "quantity": p.quantity,
                        "average_cost": p.average_cost,
                        "realized_pnl": p.realized_pnl,
                        "side": p.side.value,
                    }
                    for s, p in self._state.positions.items()
                },
            }


# Global tracker instance
_global_tracker: Optional[PositionTracker] = None


def get_position_tracker() -> PositionTracker:
    """Get or create the global position tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PositionTracker()
    return _global_tracker
