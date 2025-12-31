"""
Data Consolidator Base Classes.

Ported from QuantConnect/LEAN's consolidator framework.
Consolidators aggregate tick/bar data into higher timeframe bars.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Generic, List, Optional, TypeVar

import logging

logger = logging.getLogger(__name__)


class Resolution(Enum):
    """Data resolution/timeframe."""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class BarType(Enum):
    """Type of bar being consolidated."""
    TRADE = "trade"
    QUOTE = "quote"
    TICK = "tick"


@dataclass
class Bar:
    """
    OHLCV Bar data structure.

    Represents a single candlestick/bar of price data.
    """
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    bar_type: BarType = BarType.TRADE

    # Quote bar specific
    bid_open: Optional[Decimal] = None
    bid_high: Optional[Decimal] = None
    bid_low: Optional[Decimal] = None
    bid_close: Optional[Decimal] = None
    ask_open: Optional[Decimal] = None
    ask_high: Optional[Decimal] = None
    ask_low: Optional[Decimal] = None
    ask_close: Optional[Decimal] = None

    @property
    def is_complete(self) -> bool:
        """Check if bar has all required data."""
        return all([
            self.open is not None,
            self.high is not None,
            self.low is not None,
            self.close is not None,
        ])

    @property
    def range(self) -> Decimal:
        """Price range of the bar."""
        return self.high - self.low

    @property
    def body(self) -> Decimal:
        """Body size (absolute difference between open and close)."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """True if close > open."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """True if close < open."""
        return self.close < self.open

    @property
    def mid_price(self) -> Decimal:
        """Mid price of high and low."""
        return (self.high + self.low) / 2

    @property
    def typical_price(self) -> Decimal:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3

    @property
    def vwap(self) -> Optional[Decimal]:
        """VWAP if volume available."""
        if self.volume > 0:
            return self.typical_price
        return None


@dataclass
class Tick:
    """
    Single tick/trade data.

    Represents a single trade or quote update.
    """
    symbol: str
    price: Decimal
    quantity: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # For quote ticks
    bid_price: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_price: Optional[Decimal] = None
    ask_size: Optional[int] = None

    # Trade direction
    is_buy: Optional[bool] = None

    @property
    def is_trade(self) -> bool:
        """True if this is a trade tick."""
        return self.quantity > 0

    @property
    def is_quote(self) -> bool:
        """True if this is a quote tick."""
        return self.bid_price is not None or self.ask_price is not None

    @property
    def spread(self) -> Optional[Decimal]:
        """Bid-ask spread if quote tick."""
        if self.bid_price and self.ask_price:
            return self.ask_price - self.bid_price
        return None


# Type variable for generic consolidator
T = TypeVar('T')  # Input type
U = TypeVar('U')  # Output type


class IDataConsolidator(ABC, Generic[T, U]):
    """
    Abstract interface for data consolidators.

    Consolidators aggregate input data (ticks, bars) into
    output data (higher timeframe bars).
    """

    def __init__(self, symbol: str):
        """
        Initialize consolidator.

        Args:
            symbol: Symbol being consolidated
        """
        self.symbol = symbol
        self._working_bar: Optional[U] = None
        self._consolidated: Optional[U] = None
        self._callbacks: List[Callable[[U], None]] = []
        self._input_count: int = 0
        self._output_count: int = 0

    @property
    def working_bar(self) -> Optional[U]:
        """Current bar being built."""
        return self._working_bar

    @property
    def consolidated(self) -> Optional[U]:
        """Last consolidated (completed) bar."""
        return self._consolidated

    @property
    def input_count(self) -> int:
        """Number of data points received."""
        return self._input_count

    @property
    def output_count(self) -> int:
        """Number of bars produced."""
        return self._output_count

    @abstractmethod
    def update(self, data: T) -> Optional[U]:
        """
        Process new data point.

        Args:
            data: Input data (tick or bar)

        Returns:
            Consolidated bar if complete, None otherwise
        """
        pass

    @abstractmethod
    def should_consolidate(self, data: T) -> bool:
        """
        Check if current working bar should be consolidated.

        Args:
            data: New data point

        Returns:
            True if bar should be emitted
        """
        pass

    def on_data_consolidated(self, callback: Callable[[U], None]) -> None:
        """
        Register callback for consolidated data.

        Args:
            callback: Function to call with consolidated bar
        """
        self._callbacks.append(callback)

    def emit(self, bar: U) -> None:
        """
        Emit a consolidated bar to all callbacks.

        Args:
            bar: Completed bar to emit
        """
        self._consolidated = bar
        self._output_count += 1

        for callback in self._callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Consolidator callback error: {e}")

    def reset(self) -> None:
        """Reset consolidator state."""
        self._working_bar = None
        self._input_count = 0


class BarConsolidatorBase(IDataConsolidator[Bar, Bar]):
    """
    Base class for bar-to-bar consolidators.

    Aggregates lower timeframe bars into higher timeframe bars.
    """

    def __init__(self, symbol: str):
        super().__init__(symbol)

    def _create_new_bar(self, data: Bar) -> Bar:
        """Create a new working bar from input data."""
        return Bar(
            symbol=self.symbol,
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume,
            timestamp=data.timestamp,
            bar_type=data.bar_type,
        )

    def _update_bar(self, working: Bar, data: Bar) -> Bar:
        """Update working bar with new data."""
        return Bar(
            symbol=self.symbol,
            open=working.open,
            high=max(working.high, data.high),
            low=min(working.low, data.low),
            close=data.close,
            volume=working.volume + data.volume,
            timestamp=working.timestamp,
            end_time=data.end_time or data.timestamp,
            bar_type=working.bar_type,
        )


class TickConsolidatorBase(IDataConsolidator[Tick, Bar]):
    """
    Base class for tick-to-bar consolidators.

    Aggregates individual ticks into OHLCV bars.
    """

    def __init__(self, symbol: str):
        super().__init__(symbol)

    def _create_new_bar(self, tick: Tick) -> Bar:
        """Create a new bar from a tick."""
        return Bar(
            symbol=self.symbol,
            open=tick.price,
            high=tick.price,
            low=tick.price,
            close=tick.price,
            volume=tick.quantity,
            timestamp=tick.timestamp,
            bar_type=BarType.TICK,
        )

    def _update_bar(self, working: Bar, tick: Tick) -> Bar:
        """Update working bar with new tick."""
        return Bar(
            symbol=self.symbol,
            open=working.open,
            high=max(working.high, tick.price),
            low=min(working.low, tick.price),
            close=tick.price,
            volume=working.volume + tick.quantity,
            timestamp=working.timestamp,
            end_time=tick.timestamp,
            bar_type=working.bar_type,
        )


class IdentityDataConsolidator(IDataConsolidator[Bar, Bar]):
    """
    Pass-through consolidator that emits every bar.

    Useful for registering indicators to raw data feed.
    """

    def update(self, data: Bar) -> Optional[Bar]:
        """Pass through the bar immediately."""
        self._input_count += 1
        self.emit(data)
        return data

    def should_consolidate(self, data: Bar) -> bool:
        """Always consolidate."""
        return True


class FilteredIdentityConsolidator(IDataConsolidator[Bar, Bar]):
    """
    Identity consolidator with filtering.

    Only passes bars that match the filter criteria.
    """

    def __init__(
        self,
        symbol: str,
        filter_func: Callable[[Bar], bool],
    ):
        """
        Initialize filtered consolidator.

        Args:
            symbol: Symbol to consolidate
            filter_func: Function that returns True for bars to pass
        """
        super().__init__(symbol)
        self._filter_func = filter_func

    def update(self, data: Bar) -> Optional[Bar]:
        """Pass bar if it matches filter."""
        self._input_count += 1

        if self._filter_func(data):
            self.emit(data)
            return data
        return None

    def should_consolidate(self, data: Bar) -> bool:
        """Check filter."""
        return self._filter_func(data)

