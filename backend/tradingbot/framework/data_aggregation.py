"""
Data Aggregation - Inspired by Nautilus Trader.

Provides bar building and data aggregation patterns for trading.

Concepts from: https://github.com/nautechsystems/nautilus_trader
License: LGPL-3.0 (concepts only, clean-room implementation)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import List, Optional, Callable, Dict, Any
from enum import Enum
from collections import deque


class BarAggregation(Enum):
    """Bar aggregation type."""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    VOLUME = "volume"
    VALUE = "value"  # Dollar/notional volume


class IntervalType(Enum):
    """Interval boundary type for time bars."""
    LEFT_OPEN = "left-open"    # (start, end] - end time included
    RIGHT_OPEN = "right-open"  # [start, end) - start time included


@dataclass
class Quote:
    """Represents a bid/ask quote."""
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Tick:
    """Represents a trade tick."""
    timestamp: datetime
    price: float
    volume: float
    side: str = ""  # "buy" or "sell"


@dataclass
class Bar:
    """OHLCV Bar."""
    timestamp: datetime  # Bar close time
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int = 0
    vwap: Optional[float] = None

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class DataEngineConfig:
    """
    Configuration for data aggregation.

    Inspired by Nautilus Trader's DataEngineConfig.
    """
    time_bars_interval_type: IntervalType = IntervalType.LEFT_OPEN
    time_bars_timestamp_on_close: bool = True
    time_bars_skip_first_non_full_bar: bool = False
    time_bars_build_with_no_updates: bool = True
    validate_data_sequence: bool = False
    emit_quotes_from_book: bool = False


class BarBuilder:
    """
    Builds OHLCV bars from ticks or quotes.
    """

    def __init__(self):
        self._open: Optional[float] = None
        self._high: Optional[float] = None
        self._low: Optional[float] = None
        self._close: Optional[float] = None
        self._volume: float = 0.0
        self._tick_count: int = 0
        self._value_sum: float = 0.0  # For VWAP
        self._first_timestamp: Optional[datetime] = None
        self._last_timestamp: Optional[datetime] = None

    def update(self, price: float, volume: float = 0.0,
               timestamp: Optional[datetime] = None) -> None:
        """Update bar with new price data."""
        if self._open is None:
            self._open = price
            self._high = price
            self._low = price
            self._first_timestamp = timestamp
        else:
            self._high = max(self._high, price)
            self._low = min(self._low, price)

        self._close = price
        self._volume += volume
        self._tick_count += 1
        self._value_sum += price * volume
        self._last_timestamp = timestamp

    def update_tick(self, tick: Tick) -> None:
        """Update from tick."""
        self.update(tick.price, tick.volume, tick.timestamp)

    def update_quote(self, quote: Quote) -> None:
        """Update from quote (uses mid price)."""
        self.update(quote.mid, 0.0, quote.timestamp)

    def build(self, timestamp: Optional[datetime] = None) -> Optional[Bar]:
        """Build the bar and reset."""
        if self._open is None:
            return None

        vwap = None
        if self._volume > 0:
            vwap = self._value_sum / self._volume

        bar = Bar(
            timestamp=timestamp or self._last_timestamp or datetime.now(),
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            tick_count=self._tick_count,
            vwap=vwap,
        )

        self.reset()
        return bar

    def reset(self) -> None:
        """Reset builder state."""
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._volume = 0.0
        self._tick_count = 0
        self._value_sum = 0.0
        self._first_timestamp = None
        self._last_timestamp = None

    @property
    def is_empty(self) -> bool:
        return self._open is None


class TimeBarAggregator:
    """
    Aggregates ticks into time-based bars.

    Supports various time intervals (1min, 5min, 1hour, etc.)
    """

    def __init__(
        self,
        interval: timedelta,
        config: Optional[DataEngineConfig] = None,
        on_bar: Optional[Callable[[Bar], None]] = None,
    ):
        self._interval = interval
        self._config = config or DataEngineConfig()
        self._on_bar = on_bar
        self._builder = BarBuilder()
        self._current_bar_end: Optional[datetime] = None
        self._is_first_bar = True

    def _get_bar_end_time(self, timestamp: datetime) -> datetime:
        """Calculate the end time for the bar containing this timestamp."""
        # Align to interval boundary
        seconds = self._interval.total_seconds()

        if seconds >= 86400:  # Daily or larger
            # Align to day boundary
            return datetime.combine(timestamp.date(), time(0, 0)) + timedelta(days=1)
        elif seconds >= 3600:  # Hourly
            hours = int(seconds / 3600)
            hour_aligned = (timestamp.hour // hours) * hours + hours
            return timestamp.replace(hour=hour_aligned % 24, minute=0, second=0, microsecond=0)
        elif seconds >= 60:  # Minute-based
            minutes = int(seconds / 60)
            minute_aligned = (timestamp.minute // minutes) * minutes + minutes
            if minute_aligned >= 60:
                return timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return timestamp.replace(minute=minute_aligned, second=0, microsecond=0)
        else:  # Second-based
            second_aligned = (timestamp.second // int(seconds)) * int(seconds) + int(seconds)
            if second_aligned >= 60:
                return timestamp.replace(second=0, microsecond=0) + timedelta(minutes=1)
            return timestamp.replace(second=second_aligned, microsecond=0)

    def update(self, tick: Tick) -> Optional[Bar]:
        """
        Update with a new tick.

        Returns completed bar if interval ended.
        """
        bar_end = self._get_bar_end_time(tick.timestamp)

        # Check if we need to emit a bar
        if self._current_bar_end is not None and bar_end > self._current_bar_end:
            # New bar period - emit the completed bar
            completed_bar = self._emit_bar()

            # Skip first non-full bar if configured
            if self._is_first_bar and self._config.time_bars_skip_first_non_full_bar:
                completed_bar = None

            self._is_first_bar = False
            self._current_bar_end = bar_end
            self._builder.update_tick(tick)
            return completed_bar

        # Same bar period
        if self._current_bar_end is None:
            self._current_bar_end = bar_end

        self._builder.update_tick(tick)
        return None

    def _emit_bar(self) -> Optional[Bar]:
        """Emit the current bar."""
        if self._builder.is_empty:
            if not self._config.time_bars_build_with_no_updates:
                return None

        timestamp = self._current_bar_end
        if not self._config.time_bars_timestamp_on_close:
            timestamp = self._current_bar_end - self._interval

        bar = self._builder.build(timestamp)

        if bar and self._on_bar:
            self._on_bar(bar)

        return bar

    def flush(self) -> Optional[Bar]:
        """Flush any pending bar."""
        return self._emit_bar()


class TickBarAggregator:
    """
    Aggregates a fixed number of ticks into bars.
    """

    def __init__(
        self,
        tick_count: int,
        on_bar: Optional[Callable[[Bar], None]] = None,
    ):
        self._tick_count = tick_count
        self._on_bar = on_bar
        self._builder = BarBuilder()
        self._current_count = 0

    def update(self, tick: Tick) -> Optional[Bar]:
        """Update with a new tick."""
        self._builder.update_tick(tick)
        self._current_count += 1

        if self._current_count >= self._tick_count:
            bar = self._builder.build(tick.timestamp)
            self._current_count = 0

            if bar and self._on_bar:
                self._on_bar(bar)

            return bar

        return None

    def flush(self) -> Optional[Bar]:
        """Flush any pending bar."""
        if not self._builder.is_empty:
            return self._builder.build()
        return None


class VolumeBarAggregator:
    """
    Aggregates bars when volume threshold is reached.
    """

    def __init__(
        self,
        volume_threshold: float,
        on_bar: Optional[Callable[[Bar], None]] = None,
    ):
        self._threshold = volume_threshold
        self._on_bar = on_bar
        self._builder = BarBuilder()
        self._current_volume = 0.0

    def update(self, tick: Tick) -> Optional[Bar]:
        """Update with a new tick."""
        self._builder.update_tick(tick)
        self._current_volume += tick.volume

        if self._current_volume >= self._threshold:
            bar = self._builder.build(tick.timestamp)
            self._current_volume = 0.0

            if bar and self._on_bar:
                self._on_bar(bar)

            return bar

        return None

    def flush(self) -> Optional[Bar]:
        """Flush any pending bar."""
        if not self._builder.is_empty:
            return self._builder.build()
        return None


class DollarBarAggregator:
    """
    Aggregates bars when dollar volume (value) threshold is reached.
    """

    def __init__(
        self,
        value_threshold: float,
        on_bar: Optional[Callable[[Bar], None]] = None,
    ):
        self._threshold = value_threshold
        self._on_bar = on_bar
        self._builder = BarBuilder()
        self._current_value = 0.0

    def update(self, tick: Tick) -> Optional[Bar]:
        """Update with a new tick."""
        self._builder.update_tick(tick)
        self._current_value += tick.price * tick.volume

        if self._current_value >= self._threshold:
            bar = self._builder.build(tick.timestamp)
            self._current_value = 0.0

            if bar and self._on_bar:
                self._on_bar(bar)

            return bar

        return None

    def flush(self) -> Optional[Bar]:
        """Flush any pending bar."""
        if not self._builder.is_empty:
            return self._builder.build()
        return None


class BarResampler:
    """
    Resamples bars to a larger timeframe.

    E.g., combine 1-minute bars into 5-minute bars.
    """

    def __init__(
        self,
        source_interval: timedelta,
        target_interval: timedelta,
        on_bar: Optional[Callable[[Bar], None]] = None,
    ):
        if target_interval <= source_interval:
            raise ValueError("Target interval must be larger than source")

        self._source = source_interval
        self._target = target_interval
        self._ratio = int(target_interval / source_interval)
        self._on_bar = on_bar
        self._builder = BarBuilder()
        self._bar_count = 0

    def update(self, bar: Bar) -> Optional[Bar]:
        """Update with a source bar."""
        self._builder.update(bar.open, bar.volume / 4, bar.timestamp)
        self._builder.update(bar.high, bar.volume / 4, bar.timestamp)
        self._builder.update(bar.low, bar.volume / 4, bar.timestamp)
        self._builder.update(bar.close, bar.volume / 4, bar.timestamp)
        self._bar_count += 1

        if self._bar_count >= self._ratio:
            resampled = self._builder.build(bar.timestamp)
            self._bar_count = 0

            if resampled and self._on_bar:
                self._on_bar(resampled)

            return resampled

        return None

    def flush(self) -> Optional[Bar]:
        """Flush any pending bar."""
        if not self._builder.is_empty:
            return self._builder.build()
        return None


class QuoteAggregator:
    """
    Aggregates quotes into bar data using mid-price.
    """

    def __init__(
        self,
        interval: timedelta,
        on_bar: Optional[Callable[[Bar], None]] = None,
    ):
        self._aggregator = TimeBarAggregator(interval, on_bar=on_bar)

    def update(self, quote: Quote) -> Optional[Bar]:
        """Update with a quote."""
        # Convert quote to tick using mid price
        tick = Tick(
            timestamp=quote.timestamp,
            price=quote.mid,
            volume=0.0,
        )
        return self._aggregator.update(tick)

    def flush(self) -> Optional[Bar]:
        return self._aggregator.flush()


def create_time_aggregator(
    interval_minutes: int,
    on_bar: Optional[Callable[[Bar], None]] = None,
) -> TimeBarAggregator:
    """Factory to create a time bar aggregator."""
    return TimeBarAggregator(
        interval=timedelta(minutes=interval_minutes),
        on_bar=on_bar,
    )


def create_tick_aggregator(
    tick_count: int,
    on_bar: Optional[Callable[[Bar], None]] = None,
) -> TickBarAggregator:
    """Factory to create a tick bar aggregator."""
    return TickBarAggregator(tick_count=tick_count, on_bar=on_bar)


def create_volume_aggregator(
    volume: float,
    on_bar: Optional[Callable[[Bar], None]] = None,
) -> VolumeBarAggregator:
    """Factory to create a volume bar aggregator."""
    return VolumeBarAggregator(volume_threshold=volume, on_bar=on_bar)


def create_dollar_aggregator(
    value: float,
    on_bar: Optional[Callable[[Bar], None]] = None,
) -> DollarBarAggregator:
    """Factory to create a dollar bar aggregator."""
    return DollarBarAggregator(value_threshold=value, on_bar=on_bar)
