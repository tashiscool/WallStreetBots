"""
Trading Filters - Inspired by Nautilus Trader.

Provides filters to control when trading should occur based on:
- Economic news events
- Time of day
- Market volatility
- Liquidity conditions

Concepts from: https://github.com/nautechsystems/nautilus_trader
License: LGPL-3.0 (concepts only, clean-room implementation)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd


class NewsImpact(Enum):
    """Impact level of economic news events."""
    NONE = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4


@dataclass
class NewsEvent:
    """
    Represents an economic news event.

    Examples: FOMC announcements, NFP, CPI releases, etc.
    """
    name: str
    impact: NewsImpact
    currency: str  # e.g., "USD", "EUR"
    timestamp: datetime
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None

    @property
    def surprise(self) -> Optional[float]:
        """Calculate surprise vs forecast."""
        if self.actual is not None and self.forecast is not None:
            return self.actual - self.forecast
        return None

    def __repr__(self) -> str:
        return f"NewsEvent({self.name}, {self.impact.name}, {self.currency}, {self.timestamp})"


class TradingFilter(ABC):
    """
    Abstract base class for trading filters.

    Filters determine whether trading should be allowed at a given time.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Filter name."""
        pass

    @abstractmethod
    def should_trade(self, timestamp: datetime, **kwargs) -> bool:
        """
        Check if trading should be allowed.

        Args:
            timestamp: Current time
            **kwargs: Additional context (price, volume, etc.)

        Returns:
            True if trading is allowed
        """
        pass

    def get_reason(self) -> Optional[str]:
        """Return reason if trading is blocked."""
        return None


class EconomicNewsEventFilter(TradingFilter):
    """
    Filter trading around economic news events.

    Prevents trading during high-impact news releases to avoid
    extreme volatility and slippage.
    """

    def __init__(
        self,
        currencies: List[str],
        impacts: List[NewsImpact],
        news_data: Optional[pd.DataFrame] = None,
        blackout_before: timedelta = timedelta(minutes=30),
        blackout_after: timedelta = timedelta(minutes=15),
    ):
        """
        Args:
            currencies: Currencies to filter (e.g., ["USD", "EUR"])
            impacts: Impact levels to filter (e.g., [NewsImpact.HIGH])
            news_data: DataFrame with news events (columns: Currency, Impact, Name)
            blackout_before: Time before event to stop trading
            blackout_after: Time after event to resume trading
        """
        self._currencies = set(currencies)
        self._impacts = set(impacts)
        self._blackout_before = blackout_before
        self._blackout_after = blackout_after
        self._events: List[NewsEvent] = []
        self._last_block_reason: Optional[str] = None

        if news_data is not None:
            self._load_news_data(news_data)

    def _load_news_data(self, df: pd.DataFrame) -> None:
        """Load news events from DataFrame."""
        for idx, row in df.iterrows():
            if row.get("Currency") in self._currencies:
                impact_str = row.get("Impact", "LOW")
                try:
                    impact = NewsImpact[impact_str.upper()]
                except KeyError:
                    impact = NewsImpact.LOW

                if impact in self._impacts:
                    event = NewsEvent(
                        name=row.get("Name", "Unknown"),
                        impact=impact,
                        currency=row["Currency"],
                        timestamp=pd.Timestamp(idx).to_pydatetime() if isinstance(idx, pd.Timestamp) else idx,
                    )
                    self._events.append(event)

        self._events.sort(key=lambda e: e.timestamp)

    def add_event(self, event: NewsEvent) -> None:
        """Add a news event."""
        if event.currency in self._currencies and event.impact in self._impacts:
            self._events.append(event)
            self._events.sort(key=lambda e: e.timestamp)

    @property
    def name(self) -> str:
        return "EconomicNewsFilter"

    @property
    def events(self) -> List[NewsEvent]:
        """Return filtered events."""
        return self._events.copy()

    def next_event(self, timestamp: datetime) -> Optional[NewsEvent]:
        """Get next upcoming event."""
        for event in self._events:
            if event.timestamp > timestamp:
                return event
        return None

    def prev_event(self, timestamp: datetime) -> Optional[NewsEvent]:
        """Get most recent past event."""
        prev = None
        for event in self._events:
            if event.timestamp <= timestamp:
                prev = event
            else:
                break
        return prev

    def should_trade(self, timestamp: datetime, **kwargs) -> bool:
        """Check if trading is allowed (not in news blackout)."""
        self._last_block_reason = None

        # Check upcoming events
        next_event = self.next_event(timestamp)
        if next_event:
            time_until = next_event.timestamp - timestamp
            if time_until <= self._blackout_before:
                self._last_block_reason = (
                    f"Blackout: {next_event.name} ({next_event.impact.name}) "
                    f"in {time_until}"
                )
                return False

        # Check recent events
        prev_event = self.prev_event(timestamp)
        if prev_event:
            time_since = timestamp - prev_event.timestamp
            if time_since <= self._blackout_after:
                self._last_block_reason = (
                    f"Blackout: {prev_event.name} ({prev_event.impact.name}) "
                    f"ended {time_since} ago"
                )
                return False

        return True

    def get_reason(self) -> Optional[str]:
        return self._last_block_reason


class TimeOfDayFilter(TradingFilter):
    """
    Filter trading by time of day.

    Useful for avoiding low-liquidity periods or focusing on
    specific market sessions.
    """

    def __init__(
        self,
        allowed_start: time = time(9, 30),
        allowed_end: time = time(16, 0),
        allowed_days: Optional[List[int]] = None,  # 0=Monday, 6=Sunday
        timezone: str = "US/Eastern",
    ):
        """
        Args:
            allowed_start: Start of trading window
            allowed_end: End of trading window
            allowed_days: Days of week to trade (None = Mon-Fri)
            timezone: Timezone for time checks
        """
        self._start = allowed_start
        self._end = allowed_end
        self._days = allowed_days or [0, 1, 2, 3, 4]  # Mon-Fri
        self._timezone = timezone
        self._last_block_reason: Optional[str] = None

    @property
    def name(self) -> str:
        return "TimeOfDayFilter"

    def should_trade(self, timestamp: datetime, **kwargs) -> bool:
        """Check if current time is in allowed trading window."""
        self._last_block_reason = None

        # Check day of week
        if timestamp.weekday() not in self._days:
            self._last_block_reason = f"Trading not allowed on day {timestamp.strftime('%A')}"
            return False

        # Check time of day
        current_time = timestamp.time()

        # Handle overnight sessions (e.g., 18:00 - 09:00)
        if self._start > self._end:
            # Overnight session
            if current_time < self._start and current_time > self._end:
                self._last_block_reason = (
                    f"Outside trading hours: {current_time} not in "
                    f"{self._start}-{self._end}"
                )
                return False
        else:
            # Normal session
            if current_time < self._start or current_time > self._end:
                self._last_block_reason = (
                    f"Outside trading hours: {current_time} not in "
                    f"{self._start}-{self._end}"
                )
                return False

        return True

    def get_reason(self) -> Optional[str]:
        return self._last_block_reason


class VolatilityFilter(TradingFilter):
    """
    Filter trading based on market volatility.

    Can block trading when volatility is too high (risky)
    or too low (no opportunity).
    """

    def __init__(
        self,
        max_volatility: Optional[float] = None,
        min_volatility: Optional[float] = None,
        volatility_period: int = 20,
    ):
        """
        Args:
            max_volatility: Maximum allowed volatility (annualized %)
            min_volatility: Minimum required volatility (annualized %)
            volatility_period: Lookback period for volatility calculation
        """
        self._max_vol = max_volatility
        self._min_vol = min_volatility
        self._period = volatility_period
        self._current_volatility: Optional[float] = None
        self._last_block_reason: Optional[str] = None

    @property
    def name(self) -> str:
        return "VolatilityFilter"

    def update_volatility(self, volatility: float) -> None:
        """Update current volatility reading."""
        self._current_volatility = volatility

    def should_trade(self, timestamp: datetime, volatility: Optional[float] = None, **kwargs) -> bool:
        """Check if volatility is within acceptable range."""
        self._last_block_reason = None

        vol = volatility or self._current_volatility
        if vol is None:
            return True  # No data, allow trading

        if self._max_vol is not None and vol > self._max_vol:
            self._last_block_reason = (
                f"Volatility too high: {vol:.2f}% > {self._max_vol:.2f}%"
            )
            return False

        if self._min_vol is not None and vol < self._min_vol:
            self._last_block_reason = (
                f"Volatility too low: {vol:.2f}% < {self._min_vol:.2f}%"
            )
            return False

        return True

    def get_reason(self) -> Optional[str]:
        return self._last_block_reason


class LiquidityFilter(TradingFilter):
    """
    Filter trading based on liquidity conditions.

    Prevents trading in illiquid markets where slippage may be high.
    """

    def __init__(
        self,
        min_volume: Optional[float] = None,
        min_dollar_volume: Optional[float] = None,
        max_spread_percent: Optional[float] = None,
    ):
        """
        Args:
            min_volume: Minimum volume (shares/contracts)
            min_dollar_volume: Minimum dollar volume
            max_spread_percent: Maximum bid-ask spread as % of price
        """
        self._min_volume = min_volume
        self._min_dollar_vol = min_dollar_volume
        self._max_spread = max_spread_percent
        self._last_block_reason: Optional[str] = None

    @property
    def name(self) -> str:
        return "LiquidityFilter"

    def should_trade(
        self,
        timestamp: datetime,
        volume: Optional[float] = None,
        dollar_volume: Optional[float] = None,
        spread_percent: Optional[float] = None,
        **kwargs
    ) -> bool:
        """Check if liquidity conditions are met."""
        self._last_block_reason = None

        if self._min_volume is not None and volume is not None:
            if volume < self._min_volume:
                self._last_block_reason = (
                    f"Volume too low: {volume:,.0f} < {self._min_volume:,.0f}"
                )
                return False

        if self._min_dollar_vol is not None and dollar_volume is not None:
            if dollar_volume < self._min_dollar_vol:
                self._last_block_reason = (
                    f"Dollar volume too low: ${dollar_volume:,.0f} < ${self._min_dollar_vol:,.0f}"
                )
                return False

        if self._max_spread is not None and spread_percent is not None:
            if spread_percent > self._max_spread:
                self._last_block_reason = (
                    f"Spread too wide: {spread_percent:.3f}% > {self._max_spread:.3f}%"
                )
                return False

        return True

    def get_reason(self) -> Optional[str]:
        return self._last_block_reason


class CompositeFilter(TradingFilter):
    """
    Combines multiple filters with AND logic.

    All filters must pass for trading to be allowed.
    """

    def __init__(self, filters: List[TradingFilter]):
        self._filters = filters
        self._blocking_filter: Optional[TradingFilter] = None

    @property
    def name(self) -> str:
        return "CompositeFilter"

    def add_filter(self, filter: TradingFilter) -> None:
        """Add a filter to the composite."""
        self._filters.append(filter)

    def should_trade(self, timestamp: datetime, **kwargs) -> bool:
        """Check all filters."""
        self._blocking_filter = None

        for f in self._filters:
            if not f.should_trade(timestamp, **kwargs):
                self._blocking_filter = f
                return False

        return True

    def get_reason(self) -> Optional[str]:
        if self._blocking_filter:
            return f"{self._blocking_filter.name}: {self._blocking_filter.get_reason()}"
        return None


class RateLimitFilter(TradingFilter):
    """
    Rate limiting filter to prevent excessive trading.

    Inspired by Nautilus Trader's risk engine rate limits.
    """

    def __init__(
        self,
        max_orders_per_second: int = 100,
        max_orders_per_minute: int = 1000,
        max_orders_per_hour: int = 10000,
    ):
        self._max_per_second = max_orders_per_second
        self._max_per_minute = max_orders_per_minute
        self._max_per_hour = max_orders_per_hour
        self._order_timestamps: List[datetime] = []
        self._last_block_reason: Optional[str] = None

    @property
    def name(self) -> str:
        return "RateLimitFilter"

    def record_order(self, timestamp: datetime) -> None:
        """Record an order submission."""
        self._order_timestamps.append(timestamp)
        # Clean old timestamps
        cutoff = timestamp - timedelta(hours=1)
        self._order_timestamps = [t for t in self._order_timestamps if t > cutoff]

    def should_trade(self, timestamp: datetime, **kwargs) -> bool:
        """Check if rate limits allow more orders."""
        self._last_block_reason = None

        # Count orders in each window
        one_second_ago = timestamp - timedelta(seconds=1)
        one_minute_ago = timestamp - timedelta(minutes=1)
        one_hour_ago = timestamp - timedelta(hours=1)

        last_second = sum(1 for t in self._order_timestamps if t > one_second_ago)
        last_minute = sum(1 for t in self._order_timestamps if t > one_minute_ago)
        last_hour = sum(1 for t in self._order_timestamps if t > one_hour_ago)

        if last_second >= self._max_per_second:
            self._last_block_reason = (
                f"Rate limit: {last_second} orders/sec >= {self._max_per_second}"
            )
            return False

        if last_minute >= self._max_per_minute:
            self._last_block_reason = (
                f"Rate limit: {last_minute} orders/min >= {self._max_per_minute}"
            )
            return False

        if last_hour >= self._max_per_hour:
            self._last_block_reason = (
                f"Rate limit: {last_hour} orders/hour >= {self._max_per_hour}"
            )
            return False

        return True

    def get_reason(self) -> Optional[str]:
        return self._last_block_reason
