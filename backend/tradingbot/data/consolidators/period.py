"""
Period-based (Time) Data Consolidators.

Consolidate data into fixed time periods (1min, 5min, hourly, daily, etc.).
"""

from datetime import datetime, timedelta, time
from decimal import Decimal
from typing import Optional, Union
import pytz

from .base import (
    Bar,
    BarConsolidatorBase,
    BarType,
    Resolution,
    Tick,
    TickConsolidatorBase,
)


class TimePeriodConsolidator(BarConsolidatorBase):
    """
    Consolidate bars into fixed time periods.

    Supports minute, hour, daily, weekly, monthly periods.
    """

    def __init__(
        self,
        symbol: str,
        period: timedelta,
        start_time: Optional[time] = None,
        timezone: str = "America/New_York",
    ):
        """
        Initialize time period consolidator.

        Args:
            symbol: Symbol to consolidate
            period: Time period for each bar (e.g., timedelta(minutes=5))
            start_time: Optional start time for period alignment
            timezone: Timezone for period boundaries
        """
        super().__init__(symbol)
        self.period = period
        self.start_time = start_time
        self.timezone = pytz.timezone(timezone)
        self._period_start: Optional[datetime] = None
        self._period_end: Optional[datetime] = None

    def _get_period_start(self, timestamp: datetime) -> datetime:
        """Calculate the start of the current period."""
        # Make timezone aware if needed
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        # Calculate period start based on period duration
        seconds = int(self.period.total_seconds())

        if seconds >= 86400:  # Daily or longer
            # Align to day boundary
            period_start = timestamp.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if self.start_time:
                period_start = period_start.replace(
                    hour=self.start_time.hour,
                    minute=self.start_time.minute,
                )
        elif seconds >= 3600:  # Hourly
            # Align to hour boundary
            hours = seconds // 3600
            hour_aligned = (timestamp.hour // hours) * hours
            period_start = timestamp.replace(
                hour=hour_aligned, minute=0, second=0, microsecond=0
            )
        else:  # Sub-hourly (minutes)
            minutes = seconds // 60
            minute_aligned = (timestamp.minute // minutes) * minutes
            period_start = timestamp.replace(
                minute=minute_aligned, second=0, microsecond=0
            )

        return period_start

    def should_consolidate(self, data: Bar) -> bool:
        """Check if we should emit the current bar."""
        if self._period_end is None:
            return False

        # Make timestamp timezone aware
        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        return timestamp >= self._period_end

    def update(self, data: Bar) -> Optional[Bar]:
        """
        Process a new bar.

        Args:
            data: Input bar

        Returns:
            Consolidated bar if period complete, None otherwise
        """
        self._input_count += 1
        result = None

        # Make timestamp timezone aware
        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        # Check if we need to emit current bar
        if self._working_bar is not None and self.should_consolidate(data):
            self._working_bar.end_time = self._period_end
            result = self._working_bar
            self.emit(result)
            self._working_bar = None

        # Start new period or update existing
        if self._working_bar is None:
            self._period_start = self._get_period_start(timestamp)
            self._period_end = self._period_start + self.period
            self._working_bar = self._create_new_bar(data)
            self._working_bar.timestamp = self._period_start
        else:
            self._working_bar = self._update_bar(self._working_bar, data)

        return result


class MinuteConsolidator(TimePeriodConsolidator):
    """Consolidate to N-minute bars."""

    def __init__(
        self,
        symbol: str,
        minutes: int = 1,
        timezone: str = "America/New_York",
    ):
        """
        Initialize minute consolidator.

        Args:
            symbol: Symbol to consolidate
            minutes: Number of minutes per bar
            timezone: Timezone for period boundaries
        """
        super().__init__(
            symbol=symbol,
            period=timedelta(minutes=minutes),
            timezone=timezone,
        )
        self.minutes = minutes


class HourlyConsolidator(TimePeriodConsolidator):
    """Consolidate to N-hour bars."""

    def __init__(
        self,
        symbol: str,
        hours: int = 1,
        timezone: str = "America/New_York",
    ):
        """
        Initialize hourly consolidator.

        Args:
            symbol: Symbol to consolidate
            hours: Number of hours per bar
            timezone: Timezone for period boundaries
        """
        super().__init__(
            symbol=symbol,
            period=timedelta(hours=hours),
            timezone=timezone,
        )
        self.hours = hours


class DailyConsolidator(TimePeriodConsolidator):
    """
    Consolidate to daily bars.

    Handles market hours for equity/options.
    """

    def __init__(
        self,
        symbol: str,
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
        timezone: str = "America/New_York",
    ):
        """
        Initialize daily consolidator.

        Args:
            symbol: Symbol to consolidate
            market_open: Market open time
            market_close: Market close time
            timezone: Timezone for market hours
        """
        super().__init__(
            symbol=symbol,
            period=timedelta(days=1),
            start_time=market_open,
            timezone=timezone,
        )
        self.market_open = market_open
        self.market_close = market_close


class WeeklyConsolidator(BarConsolidatorBase):
    """
    Consolidate to weekly bars.

    Week starts on Monday by default.
    """

    def __init__(
        self,
        symbol: str,
        week_start: int = 0,  # Monday
        timezone: str = "America/New_York",
    ):
        """
        Initialize weekly consolidator.

        Args:
            symbol: Symbol to consolidate
            week_start: Day of week to start (0=Monday, 6=Sunday)
            timezone: Timezone for week boundaries
        """
        super().__init__(symbol)
        self.week_start = week_start
        self.timezone = pytz.timezone(timezone)
        self._week_start_date: Optional[datetime] = None

    def _get_week_start(self, timestamp: datetime) -> datetime:
        """Get the start of the week containing timestamp."""
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        days_since_start = (timestamp.weekday() - self.week_start) % 7
        week_start = timestamp - timedelta(days=days_since_start)
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    def should_consolidate(self, data: Bar) -> bool:
        """Check if new week started."""
        if self._week_start_date is None:
            return False

        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        new_week_start = self._get_week_start(timestamp)
        return new_week_start > self._week_start_date

    def update(self, data: Bar) -> Optional[Bar]:
        """Process a new bar."""
        self._input_count += 1
        result = None

        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        # Check if new week
        if self._working_bar is not None and self.should_consolidate(data):
            result = self._working_bar
            self.emit(result)
            self._working_bar = None

        # Start or update week bar
        if self._working_bar is None:
            self._week_start_date = self._get_week_start(timestamp)
            self._working_bar = self._create_new_bar(data)
            self._working_bar.timestamp = self._week_start_date
        else:
            self._working_bar = self._update_bar(self._working_bar, data)

        return result


class MonthlyConsolidator(BarConsolidatorBase):
    """Consolidate to monthly bars."""

    def __init__(
        self,
        symbol: str,
        timezone: str = "America/New_York",
    ):
        """
        Initialize monthly consolidator.

        Args:
            symbol: Symbol to consolidate
            timezone: Timezone for month boundaries
        """
        super().__init__(symbol)
        self.timezone = pytz.timezone(timezone)
        self._current_month: Optional[int] = None
        self._current_year: Optional[int] = None

    def should_consolidate(self, data: Bar) -> bool:
        """Check if new month started."""
        if self._current_month is None:
            return False

        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        return (
            timestamp.month != self._current_month or
            timestamp.year != self._current_year
        )

    def update(self, data: Bar) -> Optional[Bar]:
        """Process a new bar."""
        self._input_count += 1
        result = None

        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        # Check if new month
        if self._working_bar is not None and self.should_consolidate(data):
            result = self._working_bar
            self.emit(result)
            self._working_bar = None

        # Start or update month bar
        if self._working_bar is None:
            self._current_month = timestamp.month
            self._current_year = timestamp.year
            month_start = timestamp.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            self._working_bar = self._create_new_bar(data)
            self._working_bar.timestamp = month_start
        else:
            self._working_bar = self._update_bar(self._working_bar, data)

        return result


class TickTimePeriodConsolidator(TickConsolidatorBase):
    """
    Consolidate ticks into time-based bars.

    Creates OHLCV bars from raw tick data.
    """

    def __init__(
        self,
        symbol: str,
        period: timedelta,
        timezone: str = "America/New_York",
    ):
        """
        Initialize tick to time bar consolidator.

        Args:
            symbol: Symbol to consolidate
            period: Time period for each bar
            timezone: Timezone for period boundaries
        """
        super().__init__(symbol)
        self.period = period
        self.timezone = pytz.timezone(timezone)
        self._period_start: Optional[datetime] = None
        self._period_end: Optional[datetime] = None

    def _get_period_start(self, timestamp: datetime) -> datetime:
        """Calculate period start from timestamp."""
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        seconds = int(self.period.total_seconds())

        if seconds >= 3600:
            hours = seconds // 3600
            hour_aligned = (timestamp.hour // hours) * hours
            return timestamp.replace(
                hour=hour_aligned, minute=0, second=0, microsecond=0
            )
        else:
            minutes = max(1, seconds // 60)
            minute_aligned = (timestamp.minute // minutes) * minutes
            return timestamp.replace(
                minute=minute_aligned, second=0, microsecond=0
            )

    def should_consolidate(self, data: Tick) -> bool:
        """Check if period ended."""
        if self._period_end is None:
            return False

        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        return timestamp >= self._period_end

    def update(self, data: Tick) -> Optional[Bar]:
        """
        Process a new tick.

        Args:
            data: Input tick

        Returns:
            Consolidated bar if period complete, None otherwise
        """
        self._input_count += 1
        result = None

        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        # Check if period ended
        if self._working_bar is not None and self.should_consolidate(data):
            self._working_bar.end_time = self._period_end
            result = self._working_bar
            self.emit(result)
            self._working_bar = None

        # Start new bar or update existing
        if self._working_bar is None:
            self._period_start = self._get_period_start(timestamp)
            self._period_end = self._period_start + self.period
            self._working_bar = self._create_new_bar(data)
            self._working_bar.timestamp = self._period_start
        else:
            self._working_bar = self._update_bar(self._working_bar, data)

        return result


class MarketHoursConsolidator(TimePeriodConsolidator):
    """
    Time consolidator aware of market hours.

    Only consolidates during market hours, handles gaps.
    """

    def __init__(
        self,
        symbol: str,
        period: timedelta,
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
        include_extended: bool = False,
        extended_open: time = time(4, 0),
        extended_close: time = time(20, 0),
        timezone: str = "America/New_York",
    ):
        """
        Initialize market hours consolidator.

        Args:
            symbol: Symbol to consolidate
            period: Consolidation period
            market_open: Regular market open
            market_close: Regular market close
            include_extended: Include extended hours
            extended_open: Extended hours open
            extended_close: Extended hours close
            timezone: Market timezone
        """
        super().__init__(
            symbol=symbol,
            period=period,
            timezone=timezone,
        )
        self.market_open = market_open
        self.market_close = market_close
        self.include_extended = include_extended
        self.extended_open = extended_open
        self.extended_close = extended_close

    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during market hours."""
        t = timestamp.time()

        if self.include_extended:
            return self.extended_open <= t <= self.extended_close
        else:
            return self.market_open <= t <= self.market_close

    def update(self, data: Bar) -> Optional[Bar]:
        """Process bar only during market hours."""
        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        # Skip data outside market hours
        if not self._is_market_hours(timestamp):
            return None

        return super().update(data)


class SessionConsolidator(BarConsolidatorBase):
    """
    Consolidate to trading session bars.

    Creates one bar per trading session (day).
    """

    def __init__(
        self,
        symbol: str,
        session_start: time = time(9, 30),
        session_end: time = time(16, 0),
        timezone: str = "America/New_York",
    ):
        """
        Initialize session consolidator.

        Args:
            symbol: Symbol to consolidate
            session_start: Session start time
            session_end: Session end time
            timezone: Session timezone
        """
        super().__init__(symbol)
        self.session_start = session_start
        self.session_end = session_end
        self.timezone = pytz.timezone(timezone)
        self._current_date: Optional[datetime] = None

    def should_consolidate(self, data: Bar) -> bool:
        """Check if new session started."""
        if self._current_date is None:
            return False

        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        return timestamp.date() != self._current_date.date()

    def update(self, data: Bar) -> Optional[Bar]:
        """Process bar and consolidate by session."""
        self._input_count += 1
        result = None

        timestamp = data.timestamp
        if timestamp.tzinfo is None:
            timestamp = self.timezone.localize(timestamp)

        # Check if new session
        if self._working_bar is not None and self.should_consolidate(data):
            result = self._working_bar
            self.emit(result)
            self._working_bar = None

        # Start or update session bar
        if self._working_bar is None:
            self._current_date = timestamp
            session_start = timestamp.replace(
                hour=self.session_start.hour,
                minute=self.session_start.minute,
                second=0,
                microsecond=0,
            )
            self._working_bar = self._create_new_bar(data)
            self._working_bar.timestamp = session_start
        else:
            self._working_bar = self._update_bar(self._working_bar, data)

        return result

