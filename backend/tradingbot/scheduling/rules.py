"""
Scheduling Rules Module.

Ported from QuantConnect/LEAN's scheduling framework.
Provides DateRules and TimeRules for precise scheduling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Callable, Iterator, List, Optional, Set
import calendar
import pytz

import logging

logger = logging.getLogger(__name__)


class DayOfWeek(Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class IDateRule(ABC):
    """Abstract interface for date rules."""

    @abstractmethod
    def get_dates(
        self,
        start: date,
        end: date,
    ) -> Iterator[date]:
        """
        Get dates matching this rule.

        Args:
            start: Start date
            end: End date

        Yields:
            Matching dates
        """
        pass

    @abstractmethod
    def matches(self, d: date) -> bool:
        """
        Check if date matches this rule.

        Args:
            d: Date to check

        Returns:
            True if date matches
        """
        pass


class ITimeRule(ABC):
    """Abstract interface for time rules."""

    @abstractmethod
    def get_time(
        self,
        d: date,
        market_open: time,
        market_close: time,
    ) -> time:
        """
        Get the scheduled time for a date.

        Args:
            d: The date
            market_open: Market open time
            market_close: Market close time

        Returns:
            Scheduled time
        """
        pass


class EveryDayRule(IDateRule):
    """Match every day."""

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield every date in range."""
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)

    def matches(self, d: date) -> bool:
        """Always matches."""
        return True


class WeekdaysRule(IDateRule):
    """Match weekdays only (Mon-Fri)."""

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield weekdays in range."""
        current = start
        while current <= end:
            if current.weekday() < 5:
                yield current
            current += timedelta(days=1)

    def matches(self, d: date) -> bool:
        """Match weekdays."""
        return d.weekday() < 5


class WeekendRule(IDateRule):
    """Match weekends only (Sat-Sun)."""

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield weekend days in range."""
        current = start
        while current <= end:
            if current.weekday() >= 5:
                yield current
            current += timedelta(days=1)

    def matches(self, d: date) -> bool:
        """Match weekends."""
        return d.weekday() >= 5


class DayOfWeekRule(IDateRule):
    """Match specific day of week."""

    def __init__(self, day: DayOfWeek):
        """
        Initialize day of week rule.

        Args:
            day: Day to match
        """
        self.day = day

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield matching days in range."""
        current = start
        # Find first matching day
        while current.weekday() != self.day.value:
            current += timedelta(days=1)
            if current > end:
                return

        # Yield every week
        while current <= end:
            yield current
            current += timedelta(days=7)

    def matches(self, d: date) -> bool:
        """Match specific day."""
        return d.weekday() == self.day.value


class MonthStartRule(IDateRule):
    """Match first trading day of each month."""

    def __init__(self, skip_weekends: bool = True):
        """
        Initialize month start rule.

        Args:
            skip_weekends: Skip to Monday if 1st is weekend
        """
        self.skip_weekends = skip_weekends

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield first day of each month in range."""
        current = date(start.year, start.month, 1)

        while current <= end:
            first_day = current

            if self.skip_weekends:
                # Move to Monday if weekend
                while first_day.weekday() >= 5:
                    first_day += timedelta(days=1)

            if start <= first_day <= end:
                yield first_day

            # Move to next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

    def matches(self, d: date) -> bool:
        """Match first trading day of month."""
        first = date(d.year, d.month, 1)
        if self.skip_weekends:
            while first.weekday() >= 5:
                first += timedelta(days=1)
        return d == first


class MonthEndRule(IDateRule):
    """Match last trading day of each month."""

    def __init__(self, skip_weekends: bool = True):
        """
        Initialize month end rule.

        Args:
            skip_weekends: Skip to Friday if last day is weekend
        """
        self.skip_weekends = skip_weekends

    def _get_last_day(self, year: int, month: int) -> date:
        """Get last day of month."""
        last_day = calendar.monthrange(year, month)[1]
        last = date(year, month, last_day)

        if self.skip_weekends:
            while last.weekday() >= 5:
                last -= timedelta(days=1)

        return last

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield last day of each month in range."""
        current_month = start.month
        current_year = start.year

        while True:
            last_day = self._get_last_day(current_year, current_month)

            if last_day > end:
                break

            if last_day >= start:
                yield last_day

            # Move to next month
            if current_month == 12:
                current_month = 1
                current_year += 1
            else:
                current_month += 1

    def matches(self, d: date) -> bool:
        """Match last trading day of month."""
        return d == self._get_last_day(d.year, d.month)


class WeekStartRule(IDateRule):
    """Match first trading day of each week (Monday)."""

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield Mondays in range."""
        current = start
        # Find first Monday
        while current.weekday() != 0:
            current += timedelta(days=1)
            if current > end:
                return

        while current <= end:
            yield current
            current += timedelta(days=7)

    def matches(self, d: date) -> bool:
        """Match Mondays."""
        return d.weekday() == 0


class WeekEndRule(IDateRule):
    """Match last trading day of each week (Friday)."""

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield Fridays in range."""
        current = start
        # Find first Friday
        while current.weekday() != 4:
            current += timedelta(days=1)
            if current > end:
                return

        while current <= end:
            yield current
            current += timedelta(days=7)

    def matches(self, d: date) -> bool:
        """Match Fridays."""
        return d.weekday() == 4


class NthDayOfMonthRule(IDateRule):
    """Match Nth occurrence of day in month (e.g., 3rd Friday)."""

    def __init__(self, day: DayOfWeek, n: int):
        """
        Initialize Nth day rule.

        Args:
            day: Day of week
            n: Occurrence (1 = first, 2 = second, etc.)
        """
        self.day = day
        self.n = n

    def _get_nth_day(self, year: int, month: int) -> Optional[date]:
        """Get Nth occurrence of day in month."""
        first = date(year, month, 1)

        # Find first occurrence
        while first.weekday() != self.day.value:
            first += timedelta(days=1)

        # Add weeks for Nth occurrence
        result = first + timedelta(weeks=self.n - 1)

        # Check still in same month
        if result.month != month:
            return None

        return result

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield Nth day of each month in range."""
        current_month = start.month
        current_year = start.year

        while True:
            nth_day = self._get_nth_day(current_year, current_month)

            if nth_day and nth_day > end:
                break

            if nth_day and nth_day >= start:
                yield nth_day

            if current_month == 12:
                current_month = 1
                current_year += 1
            else:
                current_month += 1

            # Safety check
            if current_year > end.year + 1:
                break

    def matches(self, d: date) -> bool:
        """Match Nth occurrence."""
        nth_day = self._get_nth_day(d.year, d.month)
        return nth_day is not None and d == nth_day


class SpecificDatesRule(IDateRule):
    """Match specific dates."""

    def __init__(self, dates: List[date]):
        """
        Initialize specific dates rule.

        Args:
            dates: List of dates to match
        """
        self.dates = set(dates)

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield matching dates in range."""
        for d in sorted(self.dates):
            if start <= d <= end:
                yield d

    def matches(self, d: date) -> bool:
        """Match specific dates."""
        return d in self.dates


class ExcludeRule(IDateRule):
    """Exclude dates from another rule."""

    def __init__(self, base_rule: IDateRule, exclude: IDateRule):
        """
        Initialize exclude rule.

        Args:
            base_rule: Base date rule
            exclude: Dates to exclude
        """
        self.base_rule = base_rule
        self.exclude = exclude

    def get_dates(self, start: date, end: date) -> Iterator[date]:
        """Yield dates not excluded."""
        for d in self.base_rule.get_dates(start, end):
            if not self.exclude.matches(d):
                yield d

    def matches(self, d: date) -> bool:
        """Match if base matches and not excluded."""
        return self.base_rule.matches(d) and not self.exclude.matches(d)


# Time Rules

class AtTimeRule(ITimeRule):
    """Schedule at a specific time."""

    def __init__(self, hour: int, minute: int = 0, second: int = 0):
        """
        Initialize at time rule.

        Args:
            hour: Hour (0-23)
            minute: Minute (0-59)
            second: Second (0-59)
        """
        self.scheduled_time = time(hour, minute, second)

    def get_time(
        self,
        d: date,
        market_open: time,
        market_close: time,
    ) -> time:
        """Return the scheduled time."""
        return self.scheduled_time


class MarketOpenRule(ITimeRule):
    """Schedule at market open."""

    def __init__(self, offset_minutes: int = 0):
        """
        Initialize market open rule.

        Args:
            offset_minutes: Minutes after open (can be negative for before)
        """
        self.offset = timedelta(minutes=offset_minutes)

    def get_time(
        self,
        d: date,
        market_open: time,
        market_close: time,
    ) -> time:
        """Return market open time with offset."""
        dt = datetime.combine(d, market_open) + self.offset
        return dt.time()


class MarketCloseRule(ITimeRule):
    """Schedule relative to market close."""

    def __init__(self, offset_minutes: int = 0):
        """
        Initialize market close rule.

        Args:
            offset_minutes: Minutes before close (positive = before)
        """
        self.offset = timedelta(minutes=offset_minutes)

    def get_time(
        self,
        d: date,
        market_open: time,
        market_close: time,
    ) -> time:
        """Return market close time minus offset."""
        dt = datetime.combine(d, market_close) - self.offset
        return dt.time()


class EveryNMinutesRule(ITimeRule):
    """Schedule every N minutes during market hours."""

    def __init__(self, minutes: int = 15, start_offset: int = 0):
        """
        Initialize every N minutes rule.

        Args:
            minutes: Interval in minutes
            start_offset: Minutes after market open to start
        """
        self.minutes = minutes
        self.start_offset = start_offset

    def get_time(
        self,
        d: date,
        market_open: time,
        market_close: time,
    ) -> time:
        """Return first scheduled time (use iterator for all)."""
        dt = datetime.combine(d, market_open) + timedelta(minutes=self.start_offset)
        return dt.time()

    def get_times(
        self,
        d: date,
        market_open: time,
        market_close: time,
    ) -> Iterator[time]:
        """Yield all scheduled times during market hours."""
        start_dt = datetime.combine(d, market_open) + timedelta(minutes=self.start_offset)
        end_dt = datetime.combine(d, market_close)

        current = start_dt
        while current <= end_dt:
            yield current.time()
            current += timedelta(minutes=self.minutes)


class DateRules:
    """Factory for date rules (LEAN-style API)."""

    @staticmethod
    def every_day() -> IDateRule:
        """Every calendar day."""
        return EveryDayRule()

    @staticmethod
    def weekdays() -> IDateRule:
        """Every weekday (Mon-Fri)."""
        return WeekdaysRule()

    @staticmethod
    def monday() -> IDateRule:
        """Every Monday."""
        return DayOfWeekRule(DayOfWeek.MONDAY)

    @staticmethod
    def tuesday() -> IDateRule:
        """Every Tuesday."""
        return DayOfWeekRule(DayOfWeek.TUESDAY)

    @staticmethod
    def wednesday() -> IDateRule:
        """Every Wednesday."""
        return DayOfWeekRule(DayOfWeek.WEDNESDAY)

    @staticmethod
    def thursday() -> IDateRule:
        """Every Thursday."""
        return DayOfWeekRule(DayOfWeek.THURSDAY)

    @staticmethod
    def friday() -> IDateRule:
        """Every Friday."""
        return DayOfWeekRule(DayOfWeek.FRIDAY)

    @staticmethod
    def month_start(skip_weekends: bool = True) -> IDateRule:
        """First trading day of month."""
        return MonthStartRule(skip_weekends)

    @staticmethod
    def month_end(skip_weekends: bool = True) -> IDateRule:
        """Last trading day of month."""
        return MonthEndRule(skip_weekends)

    @staticmethod
    def week_start() -> IDateRule:
        """First day of week (Monday)."""
        return WeekStartRule()

    @staticmethod
    def week_end() -> IDateRule:
        """Last trading day of week (Friday)."""
        return WeekEndRule()

    @staticmethod
    def on(day: DayOfWeek) -> IDateRule:
        """Specific day of week."""
        return DayOfWeekRule(day)

    @staticmethod
    def nth_day(day: DayOfWeek, n: int) -> IDateRule:
        """Nth occurrence of day in month (e.g., 3rd Friday)."""
        return NthDayOfMonthRule(day, n)

    @staticmethod
    def specific_dates(dates: List[date]) -> IDateRule:
        """Specific dates only."""
        return SpecificDatesRule(dates)

    @staticmethod
    def options_expiration() -> IDateRule:
        """Third Friday of each month (standard options expiration)."""
        return NthDayOfMonthRule(DayOfWeek.FRIDAY, 3)


class TimeRules:
    """Factory for time rules (LEAN-style API)."""

    @staticmethod
    def at(hour: int, minute: int = 0, second: int = 0) -> ITimeRule:
        """At specific time."""
        return AtTimeRule(hour, minute, second)

    @staticmethod
    def market_open(offset_minutes: int = 0) -> ITimeRule:
        """At market open with optional offset."""
        return MarketOpenRule(offset_minutes)

    @staticmethod
    def after_market_open(minutes: int) -> ITimeRule:
        """Minutes after market open."""
        return MarketOpenRule(minutes)

    @staticmethod
    def market_close(offset_minutes: int = 0) -> ITimeRule:
        """At market close with offset (positive = before)."""
        return MarketCloseRule(offset_minutes)

    @staticmethod
    def before_market_close(minutes: int) -> ITimeRule:
        """Minutes before market close."""
        return MarketCloseRule(minutes)

    @staticmethod
    def every(minutes: int, start_offset: int = 0) -> EveryNMinutesRule:
        """Every N minutes during market hours."""
        return EveryNMinutesRule(minutes, start_offset)

    @staticmethod
    def noon() -> ITimeRule:
        """At noon (12:00 PM)."""
        return AtTimeRule(12, 0)

    @staticmethod
    def midnight() -> ITimeRule:
        """At midnight (00:00)."""
        return AtTimeRule(0, 0)

