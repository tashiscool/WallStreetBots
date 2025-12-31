"""
Trading Scheduler Module.

Provides scheduled execution of trading functions based on date/time rules.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
import pytz
import threading
import logging

from .rules import IDateRule, ITimeRule, DateRules, TimeRules

logger = logging.getLogger(__name__)


class ScheduleEventType(Enum):
    """Type of scheduled event."""
    BEFORE_MARKET_OPEN = "before_market_open"
    MARKET_OPEN = "market_open"
    DURING_MARKET = "during_market"
    BEFORE_MARKET_CLOSE = "before_market_close"
    MARKET_CLOSE = "market_close"
    AFTER_MARKET_CLOSE = "after_market_close"
    CUSTOM = "custom"


@dataclass
class ScheduledEvent:
    """Represents a scheduled event."""
    name: str
    date_rule: IDateRule
    time_rule: ITimeRule
    callback: Union[Callable[[], None], Callable[[], Coroutine]]
    event_type: ScheduleEventType = ScheduleEventType.CUSTOM
    enabled: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0
    is_async: bool = False

    def __post_init__(self):
        """Detect if callback is async."""
        self.is_async = asyncio.iscoroutinefunction(self.callback)


@dataclass
class MarketHours:
    """Market hours configuration."""
    open_time: time = time(9, 30)
    close_time: time = time(16, 0)
    extended_open: time = time(4, 0)
    extended_close: time = time(20, 0)
    timezone: str = "America/New_York"


class TradingScheduler:
    """
    Trading scheduler for executing functions at specific times.

    Supports both synchronous and async callbacks.
    """

    def __init__(
        self,
        market_hours: Optional[MarketHours] = None,
        timezone: str = "America/New_York",
    ):
        """
        Initialize trading scheduler.

        Args:
            market_hours: Market hours configuration
            timezone: Default timezone
        """
        self.market_hours = market_hours or MarketHours()
        self.timezone = pytz.timezone(timezone)
        self._events: Dict[str, ScheduledEvent] = {}
        self._running = False
        self._lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None

    def schedule(
        self,
        name: str,
        date_rule: IDateRule,
        time_rule: ITimeRule,
        callback: Union[Callable[[], None], Callable[[], Coroutine]],
        event_type: ScheduleEventType = ScheduleEventType.CUSTOM,
    ) -> ScheduledEvent:
        """
        Schedule a function to run based on date/time rules.

        Args:
            name: Event name (must be unique)
            date_rule: Date rule for when to run
            time_rule: Time rule for when to run
            callback: Function to call
            event_type: Type of event

        Returns:
            The scheduled event
        """
        event = ScheduledEvent(
            name=name,
            date_rule=date_rule,
            time_rule=time_rule,
            callback=callback,
            event_type=event_type,
        )

        with self._lock:
            self._events[name] = event

        logger.info(f"Scheduled event: {name}")
        return event

    def on(
        self,
        date_rule: IDateRule,
        time_rule: ITimeRule,
    ) -> Callable:
        """
        Decorator for scheduling functions.

        Usage:
            @scheduler.on(DateRules.weekdays(), TimeRules.market_open())
            def my_function():
                ...
        """
        def decorator(func: Callable) -> Callable:
            name = func.__name__
            self.schedule(name, date_rule, time_rule, func)
            return func
        return decorator

    def before_market_open(
        self,
        minutes: int = 30,
    ) -> Callable:
        """
        Decorator for functions to run before market open.

        Args:
            minutes: Minutes before open

        Usage:
            @scheduler.before_market_open(30)
            def prepare_for_open():
                ...
        """
        def decorator(func: Callable) -> Callable:
            name = f"before_open_{func.__name__}"
            self.schedule(
                name=name,
                date_rule=DateRules.weekdays(),
                time_rule=TimeRules.market_open(-minutes),
                callback=func,
                event_type=ScheduleEventType.BEFORE_MARKET_OPEN,
            )
            return func
        return decorator

    def after_market_open(
        self,
        minutes: int = 0,
    ) -> Callable:
        """
        Decorator for functions to run after market open.

        Args:
            minutes: Minutes after open
        """
        def decorator(func: Callable) -> Callable:
            name = f"after_open_{func.__name__}"
            self.schedule(
                name=name,
                date_rule=DateRules.weekdays(),
                time_rule=TimeRules.after_market_open(minutes),
                callback=func,
                event_type=ScheduleEventType.MARKET_OPEN,
            )
            return func
        return decorator

    def before_market_close(
        self,
        minutes: int = 15,
    ) -> Callable:
        """
        Decorator for functions to run before market close.

        Args:
            minutes: Minutes before close
        """
        def decorator(func: Callable) -> Callable:
            name = f"before_close_{func.__name__}"
            self.schedule(
                name=name,
                date_rule=DateRules.weekdays(),
                time_rule=TimeRules.before_market_close(minutes),
                callback=func,
                event_type=ScheduleEventType.BEFORE_MARKET_CLOSE,
            )
            return func
        return decorator

    def after_market_close(
        self,
        minutes: int = 0,
    ) -> Callable:
        """
        Decorator for functions to run after market close.

        Args:
            minutes: Minutes after close
        """
        def decorator(func: Callable) -> Callable:
            name = f"after_close_{func.__name__}"
            self.schedule(
                name=name,
                date_rule=DateRules.weekdays(),
                time_rule=TimeRules.market_close(-minutes),
                callback=func,
                event_type=ScheduleEventType.AFTER_MARKET_CLOSE,
            )
            return func
        return decorator

    def every(
        self,
        minutes: int = 15,
        during_market_hours: bool = True,
    ) -> Callable:
        """
        Decorator for periodic execution.

        Args:
            minutes: Interval in minutes
            during_market_hours: Only during market hours
        """
        def decorator(func: Callable) -> Callable:
            name = f"every_{minutes}m_{func.__name__}"
            if during_market_hours:
                date_rule = DateRules.weekdays()
            else:
                date_rule = DateRules.every_day()

            self.schedule(
                name=name,
                date_rule=date_rule,
                time_rule=TimeRules.every(minutes),
                callback=func,
                event_type=ScheduleEventType.DURING_MARKET,
            )
            return func
        return decorator

    def unschedule(self, name: str) -> bool:
        """
        Remove a scheduled event.

        Args:
            name: Event name

        Returns:
            True if removed
        """
        with self._lock:
            if name in self._events:
                del self._events[name]
                logger.info(f"Unscheduled event: {name}")
                return True
        return False

    def enable(self, name: str) -> None:
        """Enable a scheduled event."""
        if name in self._events:
            self._events[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a scheduled event."""
        if name in self._events:
            self._events[name].enabled = False

    def get_next_run_time(
        self,
        event: ScheduledEvent,
        from_time: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """
        Get next run time for an event.

        Args:
            event: The scheduled event
            from_time: Starting time (default: now)

        Returns:
            Next scheduled datetime, or None
        """
        if from_time is None:
            from_time = datetime.now(self.timezone)

        current_date = from_time.date()
        end_date = current_date + timedelta(days=365)  # Look ahead 1 year

        for d in event.date_rule.get_dates(current_date, end_date):
            scheduled_time = event.time_rule.get_time(
                d,
                self.market_hours.open_time,
                self.market_hours.close_time,
            )
            scheduled_dt = self.timezone.localize(
                datetime.combine(d, scheduled_time)
            )

            if scheduled_dt > from_time:
                return scheduled_dt

        return None

    def get_pending_events(
        self,
        within_minutes: int = 60,
    ) -> List[tuple]:
        """
        Get events scheduled to run within the next N minutes.

        Args:
            within_minutes: Look-ahead window

        Returns:
            List of (event, scheduled_time) tuples
        """
        now = datetime.now(self.timezone)
        cutoff = now + timedelta(minutes=within_minutes)
        pending = []

        for event in self._events.values():
            if not event.enabled:
                continue

            next_run = self.get_next_run_time(event, now)
            if next_run and next_run <= cutoff:
                pending.append((event, next_run))

        return sorted(pending, key=lambda x: x[1])

    async def _run_event(self, event: ScheduledEvent) -> None:
        """Execute an event callback."""
        try:
            if event.is_async:
                await event.callback()
            else:
                event.callback()

            event.last_run = datetime.now(self.timezone)
            event.run_count += 1
            logger.info(f"Executed scheduled event: {event.name}")

        except Exception as e:
            logger.error(f"Error in scheduled event {event.name}: {e}")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler started")

        while self._running:
            try:
                now = datetime.now(self.timezone)

                for event in list(self._events.values()):
                    if not event.enabled:
                        continue

                    next_run = self.get_next_run_time(event, event.last_run)
                    if next_run and next_run <= now:
                        await self._run_event(event)

                # Sleep until next check (every 1 second)
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)

        logger.info("Scheduler stopped")

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def run_now(self, name: str) -> bool:
        """
        Run an event immediately (synchronously).

        Args:
            name: Event name

        Returns:
            True if executed
        """
        event = self._events.get(name)
        if not event:
            return False

        try:
            if event.is_async:
                asyncio.run(event.callback())
            else:
                event.callback()

            event.last_run = datetime.now(self.timezone)
            event.run_count += 1
            return True

        except Exception as e:
            logger.error(f"Error running event {name}: {e}")
            return False


class LifecycleScheduler(TradingScheduler):
    """
    Scheduler with Lumibot-style lifecycle methods.

    Provides hooks for common trading lifecycle events.
    """

    def __init__(
        self,
        market_hours: Optional[MarketHours] = None,
        timezone: str = "America/New_York",
    ):
        """Initialize lifecycle scheduler."""
        super().__init__(market_hours, timezone)

        # Lifecycle callbacks
        self._on_trading_iteration: Optional[Callable] = None
        self._before_market_opens: Optional[Callable] = None
        self._before_starting_trading: Optional[Callable] = None
        self._before_market_closes: Optional[Callable] = None
        self._after_market_closes: Optional[Callable] = None

    def set_on_trading_iteration(
        self,
        callback: Callable,
        interval_minutes: int = 1,
    ) -> None:
        """Set callback for each trading iteration."""
        self._on_trading_iteration = callback
        self.schedule(
            name="_trading_iteration",
            date_rule=DateRules.weekdays(),
            time_rule=TimeRules.every(interval_minutes),
            callback=callback,
            event_type=ScheduleEventType.DURING_MARKET,
        )

    def set_before_market_opens(
        self,
        callback: Callable,
        minutes_before: int = 30,
    ) -> None:
        """Set callback for before market opens."""
        self._before_market_opens = callback
        self.schedule(
            name="_before_market_opens",
            date_rule=DateRules.weekdays(),
            time_rule=TimeRules.market_open(-minutes_before),
            callback=callback,
            event_type=ScheduleEventType.BEFORE_MARKET_OPEN,
        )

    def set_before_starting_trading(
        self,
        callback: Callable,
    ) -> None:
        """Set callback for market open."""
        self._before_starting_trading = callback
        self.schedule(
            name="_before_starting_trading",
            date_rule=DateRules.weekdays(),
            time_rule=TimeRules.market_open(),
            callback=callback,
            event_type=ScheduleEventType.MARKET_OPEN,
        )

    def set_before_market_closes(
        self,
        callback: Callable,
        minutes_before: int = 15,
    ) -> None:
        """Set callback for before market closes."""
        self._before_market_closes = callback
        self.schedule(
            name="_before_market_closes",
            date_rule=DateRules.weekdays(),
            time_rule=TimeRules.before_market_close(minutes_before),
            callback=callback,
            event_type=ScheduleEventType.BEFORE_MARKET_CLOSE,
        )

    def set_after_market_closes(
        self,
        callback: Callable,
    ) -> None:
        """Set callback for after market closes."""
        self._after_market_closes = callback
        self.schedule(
            name="_after_market_closes",
            date_rule=DateRules.weekdays(),
            time_rule=TimeRules.market_close(),
            callback=callback,
            event_type=ScheduleEventType.AFTER_MARKET_CLOSE,
        )

