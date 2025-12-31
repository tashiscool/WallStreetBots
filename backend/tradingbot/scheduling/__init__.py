"""
Trading Scheduler Module.

Ported from QuantConnect/LEAN's scheduling framework and Lumibot's lifecycle.
Provides DateRules and TimeRules for precise scheduling of trading functions.

Usage:
    from backend.tradingbot.scheduling import (
        TradingScheduler,
        DateRules,
        TimeRules,
    )

    scheduler = TradingScheduler()

    # Schedule at specific times
    scheduler.schedule(
        name="rebalance",
        date_rule=DateRules.month_start(),
        time_rule=TimeRules.after_market_open(30),
        callback=my_rebalance_function,
    )

    # Use decorators
    @scheduler.before_market_close(15)
    def close_positions():
        ...

    # Start scheduler
    await scheduler.start()
"""

from .rules import (
    DayOfWeek,
    IDateRule,
    ITimeRule,
    EveryDayRule,
    WeekdaysRule,
    WeekendRule,
    DayOfWeekRule,
    MonthStartRule,
    MonthEndRule,
    WeekStartRule,
    WeekEndRule,
    NthDayOfMonthRule,
    SpecificDatesRule,
    ExcludeRule,
    AtTimeRule,
    MarketOpenRule,
    MarketCloseRule,
    EveryNMinutesRule,
    DateRules,
    TimeRules,
)

from .scheduler import (
    ScheduleEventType,
    ScheduledEvent,
    MarketHours,
    TradingScheduler,
    LifecycleScheduler,
)

__all__ = [
    # Enums
    "DayOfWeek",
    "ScheduleEventType",
    # Interfaces
    "IDateRule",
    "ITimeRule",
    # Date Rules
    "EveryDayRule",
    "WeekdaysRule",
    "WeekendRule",
    "DayOfWeekRule",
    "MonthStartRule",
    "MonthEndRule",
    "WeekStartRule",
    "WeekEndRule",
    "NthDayOfMonthRule",
    "SpecificDatesRule",
    "ExcludeRule",
    # Time Rules
    "AtTimeRule",
    "MarketOpenRule",
    "MarketCloseRule",
    "EveryNMinutesRule",
    # Factory Classes
    "DateRules",
    "TimeRules",
    # Scheduler
    "ScheduledEvent",
    "MarketHours",
    "TradingScheduler",
    "LifecycleScheduler",
]
