"""
Extended Trading Hours Manager

Manages pre-market (4 AM ET) and after-hours (8 PM ET) trading sessions
for US equity markets.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
import pytz

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Trading session types."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


class ExtendedHoursCapability(Enum):
    """Broker capabilities for extended hours."""
    FULL = "full"  # Pre-market + after-hours
    PRE_MARKET_ONLY = "pre_market_only"
    AFTER_HOURS_ONLY = "after_hours_only"
    NONE = "none"


@dataclass
class ExtendedMarketHours:
    """Extended market hours configuration."""
    # Pre-market session
    pre_market_open: time = time(4, 0)  # 4:00 AM ET
    pre_market_close: time = time(9, 30)  # 9:30 AM ET

    # Regular session
    regular_open: time = time(9, 30)  # 9:30 AM ET
    regular_close: time = time(16, 0)  # 4:00 PM ET

    # After-hours session
    after_hours_open: time = time(16, 0)  # 4:00 PM ET
    after_hours_close: time = time(20, 0)  # 8:00 PM ET

    # Optimal trading windows
    optimal_pre_market_start: time = time(7, 0)  # 7:00 AM ET (more liquidity)
    optimal_pre_market_end: time = time(9, 30)  # 9:30 AM ET
    optimal_after_hours_start: time = time(16, 0)  # 4:00 PM ET
    optimal_after_hours_end: time = time(18, 0)  # 6:00 PM ET (best liquidity)

    # Settings
    enable_pre_market: bool = True
    enable_after_hours: bool = True
    timezone: str = "America/New_York"

    def get_session_times(self, session: TradingSession) -> tuple:
        """Get start and end times for a session."""
        if session == TradingSession.PRE_MARKET:
            return (self.pre_market_open, self.pre_market_close)
        elif session == TradingSession.REGULAR:
            return (self.regular_open, self.regular_close)
        elif session == TradingSession.AFTER_HOURS:
            return (self.after_hours_open, self.after_hours_close)
        return (None, None)


@dataclass
class SessionInfo:
    """Information about current trading session."""
    session: TradingSession
    session_start: datetime
    session_end: datetime
    is_optimal: bool
    minutes_until_close: int
    minutes_since_open: int
    next_session: TradingSession
    next_session_start: Optional[datetime] = None


@dataclass
class ExtendedHoursRisk:
    """Risk parameters for extended hours trading."""
    max_position_size_pct: float = 50.0  # Reduce position size in extended hours
    min_spread_threshold: float = 0.5  # Max acceptable spread %
    liquidity_factor: float = 0.5  # Expected liquidity vs regular
    slippage_multiplier: float = 2.0  # Expected slippage increase


@dataclass
class ExtendedHoursOrder:
    """Order with extended hours flag."""
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    order_type: str  # "market" or "limit"
    limit_price: Optional[float] = None
    extended_hours: bool = True
    time_in_force: str = "day"  # "day", "gtc", "ioc", "fok"
    session: Optional[TradingSession] = None


# US Market holidays 2024-2025
US_MARKET_HOLIDAYS = {
    # 2024
    date(2024, 1, 1): "New Year's Day",
    date(2024, 1, 15): "MLK Day",
    date(2024, 2, 19): "Presidents Day",
    date(2024, 3, 29): "Good Friday",
    date(2024, 5, 27): "Memorial Day",
    date(2024, 6, 19): "Juneteenth",
    date(2024, 7, 4): "Independence Day",
    date(2024, 9, 2): "Labor Day",
    date(2024, 11, 28): "Thanksgiving",
    date(2024, 12, 25): "Christmas",
    # 2025
    date(2025, 1, 1): "New Year's Day",
    date(2025, 1, 20): "MLK Day",
    date(2025, 2, 17): "Presidents Day",
    date(2025, 4, 18): "Good Friday",
    date(2025, 5, 26): "Memorial Day",
    date(2025, 6, 19): "Juneteenth",
    date(2025, 7, 4): "Independence Day",
    date(2025, 9, 1): "Labor Day",
    date(2025, 11, 27): "Thanksgiving",
    date(2025, 12, 25): "Christmas",
}

# Early close days (1 PM ET close)
EARLY_CLOSE_DAYS = {
    date(2024, 7, 3): "Day before Independence Day",
    date(2024, 11, 29): "Day after Thanksgiving",
    date(2024, 12, 24): "Christmas Eve",
    date(2025, 7, 3): "Day before Independence Day",
    date(2025, 11, 28): "Day after Thanksgiving",
    date(2025, 12, 24): "Christmas Eve",
}


class ExtendedHoursManager:
    """
    Manages extended trading hours for US equity markets.

    Supports:
    - Pre-market trading (4:00 AM - 9:30 AM ET)
    - Regular hours (9:30 AM - 4:00 PM ET)
    - After-hours trading (4:00 PM - 8:00 PM ET)
    """

    def __init__(
        self,
        config: Optional[ExtendedMarketHours] = None,
        risk_params: Optional[ExtendedHoursRisk] = None,
    ):
        """
        Initialize extended hours manager.

        Args:
            config: Extended hours configuration
            risk_params: Risk parameters for extended hours
        """
        self.config = config or ExtendedMarketHours()
        self.risk = risk_params or ExtendedHoursRisk()
        self.tz = pytz.timezone(self.config.timezone)

    def get_current_session(self) -> SessionInfo:
        """
        Get information about the current trading session.

        Returns:
            SessionInfo with current session details
        """
        now = datetime.now(self.tz)
        current_time = now.time()
        today = now.date()

        # Check if market is closed (weekend or holiday)
        if self._is_market_closed(today):
            return self._create_closed_session_info(now)

        # Check early close
        is_early_close = today in EARLY_CLOSE_DAYS
        effective_close = time(13, 0) if is_early_close else self.config.regular_close

        # Determine current session
        if self._time_in_range(current_time, self.config.pre_market_open, self.config.regular_open):
            session = TradingSession.PRE_MARKET if self.config.enable_pre_market else TradingSession.CLOSED
            session_start = datetime.combine(today, self.config.pre_market_open, self.tz)
            session_end = datetime.combine(today, self.config.regular_open, self.tz)
            is_optimal = self._time_in_range(
                current_time,
                self.config.optimal_pre_market_start,
                self.config.optimal_pre_market_end,
            )
            next_session = TradingSession.REGULAR
            next_session_start = session_end

        elif self._time_in_range(current_time, self.config.regular_open, effective_close):
            session = TradingSession.REGULAR
            session_start = datetime.combine(today, self.config.regular_open, self.tz)
            session_end = datetime.combine(today, effective_close, self.tz)
            is_optimal = True  # Regular hours are always optimal
            next_session = TradingSession.AFTER_HOURS if not is_early_close else TradingSession.CLOSED
            next_session_start = session_end

        elif self._time_in_range(current_time, effective_close, self.config.after_hours_close):
            session = TradingSession.AFTER_HOURS if self.config.enable_after_hours else TradingSession.CLOSED
            session_start = datetime.combine(today, effective_close, self.tz)
            session_end = datetime.combine(today, self.config.after_hours_close, self.tz)
            is_optimal = self._time_in_range(
                current_time,
                self.config.optimal_after_hours_start,
                self.config.optimal_after_hours_end,
            )
            next_session = TradingSession.CLOSED
            next_session_start = None

        else:
            return self._create_closed_session_info(now)

        # Calculate time metrics
        minutes_since_open = int((now - session_start).total_seconds() / 60)
        minutes_until_close = int((session_end - now).total_seconds() / 60)

        return SessionInfo(
            session=session,
            session_start=session_start,
            session_end=session_end,
            is_optimal=is_optimal,
            minutes_until_close=minutes_until_close,
            minutes_since_open=minutes_since_open,
            next_session=next_session,
            next_session_start=next_session_start,
        )

    def _create_closed_session_info(self, now: datetime) -> SessionInfo:
        """Create session info for closed market."""
        next_open = self.get_next_market_open()
        return SessionInfo(
            session=TradingSession.CLOSED,
            session_start=now,
            session_end=now,
            is_optimal=False,
            minutes_until_close=0,
            minutes_since_open=0,
            next_session=TradingSession.PRE_MARKET if self.config.enable_pre_market else TradingSession.REGULAR,
            next_session_start=next_open,
        )

    def is_market_open(self, include_extended: bool = True) -> bool:
        """
        Check if market is currently open.

        Args:
            include_extended: Include extended hours

        Returns:
            True if market is open
        """
        session = self.get_current_session()

        if include_extended:
            return session.session in (
                TradingSession.PRE_MARKET,
                TradingSession.REGULAR,
                TradingSession.AFTER_HOURS,
            )
        else:
            return session.session == TradingSession.REGULAR

    def is_extended_hours(self) -> bool:
        """Check if currently in extended hours."""
        session = self.get_current_session()
        return session.session in (
            TradingSession.PRE_MARKET,
            TradingSession.AFTER_HOURS,
        )

    def can_trade_extended(self, broker_capability: ExtendedHoursCapability) -> bool:
        """
        Check if extended hours trading is possible.

        Args:
            broker_capability: Broker's extended hours capability

        Returns:
            True if trading is possible now
        """
        if broker_capability == ExtendedHoursCapability.NONE:
            return self.get_current_session().session == TradingSession.REGULAR

        session = self.get_current_session()

        if session.session == TradingSession.PRE_MARKET:
            return broker_capability in (
                ExtendedHoursCapability.FULL,
                ExtendedHoursCapability.PRE_MARKET_ONLY,
            )
        elif session.session == TradingSession.AFTER_HOURS:
            return broker_capability in (
                ExtendedHoursCapability.FULL,
                ExtendedHoursCapability.AFTER_HOURS_ONLY,
            )
        elif session.session == TradingSession.REGULAR:
            return True
        else:
            return False

    def get_adjusted_position_size(
        self,
        base_size: float,
        session: Optional[TradingSession] = None,
    ) -> float:
        """
        Get position size adjusted for extended hours risk.

        Args:
            base_size: Base position size
            session: Trading session (current if None)

        Returns:
            Adjusted position size
        """
        if session is None:
            session = self.get_current_session().session

        if session == TradingSession.REGULAR:
            return base_size
        elif session in (TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS):
            return base_size * (self.risk.max_position_size_pct / 100)
        else:
            return 0  # Market closed

    def get_next_market_open(self) -> datetime:
        """Get datetime of next market open."""
        now = datetime.now(self.tz)
        check_date = now.date()

        # If it's before pre-market open today and market is open
        if now.time() < self.config.pre_market_open and not self._is_market_closed(check_date):
            if self.config.enable_pre_market:
                return datetime.combine(check_date, self.config.pre_market_open, self.tz)
            else:
                return datetime.combine(check_date, self.config.regular_open, self.tz)

        # Find next open day
        check_date += timedelta(days=1)
        while self._is_market_closed(check_date):
            check_date += timedelta(days=1)
            if (check_date - now.date()).days > 10:
                break

        if self.config.enable_pre_market:
            return datetime.combine(check_date, self.config.pre_market_open, self.tz)
        else:
            return datetime.combine(check_date, self.config.regular_open, self.tz)

    def get_schedule_for_date(self, target_date: date) -> Dict[str, Any]:
        """
        Get full trading schedule for a specific date.

        Args:
            target_date: Date to get schedule for

        Returns:
            Dictionary with session times
        """
        if self._is_market_closed(target_date):
            reason = US_MARKET_HOLIDAYS.get(target_date, "Weekend")
            return {
                "date": target_date.isoformat(),
                "is_open": False,
                "reason": reason,
                "sessions": [],
            }

        is_early_close = target_date in EARLY_CLOSE_DAYS
        effective_close = time(13, 0) if is_early_close else self.config.regular_close

        sessions = []

        if self.config.enable_pre_market:
            sessions.append({
                "name": "pre_market",
                "start": self.config.pre_market_open.isoformat(),
                "end": self.config.regular_open.isoformat(),
            })

        sessions.append({
            "name": "regular",
            "start": self.config.regular_open.isoformat(),
            "end": effective_close.isoformat(),
        })

        if self.config.enable_after_hours and not is_early_close:
            sessions.append({
                "name": "after_hours",
                "start": effective_close.isoformat(),
                "end": self.config.after_hours_close.isoformat(),
            })

        return {
            "date": target_date.isoformat(),
            "is_open": True,
            "early_close": is_early_close,
            "sessions": sessions,
        }

    def _is_market_closed(self, check_date: date) -> bool:
        """Check if market is closed on a specific date."""
        # Weekend
        if check_date.weekday() >= 5:
            return True
        # Holiday
        if check_date in US_MARKET_HOLIDAYS:
            return True
        return False

    def _time_in_range(self, check_time: time, start: time, end: time) -> bool:
        """Check if time is within range (handles midnight crossing)."""
        if start <= end:
            return start <= check_time < end
        else:
            return check_time >= start or check_time < end

    def create_extended_hours_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
    ) -> ExtendedHoursOrder:
        """
        Create an order with extended hours flag.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            qty: Quantity
            order_type: "market" or "limit" (limit recommended for extended hours)
            limit_price: Limit price (required for limit orders)

        Returns:
            ExtendedHoursOrder
        """
        session = self.get_current_session()

        # Limit orders recommended for extended hours
        if session.session in (TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS):
            if order_type == "market" and limit_price is None:
                logger.warning(
                    "Market orders not recommended in extended hours. "
                    "Consider using limit orders."
                )

        return ExtendedHoursOrder(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            extended_hours=True,
            time_in_force="day",
            session=session.session,
        )


def create_extended_hours_manager(
    enable_pre_market: bool = True,
    enable_after_hours: bool = True,
) -> ExtendedHoursManager:
    """Factory function to create extended hours manager."""
    config = ExtendedMarketHours(
        enable_pre_market=enable_pre_market,
        enable_after_hours=enable_after_hours,
    )
    return ExtendedHoursManager(config)
