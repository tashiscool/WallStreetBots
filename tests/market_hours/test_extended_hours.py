"""
Tests for Extended Hours Manager

Tests cover:
- TradingSession and ExtendedHoursCapability enums
- ExtendedMarketHours configuration
- SessionInfo dataclass
- ExtendedHoursManager session detection
- Holiday and early close handling
"""

import pytest
from datetime import date, datetime, time, timedelta
from unittest.mock import Mock, patch
import pytz

from backend.tradingbot.market_hours.extended_hours import (
    TradingSession,
    ExtendedHoursCapability,
    ExtendedMarketHours,
    SessionInfo,
    ExtendedHoursRisk,
    ExtendedHoursOrder,
    ExtendedHoursManager,
    create_extended_hours_manager,
    US_MARKET_HOLIDAYS,
    EARLY_CLOSE_DAYS,
)


class TestTradingSessionEnum:
    """Tests for TradingSession enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert TradingSession.PRE_MARKET.value == "pre_market"
        assert TradingSession.REGULAR.value == "regular"
        assert TradingSession.AFTER_HOURS.value == "after_hours"
        assert TradingSession.CLOSED.value == "closed"


class TestExtendedHoursCapabilityEnum:
    """Tests for ExtendedHoursCapability enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert ExtendedHoursCapability.FULL.value == "full"
        assert ExtendedHoursCapability.PRE_MARKET_ONLY.value == "pre_market_only"
        assert ExtendedHoursCapability.AFTER_HOURS_ONLY.value == "after_hours_only"
        assert ExtendedHoursCapability.NONE.value == "none"


class TestExtendedMarketHours:
    """Tests for ExtendedMarketHours configuration."""

    def test_default_times(self):
        """Test default market times."""
        config = ExtendedMarketHours()

        assert config.pre_market_open == time(4, 0)
        assert config.pre_market_close == time(9, 30)
        assert config.regular_open == time(9, 30)
        assert config.regular_close == time(16, 0)
        assert config.after_hours_open == time(16, 0)
        assert config.after_hours_close == time(20, 0)

    def test_optimal_times(self):
        """Test optimal trading window times."""
        config = ExtendedMarketHours()

        assert config.optimal_pre_market_start == time(7, 0)
        assert config.optimal_after_hours_end == time(18, 0)

    def test_default_settings(self):
        """Test default settings."""
        config = ExtendedMarketHours()

        assert config.enable_pre_market is True
        assert config.enable_after_hours is True
        assert config.timezone == "America/New_York"

    def test_get_session_times_pre_market(self):
        """Test getting pre-market session times."""
        config = ExtendedMarketHours()
        start, end = config.get_session_times(TradingSession.PRE_MARKET)

        assert start == time(4, 0)
        assert end == time(9, 30)

    def test_get_session_times_regular(self):
        """Test getting regular session times."""
        config = ExtendedMarketHours()
        start, end = config.get_session_times(TradingSession.REGULAR)

        assert start == time(9, 30)
        assert end == time(16, 0)

    def test_get_session_times_after_hours(self):
        """Test getting after-hours session times."""
        config = ExtendedMarketHours()
        start, end = config.get_session_times(TradingSession.AFTER_HOURS)

        assert start == time(16, 0)
        assert end == time(20, 0)

    def test_get_session_times_closed(self):
        """Test getting closed session times."""
        config = ExtendedMarketHours()
        start, end = config.get_session_times(TradingSession.CLOSED)

        assert start is None
        assert end is None


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_session_info_creation(self):
        """Test creating session info."""
        now = datetime.now()
        info = SessionInfo(
            session=TradingSession.REGULAR,
            session_start=now,
            session_end=now + timedelta(hours=6),
            is_optimal=True,
            minutes_until_close=360,
            minutes_since_open=30,
            next_session=TradingSession.AFTER_HOURS,
        )

        assert info.session == TradingSession.REGULAR
        assert info.is_optimal is True
        assert info.minutes_until_close == 360


class TestExtendedHoursRisk:
    """Tests for ExtendedHoursRisk dataclass."""

    def test_risk_defaults(self):
        """Test default risk parameters."""
        risk = ExtendedHoursRisk()

        assert risk.max_position_size_pct == 50.0
        assert risk.min_spread_threshold == 0.5
        assert risk.liquidity_factor == 0.5
        assert risk.slippage_multiplier == 2.0


class TestExtendedHoursOrder:
    """Tests for ExtendedHoursOrder dataclass."""

    def test_order_creation(self):
        """Test creating extended hours order."""
        order = ExtendedHoursOrder(
            symbol="AAPL",
            side="buy",
            qty=100.0,
            order_type="limit",
            limit_price=150.0,
            extended_hours=True,
        )

        assert order.symbol == "AAPL"
        assert order.extended_hours is True
        assert order.time_in_force == "day"


class TestExtendedHoursManager:
    """Tests for ExtendedHoursManager."""

    def test_manager_creation_defaults(self):
        """Test creating manager with defaults."""
        manager = ExtendedHoursManager()
        assert manager.config.enable_pre_market is True
        assert manager.config.enable_after_hours is True

    def test_manager_creation_custom_config(self):
        """Test creating manager with custom config."""
        config = ExtendedMarketHours(enable_pre_market=False)
        manager = ExtendedHoursManager(config=config)
        assert manager.config.enable_pre_market is False

    def test_time_in_range_normal(self):
        """Test time_in_range for normal range."""
        manager = ExtendedHoursManager()
        assert manager._time_in_range(time(10, 0), time(9, 0), time(16, 0)) is True
        assert manager._time_in_range(time(8, 0), time(9, 0), time(16, 0)) is False
        assert manager._time_in_range(time(17, 0), time(9, 0), time(16, 0)) is False

    def test_is_market_closed_weekend(self):
        """Test market is closed on weekend."""
        manager = ExtendedHoursManager()
        # Saturday
        saturday = date(2024, 12, 28)
        assert manager._is_market_closed(saturday) is True

    def test_is_market_closed_holiday(self):
        """Test market is closed on holiday."""
        manager = ExtendedHoursManager()
        # Christmas 2024
        christmas = date(2024, 12, 25)
        assert manager._is_market_closed(christmas) is True

    def test_is_market_closed_regular_day(self):
        """Test market is open on regular day."""
        manager = ExtendedHoursManager()
        # A Monday
        monday = date(2024, 12, 30)
        assert manager._is_market_closed(monday) is False

    def test_is_extended_hours(self):
        """Test is_extended_hours check."""
        manager = ExtendedHoursManager()

        # Mock get_current_session to return pre-market
        with patch.object(manager, 'get_current_session') as mock_session:
            mock_session.return_value = SessionInfo(
                session=TradingSession.PRE_MARKET,
                session_start=datetime.now(),
                session_end=datetime.now(),
                is_optimal=False,
                minutes_until_close=0,
                minutes_since_open=0,
                next_session=TradingSession.REGULAR,
            )
            assert manager.is_extended_hours() is True

            mock_session.return_value = SessionInfo(
                session=TradingSession.REGULAR,
                session_start=datetime.now(),
                session_end=datetime.now(),
                is_optimal=True,
                minutes_until_close=0,
                minutes_since_open=0,
                next_session=TradingSession.AFTER_HOURS,
            )
            assert manager.is_extended_hours() is False

    def test_is_market_open_with_extended(self):
        """Test is_market_open including extended hours."""
        manager = ExtendedHoursManager()

        with patch.object(manager, 'get_current_session') as mock_session:
            mock_session.return_value = SessionInfo(
                session=TradingSession.PRE_MARKET,
                session_start=datetime.now(),
                session_end=datetime.now(),
                is_optimal=False,
                minutes_until_close=0,
                minutes_since_open=0,
                next_session=TradingSession.REGULAR,
            )
            assert manager.is_market_open(include_extended=True) is True
            assert manager.is_market_open(include_extended=False) is False

    def test_can_trade_extended_full(self):
        """Test can_trade_extended with full capability."""
        manager = ExtendedHoursManager()

        with patch.object(manager, 'get_current_session') as mock_session:
            mock_session.return_value = SessionInfo(
                session=TradingSession.PRE_MARKET,
                session_start=datetime.now(),
                session_end=datetime.now(),
                is_optimal=False,
                minutes_until_close=0,
                minutes_since_open=0,
                next_session=TradingSession.REGULAR,
            )
            assert manager.can_trade_extended(ExtendedHoursCapability.FULL) is True

    def test_can_trade_extended_none(self):
        """Test can_trade_extended with no capability."""
        manager = ExtendedHoursManager()

        with patch.object(manager, 'get_current_session') as mock_session:
            mock_session.return_value = SessionInfo(
                session=TradingSession.PRE_MARKET,
                session_start=datetime.now(),
                session_end=datetime.now(),
                is_optimal=False,
                minutes_until_close=0,
                minutes_since_open=0,
                next_session=TradingSession.REGULAR,
            )
            assert manager.can_trade_extended(ExtendedHoursCapability.NONE) is False

    def test_can_trade_extended_pre_market_only(self):
        """Test can_trade_extended with pre-market only."""
        manager = ExtendedHoursManager()

        with patch.object(manager, 'get_current_session') as mock_session:
            # Pre-market session
            mock_session.return_value = SessionInfo(
                session=TradingSession.PRE_MARKET,
                session_start=datetime.now(),
                session_end=datetime.now(),
                is_optimal=False,
                minutes_until_close=0,
                minutes_since_open=0,
                next_session=TradingSession.REGULAR,
            )
            assert manager.can_trade_extended(ExtendedHoursCapability.PRE_MARKET_ONLY) is True

            # After-hours session
            mock_session.return_value = SessionInfo(
                session=TradingSession.AFTER_HOURS,
                session_start=datetime.now(),
                session_end=datetime.now(),
                is_optimal=False,
                minutes_until_close=0,
                minutes_since_open=0,
                next_session=TradingSession.CLOSED,
            )
            assert manager.can_trade_extended(ExtendedHoursCapability.PRE_MARKET_ONLY) is False

    def test_get_adjusted_position_size_regular(self):
        """Test position size adjustment during regular hours."""
        manager = ExtendedHoursManager()
        size = manager.get_adjusted_position_size(1000.0, TradingSession.REGULAR)
        assert size == 1000.0

    def test_get_adjusted_position_size_extended(self):
        """Test position size adjustment during extended hours."""
        manager = ExtendedHoursManager()
        size = manager.get_adjusted_position_size(1000.0, TradingSession.PRE_MARKET)
        # 50% reduction
        assert size == 500.0

    def test_get_adjusted_position_size_closed(self):
        """Test position size adjustment when closed."""
        manager = ExtendedHoursManager()
        size = manager.get_adjusted_position_size(1000.0, TradingSession.CLOSED)
        assert size == 0

    def test_get_schedule_for_date_holiday(self):
        """Test schedule for a holiday."""
        manager = ExtendedHoursManager()
        christmas = date(2024, 12, 25)
        schedule = manager.get_schedule_for_date(christmas)

        assert schedule["is_open"] is False
        assert schedule["reason"] == "Christmas"
        assert len(schedule["sessions"]) == 0

    def test_get_schedule_for_date_regular(self):
        """Test schedule for a regular day."""
        manager = ExtendedHoursManager()
        monday = date(2024, 12, 30)
        schedule = manager.get_schedule_for_date(monday)

        assert schedule["is_open"] is True
        assert schedule["early_close"] is False
        assert len(schedule["sessions"]) == 3  # pre, regular, after

    def test_get_schedule_for_date_early_close(self):
        """Test schedule for an early close day."""
        manager = ExtendedHoursManager()
        christmas_eve = date(2024, 12, 24)
        schedule = manager.get_schedule_for_date(christmas_eve)

        assert schedule["is_open"] is True
        assert schedule["early_close"] is True

    def test_create_extended_hours_order(self):
        """Test creating an extended hours order."""
        manager = ExtendedHoursManager()

        with patch.object(manager, 'get_current_session') as mock_session:
            mock_session.return_value = SessionInfo(
                session=TradingSession.PRE_MARKET,
                session_start=datetime.now(),
                session_end=datetime.now(),
                is_optimal=False,
                minutes_until_close=0,
                minutes_since_open=0,
                next_session=TradingSession.REGULAR,
            )

            order = manager.create_extended_hours_order(
                symbol="AAPL",
                side="buy",
                qty=100,
                order_type="limit",
                limit_price=150.0,
            )

            assert order.symbol == "AAPL"
            assert order.extended_hours is True
            assert order.session == TradingSession.PRE_MARKET

    def test_get_next_market_open(self):
        """Test getting next market open."""
        manager = ExtendedHoursManager()
        next_open = manager.get_next_market_open()

        assert next_open is not None
        assert next_open > datetime.now(pytz.timezone("America/New_York"))


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_extended_hours_manager_defaults(self):
        """Test factory with defaults."""
        manager = create_extended_hours_manager()
        assert manager.config.enable_pre_market is True
        assert manager.config.enable_after_hours is True

    def test_create_extended_hours_manager_custom(self):
        """Test factory with custom settings."""
        manager = create_extended_hours_manager(
            enable_pre_market=False,
            enable_after_hours=True,
        )
        assert manager.config.enable_pre_market is False
        assert manager.config.enable_after_hours is True


class TestMarketHolidaysData:
    """Tests for market holidays data."""

    def test_holidays_exist(self):
        """Test holidays data exists."""
        assert len(US_MARKET_HOLIDAYS) > 0

    def test_christmas_in_holidays(self):
        """Test Christmas is in holidays."""
        christmas_2024 = date(2024, 12, 25)
        assert christmas_2024 in US_MARKET_HOLIDAYS

    def test_early_close_days_exist(self):
        """Test early close days data exists."""
        assert len(EARLY_CLOSE_DAYS) > 0

    def test_christmas_eve_early_close(self):
        """Test Christmas Eve is an early close day."""
        christmas_eve_2024 = date(2024, 12, 24)
        assert christmas_eve_2024 in EARLY_CLOSE_DAYS
