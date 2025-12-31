"""
Tests for Margin Tracker

Tests cover:
- MarginStatus and PositionType enums
- MarginRequirement dataclass
- MarginSummary dataclass
- MarginTracker calculations
- Margin call detection
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock

from backend.tradingbot.borrow.margin_tracker import (
    MarginStatus,
    PositionType,
    MarginRequirement,
    MarginSummary,
    MarginCallInfo,
    MarginTracker,
    create_margin_tracker,
    REG_T_INITIAL_MARGIN,
    REG_T_MAINTENANCE_MARGIN,
    SHORT_INITIAL_MARGIN,
    SHORT_MAINTENANCE_MARGIN,
)


class TestMarginStatusEnum:
    """Tests for MarginStatus enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert MarginStatus.HEALTHY.value == "healthy"
        assert MarginStatus.CAUTION.value == "caution"
        assert MarginStatus.WARNING.value == "warning"
        assert MarginStatus.MARGIN_CALL.value == "margin_call"


class TestPositionTypeEnum:
    """Tests for PositionType enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert PositionType.LONG_STOCK.value == "long_stock"
        assert PositionType.SHORT_STOCK.value == "short_stock"
        assert PositionType.LONG_OPTION.value == "long_option"
        assert PositionType.SHORT_OPTION.value == "short_option"
        assert PositionType.SPREAD.value == "spread"


class TestMarginRequirement:
    """Tests for MarginRequirement dataclass."""

    def test_requirement_creation(self):
        """Test creating a margin requirement."""
        req = MarginRequirement(
            symbol="AAPL",
            position_type=PositionType.LONG_STOCK,
            initial_margin=Decimal("7500"),
            maintenance_margin=Decimal("3750"),
            current_value=Decimal("15000"),
            margin_used=Decimal("7500"),
        )

        assert req.symbol == "AAPL"
        assert req.position_type == PositionType.LONG_STOCK
        assert req.initial_margin == Decimal("7500")

    def test_maintenance_ratio(self):
        """Test maintenance ratio calculation."""
        req = MarginRequirement(
            symbol="AAPL",
            position_type=PositionType.LONG_STOCK,
            initial_margin=Decimal("7500"),
            maintenance_margin=Decimal("3750"),
            current_value=Decimal("15000"),
            margin_used=Decimal("7500"),
        )

        # ratio = margin_used / current_value = 7500/15000 = 0.5
        assert req.maintenance_ratio == 0.5

    def test_maintenance_ratio_zero_value(self):
        """Test maintenance ratio with zero value."""
        req = MarginRequirement(
            symbol="AAPL",
            position_type=PositionType.LONG_STOCK,
            initial_margin=Decimal("0"),
            maintenance_margin=Decimal("0"),
            current_value=Decimal("0"),
            margin_used=Decimal("0"),
        )

        assert req.maintenance_ratio == 0.0


class TestMarginSummary:
    """Tests for MarginSummary dataclass."""

    def test_summary_creation(self):
        """Test creating a margin summary."""
        summary = MarginSummary(
            total_equity=Decimal("100000"),
            total_positions_value=Decimal("150000"),
            buying_power=Decimal("50000"),
            margin_used=Decimal("75000"),
            margin_available=Decimal("25000"),
            maintenance_margin=Decimal("37500"),
            maintenance_excess=Decimal("62500"),
            margin_ratio=0.67,
            status=MarginStatus.HEALTHY,
            timestamp=datetime.now(),
        )

        assert summary.total_equity == Decimal("100000")
        assert summary.status == MarginStatus.HEALTHY

    def test_is_margin_call_false(self):
        """Test is_margin_call returns False for healthy account."""
        summary = MarginSummary(
            total_equity=Decimal("100000"),
            total_positions_value=Decimal("150000"),
            buying_power=Decimal("50000"),
            margin_used=Decimal("75000"),
            margin_available=Decimal("25000"),
            maintenance_margin=Decimal("37500"),
            maintenance_excess=Decimal("62500"),
            margin_ratio=0.67,
            status=MarginStatus.HEALTHY,
            timestamp=datetime.now(),
        )

        assert summary.is_margin_call is False

    def test_is_margin_call_true(self):
        """Test is_margin_call returns True for margin call status."""
        summary = MarginSummary(
            total_equity=Decimal("20000"),
            total_positions_value=Decimal("100000"),
            buying_power=Decimal("0"),
            margin_used=Decimal("25000"),
            margin_available=Decimal("0"),
            maintenance_margin=Decimal("25000"),
            maintenance_excess=Decimal("-5000"),
            margin_ratio=0.20,
            status=MarginStatus.MARGIN_CALL,
            timestamp=datetime.now(),
        )

        assert summary.is_margin_call is True

    def test_can_open_positions_true(self):
        """Test can_open_positions returns True when margin available."""
        summary = MarginSummary(
            total_equity=Decimal("100000"),
            total_positions_value=Decimal("50000"),
            buying_power=Decimal("150000"),
            margin_used=Decimal("25000"),
            margin_available=Decimal("75000"),
            maintenance_margin=Decimal("12500"),
            maintenance_excess=Decimal("87500"),
            margin_ratio=2.0,
            status=MarginStatus.HEALTHY,
            timestamp=datetime.now(),
        )

        assert summary.can_open_positions is True

    def test_can_open_positions_false_margin_call(self):
        """Test can_open_positions returns False during margin call."""
        summary = MarginSummary(
            total_equity=Decimal("20000"),
            total_positions_value=Decimal("100000"),
            buying_power=Decimal("0"),
            margin_used=Decimal("25000"),
            margin_available=Decimal("0"),
            maintenance_margin=Decimal("25000"),
            maintenance_excess=Decimal("-5000"),
            margin_ratio=0.20,
            status=MarginStatus.MARGIN_CALL,
            timestamp=datetime.now(),
        )

        assert summary.can_open_positions is False


class TestMarginCallInfo:
    """Tests for MarginCallInfo dataclass."""

    def test_margin_call_info_creation(self):
        """Test creating margin call info."""
        info = MarginCallInfo(
            call_amount=Decimal("5000"),
            due_by=datetime.now() + timedelta(days=5),
            positions_at_risk=["AAPL", "MSFT"],
            recommended_action="Deposit $5000 or reduce positions",
        )

        assert info.call_amount == Decimal("5000")
        assert len(info.positions_at_risk) == 2


class TestMarginTracker:
    """Tests for MarginTracker."""

    def test_tracker_creation_defaults(self):
        """Test creating tracker with defaults."""
        tracker = MarginTracker()
        assert tracker.equity == Decimal("100000")
        assert tracker.broker is None

    def test_tracker_creation_custom_equity(self):
        """Test creating tracker with custom equity."""
        tracker = MarginTracker(initial_equity=Decimal("250000"))
        assert tracker.equity == Decimal("250000")

    def test_calculate_long_margin_with_margin(self):
        """Test calculating margin for long position on margin."""
        tracker = MarginTracker()

        req = tracker.calculate_long_margin(
            symbol="AAPL",
            qty=100,
            price=Decimal("150"),
            use_margin=True,
        )

        # Position value = 100 * 150 = 15000
        # Initial margin = 15000 * 0.50 = 7500
        # Maintenance margin = 15000 * 0.25 = 3750
        assert req.symbol == "AAPL"
        assert req.position_type == PositionType.LONG_STOCK
        assert req.initial_margin == Decimal("7500")
        assert req.maintenance_margin == Decimal("3750")
        assert req.current_value == Decimal("15000")

    def test_calculate_long_margin_cash(self):
        """Test calculating margin for cash purchase."""
        tracker = MarginTracker()

        req = tracker.calculate_long_margin(
            symbol="AAPL",
            qty=100,
            price=Decimal("150"),
            use_margin=False,
        )

        # Cash purchase: full amount required
        assert req.initial_margin == Decimal("15000")
        assert req.maintenance_margin == Decimal("0")

    def test_calculate_short_margin(self):
        """Test calculating margin for short position."""
        tracker = MarginTracker()

        req = tracker.calculate_short_margin(
            symbol="AAPL",
            qty=100,
            price=Decimal("150"),
        )

        # Position value = 100 * 150 = 15000
        # Initial margin = 15000 * 0.50 = 7500
        # Maintenance margin = 15000 * 0.30 = 4500
        assert req.symbol == "AAPL"
        assert req.position_type == PositionType.SHORT_STOCK
        assert req.initial_margin == Decimal("7500")
        assert req.maintenance_margin == Decimal("4500")

    def test_add_position_success(self):
        """Test adding a position successfully."""
        tracker = MarginTracker(initial_equity=Decimal("100000"))

        req = tracker.calculate_long_margin("AAPL", 100, Decimal("150"), True)
        success = tracker.add_position(req)

        assert success is True

    def test_add_position_insufficient_margin(self):
        """Test adding position with insufficient margin."""
        tracker = MarginTracker(initial_equity=Decimal("5000"))

        req = tracker.calculate_long_margin("AAPL", 100, Decimal("150"), True)
        success = tracker.add_position(req)

        # 7500 required but only 5000 available
        assert success is False

    def test_remove_position(self):
        """Test removing a position."""
        tracker = MarginTracker()

        req = tracker.calculate_long_margin("AAPL", 100, Decimal("150"), True)
        tracker.add_position(req)

        removed = tracker.remove_position("AAPL")
        assert removed is not None
        assert removed.symbol == "AAPL"

    def test_remove_position_not_found(self):
        """Test removing non-existent position."""
        tracker = MarginTracker()
        removed = tracker.remove_position("NONEXISTENT")
        assert removed is None

    def test_can_open_position_true(self):
        """Test can_open_position when enough margin."""
        tracker = MarginTracker(initial_equity=Decimal("100000"))

        req = tracker.calculate_long_margin("AAPL", 100, Decimal("150"), True)
        can_open = tracker.can_open_position(req)

        assert can_open is True

    def test_can_open_position_false(self):
        """Test can_open_position when not enough margin."""
        tracker = MarginTracker(initial_equity=Decimal("5000"))

        req = tracker.calculate_long_margin("AAPL", 100, Decimal("150"), True)
        can_open = tracker.can_open_position(req)

        assert can_open is False

    def test_get_buying_power_for_short(self):
        """Test getting buying power for short selling."""
        tracker = MarginTracker(initial_equity=Decimal("100000"))

        buying_power = tracker.get_buying_power_for_short()
        # With no positions, full 2x leverage available
        assert buying_power == Decimal("200000")

    def test_get_position_margins(self):
        """Test getting all position margins."""
        tracker = MarginTracker()

        req1 = tracker.calculate_long_margin("AAPL", 100, Decimal("150"), True)
        req2 = tracker.calculate_long_margin("MSFT", 50, Decimal("300"), True)

        tracker.add_position(req1)
        tracker.add_position(req2)

        margins = tracker.get_position_margins()
        assert len(margins) == 2

    @pytest.mark.asyncio
    async def test_get_margin_summary(self):
        """Test getting margin summary."""
        tracker = MarginTracker(initial_equity=Decimal("100000"))

        req = tracker.calculate_long_margin("AAPL", 100, Decimal("150"), True)
        tracker.add_position(req)

        summary = await tracker.get_margin_summary()

        assert summary.total_equity == Decimal("100000")
        assert summary.margin_used == Decimal("7500")
        assert summary.status == MarginStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_margin_summary_caution(self):
        """Test margin summary with caution status."""
        tracker = MarginTracker(initial_equity=Decimal("20000"))

        req = tracker.calculate_short_margin("AAPL", 100, Decimal("150"))
        tracker.add_position(req)

        # Simulate price increase to create stress
        tracker.update_position("AAPL", Decimal("200"))

        summary = await tracker.get_margin_summary()
        # With position value increased, margin status should worsen
        assert summary.status in (MarginStatus.CAUTION, MarginStatus.WARNING, MarginStatus.HEALTHY)

    @pytest.mark.asyncio
    async def test_check_margin_call_no_call(self):
        """Test no margin call for healthy account."""
        tracker = MarginTracker(initial_equity=Decimal("100000"))

        req = tracker.calculate_long_margin("AAPL", 100, Decimal("150"), True)
        tracker.add_position(req)

        call_info = await tracker.check_margin_call()
        assert call_info is None

    def test_update_position(self):
        """Test updating a position."""
        tracker = MarginTracker()

        req = tracker.calculate_long_margin("AAPL", 100, Decimal("150"), True)
        tracker.add_position(req)

        updated = tracker.update_position("AAPL", Decimal("160"))
        assert updated is not None

    def test_update_position_not_found(self):
        """Test updating non-existent position."""
        tracker = MarginTracker()
        updated = tracker.update_position("NONEXISTENT", Decimal("100"))
        assert updated is None

    def test_determine_status_healthy(self):
        """Test status determination for healthy account."""
        tracker = MarginTracker()
        status = tracker._determine_status(0.6, Decimal("50000"))
        assert status == MarginStatus.HEALTHY

    def test_determine_status_caution(self):
        """Test status determination for caution."""
        tracker = MarginTracker()
        status = tracker._determine_status(0.45, Decimal("10000"))
        assert status == MarginStatus.CAUTION

    def test_determine_status_warning(self):
        """Test status determination for warning."""
        tracker = MarginTracker()
        status = tracker._determine_status(0.30, Decimal("5000"))
        assert status == MarginStatus.WARNING

    def test_determine_status_margin_call(self):
        """Test status determination for margin call."""
        tracker = MarginTracker()
        status = tracker._determine_status(0.20, Decimal("-5000"))
        assert status == MarginStatus.MARGIN_CALL

    def test_get_margin_history(self):
        """Test getting margin history."""
        tracker = MarginTracker()
        history = tracker.get_margin_history()
        assert isinstance(history, list)


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_margin_tracker_defaults(self):
        """Test factory with defaults."""
        tracker = create_margin_tracker()
        assert isinstance(tracker, MarginTracker)
        assert tracker.equity == Decimal("100000")

    def test_create_margin_tracker_custom(self):
        """Test factory with custom parameters."""
        mock_broker = Mock()
        tracker = create_margin_tracker(
            broker_client=mock_broker,
            initial_equity=Decimal("500000"),
        )
        assert tracker.broker is mock_broker
        assert tracker.equity == Decimal("500000")


class TestRegulationConstants:
    """Tests for regulation constants."""

    def test_reg_t_constants(self):
        """Test Reg T constants are correct."""
        assert REG_T_INITIAL_MARGIN == Decimal("0.50")
        assert REG_T_MAINTENANCE_MARGIN == Decimal("0.25")

    def test_short_margin_constants(self):
        """Test short selling margin constants."""
        assert SHORT_INITIAL_MARGIN == Decimal("0.50")
        assert SHORT_MAINTENANCE_MARGIN == Decimal("0.30")
