"""
Tests for Enhanced Short Selling and Borrow Management.

Tests margin tracking and borrow functionality:
- Borrow client initialization
- Dynamic borrow rate tracking
- Margin requirement calculations
- Short squeeze detection
- Reg SHO compliance
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
from datetime import datetime, timedelta

from backend.tradingbot.borrow.enhanced_borrow_client import (
    EnhancedBorrowClient,
    EnhancedLocateQuote,
    BorrowDifficulty,
    ShortSqueezeRisk,
    ShortPosition,
    BorrowRateHistory,
)
from backend.tradingbot.borrow.margin_tracker import (
    MarginTracker,
    MarginRequirement,
    MarginSummary,
    MarginCallInfo,
    MarginStatus,
    PositionType,
)


class TestEnhancedLocateQuote:
    """Tests for EnhancedLocateQuote dataclass."""

    def test_locate_quote_initialization(self):
        """Test locate quote initializes correctly."""
        quote = EnhancedLocateQuote(
            symbol="GME",
            available=True,
            shares_available=50000,
            borrow_rate_bps=150.0,  # 150 basis points = 1.5%
            borrow_rate_pct=1.50,
            difficulty=BorrowDifficulty.HARD_TO_BORROW,
            timestamp=datetime.now()
        )
        assert quote.symbol == "GME"
        assert quote.borrow_rate_bps == 150.0
        assert quote.difficulty == BorrowDifficulty.HARD_TO_BORROW

    def test_daily_cost_per_share(self):
        """Test daily borrow cost calculation."""
        quote = EnhancedLocateQuote(
            symbol="AMC",
            available=True,
            shares_available=100000,
            borrow_rate_bps=3650.0,  # 36.5% annual
            borrow_rate_pct=36.50,
            difficulty=BorrowDifficulty.HARD_TO_BORROW,
            timestamp=datetime.now()
        )
        # Daily cost per share at $100 = 100 * (36.5/100) / 365
        daily_cost = quote.daily_cost_per_share(Decimal("100.00"))
        expected = Decimal("100.00") * Decimal("36.50") / 100 / 365
        assert abs(daily_cost - expected) < Decimal("0.001")

    def test_is_htb(self):
        """Test hard-to-borrow detection."""
        easy_quote = EnhancedLocateQuote(
            symbol="AAPL",
            available=True,
            shares_available=1000000,
            borrow_rate_bps=15.0,
            borrow_rate_pct=0.15,
            difficulty=BorrowDifficulty.EASY_TO_BORROW,
            timestamp=datetime.now()
        )
        assert easy_quote.is_htb is False

        htb_quote = EnhancedLocateQuote(
            symbol="GME",
            available=True,
            shares_available=5000,
            borrow_rate_bps=200.0,
            borrow_rate_pct=2.00,
            difficulty=BorrowDifficulty.HARD_TO_BORROW,
            timestamp=datetime.now()
        )
        assert htb_quote.is_htb is True


class TestBorrowDifficulty:
    """Tests for BorrowDifficulty enumeration."""

    def test_difficulty_levels(self):
        """Test all difficulty levels exist."""
        assert BorrowDifficulty.EASY_TO_BORROW.value == "easy"
        assert BorrowDifficulty.MODERATE.value == "moderate"
        assert BorrowDifficulty.HARD_TO_BORROW.value == "htb"
        assert BorrowDifficulty.VERY_HARD.value == "very_hard"
        assert BorrowDifficulty.NO_BORROW.value == "no_borrow"


class TestEnhancedBorrowClient:
    """Tests for EnhancedBorrowClient."""

    def test_client_initialization(self):
        """Test borrow client initializes correctly."""
        client = EnhancedBorrowClient()
        assert client is not None
        assert client.use_real_rates is False

    def test_client_with_broker(self):
        """Test client with broker connection."""
        mock_broker = Mock()
        client = EnhancedBorrowClient(
            broker_client=mock_broker,
            use_real_rates=True
        )
        assert client.use_real_rates is True

    @pytest.mark.asyncio
    async def test_get_locate_quote(self):
        """Test fetching locate quote for a symbol."""
        client = EnhancedBorrowClient()

        quote = await client.get_locate_quote("AAPL", 100)
        assert quote.symbol == "AAPL"
        assert quote.available is True

    @pytest.mark.asyncio
    async def test_get_locate_quote_htb(self):
        """Test fetching locate quote for HTB stock."""
        client = EnhancedBorrowClient()

        quote = await client.get_locate_quote("GME", 100)
        assert quote.symbol == "GME"
        assert quote.is_htb is True  # GME is in KNOWN_HTB_STOCKS

    def test_classify_difficulty(self):
        """Test difficulty classification by rate."""
        client = EnhancedBorrowClient()

        # Easy to borrow
        assert client._classify_difficulty(20.0) == BorrowDifficulty.EASY_TO_BORROW
        # Moderate
        assert client._classify_difficulty(50.0) == BorrowDifficulty.MODERATE
        # Hard to borrow
        assert client._classify_difficulty(150.0) == BorrowDifficulty.HARD_TO_BORROW
        # Very hard
        assert client._classify_difficulty(400.0) == BorrowDifficulty.VERY_HARD
        # No borrow
        assert client._classify_difficulty(600.0) == BorrowDifficulty.NO_BORROW


class TestMarginRequirement:
    """Tests for MarginRequirement dataclass."""

    def test_margin_requirement_initialization(self):
        """Test margin requirement initializes correctly."""
        req = MarginRequirement(
            symbol="TSLA",
            position_type=PositionType.SHORT_STOCK,
            initial_margin=Decimal("25000.00"),  # 50% of 50k
            maintenance_margin=Decimal("15000.00"),  # 30% of 50k
            current_value=Decimal("50000.00"),
            margin_used=Decimal("25000.00")
        )
        assert req.symbol == "TSLA"
        assert req.initial_margin == Decimal("25000.00")

    def test_maintenance_ratio(self):
        """Test maintenance ratio calculation."""
        req = MarginRequirement(
            symbol="GME",
            position_type=PositionType.SHORT_STOCK,
            initial_margin=Decimal("50000.00"),
            maintenance_margin=Decimal("30000.00"),
            current_value=Decimal("100000.00"),
            margin_used=Decimal("50000.00")
        )
        # Ratio = margin_used / current_value = 50000 / 100000 = 0.5
        assert req.maintenance_ratio == 0.5


class TestMarginTracker:
    """Tests for MarginTracker."""

    def test_tracker_initialization(self):
        """Test margin tracker initializes correctly."""
        tracker = MarginTracker()
        assert tracker is not None
        assert tracker.equity == Decimal("100000")  # Default

    def test_tracker_with_custom_equity(self):
        """Test tracker with custom initial equity."""
        tracker = MarginTracker(initial_equity=Decimal("500000"))
        assert tracker.equity == Decimal("500000")

    def test_calculate_short_margin(self):
        """Test short position margin calculation."""
        tracker = MarginTracker()
        margin = tracker.calculate_short_margin(
            symbol="AAPL",
            qty=100,
            price=Decimal("150.00")
        )

        # Position value = 100 * 150 = 15000
        # Initial margin (50%) = 7500
        # Maintenance margin (30%) = 4500
        assert margin.symbol == "AAPL"
        assert margin.position_type == PositionType.SHORT_STOCK
        assert margin.current_value == Decimal("15000")
        assert margin.initial_margin == Decimal("7500.00")
        assert margin.maintenance_margin == Decimal("4500.00")

    def test_calculate_long_margin(self):
        """Test long position margin calculation."""
        tracker = MarginTracker()
        margin = tracker.calculate_long_margin(
            symbol="MSFT",
            qty=50,
            price=Decimal("300.00"),
            use_margin=True
        )

        # Position value = 50 * 300 = 15000
        # Initial margin (50%) = 7500
        # Maintenance margin (25%) = 3750
        assert margin.symbol == "MSFT"
        assert margin.position_type == PositionType.LONG_STOCK
        assert margin.current_value == Decimal("15000")

    def test_add_position(self):
        """Test adding a position to tracker."""
        tracker = MarginTracker()
        margin = tracker.calculate_short_margin("AAPL", 100, Decimal("150.00"))
        result = tracker.add_position(margin)
        assert result is True

    def test_can_open_position(self):
        """Test checking if position can be opened."""
        tracker = MarginTracker(initial_equity=Decimal("10000"))
        small_margin = tracker.calculate_short_margin("AAPL", 10, Decimal("150.00"))
        assert tracker.can_open_position(small_margin) is True

        large_margin = tracker.calculate_short_margin("TSLA", 1000, Decimal("200.00"))
        assert tracker.can_open_position(large_margin) is False

    def test_get_buying_power_for_short(self):
        """Test buying power calculation for shorts."""
        tracker = MarginTracker(initial_equity=Decimal("100000"))
        buying_power = tracker.get_buying_power_for_short()
        # With 2:1 leverage: 100000 * 2 = 200000
        assert buying_power == Decimal("200000")


class TestMarginSummary:
    """Tests for MarginSummary dataclass."""

    def test_summary_initialization(self):
        """Test summary initializes correctly."""
        summary = MarginSummary(
            total_equity=Decimal("100000"),
            total_positions_value=Decimal("150000"),
            buying_power=Decimal("50000"),
            margin_used=Decimal("75000"),
            margin_available=Decimal("25000"),
            maintenance_margin=Decimal("45000"),
            maintenance_excess=Decimal("55000"),
            margin_ratio=0.67,
            status=MarginStatus.HEALTHY,
            timestamp=datetime.now()
        )
        assert summary.is_margin_call is False
        assert summary.can_open_positions is True

    def test_margin_call_status(self):
        """Test margin call detection."""
        summary = MarginSummary(
            total_equity=Decimal("20000"),
            total_positions_value=Decimal("100000"),
            buying_power=Decimal("0"),
            margin_used=Decimal("100000"),
            margin_available=Decimal("0"),
            maintenance_margin=Decimal("25000"),
            maintenance_excess=Decimal("-5000"),
            margin_ratio=0.20,
            status=MarginStatus.MARGIN_CALL,
            timestamp=datetime.now()
        )
        assert summary.is_margin_call is True
        assert summary.can_open_positions is False


class TestShortPosition:
    """Tests for ShortPosition dataclass."""

    def test_short_position_initialization(self):
        """Test short position initializes correctly."""
        position = ShortPosition(
            symbol="GME",
            qty=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(),
            borrow_rate_bps=150.0
        )
        assert position.symbol == "GME"
        assert position.qty == 100

    def test_days_held(self):
        """Test days held calculation."""
        entry = datetime.now() - timedelta(days=5)
        position = ShortPosition(
            symbol="AMC",
            qty=50,
            entry_price=Decimal("10.00"),
            entry_date=entry,
            borrow_rate_bps=100.0
        )
        assert position.days_held == 5

    def test_unrealized_pnl(self):
        """Test unrealized PnL calculation."""
        position = ShortPosition(
            symbol="GME",
            qty=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(),
            borrow_rate_bps=150.0,
            current_price=Decimal("140.00"),  # Price dropped
            accrued_borrow_cost=Decimal("50.00")
        )
        # PnL = (entry - current) * qty - borrow_cost = (150 - 140) * 100 - 50 = 950
        assert position.unrealized_pnl == Decimal("950")


class TestShortSqueezeDetection:
    """Tests for short squeeze risk detection."""

    def test_detect_squeeze_risk(self):
        """Test squeeze risk detection."""
        client = EnhancedBorrowClient()

        # High risk quote
        high_risk_quote = EnhancedLocateQuote(
            symbol="GME",
            available=True,
            shares_available=1000,
            borrow_rate_bps=250.0,
            borrow_rate_pct=2.5,
            difficulty=BorrowDifficulty.VERY_HARD,
            timestamp=datetime.now(),
            utilization_pct=85.0,
            days_to_cover=6.0,
        )
        risk = client.detect_squeeze_risk(high_risk_quote)
        assert risk in (ShortSqueezeRisk.HIGH, ShortSqueezeRisk.EXTREME)

    def test_low_squeeze_risk(self):
        """Test low squeeze risk detection."""
        client = EnhancedBorrowClient()

        # Low risk quote
        low_risk_quote = EnhancedLocateQuote(
            symbol="AAPL",
            available=True,
            shares_available=10000000,
            borrow_rate_bps=15.0,
            borrow_rate_pct=0.15,
            difficulty=BorrowDifficulty.EASY_TO_BORROW,
            timestamp=datetime.now(),
            utilization_pct=5.0,
            days_to_cover=0.5,
        )
        risk = client.detect_squeeze_risk(low_risk_quote)
        assert risk == ShortSqueezeRisk.LOW
