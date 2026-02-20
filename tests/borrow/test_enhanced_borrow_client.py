"""
Tests for Enhanced Borrow Client

Tests cover:
- BorrowDifficulty and ShortSqueezeRisk enums
- EnhancedLocateQuote dataclass
- ShortPosition tracking
- EnhancedBorrowClient rate estimation
- HTB detection and squeeze risk
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from backend.tradingbot.borrow.enhanced_borrow_client import (
    BorrowDifficulty,
    ShortSqueezeRisk,
    EnhancedLocateQuote,
    ShortPosition,
    BorrowRateHistory,
    EnhancedBorrowClient,
    create_enhanced_borrow_client,
    KNOWN_HTB_STOCKS,
    DEFAULT_RATES,
)


class TestBorrowDifficultyEnum:
    """Tests for BorrowDifficulty enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert BorrowDifficulty.EASY_TO_BORROW.value == "easy"
        assert BorrowDifficulty.MODERATE.value == "moderate"
        assert BorrowDifficulty.HARD_TO_BORROW.value == "htb"
        assert BorrowDifficulty.VERY_HARD.value == "very_hard"
        assert BorrowDifficulty.NO_BORROW.value == "no_borrow"

    def test_enum_count(self):
        """Test correct number of enum values."""
        assert len(BorrowDifficulty) == 5


class TestShortSqueezeRiskEnum:
    """Tests for ShortSqueezeRisk enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert ShortSqueezeRisk.LOW.value == "low"
        assert ShortSqueezeRisk.MODERATE.value == "moderate"
        assert ShortSqueezeRisk.HIGH.value == "high"
        assert ShortSqueezeRisk.EXTREME.value == "extreme"


class TestEnhancedLocateQuote:
    """Tests for EnhancedLocateQuote dataclass."""

    def test_quote_creation(self):
        """Test creating a locate quote."""
        quote = EnhancedLocateQuote(
            symbol="AAPL",
            available=True,
            shares_available=100000,
            borrow_rate_bps=25.0,
            borrow_rate_pct=0.25,
            difficulty=BorrowDifficulty.EASY_TO_BORROW,
            timestamp=datetime.now(),
        )

        assert quote.symbol == "AAPL"
        assert quote.available is True
        assert quote.shares_available == 100000
        assert quote.borrow_rate_bps == 25.0

    def test_is_htb_property_false(self):
        """Test is_htb returns False for easy borrow."""
        quote = EnhancedLocateQuote(
            symbol="AAPL",
            available=True,
            shares_available=100000,
            borrow_rate_bps=25.0,
            borrow_rate_pct=0.25,
            difficulty=BorrowDifficulty.EASY_TO_BORROW,
            timestamp=datetime.now(),
        )
        assert quote.is_htb is False

    def test_is_htb_property_true(self):
        """Test is_htb returns True for hard to borrow."""
        quote = EnhancedLocateQuote(
            symbol="GME",
            available=True,
            shares_available=1000,
            borrow_rate_bps=150.0,
            borrow_rate_pct=1.5,
            difficulty=BorrowDifficulty.HARD_TO_BORROW,
            timestamp=datetime.now(),
        )
        assert quote.is_htb is True

    def test_is_htb_very_hard(self):
        """Test is_htb returns True for very hard to borrow."""
        quote = EnhancedLocateQuote(
            symbol="KOSS",
            available=True,
            shares_available=500,
            borrow_rate_bps=300.0,
            borrow_rate_pct=3.0,
            difficulty=BorrowDifficulty.VERY_HARD,
            timestamp=datetime.now(),
        )
        assert quote.is_htb is True

    def test_daily_cost_per_share(self):
        """Test daily cost calculation."""
        quote = EnhancedLocateQuote(
            symbol="AAPL",
            available=True,
            shares_available=100000,
            borrow_rate_bps=36.5,  # 0.365% annual = 0.001%/day
            borrow_rate_pct=0.365,
            difficulty=BorrowDifficulty.EASY_TO_BORROW,
            timestamp=datetime.now(),
        )

        cost = quote.daily_cost_per_share(Decimal("100.00"))
        # 100 * (0.365/100/365) = 100 * 0.00001 = 0.001
        assert cost == Decimal("0.001")


class TestShortPosition:
    """Tests for ShortPosition dataclass."""

    def test_position_creation(self):
        """Test creating a short position."""
        position = ShortPosition(
            symbol="AAPL",
            qty=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now() - timedelta(days=5),
            borrow_rate_bps=25.0,
        )

        assert position.symbol == "AAPL"
        assert position.qty == 100
        assert position.entry_price == Decimal("150.00")

    def test_days_held(self):
        """Test days_held calculation."""
        position = ShortPosition(
            symbol="AAPL",
            qty=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now() - timedelta(days=10),
            borrow_rate_bps=25.0,
        )

        assert position.days_held == 10

    def test_unrealized_pnl_profit(self):
        """Test unrealized P&L when in profit."""
        position = ShortPosition(
            symbol="AAPL",
            qty=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now() - timedelta(days=5),
            borrow_rate_bps=25.0,
            current_price=Decimal("140.00"),
            accrued_borrow_cost=Decimal("10.00"),
        )

        # Profit from short = (entry - current) * qty - costs
        # (150 - 140) * 100 - 10 = 1000 - 10 = 990
        assert position.unrealized_pnl == Decimal("990.00")

    def test_unrealized_pnl_loss(self):
        """Test unrealized P&L when in loss."""
        position = ShortPosition(
            symbol="AAPL",
            qty=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now() - timedelta(days=5),
            borrow_rate_bps=25.0,
            current_price=Decimal("160.00"),
            accrued_borrow_cost=Decimal("10.00"),
        )

        # Loss from short = (entry - current) * qty - costs
        # (150 - 160) * 100 - 10 = -1000 - 10 = -1010
        assert position.unrealized_pnl == Decimal("-1010.00")

    def test_unrealized_pnl_none(self):
        """Test unrealized P&L when no current price."""
        position = ShortPosition(
            symbol="AAPL",
            qty=100,
            entry_price=Decimal("150.00"),
            entry_date=datetime.now(),
            borrow_rate_bps=25.0,
        )

        assert position.unrealized_pnl is None


class TestBorrowRateHistory:
    """Tests for BorrowRateHistory dataclass."""

    def test_history_creation(self):
        """Test creating rate history."""
        history = BorrowRateHistory(
            symbol="GME",
            rate_bps=150.0,
            availability=5000,
            timestamp=datetime.now(),
        )

        assert history.symbol == "GME"
        assert history.rate_bps == 150.0
        assert history.availability == 5000


class TestEnhancedBorrowClient:
    """Tests for EnhancedBorrowClient."""

    def test_client_creation_no_broker(self):
        """Test creating client without broker."""
        client = EnhancedBorrowClient()
        assert client.broker is None
        assert client.use_real_rates is False

    def test_client_creation_with_broker(self):
        """Test creating client with broker."""
        mock_broker = Mock()
        client = EnhancedBorrowClient(broker_client=mock_broker, use_real_rates=True)
        assert client.broker is mock_broker
        assert client.use_real_rates is True

    def test_classify_difficulty_easy(self):
        """Test difficulty classification for easy to borrow."""
        client = EnhancedBorrowClient()
        assert client._classify_difficulty(20.0) == BorrowDifficulty.EASY_TO_BORROW

    def test_classify_difficulty_moderate(self):
        """Test difficulty classification for moderate."""
        client = EnhancedBorrowClient()
        assert client._classify_difficulty(50.0) == BorrowDifficulty.MODERATE

    def test_classify_difficulty_htb(self):
        """Test difficulty classification for HTB."""
        client = EnhancedBorrowClient()
        assert client._classify_difficulty(100.0) == BorrowDifficulty.HARD_TO_BORROW

    def test_classify_difficulty_very_hard(self):
        """Test difficulty classification for very hard."""
        client = EnhancedBorrowClient()
        assert client._classify_difficulty(300.0) == BorrowDifficulty.VERY_HARD

    def test_classify_difficulty_no_borrow(self):
        """Test difficulty classification for no borrow."""
        client = EnhancedBorrowClient()
        assert client._classify_difficulty(600.0) == BorrowDifficulty.NO_BORROW

    def test_estimate_rate_mega_cap(self):
        """Test rate estimation for mega cap stocks."""
        client = EnhancedBorrowClient()
        rate = client._estimate_rate_by_characteristics("AAPL")
        assert rate == DEFAULT_RATES["mega_cap"]

    def test_estimate_rate_large_cap(self):
        """Test rate estimation for large cap stocks."""
        client = EnhancedBorrowClient()
        rate = client._estimate_rate_by_characteristics("JPM")
        assert rate == DEFAULT_RATES["large_cap"]

    def test_estimate_rate_short_symbol(self):
        """Test rate estimation for short symbol (assumed large cap)."""
        client = EnhancedBorrowClient()
        rate = client._estimate_rate_by_characteristics("XYZ")
        assert rate == DEFAULT_RATES["large_cap"]

    def test_estimate_rate_mid_cap(self):
        """Test rate estimation for mid cap (4 letter symbol)."""
        client = EnhancedBorrowClient()
        rate = client._estimate_rate_by_characteristics("ABCD")
        assert rate == DEFAULT_RATES["mid_cap"]

    def test_estimate_rate_small_cap(self):
        """Test rate estimation for small cap (5+ letter symbol)."""
        client = EnhancedBorrowClient()
        rate = client._estimate_rate_by_characteristics("ABCDE")
        assert rate == DEFAULT_RATES["small_cap"]

    @pytest.mark.asyncio
    async def test_get_locate_quote_htb(self):
        """Test getting locate quote for known HTB stock."""
        client = EnhancedBorrowClient()
        quote = await client.get_locate_quote("GME", 100)

        assert quote.symbol == "GME"
        assert quote.is_htb is True
        assert quote.borrow_rate_bps >= 50  # Should be high for GME

    @pytest.mark.asyncio
    async def test_get_locate_quote_etb(self):
        """Test getting locate quote for easy to borrow stock."""
        client = EnhancedBorrowClient()
        quote = await client.get_locate_quote("AAPL", 100)

        assert quote.symbol == "AAPL"
        assert quote.is_htb is False
        assert quote.borrow_rate_bps <= 30

    @pytest.mark.asyncio
    async def test_get_locate_quote_caching(self):
        """Test that quotes are cached."""
        client = EnhancedBorrowClient()

        quote1 = await client.get_locate_quote("AAPL", 100)
        quote2 = await client.get_locate_quote("AAPL", 100)

        # Same cached quote
        assert quote1.timestamp == quote2.timestamp

    @pytest.mark.asyncio
    async def test_get_locate_quote_refresh(self):
        """Test that refresh bypasses cache."""
        client = EnhancedBorrowClient()

        quote1 = await client.get_locate_quote("AAPL", 100)
        quote2 = await client.get_locate_quote("AAPL", 100, refresh=True)

        # Different timestamps due to refresh
        assert quote2.timestamp >= quote1.timestamp

    @pytest.mark.asyncio
    async def test_open_short_position(self):
        """Test opening a short position."""
        client = EnhancedBorrowClient()

        position = await client.open_short_position(
            symbol="AAPL",
            qty=100,
            entry_price=Decimal("150.00"),
        )

        assert position.symbol == "AAPL"
        assert position.qty == 100
        assert position.entry_price == Decimal("150.00")
        assert position.borrow_rate_bps > 0

    @pytest.mark.asyncio
    async def test_get_position(self):
        """Test getting a tracked position."""
        client = EnhancedBorrowClient()

        await client.open_short_position("AAPL", 100, Decimal("150.00"))
        position = client.get_position("AAPL")

        assert position is not None
        assert position.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_position_not_found(self):
        """Test getting position that doesn't exist."""
        client = EnhancedBorrowClient()
        position = client.get_position("NONEXISTENT")

        assert position is None

    @pytest.mark.asyncio
    async def test_close_position(self):
        """Test closing a position."""
        client = EnhancedBorrowClient()

        await client.open_short_position("AAPL", 100, Decimal("150.00"))
        closed = client.close_position("AAPL")

        assert closed is not None
        assert closed.symbol == "AAPL"

        # Should be removed
        assert client.get_position("AAPL") is None

    @pytest.mark.asyncio
    async def test_get_all_positions(self):
        """Test getting all positions."""
        client = EnhancedBorrowClient()

        await client.open_short_position("AAPL", 100, Decimal("150.00"))
        await client.open_short_position("MSFT", 50, Decimal("300.00"))

        positions = client.get_all_positions()
        assert len(positions) == 2

    @pytest.mark.asyncio
    async def test_update_position_costs(self):
        """Test updating position costs."""
        client = EnhancedBorrowClient()

        await client.open_short_position("AAPL", 100, Decimal("150.00"))
        updated = await client.update_position_costs("AAPL", Decimal("145.00"))

        assert updated is not None
        assert updated.current_price == Decimal("145.00")
        assert updated.mark_to_market == Decimal("500.00")  # (150-145)*100

    @pytest.mark.asyncio
    async def test_get_total_borrow_costs(self):
        """Test getting total borrow costs."""
        client = EnhancedBorrowClient()

        # Create a position with some days held
        pos = await client.open_short_position("AAPL", 100, Decimal("150.00"))
        pos.entry_date = datetime.now() - timedelta(days=30)

        total = await client.get_total_borrow_costs()
        assert total >= Decimal("0")

    def test_get_rate_history_empty(self):
        """Test rate history when empty."""
        client = EnhancedBorrowClient()
        history = client.get_rate_history("AAPL")
        assert history == []

    @pytest.mark.asyncio
    async def test_get_rate_history_with_data(self):
        """Test rate history after fetching quotes."""
        client = EnhancedBorrowClient()

        await client.get_locate_quote("AAPL", 100)
        history = client.get_rate_history("AAPL")

        assert len(history) >= 1
        assert history[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_scan_for_htb_opportunities(self):
        """Test scanning for HTB opportunities."""
        client = EnhancedBorrowClient()

        symbols = ["GME", "AAPL", "AMC", "MSFT"]
        htb_quotes = await client.scan_for_htb_opportunities(symbols)

        # GME and AMC are known HTB
        htb_symbols = [q.symbol for q in htb_quotes]
        assert "GME" in htb_symbols
        assert "AMC" in htb_quotes or len(htb_quotes) >= 1

    def test_detect_squeeze_risk_low(self):
        """Test squeeze risk detection for low risk."""
        client = EnhancedBorrowClient()

        quote = EnhancedLocateQuote(
            symbol="AAPL",
            available=True,
            shares_available=1000000,
            borrow_rate_bps=20.0,
            borrow_rate_pct=0.2,
            difficulty=BorrowDifficulty.EASY_TO_BORROW,
            timestamp=datetime.now(),
            utilization_pct=10.0,
            days_to_cover=0.5,
        )

        risk = client.detect_squeeze_risk(quote)
        assert risk == ShortSqueezeRisk.LOW

    def test_detect_squeeze_risk_extreme(self):
        """Test squeeze risk detection for extreme risk."""
        client = EnhancedBorrowClient()

        quote = EnhancedLocateQuote(
            symbol="GME",
            available=True,
            shares_available=1000,
            borrow_rate_bps=250.0,
            borrow_rate_pct=2.5,
            difficulty=BorrowDifficulty.VERY_HARD,
            timestamp=datetime.now(),
            utilization_pct=90.0,
            days_to_cover=8.0,
        )

        risk = client.detect_squeeze_risk(quote)
        assert risk == ShortSqueezeRisk.EXTREME


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_enhanced_borrow_client_no_broker(self):
        """Test factory without broker."""
        client = create_enhanced_borrow_client()
        assert isinstance(client, EnhancedBorrowClient)
        assert client.broker is None

    def test_create_enhanced_borrow_client_with_broker(self):
        """Test factory with broker."""
        mock_broker = Mock()
        client = create_enhanced_borrow_client(broker_client=mock_broker)
        assert client.broker is mock_broker


class TestKnownHTBStocks:
    """Tests for known HTB stock data."""

    def test_known_htb_stocks_exist(self):
        """Test known HTB stocks data exists."""
        assert "GME" in KNOWN_HTB_STOCKS
        assert "AMC" in KNOWN_HTB_STOCKS

    def test_known_htb_stocks_structure(self):
        """Test known HTB stocks have correct structure."""
        for data in KNOWN_HTB_STOCKS.values():
            assert "base_rate" in data
            assert "volatility" in data
            assert data["base_rate"] > 0
            assert 0 <= data["volatility"] <= 1
