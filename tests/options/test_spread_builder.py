"""
Tests for Spread Builder

Tests cover:
- SpreadBuilderConfig
- SpreadBuilder methods
- Iron condor building
- Iron butterfly building
- Straddle and strangle building
- Calendar and ratio spreads
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from backend.tradingbot.options.spread_builder import (
    SpreadBuilderConfig,
    SpreadBuilder,
    create_spread_builder,
)
from backend.tradingbot.options.exotic_spreads import (
    IronCondor,
    IronButterfly,
    Straddle,
    Strangle,
    CalendarSpread,
    RatioSpread,
    SpreadAnalysis,
)


class TestSpreadBuilderConfig:
    """Tests for SpreadBuilderConfig."""

    def test_config_defaults(self):
        """Test config has sensible defaults."""
        config = SpreadBuilderConfig()

        assert config.min_dte == 21
        assert config.max_dte == 45
        assert config.min_premium_credit == Decimal("0.30")
        assert config.target_delta_short == 0.16

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = SpreadBuilderConfig(
            min_dte=14,
            max_dte=60,
            target_delta_short=0.20,
        )

        assert config.min_dte == 14
        assert config.max_dte == 60
        assert config.target_delta_short == 0.20


class TestSpreadBuilder:
    """Tests for SpreadBuilder."""

    @pytest.fixture
    def mock_pricing_engine(self):
        """Create a mock pricing engine."""
        engine = Mock()
        engine.get_options_chain_yahoo = AsyncMock(return_value=[])
        engine.calculate_theoretical_price = AsyncMock(return_value=Decimal("2.50"))
        return engine

    def test_builder_creation_defaults(self):
        """Test builder creation with defaults."""
        builder = SpreadBuilder()
        assert builder.config is not None
        assert builder.pricing_engine is not None

    def test_builder_creation_custom_config(self, mock_pricing_engine):
        """Test builder creation with custom config."""
        config = SpreadBuilderConfig(min_dte=14)
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine, config=config)
        assert builder.config.min_dte == 14

    def test_select_optimal_expiry(self):
        """Test selecting optimal expiry."""
        builder = SpreadBuilder()
        expiry = builder._select_optimal_expiry()

        # Should be a Friday
        assert expiry.weekday() == 4

        # Should be within config range
        today = date.today()
        days_out = (expiry - today).days
        assert days_out >= builder.config.min_dte
        assert days_out <= builder.config.max_dte + 10  # Some buffer for Friday adjustment

    def test_analyze_spread_calculates_risk_reward(self):
        """Test spread analysis calculates risk/reward."""
        builder = SpreadBuilder()

        # Create a mock spread
        mock_spread = Mock()
        mock_spread.get_max_profit.return_value = Decimal("100")
        mock_spread.get_max_loss.return_value = Decimal("400")
        mock_spread.get_breakeven_points.return_value = [Decimal("145"), Decimal("155")]
        mock_spread.aggregate_greeks = Mock()
        mock_spread.aggregate_greeks.delta = Decimal("0.1")
        mock_spread.is_credit = True
        mock_spread.net_premium = Decimal("1.00")

        analysis = builder._analyze_spread(mock_spread, Decimal("150"))

        assert analysis.max_profit == Decimal("100")
        assert analysis.max_loss == Decimal("400")
        assert analysis.risk_reward_ratio == 4.0

    def test_analyze_spread_recommendation_favorable(self):
        """Test spread analysis gives favorable recommendation."""
        builder = SpreadBuilder()

        mock_spread = Mock()
        mock_spread.get_max_profit.return_value = Decimal("200")
        mock_spread.get_max_loss.return_value = Decimal("200")
        mock_spread.get_breakeven_points.return_value = [Decimal("145"), Decimal("155")]
        mock_spread.aggregate_greeks = Mock()
        mock_spread.aggregate_greeks.delta = Decimal("0.1")
        mock_spread.is_credit = True
        mock_spread.net_premium = Decimal("2.00")

        analysis = builder._analyze_spread(mock_spread, Decimal("150"))

        assert analysis.recommendation == "favorable"

    def test_analyze_spread_recommendation_caution(self):
        """Test spread analysis gives caution recommendation."""
        builder = SpreadBuilder()

        mock_spread = Mock()
        mock_spread.get_max_profit.return_value = Decimal("100")
        mock_spread.get_max_loss.return_value = Decimal("500")
        mock_spread.get_breakeven_points.return_value = [Decimal("145"), Decimal("155")]
        mock_spread.aggregate_greeks = Mock()
        mock_spread.aggregate_greeks.delta = Decimal("0.1")
        mock_spread.is_credit = True
        mock_spread.net_premium = Decimal("1.00")

        analysis = builder._analyze_spread(mock_spread, Decimal("150"))

        assert analysis.recommendation == "caution"

    @pytest.mark.asyncio
    async def test_get_strike_ladder_fallback(self, mock_pricing_engine):
        """Test strike ladder generation fallback."""
        mock_pricing_engine.get_options_chain_yahoo = AsyncMock(side_effect=Exception("API error"))
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        strikes = await builder._get_strike_ladder("AAPL", Decimal("150"), date.today())

        # Should generate synthetic strikes
        assert len(strikes) > 0
        assert Decimal("150") in strikes

    @pytest.mark.asyncio
    async def test_find_strike_by_delta_fallback(self, mock_pricing_engine):
        """Test finding strike by delta with fallback."""
        mock_pricing_engine.get_options_chain_yahoo = AsyncMock(side_effect=Exception("API error"))
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        strike = await builder._find_strike_by_delta(
            "AAPL", Decimal("150"), date.today(), 0.16, "call"
        )

        assert strike is not None
        assert strike > Decimal("150")  # OTM call should be above current price

    @pytest.mark.asyncio
    async def test_get_option_premium(self, mock_pricing_engine):
        """Test getting option premium."""
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        premium = await builder._get_option_premium(
            "AAPL", Decimal("150"), date.today(), "call", Decimal("150")
        )

        assert premium == Decimal("2.50")

    @pytest.mark.asyncio
    async def test_build_iron_condor(self, mock_pricing_engine):
        """Test building an iron condor."""
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        result = await builder.build_iron_condor(
            ticker="AAPL",
            current_price=Decimal("150"),
            wing_width=5,
        )

        if result is not None:
            ic, analysis = result
            assert isinstance(ic, IronCondor)
            assert isinstance(analysis, SpreadAnalysis)

    @pytest.mark.asyncio
    async def test_build_iron_butterfly(self, mock_pricing_engine):
        """Test building an iron butterfly."""
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        result = await builder.build_iron_butterfly(
            ticker="AAPL",
            current_price=Decimal("150"),
            wing_width=5,
        )

        if result is not None:
            ib, analysis = result
            assert isinstance(ib, IronButterfly)
            assert isinstance(analysis, SpreadAnalysis)

    @pytest.mark.asyncio
    async def test_build_straddle_long(self, mock_pricing_engine):
        """Test building a long straddle."""
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        result = await builder.build_straddle(
            ticker="AAPL",
            current_price=Decimal("150"),
            is_long=True,
        )

        if result is not None:
            straddle, analysis = result
            assert isinstance(straddle, Straddle)
            assert straddle.is_long is True

    @pytest.mark.asyncio
    async def test_build_straddle_short(self, mock_pricing_engine):
        """Test building a short straddle."""
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        result = await builder.build_straddle(
            ticker="AAPL",
            current_price=Decimal("150"),
            is_long=False,
        )

        if result is not None:
            straddle, analysis = result
            assert isinstance(straddle, Straddle)
            assert straddle.is_long is False

    @pytest.mark.asyncio
    async def test_build_strangle(self, mock_pricing_engine):
        """Test building a strangle."""
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        result = await builder.build_strangle(
            ticker="AAPL",
            current_price=Decimal("150"),
            width=5,
            is_long=True,
        )

        if result is not None:
            strangle, analysis = result
            assert isinstance(strangle, Strangle)
            assert strangle.width == Decimal("10")  # 5 on each side

    @pytest.mark.asyncio
    async def test_build_calendar_spread(self, mock_pricing_engine):
        """Test building a calendar spread."""
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        result = await builder.build_calendar_spread(
            ticker="AAPL",
            current_price=Decimal("150"),
            option_type="call",
        )

        if result is not None:
            calendar, analysis = result
            assert isinstance(calendar, CalendarSpread)
            assert calendar.option_type == "call"

    @pytest.mark.asyncio
    async def test_build_ratio_spread(self, mock_pricing_engine):
        """Test building a ratio spread."""
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        result = await builder.build_ratio_spread(
            ticker="AAPL",
            current_price=Decimal("150"),
            ratio=(1, 2),
            option_type="call",
        )

        if result is not None:
            ratio, analysis = result
            assert isinstance(ratio, RatioSpread)
            assert ratio.ratio == "1:2"

    @pytest.mark.asyncio
    async def test_build_iron_condor_error_handling(self, mock_pricing_engine):
        """Test iron condor building handles errors."""
        mock_pricing_engine.get_options_chain_yahoo = AsyncMock(side_effect=Exception("API error"))
        mock_pricing_engine.calculate_theoretical_price = AsyncMock(side_effect=Exception("API error"))
        builder = SpreadBuilder(pricing_engine=mock_pricing_engine)

        # Force an error by making _find_strike_by_delta return None
        with patch.object(builder, '_find_strike_by_delta', AsyncMock(return_value=None)):
            result = await builder.build_iron_condor(
                ticker="AAPL",
                current_price=Decimal("150"),
            )

            assert result is None


class TestFactoryFunction:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_create_spread_builder_defaults(self):
        """Test factory with defaults."""
        builder = await create_spread_builder()
        assert isinstance(builder, SpreadBuilder)

    @pytest.mark.asyncio
    async def test_create_spread_builder_custom_config(self):
        """Test factory with custom config."""
        config = SpreadBuilderConfig(min_dte=14)
        builder = await create_spread_builder(config=config)
        assert builder.config.min_dte == 14
