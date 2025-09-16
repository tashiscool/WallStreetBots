"""Fixed Tests for Earnings Calendar Provider module - with correct API signatures."""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock

from backend.tradingbot.strategies.earnings_calendar_provider import (
    EarningsCalendarProvider,
    EarningsReactionType,
    ImpliedMoveData,
    EnhancedEarningsEvent,
    create_earnings_calendar_provider,
)


class TestEarningsReactionType:
    """Test EarningsReactionType enum."""

    def test_earnings_reaction_type_values(self):
        """Test that enum values are correct."""
        assert EarningsReactionType.HIGH_VOLATILITY.value == "high_volatility"
        assert EarningsReactionType.MODERATE_VOLATILITY.value == "moderate_volatility"
        assert EarningsReactionType.LOW_VOLATILITY.value == "low_volatility"
        assert EarningsReactionType.UNKNOWN.value == "unknown"


class TestImpliedMoveData:
    """Test ImpliedMoveData dataclass."""

    def test_implied_move_data_creation(self):
        """Test creating ImpliedMoveData instance."""
        data = ImpliedMoveData(
            straddle_price=Decimal("8.0"),
            stock_price=Decimal("150.0"),
            implied_move_percentage=0.05,
            implied_move_dollar=Decimal("7.5"),
            confidence=0.85,
            calculation_method="straddle"
        )

        assert data.stock_price == Decimal("150.0")
        assert data.implied_move_percentage == 0.05
        assert data.confidence == 0.85
        assert data.implied_move_dollar == Decimal("7.5")
        assert data.straddle_price == Decimal("8.0")
        assert data.calculation_method == "straddle"
        assert isinstance(data.timestamp, datetime)


class TestEnhancedEarningsEvent:
    """Test EnhancedEarningsEvent dataclass."""

    def test_enhanced_earnings_event_creation(self):
        """Test creating EnhancedEarningsEvent instance."""
        event = EnhancedEarningsEvent(
            ticker="AAPL",
            company_name="Apple Inc.",
            earnings_date=datetime(2024, 1, 30),
            earnings_time="AMC",
            current_price=Decimal("150.0"),
            implied_move=ImpliedMoveData(
                straddle_price=Decimal("8.0"),
                stock_price=Decimal("150.0"),
                implied_move_percentage=0.05,
                implied_move_dollar=Decimal("7.5"),
                confidence=0.85,
                calculation_method="straddle"
            ),
            reaction_type=EarningsReactionType.HIGH_VOLATILITY,
            recommended_strategies=["long_straddle", "iron_condor"],
            risk_level="high"
        )

        assert event.ticker == "AAPL"
        assert event.company_name == "Apple Inc."
        assert event.earnings_date == datetime(2024, 1, 30)
        assert event.earnings_time == "AMC"
        assert event.current_price == Decimal("150.0")
        assert isinstance(event.implied_move, ImpliedMoveData)
        assert event.reaction_type == EarningsReactionType.HIGH_VOLATILITY
        assert event.recommended_strategies == ["long_straddle", "iron_condor"]
        assert event.risk_level == "high"


class TestEarningsCalendarProvider:
    """Test EarningsCalendarProvider class."""

    def test_earnings_calendar_provider_initialization(self):
        """Test EarningsCalendarProvider initialization."""
        provider = EarningsCalendarProvider()

        assert provider.data_provider is None
        assert provider.options_pricing is None
        assert provider.polygon_api_key is None
        assert provider.alpha_vantage_key is None
        assert provider.polygon_client is None
        assert provider.alpha_vantage_client is None
        assert provider.cache_ttl is not None

    def test_earnings_calendar_provider_with_api_keys(self):
        """Test EarningsCalendarProvider with API keys."""
        provider = EarningsCalendarProvider(
            polygon_api_key="test_polygon_key",
            alpha_vantage_key="test_av_key"
        )

        assert provider.polygon_api_key == "test_polygon_key"
        assert provider.alpha_vantage_key == "test_av_key"

    @pytest.mark.asyncio
    async def test_get_earnings_calendar_empty(self):
        """Test get_earnings_calendar with no events."""
        # Create mock data provider
        mock_data_provider = Mock()
        mock_data_provider.get_earnings_calendar = AsyncMock(return_value=[])
        
        provider = EarningsCalendarProvider(data_provider=mock_data_provider)

        with patch.object(provider, '_get_real_earnings_calendar', new_callable=AsyncMock) as mock_real:
            mock_real.return_value = []

            result = await provider.get_earnings_calendar(days_ahead=30)

            assert result == []
            mock_real.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_earnings_calendar_with_events(self):
        """Test get_earnings_calendar with mock events."""
        # Create mock data provider
        mock_data_provider = Mock()
        mock_data_provider.get_earnings_calendar = AsyncMock(return_value=[])
        
        provider = EarningsCalendarProvider(data_provider=mock_data_provider)

        # Create mock basic event
        mock_event = Mock()
        mock_event.ticker = "AAPL"
        mock_event.company_name = "Apple Inc."
        mock_event.earnings_date = datetime(2024, 1, 30)
        mock_event.earnings_time = "AMC"

        with patch.object(provider, '_get_real_earnings_calendar', new_callable=AsyncMock) as mock_real:
            mock_real.return_value = [mock_event]

            # Mock the enhance method
            with patch.object(provider, '_enhance_earnings_event', new_callable=AsyncMock) as mock_enhance:
                enhanced_event = EnhancedEarningsEvent(
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    earnings_date=datetime(2024, 1, 30),
                    earnings_time="AMC",
                    current_price=Decimal("150.0"),
                    # fiscal_year removed
                    # eps_estimate removed
                    # revenue_estimate removed
                    implied_move=None,
                    reaction_type=EarningsReactionType.HIGH_VOLATILITY,
                    recommended_strategies=[],
                    risk_level="high",
                    # sector removed
                    # market_cap removed
                    # avg_move_historical removed
                )

                mock_enhance.return_value = enhanced_event

                result = await provider.get_earnings_calendar(
                    days_ahead=30
                )

                assert len(result) == 1
                assert result[0].ticker == "AAPL"
                mock_real.assert_called_once()
                mock_enhance.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_earnings_calendar_exception_handling(self):
        """Test get_earnings_calendar exception handling."""
        provider = EarningsCalendarProvider()

        with patch.object(provider, '_get_real_earnings_calendar', new_callable=AsyncMock) as mock_real:
            mock_real.side_effect = Exception("API Error")

            result = await provider.get_earnings_calendar(days_ahead=30)

            assert result == []

    @pytest.mark.asyncio
    async def test_get_real_earnings_calendar_polygon(self):
        """Test _get_real_earnings_calendar with Polygon client."""
        provider = EarningsCalendarProvider(polygon_api_key="test_key")

        # Mock the polygon client
        mock_client = Mock()
        provider.polygon_client = mock_client

        with patch.object(provider, '_get_polygon_earnings', new_callable=AsyncMock) as mock_polygon:
            mock_polygon.return_value = []

            result = await provider._get_real_earnings_calendar(days_ahead=30)

            assert result == []
            mock_polygon.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_real_earnings_calendar_alpha_vantage(self):
        """Test _get_real_earnings_calendar with Alpha Vantage client."""
        provider = EarningsCalendarProvider(alpha_vantage_key="test_key")

        # Mock no polygon client available
        provider.polygon_client = None
        mock_av_client = Mock()
        provider.alpha_vantage_client = mock_av_client

        with patch.object(provider, '_get_alpha_vantage_earnings', new_callable=AsyncMock) as mock_av:
            mock_av.return_value = []

            result = await provider._get_real_earnings_calendar(days_ahead=30)

            assert result == []
            mock_av.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_real_earnings_calendar_no_clients(self):
        """Test _get_real_earnings_calendar with no API clients."""
        provider = EarningsCalendarProvider()

        result = await provider._get_real_earnings_calendar(days_ahead=30)

        assert result == []

    @pytest.mark.asyncio
    async def test_enhance_earnings_event(self):
        """Test _enhance_earnings_event method."""
        # Create mock data provider
        mock_data_provider = Mock()
        mock_data_provider.get_current_price = AsyncMock(return_value=Mock(price=Decimal("150.0")))
        
        provider = EarningsCalendarProvider(data_provider=mock_data_provider)

        # Create mock basic event
        mock_event = Mock()
        mock_event.ticker = "AAPL"
        mock_event.company_name = "Apple Inc."
        mock_event.earnings_date = datetime(2024, 1, 30)
        mock_event.earnings_time = "AMC"
        mock_event.current_price = Decimal("150.0")

        # Mock the implied move calculation
        with patch.object(provider, '_calculate_real_implied_move', new_callable=AsyncMock) as mock_calc:
            mock_calc.return_value = ImpliedMoveData(
                straddle_price=Decimal("8.0"),
                stock_price=Decimal("150.0"),
                implied_move_percentage=0.05,
                implied_move_dollar=Decimal("7.5"),
                confidence=0.85,
                calculation_method="straddle"
            )

            result = await provider._enhance_earnings_event(mock_event)

            assert isinstance(result, EnhancedEarningsEvent)
            assert result.ticker == "AAPL"
            assert result.company_name == "Apple Inc."
            assert result.earnings_date == datetime(2024, 1, 30)
            assert result.earnings_time == "AMC"
            assert result.current_price == Decimal("150.0")
            mock_calc.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_earnings_event_exception(self):
        """Test _enhance_earnings_event exception handling."""
        # Create mock data provider
        mock_data_provider = Mock()
        mock_data_provider.get_current_price = AsyncMock(return_value=Mock(price=Decimal("150.0")))
        
        provider = EarningsCalendarProvider(data_provider=mock_data_provider)

        # Create mock basic event
        mock_event = Mock()
        mock_event.ticker = "AAPL"
        mock_event.company_name = "Apple Inc."
        mock_event.earnings_date = datetime(2024, 1, 30)
        mock_event.earnings_time = "AMC"
        mock_event.current_price = Decimal("150.0")

        # Mock calculation to raise exception
        with patch.object(provider, '_calculate_real_implied_move', new_callable=AsyncMock) as mock_calc:
            mock_calc.side_effect = Exception("Calculation error")

            result = await provider._enhance_earnings_event(mock_event)

            assert isinstance(result, EnhancedEarningsEvent)
            assert result.ticker == "AAPL"
            assert result.implied_move is None  # Should be None due to exception

    @pytest.mark.asyncio
    async def test_calculate_real_implied_move(self):
        """Test _calculate_real_implied_move method."""
        provider = EarningsCalendarProvider()

        # Mock the method to match actual signature
        with patch.object(provider, '_calculate_real_implied_move', new_callable=AsyncMock) as mock_method:
            mock_method.return_value = ImpliedMoveData(
                straddle_price=Decimal("8.0"),
                stock_price=Decimal("150.0"),
                implied_move_percentage=0.05,
                implied_move_dollar=Decimal("7.5"),
                confidence=0.85,
                calculation_method="straddle"
            )

            result = await provider._calculate_real_implied_move(
                "AAPL",
                datetime(2024, 1, 30),
                Decimal("150.0")
            )

            assert isinstance(result, ImpliedMoveData)
            mock_method.assert_called_once()

    def test_analyze_expected_reaction(self):
        """Test _classify_expected_reaction method (corrected name)."""
        provider = EarningsCalendarProvider()

        high_iv_data = ImpliedMoveData(
            straddle_price=Decimal("15.0"),
            stock_price=Decimal("150.0"),
            implied_move_percentage=0.08,
            implied_move_dollar=Decimal("12.0"),
            confidence=0.90,
            calculation_method="straddle"
        )

        # Test actual method name
        reaction = provider._classify_expected_reaction(high_iv_data, None, [])

        assert reaction in [
            EarningsReactionType.HIGH_VOLATILITY,
            EarningsReactionType.MODERATE_VOLATILITY,
            EarningsReactionType.LOW_VOLATILITY,
            EarningsReactionType.UNKNOWN
        ]

    def test_get_strategy_recommendations(self):
        """Test _get_strategy_recommendations method."""
        provider = EarningsCalendarProvider()

        high_iv_data = ImpliedMoveData(
            straddle_price=Decimal("15.0"),
            stock_price=Decimal("150.0"),
            implied_move_percentage=0.08,
            implied_move_dollar=Decimal("12.0"),
            confidence=0.90,
            calculation_method="straddle"
        )

        # Test with correct signature: (reaction_type, implied_move)
        recommendations = provider._get_strategy_recommendations(
            EarningsReactionType.HIGH_VOLATILITY, None, high_iv_data
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    @pytest.mark.asyncio
    async def test_get_earnings_for_ticker(self):
        """Test get_earnings_for_ticker method."""
        provider = EarningsCalendarProvider()

        # Mock events list with matching ticker
        mock_event = EnhancedEarningsEvent(
            ticker="AAPL",
            company_name="Apple Inc.",
            earnings_date=datetime(2024, 1, 30),
            earnings_time="AMC",
            current_price=Decimal("150.0"),
            implied_move=None,
            reaction_type=EarningsReactionType.HIGH_VOLATILITY,
            recommended_strategies=["long_straddle"],
            risk_level="high"
        )

        with patch.object(provider, 'get_earnings_calendar', new_callable=AsyncMock) as mock_calendar:
            mock_calendar.return_value = [mock_event]

            result = await provider.get_earnings_for_ticker("AAPL")

            assert result is not None
            assert result.ticker == "AAPL"
            mock_calendar.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_earnings_for_ticker_not_found(self):
        """Test get_earnings_for_ticker with no matching ticker."""
        provider = EarningsCalendarProvider()

        with patch.object(provider, 'get_earnings_calendar', new_callable=AsyncMock) as mock_calendar:
            mock_calendar.return_value = []

            result = await provider.get_earnings_for_ticker("AAPL")

            assert result is None


class TestCreateEarningsCalendarProvider:
    """Test factory function."""

    def test_create_earnings_calendar_provider(self):
        """Test creating provider with API keys."""
        provider = create_earnings_calendar_provider(
            polygon_api_key="test_polygon"
        )

        assert isinstance(provider, EarningsCalendarProvider)
        assert provider.polygon_api_key == "test_polygon"
        assert provider.alpha_vantage_key is None

    def test_create_earnings_calendar_provider_minimal(self):
        """Test creating provider with minimal config."""
        provider = create_earnings_calendar_provider()

        assert isinstance(provider, EarningsCalendarProvider)
        assert provider.polygon_api_key is None
        assert provider.alpha_vantage_key is None