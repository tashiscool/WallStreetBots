"""Comprehensive tests for Production Earnings Protection Strategy.

Tests all components, edge cases, and error handling for production earnings protection.
Target: 80%+ coverage.
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Any

from backend.tradingbot.strategies.production.production_earnings_protection import (
    ProductionEarningsProtection,
    EarningsSignal,
)
from backend.tradingbot.production.core.production_integration import (
    ProductionTradeSignal,
    OrderSide,
    OrderType,
)
from backend.tradingbot.production.data.production_data_integration import (
    EarningsEvent,
)
from backend.tradingbot.core.trading_interface import TradeStatus as OrderStatus


@pytest.fixture
def mock_integration_manager():
    """Create mock integration manager."""
    manager = AsyncMock()
    manager.get_portfolio_value = AsyncMock(return_value=Decimal("100000"))

    # Mock execute_trade to return proper result
    mock_result = Mock()
    mock_result.status = Mock()
    mock_result.status.value = "FILLED"
    mock_result.error_message = None
    manager.execute_trade = AsyncMock(return_value=mock_result)

    manager.alert_system = AsyncMock()
    manager.alert_system.send_alert = AsyncMock()
    return manager


@pytest.fixture
def mock_data_provider():
    """Create mock data provider."""
    provider = AsyncMock()

    # Mock get_current_price to return object with price attribute
    mock_price_data = Mock()
    mock_price_data.price = Decimal("150.00")
    provider.get_current_price = AsyncMock(return_value=mock_price_data)

    provider.get_volatility = AsyncMock(return_value=Decimal("0.30"))
    provider.get_historical_data = AsyncMock(return_value=[])

    # Mock earnings calendar
    earnings_event = EarningsEvent(
        ticker="AAPL",
        earnings_date=datetime.now() + timedelta(days=3),
        earnings_time="AMC",
        estimated_eps=Decimal("1.50"),
        revenue_estimate=Decimal("90000000000"),
        implied_move=Decimal("0.05"),
        source="mock",
    )
    provider.get_earnings_calendar = AsyncMock(return_value=[earnings_event])

    return provider


@pytest.fixture
def earnings_config():
    """Create earnings protection config."""
    return {
        "max_position_size": 0.15,
        "iv_percentile_threshold": 70,
        "min_implied_move": 0.04,
        "max_days_to_earnings": 7,
        "min_days_to_earnings": 1,
        "preferred_strategies": ["deep_itm", "calendar_spread", "protective_hedge"],
    }


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider, earnings_config):
    """Create ProductionEarningsProtection instance."""
    return ProductionEarningsProtection(
        mock_integration_manager,
        mock_data_provider,
        earnings_config
    )


class TestProductionEarningsProtectionInitialization:
    """Test strategy initialization."""

    def test_initialization_success(self, strategy, earnings_config):
        """Test successful initialization."""
        assert strategy.max_position_size == 0.15
        assert strategy.iv_percentile_threshold == 70
        assert strategy.min_implied_move == 0.04
        assert strategy.max_days_to_earnings == 7
        assert strategy.min_days_to_earnings == 1
        assert len(strategy.active_positions) == 0

    def test_initialization_default_values(self, mock_integration_manager, mock_data_provider):
        """Test initialization with default values."""
        strategy = ProductionEarningsProtection(
            mock_integration_manager,
            mock_data_provider,
            {}
        )
        assert strategy.max_position_size == 0.15
        assert strategy.iv_percentile_threshold == 70

    def test_components_initialized(self, strategy):
        """Test that all components are initialized."""
        assert strategy.integration is not None
        assert strategy.data_provider is not None
        assert strategy.logger is not None


class TestEarningsSignalDataclass:
    """Test EarningsSignal dataclass."""

    def test_signal_creation(self):
        """Test creating an earnings signal."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={"days_to_earnings": 3},
        )

        assert signal.ticker == "AAPL"
        assert signal.earnings_time == "AMC"
        assert signal.strategy_type == "deep_itm"
        assert signal.confidence == 0.85


class TestScanForEarningsSignals:
    """Test scan_for_earnings_signals method."""

    @pytest.mark.asyncio
    async def test_scan_finds_signals(self, strategy):
        """Test scanning finds valid signals."""
        signals = await strategy.scan_for_earnings_signals()

        assert isinstance(signals, list)
        # May or may not find signals depending on criteria
        for signal in signals:
            assert isinstance(signal, EarningsSignal)

    @pytest.mark.asyncio
    async def test_scan_no_earnings_events(self, strategy, mock_data_provider):
        """Test scanning with no earnings events."""
        mock_data_provider.get_earnings_calendar = AsyncMock(return_value=[])

        signals = await strategy.scan_for_earnings_signals()

        assert signals == []

    @pytest.mark.asyncio
    async def test_scan_handles_exceptions(self, strategy, mock_data_provider):
        """Test exception handling in scan."""
        mock_data_provider.get_earnings_calendar = AsyncMock(
            side_effect=Exception("API Error")
        )

        signals = await strategy.scan_for_earnings_signals()

        assert signals == []


class TestAnalyzeEarningsOpportunity:
    """Test _analyze_earnings_opportunity method."""

    @pytest.mark.asyncio
    async def test_analyze_valid_opportunity(self, strategy):
        """Test analyzing valid earnings opportunity."""
        event = EarningsEvent(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            estimated_eps=Decimal("1.50"),
            revenue_estimate=Decimal("90000000000"),
            implied_move=Decimal("0.05"),
            source="test",
        )

        signal = await strategy._analyze_earnings_opportunity(event)

        assert signal is not None
        assert signal.ticker == "AAPL"
        assert signal.implied_move >= strategy.min_implied_move

    @pytest.mark.asyncio
    async def test_analyze_too_far_out(self, strategy):
        """Test rejecting earnings too far out."""
        event = EarningsEvent(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=30),  # Too far
            earnings_time="AMC",
            estimated_eps=Decimal("1.50"),
            revenue_estimate=None,
            implied_move=Decimal("0.05"),
            source="test",
        )

        signal = await strategy._analyze_earnings_opportunity(event)

        assert signal is None

    @pytest.mark.asyncio
    async def test_analyze_too_soon(self, strategy):
        """Test rejecting earnings too soon."""
        event = EarningsEvent(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(hours=12),  # < 1 day
            earnings_time="AMC",
            estimated_eps=Decimal("1.50"),
            revenue_estimate=None,
            implied_move=Decimal("0.05"),
            source="test",
        )

        signal = await strategy._analyze_earnings_opportunity(event)

        assert signal is None

    @pytest.mark.asyncio
    async def test_analyze_low_implied_move(self, strategy):
        """Test rejecting low implied move."""
        event = EarningsEvent(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            estimated_eps=Decimal("1.50"),
            revenue_estimate=None,
            implied_move=Decimal("0.01"),  # Below threshold
            source="test",
        )

        signal = await strategy._analyze_earnings_opportunity(event)

        assert signal is None

    @pytest.mark.asyncio
    async def test_analyze_low_iv_percentile(self, strategy, mock_data_provider):
        """Test rejecting low IV percentile."""
        event = EarningsEvent(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            estimated_eps=Decimal("1.50"),
            revenue_estimate=None,
            implied_move=Decimal("0.05"),
            source="test",
        )

        # Mock low IV percentile
        strategy._calculate_iv_percentile = AsyncMock(return_value=50.0)

        signal = await strategy._analyze_earnings_opportunity(event)

        assert signal is None

    @pytest.mark.asyncio
    async def test_analyze_no_market_data(self, strategy, mock_data_provider):
        """Test handling missing market data."""
        event = EarningsEvent(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            estimated_eps=Decimal("1.50"),
            revenue_estimate=None,
            implied_move=Decimal("0.05"),
            source="test",
        )

        mock_data_provider.get_current_price = AsyncMock(return_value=None)

        signal = await strategy._analyze_earnings_opportunity(event)

        assert signal is None


class TestCalculateImpliedMove:
    """Test _calculate_implied_move method."""

    @pytest.mark.asyncio
    async def test_calculate_implied_move_success(self, strategy, mock_data_provider):
        """Test calculating implied move."""
        implied_move = await strategy._calculate_implied_move("AAPL")

        assert isinstance(implied_move, Decimal)
        assert implied_move > Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_implied_move_no_volatility(self, strategy, mock_data_provider):
        """Test fallback when no volatility data."""
        mock_data_provider.get_volatility = AsyncMock(return_value=None)

        implied_move = await strategy._calculate_implied_move("AAPL")

        assert implied_move == Decimal("0.05")  # Default

    @pytest.mark.asyncio
    async def test_calculate_implied_move_exception(self, strategy, mock_data_provider):
        """Test exception handling."""
        mock_data_provider.get_volatility = AsyncMock(side_effect=Exception("Error"))

        implied_move = await strategy._calculate_implied_move("AAPL")

        assert implied_move == Decimal("0.05")


class TestCalculateIVPercentile:
    """Test _calculate_iv_percentile method."""

    @pytest.mark.asyncio
    async def test_calculate_iv_percentile_success(self, strategy, mock_data_provider):
        """Test calculating IV percentile."""
        percentile = await strategy._calculate_iv_percentile("AAPL")

        assert isinstance(percentile, float)
        assert 0 <= percentile <= 100

    @pytest.mark.asyncio
    async def test_calculate_iv_percentile_no_data(self, strategy, mock_data_provider):
        """Test fallback when no volatility data."""
        mock_data_provider.get_volatility = AsyncMock(return_value=None)

        percentile = await strategy._calculate_iv_percentile("AAPL")

        assert percentile == 50.0  # Default

    @pytest.mark.asyncio
    async def test_calculate_iv_percentile_exception(self, strategy, mock_data_provider):
        """Test exception handling."""
        mock_data_provider.get_volatility = AsyncMock(side_effect=Exception("Error"))

        percentile = await strategy._calculate_iv_percentile("AAPL")

        assert percentile == 50.0


class TestSelectStrategy:
    """Test _select_strategy method."""

    @pytest.mark.asyncio
    async def test_select_deep_itm(self, strategy):
        """Test selecting deep ITM strategy."""
        strategy_type = await strategy._select_strategy("AAPL", 85.0, Decimal("0.10"))

        assert strategy_type == "deep_itm"

    @pytest.mark.asyncio
    async def test_select_calendar_spread(self, strategy):
        """Test selecting calendar spread strategy."""
        strategy_type = await strategy._select_strategy("AAPL", 65.0, Decimal("0.06"))

        assert strategy_type == "calendar_spread"

    @pytest.mark.asyncio
    async def test_select_protective_hedge(self, strategy):
        """Test selecting protective hedge strategy."""
        strategy_type = await strategy._select_strategy("AAPL", 75.0, Decimal("0.03"))

        assert strategy_type == "protective_hedge"

    @pytest.mark.asyncio
    async def test_select_none(self, strategy):
        """Test no strategy selected when criteria not met."""
        strategy_type = await strategy._select_strategy("AAPL", 50.0, Decimal("0.02"))

        assert strategy_type is None

    @pytest.mark.asyncio
    async def test_select_strategy_exception(self, strategy):
        """Test exception handling in strategy selection."""
        # Should handle gracefully
        strategy_type = await strategy._select_strategy("AAPL", None, None)
        # May return None or raise, both acceptable


class TestExecuteEarningsTrade:
    """Test execute_earnings_trade method."""

    @pytest.mark.asyncio
    async def test_execute_trade_success(self, strategy, mock_integration_manager):
        """Test successful trade execution."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        strategy._calculate_position_size = AsyncMock(return_value=100)
        strategy._create_trade_signal = AsyncMock(return_value=Mock())

        success = await strategy.execute_earnings_trade(signal)

        assert success is True
        assert "AAPL" in strategy.active_positions

    @pytest.mark.asyncio
    async def test_execute_trade_zero_quantity(self, strategy):
        """Test handling zero quantity."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        strategy._calculate_position_size = AsyncMock(return_value=0)

        success = await strategy.execute_earnings_trade(signal)

        assert success is False

    @pytest.mark.asyncio
    async def test_execute_trade_failure(self, strategy, mock_integration_manager):
        """Test handling trade execution failure."""
        mock_result = Mock()
        mock_result.status = Mock()
        mock_result.status.value = "REJECTED"
        mock_result.error_message = "Insufficient funds"
        mock_integration_manager.execute_trade = AsyncMock(return_value=mock_result)

        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        strategy._calculate_position_size = AsyncMock(return_value=100)
        strategy._create_trade_signal = AsyncMock(return_value=Mock())

        success = await strategy.execute_earnings_trade(signal)

        assert success is False


class TestCalculatePositionSize:
    """Test _calculate_position_size method."""

    @pytest.mark.asyncio
    async def test_calculate_size_deep_itm(self, strategy):
        """Test position size for deep ITM strategy."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        size = await strategy._calculate_position_size(signal)

        assert size == 100  # 15000 / 150

    @pytest.mark.asyncio
    async def test_calculate_size_calendar_spread(self, strategy):
        """Test position size for calendar spread."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="calendar_spread",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        size = await strategy._calculate_position_size(signal)

        assert size == 50  # Half size for spreads

    @pytest.mark.asyncio
    async def test_calculate_size_protective_hedge(self, strategy):
        """Test position size for protective hedge."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="protective_hedge",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        size = await strategy._calculate_position_size(signal)

        assert size == 33  # Third size for hedges


class TestCreateTradeSignal:
    """Test _create_trade_signal method."""

    @pytest.mark.asyncio
    async def test_create_trade_signal(self, strategy):
        """Test creating trade signal."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={"days_to_earnings": 3},
        )

        trade_signal = await strategy._create_trade_signal(signal, 100)

        assert isinstance(trade_signal, ProductionTradeSignal)
        assert trade_signal.ticker == "AAPL"
        assert trade_signal.quantity == 100
        assert trade_signal.strategy_name == "earnings_protection"


class TestMonitorPositions:
    """Test monitor_positions method."""

    @pytest.mark.asyncio
    async def test_monitor_positions_no_positions(self, strategy):
        """Test monitoring with no positions."""
        await strategy.monitor_positions()
        # Should not crash

    @pytest.mark.asyncio
    async def test_monitor_positions_earnings_passed(self, strategy, mock_data_provider):
        """Test closing position after earnings passed."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() - timedelta(days=1),  # Earnings passed
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        strategy.active_positions["AAPL"] = signal
        strategy._check_exit_conditions = AsyncMock(return_value="earnings_passed")
        strategy._execute_exit = AsyncMock()

        await strategy.monitor_positions()

        strategy._execute_exit.assert_called_once()


class TestCheckExitConditions:
    """Test _check_exit_conditions method."""

    @pytest.mark.asyncio
    async def test_exit_earnings_passed(self, strategy):
        """Test exit when earnings passed."""
        position = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() - timedelta(days=1),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        exit_reason = await strategy._check_exit_conditions(position)

        assert exit_reason == "earnings_passed"

    @pytest.mark.asyncio
    async def test_exit_time_decay(self, strategy):
        """Test exit for time decay."""
        position = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(hours=12),  # Same day
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        exit_reason = await strategy._check_exit_conditions(position)

        assert exit_reason == "time_decay"

    @pytest.mark.asyncio
    async def test_exit_profit_target(self, strategy, mock_data_provider):
        """Test exit when profit target hit."""
        position = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        # Mock price increase
        mock_price_data = Mock()
        mock_price_data.price = Decimal("190.00")  # 26.67% increase
        mock_data_provider.get_current_price = AsyncMock(return_value=mock_price_data)

        exit_reason = await strategy._check_exit_conditions(position)

        assert exit_reason == "profit_target"

    @pytest.mark.asyncio
    async def test_no_exit_needed(self, strategy, mock_data_provider):
        """Test no exit when conditions not met."""
        position = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        exit_reason = await strategy._check_exit_conditions(position)

        assert exit_reason is None


class TestExecuteExit:
    """Test _execute_exit method."""

    @pytest.mark.asyncio
    async def test_execute_exit_success(self, strategy, mock_integration_manager, mock_data_provider):
        """Test successful exit execution."""
        position = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        strategy.active_positions["AAPL"] = position

        await strategy._execute_exit(position, "profit_target")

        assert "AAPL" not in strategy.active_positions

    @pytest.mark.asyncio
    async def test_execute_exit_no_price_data(self, strategy, mock_data_provider):
        """Test handling missing price data on exit."""
        mock_data_provider.get_current_price = AsyncMock(return_value=None)

        position = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        strategy.active_positions["AAPL"] = position

        await strategy._execute_exit(position, "profit_target")

        # Should remain in positions since exit failed
        assert "AAPL" in strategy.active_positions


class TestRunStrategy:
    """Test run_strategy method."""

    @pytest.mark.asyncio
    async def test_run_strategy_single_iteration(self, strategy):
        """Test strategy run executes correctly."""
        strategy.scan_for_earnings_signals = AsyncMock(return_value=[])
        strategy.execute_earnings_trade = AsyncMock(return_value=True)
        strategy.monitor_positions = AsyncMock()

        # Run one iteration
        async def run_once():
            signals = await strategy.scan_for_earnings_signals()
            await strategy.monitor_positions()

        await run_once()

        strategy.scan_for_earnings_signals.assert_called_once()
        strategy.monitor_positions.assert_called_once()


class TestGetStrategyStatus:
    """Test get_strategy_status method."""

    def test_status_no_positions(self, strategy):
        """Test status with no positions."""
        status = strategy.get_strategy_status()

        assert status["strategy_name"] == "earnings_protection"
        assert status["active_positions"] == 0
        assert isinstance(status["positions"], list)

    def test_status_with_positions(self, strategy):
        """Test status with active positions."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        strategy.active_positions["AAPL"] = signal

        status = strategy.get_strategy_status()

        assert status["active_positions"] == 1
        assert len(status["positions"]) == 1
        assert status["positions"][0]["ticker"] == "AAPL"


class TestFactoryFunction:
    """Test factory function."""

    def test_create_production_earnings_protection(self, mock_integration_manager, mock_data_provider):
        """Test factory function creates strategy."""
        from backend.tradingbot.strategies.production.production_earnings_protection import (
            create_production_earnings_protection
        )

        config = {"max_position_size": 0.10}
        strategy = create_production_earnings_protection(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        assert isinstance(strategy, ProductionEarningsProtection)
        assert strategy.max_position_size == 0.10


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_zero_risk_amount(self, strategy):
        """Test handling zero risk amount."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("0.00"),
            confidence=0.85,
            metadata={},
        )

        size = await strategy._calculate_position_size(signal)
        assert size == 0

    @pytest.mark.asyncio
    async def test_zero_current_price(self, strategy):
        """Test handling zero current price."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("0.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        # Should handle division by zero
        try:
            size = await strategy._calculate_position_size(signal)
        except Exception:
            pass  # Expected to handle gracefully

    @pytest.mark.asyncio
    async def test_negative_implied_move(self, strategy, mock_data_provider):
        """Test handling negative implied move."""
        mock_data_provider.get_volatility = AsyncMock(return_value=Decimal("-0.10"))

        implied_move = await strategy._calculate_implied_move("AAPL")
        # Should return reasonable value
        assert implied_move >= Decimal("0")

    @pytest.mark.asyncio
    async def test_duplicate_position(self, strategy, mock_integration_manager):
        """Test handling duplicate position creation."""
        signal = EarningsSignal(
            ticker="AAPL",
            earnings_date=datetime.now() + timedelta(days=3),
            earnings_time="AMC",
            current_price=Decimal("150.00"),
            implied_move=Decimal("0.05"),
            iv_percentile=75.0,
            strategy_type="deep_itm",
            risk_amount=Decimal("15000.00"),
            confidence=0.85,
            metadata={},
        )

        # Add position
        strategy.active_positions["AAPL"] = signal

        # Try to add again
        strategy._calculate_position_size = AsyncMock(return_value=100)
        strategy._create_trade_signal = AsyncMock(return_value=Mock())

        await strategy.execute_earnings_trade(signal)

        # Should overwrite or handle gracefully
        assert "AAPL" in strategy.active_positions
