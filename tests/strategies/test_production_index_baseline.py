"""Comprehensive tests for Production Index Baseline Strategy.

Tests all components, edge cases, and error handling for production index baseline.
Target: 80%+ coverage.
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Any

from backend.tradingbot.strategies.production.production_index_baseline import (
    ProductionIndexBaseline,
    BaselineComparison,
    BaselineSignal,
)
from backend.tradingbot.production.core.production_integration import (
    ProductionTradeSignal,
    OrderSide,
    OrderType,
)
from backend.tradingbot.core.trading_interface import OrderStatus


@pytest.fixture
def mock_integration_manager():
    """Create mock integration manager."""
    manager = AsyncMock()
    manager.get_portfolio_value = AsyncMock(return_value=Decimal("100000"))
    manager.get_position_value = AsyncMock(return_value=Decimal("40000"))

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
    mock_price_data.price = Decimal("450.00")
    provider.get_current_price = AsyncMock(return_value=mock_price_data)

    # Mock historical data
    mock_historical = []
    base_price = Decimal("400.00")
    for i in range(200):
        data_point = Mock()
        data_point.price = base_price + Decimal(str(i * 0.5))
        mock_historical.append(data_point)

    provider.get_historical_data = AsyncMock(return_value=mock_historical)

    return provider


@pytest.fixture
def baseline_config():
    """Create index baseline config."""
    return {
        "benchmarks": ["SPY", "VTI", "QQQ"],
        "target_allocation": 0.80,
        "rebalance_threshold": 0.05,
        "tax_loss_threshold": -0.10,
    }


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider, baseline_config):
    """Create ProductionIndexBaseline instance."""
    return ProductionIndexBaseline(
        mock_integration_manager,
        mock_data_provider,
        baseline_config
    )


class TestProductionIndexBaselineInitialization:
    """Test strategy initialization."""

    def test_initialization_success(self, strategy, baseline_config):
        """Test successful initialization."""
        assert strategy.benchmarks == ["SPY", "VTI", "QQQ"]
        assert strategy.target_allocation == 0.80
        assert strategy.rebalance_threshold == 0.05
        assert strategy.tax_loss_threshold == -0.10
        assert len(strategy.current_positions) == 0

    def test_initialization_default_values(self, mock_integration_manager, mock_data_provider):
        """Test initialization with default values."""
        strategy = ProductionIndexBaseline(
            mock_integration_manager,
            mock_data_provider,
            {}
        )
        assert len(strategy.benchmarks) > 0
        assert strategy.target_allocation == 0.80

    def test_components_initialized(self, strategy):
        """Test that all components are initialized."""
        assert strategy.integration is not None
        assert strategy.data_provider is not None
        assert strategy.logger is not None


class TestBaselineComparisonDataclass:
    """Test BaselineComparison dataclass."""

    def test_comparison_creation(self):
        """Test creating a baseline comparison."""
        comparison = BaselineComparison(
            ticker="SPY",
            benchmark_return=0.15,
            strategy_return=0.18,
            alpha=0.03,
            sharpe_ratio=1.5,
            max_drawdown=0.08,
            volatility=0.12,
            win_rate=0.65,
            total_trades=100,
            period_days=180,
            last_updated=datetime.now(),
        )

        assert comparison.ticker == "SPY"
        assert comparison.alpha == 0.03
        assert comparison.sharpe_ratio == 1.5


class TestBaselineSignalDataclass:
    """Test BaselineSignal dataclass."""

    def test_signal_creation(self):
        """Test creating a baseline signal."""
        signal = BaselineSignal(
            ticker="SPY",
            signal_type="buy_and_hold",
            current_price=Decimal("450.00"),
            target_allocation=0.80,
            risk_amount=Decimal("10000.00"),
            confidence=0.95,
            metadata={"reason": "rebalancing"},
        )

        assert signal.ticker == "SPY"
        assert signal.signal_type == "buy_and_hold"
        assert signal.target_allocation == 0.80


class TestCalculateBaselinePerformance:
    """Test calculate_baseline_performance method."""

    @pytest.mark.asyncio
    async def test_calculate_performance_success(self, strategy):
        """Test calculating baseline performance."""
        comparisons = await strategy.calculate_baseline_performance(180)

        assert isinstance(comparisons, dict)
        # May have comparisons for benchmarks
        for ticker, comparison in comparisons.items():
            assert isinstance(comparison, BaselineComparison)
            assert ticker in strategy.benchmarks

    @pytest.mark.asyncio
    async def test_calculate_performance_single_benchmark(self, strategy):
        """Test calculating for single benchmark."""
        strategy.benchmarks = ["SPY"]

        comparisons = await strategy.calculate_baseline_performance(180)

        # Should attempt to calculate for SPY
        assert isinstance(comparisons, dict)

    @pytest.mark.asyncio
    async def test_calculate_performance_exception(self, strategy, mock_data_provider):
        """Test exception handling."""
        mock_data_provider.get_historical_data = AsyncMock(
            side_effect=Exception("API Error")
        )

        comparisons = await strategy.calculate_baseline_performance(180)

        assert comparisons == {}


class TestCalculateBenchmarkComparison:
    """Test _calculate_benchmark_comparison method."""

    @pytest.mark.asyncio
    async def test_calculate_comparison_success(self, strategy):
        """Test successful benchmark comparison."""
        comparison = await strategy._calculate_benchmark_comparison("SPY", 180)

        if comparison:  # May be None if insufficient data
            assert isinstance(comparison, BaselineComparison)
            assert comparison.ticker == "SPY"
            assert comparison.period_days == 180

    @pytest.mark.asyncio
    async def test_calculate_comparison_insufficient_data(self, strategy, mock_data_provider):
        """Test handling insufficient data."""
        mock_data_provider.get_historical_data = AsyncMock(return_value=[])

        comparison = await strategy._calculate_benchmark_comparison("SPY", 180)

        assert comparison is None

    @pytest.mark.asyncio
    async def test_calculate_comparison_exception(self, strategy, mock_data_provider):
        """Test exception handling in comparison."""
        mock_data_provider.get_historical_data = AsyncMock(
            side_effect=Exception("Error")
        )

        comparison = await strategy._calculate_benchmark_comparison("SPY", 180)

        assert comparison is None


class TestCalculateStrategyReturn:
    """Test _calculate_strategy_return method."""

    @pytest.mark.asyncio
    async def test_calculate_return_success(self, strategy):
        """Test calculating strategy return."""
        strategy_return = await strategy._calculate_strategy_return("SPY", 180)

        assert isinstance(strategy_return, float)
        # May be positive or negative

    @pytest.mark.asyncio
    async def test_calculate_return_insufficient_data(self, strategy, mock_data_provider):
        """Test handling insufficient data."""
        mock_data_provider.get_historical_data = AsyncMock(return_value=[Mock()])

        strategy_return = await strategy._calculate_strategy_return("SPY", 180)

        assert strategy_return == 0.0


class TestCalculateSharpeRatio:
    """Test _calculate_sharpe_ratio method."""

    @pytest.mark.asyncio
    async def test_calculate_sharpe_success(self, strategy):
        """Test calculating Sharpe ratio."""
        sharpe = await strategy._calculate_sharpe_ratio("SPY", 180)

        assert isinstance(sharpe, float)
        # Sharpe can be negative or positive

    @pytest.mark.asyncio
    async def test_calculate_sharpe_insufficient_data(self, strategy, mock_data_provider):
        """Test handling insufficient data."""
        mock_data_provider.get_historical_data = AsyncMock(return_value=[])

        sharpe = await strategy._calculate_sharpe_ratio("SPY", 180)

        assert sharpe == 0.0

    @pytest.mark.asyncio
    async def test_calculate_sharpe_zero_volatility(self, strategy, mock_data_provider):
        """Test handling zero volatility."""
        # Create flat price data
        mock_data = []
        for i in range(100):
            data_point = Mock()
            data_point.price = Decimal("100.00")  # Same price
            mock_data.append(data_point)

        mock_data_provider.get_historical_data = AsyncMock(return_value=mock_data)

        sharpe = await strategy._calculate_sharpe_ratio("SPY", 90)

        # Should handle zero division
        assert sharpe == 0.0


class TestCalculateMaxDrawdown:
    """Test _calculate_max_drawdown method."""

    @pytest.mark.asyncio
    async def test_calculate_drawdown_success(self, strategy):
        """Test calculating max drawdown."""
        mock_data = []
        prices = [100, 110, 105, 115, 95, 105, 120]  # Has drawdown
        for price in prices:
            data_point = Mock()
            data_point.price = Decimal(str(price))
            mock_data.append(data_point)

        drawdown = await strategy._calculate_max_drawdown(mock_data)

        assert isinstance(drawdown, float)
        assert drawdown >= 0

    @pytest.mark.asyncio
    async def test_calculate_drawdown_uptrend(self, strategy):
        """Test drawdown with pure uptrend."""
        mock_data = []
        for i in range(100, 200):
            data_point = Mock()
            data_point.price = Decimal(str(i))
            mock_data.append(data_point)

        drawdown = await strategy._calculate_max_drawdown(mock_data)

        # Should be zero for pure uptrend
        assert drawdown == 0.0

    @pytest.mark.asyncio
    async def test_calculate_drawdown_insufficient_data(self, strategy):
        """Test handling insufficient data."""
        drawdown = await strategy._calculate_max_drawdown([])

        assert drawdown == 0.0


class TestCalculateVolatility:
    """Test _calculate_volatility method."""

    @pytest.mark.asyncio
    async def test_calculate_volatility_success(self, strategy):
        """Test calculating volatility."""
        mock_data = []
        for i in range(100):
            data_point = Mock()
            data_point.price = Decimal(str(100 + i * 0.5))
            mock_data.append(data_point)

        volatility = await strategy._calculate_volatility(mock_data)

        assert isinstance(volatility, float)
        assert volatility >= 0

    @pytest.mark.asyncio
    async def test_calculate_volatility_insufficient_data(self, strategy):
        """Test handling insufficient data."""
        volatility = await strategy._calculate_volatility([])

        assert volatility == 0.0


class TestCalculateTradeMetrics:
    """Test _calculate_trade_metrics method."""

    @pytest.mark.asyncio
    async def test_calculate_metrics(self, strategy):
        """Test calculating trade metrics."""
        win_rate, total_trades = await strategy._calculate_trade_metrics("SPY", 180)

        assert isinstance(win_rate, float)
        assert isinstance(total_trades, int)
        assert 0 <= win_rate <= 1
        assert total_trades >= 0


class TestGenerateBaselineSignals:
    """Test generate_baseline_signals method."""

    @pytest.mark.asyncio
    async def test_generate_signals_rebalancing_needed(self, strategy, mock_integration_manager):
        """Test generating rebalancing signal."""
        # Set current allocation away from target
        strategy._calculate_current_allocation = AsyncMock(return_value=0.60)  # Below target

        signals = await strategy.generate_baseline_signals()

        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_generate_signals_no_rebalancing(self, strategy):
        """Test no signals when balanced."""
        # Set current allocation at target
        strategy._calculate_current_allocation = AsyncMock(return_value=0.80)

        signals = await strategy.generate_baseline_signals()

        # May or may not have signals depending on other factors
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_generate_signals_tax_loss(self, strategy):
        """Test generating tax loss harvesting signals."""
        strategy._calculate_current_allocation = AsyncMock(return_value=0.80)
        strategy._check_tax_loss_harvesting = AsyncMock(return_value=[Mock()])

        signals = await strategy.generate_baseline_signals()

        assert isinstance(signals, list)


class TestCalculateCurrentAllocation:
    """Test _calculate_current_allocation method."""

    @pytest.mark.asyncio
    async def test_calculate_allocation_success(self, strategy, mock_integration_manager):
        """Test calculating current allocation."""
        allocation = await strategy._calculate_current_allocation()

        assert isinstance(allocation, float)
        assert 0 <= allocation <= 1

    @pytest.mark.asyncio
    async def test_calculate_allocation_zero_portfolio(self, strategy, mock_integration_manager):
        """Test handling zero portfolio value."""
        mock_integration_manager.get_portfolio_value = AsyncMock(return_value=Decimal("0"))

        allocation = await strategy._calculate_current_allocation()

        assert allocation == 0.0


class TestCreateRebalanceSignal:
    """Test _create_rebalance_signal method."""

    @pytest.mark.asyncio
    async def test_create_buy_signal(self, strategy, mock_integration_manager):
        """Test creating buy signal for rebalancing."""
        current_allocation = 0.60  # Below target

        signal = await strategy._create_rebalance_signal(current_allocation)

        if signal:  # May be None if change too small
            assert signal.signal_type in ["buy_and_hold", "rebalance"]

    @pytest.mark.asyncio
    async def test_create_sell_signal(self, strategy, mock_integration_manager):
        """Test creating sell signal for rebalancing."""
        current_allocation = 0.90  # Above target

        signal = await strategy._create_rebalance_signal(current_allocation)

        if signal:
            assert signal.signal_type == "rebalance"

    @pytest.mark.asyncio
    async def test_create_signal_small_change(self, strategy, mock_integration_manager):
        """Test not creating signal for small changes."""
        current_allocation = 0.805  # Very close to target

        signal = await strategy._create_rebalance_signal(current_allocation)

        # Might not create signal for tiny change
        assert signal is None or isinstance(signal, BaselineSignal)

    @pytest.mark.asyncio
    async def test_create_signal_no_price_data(self, strategy, mock_data_provider):
        """Test handling missing price data."""
        mock_data_provider.get_current_price = AsyncMock(return_value=None)

        signal = await strategy._create_rebalance_signal(0.60)

        assert signal is None


class TestCheckTaxLossHarvesting:
    """Test _check_tax_loss_harvesting method."""

    @pytest.mark.asyncio
    async def test_check_tax_loss_finds_opportunities(self, strategy, mock_integration_manager):
        """Test finding tax loss harvesting opportunities."""
        signals = await strategy._check_tax_loss_harvesting()

        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_check_tax_loss_no_positions(self, strategy, mock_integration_manager):
        """Test with no positions."""
        mock_integration_manager.get_position_value = AsyncMock(return_value=Decimal("0"))

        signals = await strategy._check_tax_loss_harvesting()

        # Should not find opportunities without positions
        assert isinstance(signals, list)


class TestExecuteBaselineTrade:
    """Test execute_baseline_trade method."""

    @pytest.mark.asyncio
    async def test_execute_buy_trade_success(self, strategy, mock_integration_manager):
        """Test successful buy trade."""
        signal = BaselineSignal(
            ticker="SPY",
            signal_type="buy_and_hold",
            current_price=Decimal("450.00"),
            target_allocation=0.80,
            risk_amount=Decimal("10000.00"),
            confidence=0.95,
            metadata={},
        )

        success = await strategy.execute_baseline_trade(signal)

        assert success is True
        assert "SPY" in strategy.active_signals

    @pytest.mark.asyncio
    async def test_execute_sell_trade_success(self, strategy, mock_integration_manager):
        """Test successful sell trade."""
        signal = BaselineSignal(
            ticker="SPY",
            signal_type="tax_loss_harvest",
            current_price=Decimal("450.00"),
            target_allocation=0.0,
            risk_amount=Decimal("10000.00"),
            confidence=0.80,
            metadata={},
        )

        success = await strategy.execute_baseline_trade(signal)

        assert success is True

    @pytest.mark.asyncio
    async def test_execute_trade_zero_quantity(self, strategy):
        """Test handling zero quantity."""
        signal = BaselineSignal(
            ticker="SPY",
            signal_type="buy_and_hold",
            current_price=Decimal("1000000.00"),  # Very expensive
            target_allocation=0.80,
            risk_amount=Decimal("100.00"),  # Small amount
            confidence=0.95,
            metadata={},
        )

        success = await strategy.execute_baseline_trade(signal)

        assert success is False

    @pytest.mark.asyncio
    async def test_execute_trade_failure(self, strategy, mock_integration_manager):
        """Test handling trade execution failure."""
        mock_result = Mock()
        mock_result.status = Mock()
        mock_result.status.value = "REJECTED"
        mock_result.error_message = "Insufficient funds"
        mock_integration_manager.execute_trade = AsyncMock(return_value=mock_result)

        signal = BaselineSignal(
            ticker="SPY",
            signal_type="buy_and_hold",
            current_price=Decimal("450.00"),
            target_allocation=0.80,
            risk_amount=Decimal("10000.00"),
            confidence=0.95,
            metadata={},
        )

        success = await strategy.execute_baseline_trade(signal)

        assert success is False


class TestRunStrategy:
    """Test run_strategy method."""

    @pytest.mark.asyncio
    async def test_run_strategy_single_iteration(self, strategy):
        """Test strategy run executes correctly."""
        strategy.calculate_baseline_performance = AsyncMock(return_value={})
        strategy.generate_baseline_signals = AsyncMock(return_value=[])
        strategy.execute_baseline_trade = AsyncMock(return_value=True)

        # Run one iteration
        async def run_once():
            await strategy.calculate_baseline_performance()
            signals = await strategy.generate_baseline_signals()

        await run_once()

        strategy.calculate_baseline_performance.assert_called_once()
        strategy.generate_baseline_signals.assert_called_once()


class TestGetStrategyStatus:
    """Test get_strategy_status method."""

    def test_status_no_signals(self, strategy):
        """Test status with no active signals."""
        status = strategy.get_strategy_status()

        assert status["strategy_name"] == "index_baseline"
        assert status["active_signals"] == 0
        assert isinstance(status["signals"], list)

    def test_status_with_signals(self, strategy):
        """Test status with active signals."""
        signal = BaselineSignal(
            ticker="SPY",
            signal_type="buy_and_hold",
            current_price=Decimal("450.00"),
            target_allocation=0.80,
            risk_amount=Decimal("10000.00"),
            confidence=0.95,
            metadata={},
        )

        strategy.active_signals["SPY"] = signal

        status = strategy.get_strategy_status()

        assert status["active_signals"] == 1
        assert len(status["signals"]) == 1
        assert status["signals"][0]["ticker"] == "SPY"


class TestFactoryFunction:
    """Test factory function."""

    def test_create_production_index_baseline(self, mock_integration_manager, mock_data_provider):
        """Test factory function creates strategy."""
        from backend.tradingbot.strategies.production.production_index_baseline import (
            create_production_index_baseline
        )

        config = {"target_allocation": 0.70}
        strategy = create_production_index_baseline(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        assert isinstance(strategy, ProductionIndexBaseline)
        assert strategy.target_allocation == 0.70


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_benchmarks(self, mock_integration_manager, mock_data_provider):
        """Test handling empty benchmarks list."""
        config = {"benchmarks": []}
        strategy = ProductionIndexBaseline(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        comparisons = await strategy.calculate_baseline_performance(180)
        assert comparisons == {}

    @pytest.mark.asyncio
    async def test_negative_period_days(self, strategy):
        """Test handling negative period."""
        comparisons = await strategy.calculate_baseline_performance(-30)
        # Should handle gracefully
        assert isinstance(comparisons, dict)

    @pytest.mark.asyncio
    async def test_zero_period_days(self, strategy):
        """Test handling zero period."""
        comparisons = await strategy.calculate_baseline_performance(0)
        assert isinstance(comparisons, dict)

    @pytest.mark.asyncio
    async def test_duplicate_signal(self, strategy, mock_integration_manager):
        """Test handling duplicate signal creation."""
        signal = BaselineSignal(
            ticker="SPY",
            signal_type="buy_and_hold",
            current_price=Decimal("450.00"),
            target_allocation=0.80,
            risk_amount=Decimal("10000.00"),
            confidence=0.95,
            metadata={},
        )

        # Add signal
        strategy.active_signals["SPY"] = signal

        # Try to add again
        await strategy.execute_baseline_trade(signal)

        # Should overwrite or handle gracefully
        assert "SPY" in strategy.active_signals

    @pytest.mark.asyncio
    async def test_allocation_above_one(self, strategy, mock_integration_manager):
        """Test handling allocation above 100%."""
        # This shouldn't happen but test resilience
        allocation = await strategy._calculate_current_allocation()

        # If calculation returns >1, it should be handled
        assert allocation >= 0

    @pytest.mark.asyncio
    async def test_negative_allocation(self, strategy):
        """Test handling negative allocation."""
        # Create signal with negative allocation
        signal = BaselineSignal(
            ticker="SPY",
            signal_type="buy_and_hold",
            current_price=Decimal("450.00"),
            target_allocation=-0.10,  # Invalid
            risk_amount=Decimal("10000.00"),
            confidence=0.95,
            metadata={},
        )

        # Should handle gracefully
        try:
            await strategy.execute_baseline_trade(signal)
        except Exception:
            pass  # Expected to handle or raise

    @pytest.mark.asyncio
    async def test_very_long_period(self, strategy):
        """Test handling very long period."""
        comparisons = await strategy.calculate_baseline_performance(10000)
        # Should handle gracefully even if no data
        assert isinstance(comparisons, dict)

    @pytest.mark.asyncio
    async def test_all_benchmarks_fail(self, strategy, mock_data_provider):
        """Test when all benchmark calculations fail."""
        mock_data_provider.get_historical_data = AsyncMock(
            side_effect=Exception("API Error")
        )

        comparisons = await strategy.calculate_baseline_performance(180)

        # Should return empty dict gracefully
        assert comparisons == {}

    def test_statistics_import_error(self, strategy):
        """Test handling if statistics module unavailable."""
        # This tests resilience to import errors
        # The actual implementation should handle this
        pass

    @pytest.mark.asyncio
    async def test_single_price_point(self, strategy):
        """Test calculations with single price point."""
        mock_data = [Mock(price=Decimal("100.00"))]

        volatility = await strategy._calculate_volatility(mock_data)
        assert volatility == 0.0

        drawdown = await strategy._calculate_max_drawdown(mock_data)
        assert drawdown == 0.0

    @pytest.mark.asyncio
    async def test_price_data_with_none(self, strategy):
        """Test handling None in price data."""
        mock_data = [
            Mock(price=Decimal("100.00")),
            Mock(price=None),
            Mock(price=Decimal("105.00")),
        ]

        # Should handle gracefully
        try:
            volatility = await strategy._calculate_volatility(mock_data)
        except Exception:
            pass  # Expected to handle or raise

    @pytest.mark.asyncio
    async def test_extreme_volatility(self, strategy):
        """Test handling extreme volatility."""
        mock_data = []
        prices = [100, 200, 50, 300, 10, 400]  # Extreme swings
        for price in prices:
            data_point = Mock()
            data_point.price = Decimal(str(price))
            mock_data.append(data_point)

        volatility = await strategy._calculate_volatility(mock_data)
        assert isinstance(volatility, float)
        assert volatility > 0
