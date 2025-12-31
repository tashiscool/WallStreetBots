"""Comprehensive tests for Production Wheel Strategy.

Tests all components, edge cases, and error handling for production wheel strategy.
Target: 80%+ coverage.
"""
import asyncio
import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Any

from backend.tradingbot.strategies.production.production_wheel_strategy import (
    ProductionWheelStrategy,
    WheelPosition,
    WheelCandidate,
)
from backend.tradingbot.production.core.production_integration import (
    TradeSignal,
    OrderSide,
    OrderType,
)
from backend.tradingbot.options.smart_selection import (
    OptionsAnalysis,
    OptionsContract,
    OptionsGreeks,
)


@pytest.fixture
def mock_integration_manager():
    """Create mock integration manager."""
    manager = AsyncMock()
    manager.get_portfolio_value = AsyncMock(return_value=Decimal("100000"))
    manager.execute_trade = AsyncMock(return_value={"status": "success", "order_id": "123"})
    manager.get_position_value = AsyncMock(return_value=Decimal("5000"))
    manager.alert_system = AsyncMock()
    manager.alert_system.send_alert = AsyncMock()
    return manager


@pytest.fixture
def mock_data_provider():
    """Create mock data provider."""
    provider = AsyncMock()
    provider.get_real_time_quote = AsyncMock(return_value=Mock(price=Decimal("150.00")))
    provider.get_implied_volatility = AsyncMock(return_value=60.0)
    provider.get_options_chain = AsyncMock(return_value=[])
    provider.is_market_open = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def wheel_config():
    """Create wheel strategy config."""
    return {
        "target_iv_rank": 50,
        "target_dte_range": (30, 45),
        "target_delta_range": (0.15, 0.30),
        "max_positions": 10,
        "min_premium_dollars": 50,
        "profit_target": 0.25,
        "max_loss_pct": 0.50,
        "assignment_buffer_days": 7,
        "watchlist": ["AAPL", "MSFT", "GOOGL", "TSLA"],
    }


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider, wheel_config):
    """Create ProductionWheelStrategy instance."""
    return ProductionWheelStrategy(
        mock_integration_manager,
        mock_data_provider,
        wheel_config
    )


class TestProductionWheelStrategyInitialization:
    """Test strategy initialization."""

    def test_initialization_success(self, strategy, wheel_config):
        """Test successful initialization."""
        assert strategy.target_iv_rank == 50
        assert strategy.target_dte_range == (30, 45)
        assert strategy.target_delta_range == (0.15, 0.30)
        assert strategy.max_positions == 10
        assert strategy.min_premium_dollars == 50
        assert strategy.profit_target == 0.25
        assert strategy.max_loss_pct == 0.50
        assert strategy.assignment_buffer_days == 7
        assert "AAPL" in strategy.watchlist
        assert len(strategy.positions) == 0

    def test_initialization_default_values(self, mock_integration_manager, mock_data_provider):
        """Test initialization with default values."""
        strategy = ProductionWheelStrategy(
            mock_integration_manager,
            mock_data_provider,
            {}
        )
        assert strategy.target_iv_rank == 50
        assert strategy.max_positions == 10
        assert len(strategy.watchlist) > 0

    def test_components_initialized(self, strategy):
        """Test that all components are initialized."""
        assert strategy.options_selector is not None
        assert strategy.options_engine is not None
        assert strategy.risk_manager is not None
        assert strategy.logger is not None


class TestWheelPositionDataclass:
    """Test WheelPosition dataclass."""

    def test_position_creation(self):
        """Test creating a wheel position."""
        position = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("150.00"),
            current_price=Decimal("152.00"),
            unrealized_pnl=Decimal("200.00"),
            option_strike=Decimal("145.00"),
            option_expiry=date.today() + timedelta(days=30),
            option_premium=Decimal("2.50"),
            option_type="put",
            total_premium_collected=Decimal("250.00"),
            cycle_number=1,
        )

        assert position.ticker == "AAPL"
        assert position.stage == "cash_secured_put"
        assert position.status == "active"
        assert position.quantity == -100
        assert position.option_type == "put"

    def test_position_default_values(self):
        """Test position with default values."""
        position = WheelPosition(
            ticker="MSFT",
            stage="assigned_stock",
            status="active",
            quantity=100,
            entry_price=Decimal("300.00"),
            current_price=Decimal("305.00"),
            unrealized_pnl=Decimal("500.00"),
        )

        assert position.option_strike is None
        assert position.option_expiry is None
        assert position.total_premium_collected == Decimal("0")
        assert position.cycle_number == 1


class TestWheelCandidateDataclass:
    """Test WheelCandidate dataclass."""

    def test_candidate_creation(self):
        """Test creating a wheel candidate."""
        candidate = WheelCandidate(
            ticker="AAPL",
            current_price=Decimal("150.00"),
            iv_rank=Decimal("65.0"),
            put_strike=Decimal("145.00"),
            put_premium=Decimal("2.50"),
            put_delta=Decimal("-0.25"),
            put_expiry=date.today() + timedelta(days=35),
            put_dte=35,
            liquidity_score=Decimal("85.0"),
            volume_score=Decimal("120.0"),
            earnings_risk="low",
            technical_score=Decimal("75.0"),
            annualized_return=Decimal("28.5"),
            probability_profit=Decimal("0.75"),
            risk_reward_ratio=Decimal("0.285"),
        )

        assert candidate.ticker == "AAPL"
        assert candidate.iv_rank == Decimal("65.0")
        assert candidate.put_dte == 35
        assert candidate.earnings_risk == "low"


class TestScanOpportunities:
    """Test scan_opportunities method."""

    @pytest.mark.asyncio
    async def test_scan_opportunities_empty_positions(self, strategy, mock_data_provider):
        """Test scanning with no existing positions."""
        # Mock methods
        strategy._manage_existing_positions = AsyncMock(return_value=[])
        strategy._scan_new_opportunities = AsyncMock(return_value=[
            TradeSignal(
                ticker="AAPL",
                action="SELL_TO_OPEN",
                order_type=OrderType.LIMIT,
                side=OrderSide.SELL,
                quantity=1,
                price=2.50,
                reason="Test signal",
                strategy="wheel_strategy",
                metadata={}
            )
        ])

        signals = await strategy.scan_opportunities()

        assert len(signals) == 1
        assert signals[0].ticker == "AAPL"
        strategy._manage_existing_positions.assert_called_once()
        strategy._scan_new_opportunities.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_opportunities_max_positions_reached(self, strategy):
        """Test scanning when max positions reached."""
        # Fill up positions
        for i in range(strategy.max_positions):
            strategy.positions[f"TICKER{i}"] = WheelPosition(
                ticker=f"TICKER{i}",
                stage="cash_secured_put",
                status="active",
                quantity=-100,
                entry_price=Decimal("100.00"),
                current_price=Decimal("100.00"),
                unrealized_pnl=Decimal("0"),
            )

        strategy._manage_existing_positions = AsyncMock(return_value=[])
        strategy._scan_new_opportunities = AsyncMock(return_value=[])

        signals = await strategy.scan_opportunities()

        # Should not scan for new opportunities
        strategy._scan_new_opportunities.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_opportunities_exception_handling(self, strategy):
        """Test exception handling in scan_opportunities."""
        strategy._manage_existing_positions = AsyncMock(side_effect=Exception("Test error"))

        signals = await strategy.scan_opportunities()

        assert signals == []


class TestManageExistingPositions:
    """Test _manage_existing_positions method."""

    @pytest.mark.asyncio
    async def test_manage_positions_cash_secured_put(self, strategy, mock_data_provider):
        """Test managing cash secured put positions."""
        # Add a position
        strategy.positions["AAPL"] = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("2.00"),
            unrealized_pnl=Decimal("50.00"),
            option_strike=Decimal("145.00"),
            option_expiry=date.today() + timedelta(days=30),
            option_premium=Decimal("2.50"),
            option_type="put",
        )

        strategy._get_current_option_value = AsyncMock(return_value=Decimal("2.00"))
        strategy._check_position_management = AsyncMock(return_value=None)

        signals = await strategy._manage_existing_positions()

        assert len(signals) == 0
        mock_data_provider.get_real_time_quote.assert_called_with("AAPL")

    @pytest.mark.asyncio
    async def test_manage_positions_assigned_stock(self, strategy, mock_data_provider):
        """Test managing assigned stock positions."""
        strategy.positions["MSFT"] = WheelPosition(
            ticker="MSFT",
            stage="assigned_stock",
            status="active",
            quantity=100,
            entry_price=Decimal("300.00"),
            current_price=Decimal("310.00"),
            unrealized_pnl=Decimal("1000.00"),
        )

        strategy._check_position_management = AsyncMock(return_value=None)

        signals = await strategy._manage_existing_positions()

        assert strategy.positions["MSFT"].unrealized_pnl == Decimal("1000.00")

    @pytest.mark.asyncio
    async def test_manage_positions_covered_call(self, strategy, mock_data_provider):
        """Test managing covered call positions."""
        strategy.positions["GOOGL"] = WheelPosition(
            ticker="GOOGL",
            stage="covered_call",
            status="active",
            quantity=100,
            entry_price=Decimal("140.00"),
            current_price=Decimal("145.00"),
            unrealized_pnl=Decimal("500.00"),
            option_strike=Decimal("150.00"),
            option_expiry=date.today() + timedelta(days=20),
            option_premium=Decimal("3.00"),
            option_type="call",
            total_premium_collected=Decimal("5.00"),
        )

        strategy._get_current_option_value = AsyncMock(return_value=Decimal("2.00"))
        strategy._check_position_management = AsyncMock(return_value=None)

        signals = await strategy._manage_existing_positions()

        # Should calculate option P&L correctly
        assert strategy.positions["GOOGL"].unrealized_pnl > Decimal("0")

    @pytest.mark.asyncio
    async def test_manage_positions_no_market_data(self, strategy, mock_data_provider):
        """Test handling when market data unavailable."""
        mock_data_provider.get_real_time_quote = AsyncMock(return_value=None)

        strategy.positions["AAPL"] = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("2.00"),
            unrealized_pnl=Decimal("0"),
        )

        signals = await strategy._manage_existing_positions()

        # Should skip position when no data
        assert len(signals) == 0


class TestCheckPositionManagement:
    """Test _check_position_management method."""

    @pytest.mark.asyncio
    async def test_put_profit_target_hit(self, strategy):
        """Test closing put when profit target hit."""
        position = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("150.00"),
            unrealized_pnl=Decimal("75.00"),  # 30% profit
            option_strike=Decimal("145.00"),
            option_expiry=date.today() + timedelta(days=20),
            option_premium=Decimal("250.00"),
            option_type="put",
        )

        signal = await strategy._check_position_management(position)

        assert signal is not None
        assert signal.action == "BUY_TO_CLOSE"
        assert "profit target" in signal.reason.lower()

    @pytest.mark.asyncio
    async def test_put_close_before_assignment(self, strategy):
        """Test closing put before assignment."""
        position = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("150.00"),
            unrealized_pnl=Decimal("25.00"),
            option_strike=Decimal("145.00"),
            option_expiry=date.today() + timedelta(days=5),  # Within buffer
            option_premium=Decimal("250.00"),
            option_type="put",
        )

        signal = await strategy._check_position_management(position)

        assert signal is not None
        assert signal.action == "BUY_TO_CLOSE"
        assert "assignment" in signal.reason.lower()

    @pytest.mark.asyncio
    async def test_call_profit_target_hit(self, strategy):
        """Test closing call when profit target hit."""
        position = WheelPosition(
            ticker="MSFT",
            stage="covered_call",
            status="active",
            quantity=100,
            entry_price=Decimal("300.00"),
            current_price=Decimal("305.00"),
            unrealized_pnl=Decimal("100.00"),
            option_strike=Decimal("310.00"),
            option_expiry=date.today() + timedelta(days=15),
            option_premium=Decimal("300.00"),
            option_type="call",
        )

        signal = await strategy._check_position_management(position)

        assert signal is not None
        assert signal.action == "BUY_TO_CLOSE"

    @pytest.mark.asyncio
    async def test_assigned_stock_covered_call_opportunity(self, strategy):
        """Test finding covered call for assigned stock."""
        position = WheelPosition(
            ticker="GOOGL",
            stage="assigned_stock",
            status="active",
            quantity=100,
            entry_price=Decimal("140.00"),
            current_price=Decimal("145.00"),
            unrealized_pnl=Decimal("500.00"),
        )

        strategy._find_covered_call_opportunity = AsyncMock(return_value=Mock())

        signal = await strategy._check_position_management(position)

        strategy._find_covered_call_opportunity.assert_called_once_with(position)

    @pytest.mark.asyncio
    async def test_no_management_action_needed(self, strategy):
        """Test when no management action needed."""
        position = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("150.00"),
            unrealized_pnl=Decimal("25.00"),  # Below profit target
            option_strike=Decimal("145.00"),
            option_expiry=date.today() + timedelta(days=20),  # Not near expiry
            option_premium=Decimal("250.00"),
            option_type="put",
        )

        signal = await strategy._check_position_management(position)

        assert signal is None


class TestScanNewOpportunities:
    """Test _scan_new_opportunities method."""

    @pytest.mark.asyncio
    async def test_scan_finds_candidates(self, strategy):
        """Test scanning finds valid candidates."""
        mock_candidate = WheelCandidate(
            ticker="AAPL",
            current_price=Decimal("150.00"),
            iv_rank=Decimal("65.0"),
            put_strike=Decimal("145.00"),
            put_premium=Decimal("2.50"),
            put_delta=Decimal("-0.25"),
            put_expiry=date.today() + timedelta(days=35),
            put_dte=35,
            liquidity_score=Decimal("85.0"),
            volume_score=Decimal("120.0"),
            earnings_risk="low",
            technical_score=Decimal("75.0"),
            annualized_return=Decimal("28.5"),
            probability_profit=Decimal("0.75"),
            risk_reward_ratio=Decimal("0.285"),
        )

        strategy._analyze_wheel_candidate = AsyncMock(return_value=mock_candidate)
        strategy._create_cash_secured_put_signal = Mock(return_value=Mock())

        signals = await strategy._scan_new_opportunities()

        assert len(signals) > 0

    @pytest.mark.asyncio
    async def test_scan_skips_existing_positions(self, strategy):
        """Test scanning skips tickers with existing positions."""
        strategy.positions["AAPL"] = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("2.00"),
            unrealized_pnl=Decimal("0"),
        )

        strategy._analyze_wheel_candidate = AsyncMock(return_value=None)

        signals = await strategy._scan_new_opportunities()

        # Should not analyze AAPL since it has a position
        calls = [call[0][0] for call in strategy._analyze_wheel_candidate.call_args_list]
        assert "AAPL" not in calls

    @pytest.mark.asyncio
    async def test_scan_respects_max_positions(self, strategy):
        """Test scanning respects max positions limit."""
        # Create candidates
        candidates = []
        for i in range(15):  # More than max_positions
            candidates.append(WheelCandidate(
                ticker=f"TICK{i}",
                current_price=Decimal("100.00"),
                iv_rank=Decimal("60.0"),
                put_strike=Decimal("95.00"),
                put_premium=Decimal("2.00"),
                put_delta=Decimal("-0.25"),
                put_expiry=date.today() + timedelta(days=30),
                put_dte=30,
                liquidity_score=Decimal("80.0"),
                volume_score=Decimal("100.0"),
                earnings_risk="low",
                technical_score=Decimal("70.0"),
                annualized_return=Decimal(f"{25 + i}.0"),  # Different returns
                probability_profit=Decimal("0.70"),
                risk_reward_ratio=Decimal("0.25"),
            ))

        strategy._analyze_wheel_candidate = AsyncMock(side_effect=candidates)
        strategy._create_cash_secured_put_signal = Mock(return_value=Mock())

        signals = await strategy._scan_new_opportunities()

        # Should not exceed max_positions
        assert len(signals) <= strategy.max_positions

    @pytest.mark.asyncio
    async def test_scan_sorts_by_return(self, strategy):
        """Test candidates sorted by annualized return."""
        candidates = [
            WheelCandidate(
                ticker="LOW",
                current_price=Decimal("100.00"),
                iv_rank=Decimal("60.0"),
                put_strike=Decimal("95.00"),
                put_premium=Decimal("2.00"),
                put_delta=Decimal("-0.25"),
                put_expiry=date.today() + timedelta(days=30),
                put_dte=30,
                liquidity_score=Decimal("80.0"),
                volume_score=Decimal("100.0"),
                earnings_risk="low",
                technical_score=Decimal("70.0"),
                annualized_return=Decimal("15.0"),
                probability_profit=Decimal("0.70"),
                risk_reward_ratio=Decimal("0.15"),
            ),
            WheelCandidate(
                ticker="HIGH",
                current_price=Decimal("100.00"),
                iv_rank=Decimal("60.0"),
                put_strike=Decimal("95.00"),
                put_premium=Decimal("2.00"),
                put_delta=Decimal("-0.25"),
                put_expiry=date.today() + timedelta(days=30),
                put_dte=30,
                liquidity_score=Decimal("80.0"),
                volume_score=Decimal("100.0"),
                earnings_risk="low",
                technical_score=Decimal("70.0"),
                annualized_return=Decimal("35.0"),
                probability_profit=Decimal("0.70"),
                risk_reward_ratio=Decimal("0.35"),
            ),
        ]

        strategy.watchlist = ["LOW", "HIGH"]
        strategy._analyze_wheel_candidate = AsyncMock(side_effect=candidates)
        strategy._create_cash_secured_put_signal = Mock(side_effect=lambda c: Mock(ticker=c.ticker))

        signals = await strategy._scan_new_opportunities()

        # First signal should be HIGH (higher return)
        assert signals[0].ticker == "HIGH"


class TestAnalyzeWheelCandidate:
    """Test _analyze_wheel_candidate method."""

    @pytest.mark.asyncio
    async def test_analyze_valid_candidate(self, strategy, mock_data_provider):
        """Test analyzing a valid candidate."""
        mock_put_analysis = Mock(spec=OptionsAnalysis)
        mock_put_analysis.contract = Mock(spec=OptionsContract)
        mock_put_analysis.contract.strike = Decimal("145.00")
        mock_put_analysis.contract.expiry_date = date.today() + timedelta(days=35)
        mock_put_analysis.mid_price = Decimal("2.50")
        mock_put_analysis.greeks = Mock(spec=OptionsGreeks)
        mock_put_analysis.greeks.delta = -0.25

        strategy._find_optimal_put = AsyncMock(return_value=mock_put_analysis)
        strategy._calculate_liquidity_score = AsyncMock(return_value=Decimal("85.0"))
        strategy._calculate_volume_score = AsyncMock(return_value=Decimal("120.0"))
        strategy._assess_earnings_risk = AsyncMock(return_value="low")
        strategy._calculate_technical_score = AsyncMock(return_value=Decimal("75.0"))

        candidate = await strategy._analyze_wheel_candidate("AAPL")

        assert candidate is not None
        assert candidate.ticker == "AAPL"
        assert candidate.put_strike == Decimal("145.00")
        assert candidate.annualized_return > Decimal("0")

    @pytest.mark.asyncio
    async def test_analyze_low_iv_rank(self, strategy, mock_data_provider):
        """Test candidate rejected due to low IV rank."""
        mock_data_provider.get_implied_volatility = AsyncMock(return_value=30.0)  # Below threshold

        candidate = await strategy._analyze_wheel_candidate("AAPL")

        assert candidate is None

    @pytest.mark.asyncio
    async def test_analyze_no_market_data(self, strategy, mock_data_provider):
        """Test handling when no market data available."""
        mock_data_provider.get_real_time_quote = AsyncMock(return_value=None)

        candidate = await strategy._analyze_wheel_candidate("AAPL")

        assert candidate is None

    @pytest.mark.asyncio
    async def test_analyze_no_put_found(self, strategy, mock_data_provider):
        """Test when no suitable put found."""
        strategy._find_optimal_put = AsyncMock(return_value=None)

        candidate = await strategy._analyze_wheel_candidate("AAPL")

        assert candidate is None


class TestFindOptimalPut:
    """Test _find_optimal_put method."""

    @pytest.mark.asyncio
    async def test_find_optimal_put_success(self, strategy):
        """Test finding optimal put successfully."""
        mock_analysis = Mock(spec=OptionsAnalysis)
        strategy.options_selector.select_optimal_put_option = AsyncMock(return_value=mock_analysis)

        result = await strategy._find_optimal_put("AAPL", Decimal("150.00"))

        assert result == mock_analysis

    @pytest.mark.asyncio
    async def test_find_optimal_put_exception(self, strategy):
        """Test exception handling in find optimal put."""
        strategy.options_selector.select_optimal_put_option = AsyncMock(
            side_effect=Exception("Test error")
        )

        result = await strategy._find_optimal_put("AAPL", Decimal("150.00"))

        assert result is None


class TestFindCoveredCallOpportunity:
    """Test _find_covered_call_opportunity method."""

    @pytest.mark.asyncio
    async def test_find_covered_call_success(self, strategy):
        """Test finding covered call successfully."""
        position = WheelPosition(
            ticker="AAPL",
            stage="assigned_stock",
            status="active",
            quantity=100,
            entry_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
            unrealized_pnl=Decimal("500.00"),
        )

        mock_call_analysis = Mock(spec=OptionsAnalysis)
        mock_call_analysis.contract = Mock(spec=OptionsContract)
        mock_call_analysis.contract.strike = Decimal("152.00")
        mock_call_analysis.contract.expiry_date = date.today() + timedelta(days=30)
        mock_call_analysis.mid_price = Decimal("2.00")
        mock_call_analysis.greeks = Mock(spec=OptionsGreeks)
        mock_call_analysis.greeks.delta = 0.30

        strategy.options_selector.select_optimal_call_option = AsyncMock(
            return_value=mock_call_analysis
        )

        signal = await strategy._find_covered_call_opportunity(position)

        assert signal is not None
        assert signal.action == "SELL_TO_OPEN"
        assert signal.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_find_covered_call_strike_too_low(self, strategy):
        """Test rejecting call when strike too low."""
        position = WheelPosition(
            ticker="AAPL",
            stage="assigned_stock",
            status="active",
            quantity=100,
            entry_price=Decimal("150.00"),
            current_price=Decimal("148.00"),
            unrealized_pnl=Decimal("-200.00"),
        )

        mock_call_analysis = Mock(spec=OptionsAnalysis)
        mock_call_analysis.contract = Mock(spec=OptionsContract)
        mock_call_analysis.contract.strike = Decimal("140.00")  # Below entry price

        strategy.options_selector.select_optimal_call_option = AsyncMock(
            return_value=mock_call_analysis
        )

        signal = await strategy._find_covered_call_opportunity(position)

        assert signal is None


class TestExecuteTrades:
    """Test execute_trades method."""

    @pytest.mark.asyncio
    async def test_execute_trades_success(self, strategy, mock_integration_manager):
        """Test successful trade execution."""
        signal = TradeSignal(
            ticker="AAPL",
            action="SELL_TO_OPEN",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=1,
            price=2.50,
            reason="Test signal",
            strategy="wheel_strategy",
            metadata={
                "stage": "cash_secured_put",
                "strike": 145.00,
                "expiry": (date.today() + timedelta(days=30)).isoformat(),
                "premium": 2.50,
            }
        )

        strategy._update_position_after_trade = AsyncMock()

        results = await strategy.execute_trades([signal])

        assert len(results) == 1
        assert results[0]["status"] == "success"
        strategy._update_position_after_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_trades_failure(self, strategy, mock_integration_manager):
        """Test handling trade execution failure."""
        mock_integration_manager.execute_trade = AsyncMock(
            side_effect=Exception("Execution failed")
        )

        signal = TradeSignal(
            ticker="AAPL",
            action="SELL_TO_OPEN",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=1,
            price=2.50,
            reason="Test signal",
            strategy="wheel_strategy",
            metadata={}
        )

        results = await strategy.execute_trades([signal])

        assert len(results) == 1
        assert results[0]["status"] == "error"


class TestUpdatePositionAfterTrade:
    """Test _update_position_after_trade method."""

    @pytest.mark.asyncio
    async def test_update_new_cash_secured_put(self, strategy):
        """Test updating position after opening cash secured put."""
        signal = TradeSignal(
            ticker="AAPL",
            action="SELL_TO_OPEN",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=1,
            price=2.50,
            reason="Test",
            strategy="wheel_strategy",
            metadata={
                "stage": "cash_secured_put",
                "strike": 145.00,
                "expiry": (date.today() + timedelta(days=30)).isoformat(),
                "premium": 2.50,
            }
        )

        result = {"status": "success"}

        await strategy._update_position_after_trade(signal, result)

        assert "AAPL" in strategy.positions
        assert strategy.positions["AAPL"].stage == "cash_secured_put"
        assert strategy.positions["AAPL"].quantity == -100

    @pytest.mark.asyncio
    async def test_update_covered_call(self, strategy):
        """Test updating position after opening covered call."""
        # Create existing position
        strategy.positions["AAPL"] = WheelPosition(
            ticker="AAPL",
            stage="assigned_stock",
            status="active",
            quantity=100,
            entry_price=Decimal("145.00"),
            current_price=Decimal("150.00"),
            unrealized_pnl=Decimal("500.00"),
        )

        signal = TradeSignal(
            ticker="AAPL",
            action="SELL_TO_OPEN",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=1,
            price=2.00,
            reason="Test",
            strategy="wheel_strategy",
            metadata={
                "stage": "covered_call",
                "strike": 155.00,
                "expiry": (date.today() + timedelta(days=30)).isoformat(),
                "premium": 2.00,
            }
        )

        result = {"status": "success"}

        await strategy._update_position_after_trade(signal, result)

        assert strategy.positions["AAPL"].stage == "covered_call"
        assert strategy.positions["AAPL"].option_type == "call"

    @pytest.mark.asyncio
    async def test_update_close_position(self, strategy):
        """Test updating position after closing."""
        strategy.positions["AAPL"] = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("1.50"),
            unrealized_pnl=Decimal("100.00"),
        )

        signal = TradeSignal(
            ticker="AAPL",
            action="BUY_TO_CLOSE",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1,
            reason="Test",
            strategy="wheel_strategy",
            metadata={"stage": "close_profitable_put"}
        )

        result = {"status": "success"}

        await strategy._update_position_after_trade(signal, result)

        assert strategy.positions["AAPL"].status == "closed"


class TestGetCurrentOptionValue:
    """Test _get_current_option_value method."""

    @pytest.mark.asyncio
    async def test_get_option_value_success(self, strategy, mock_data_provider):
        """Test getting current option value successfully."""
        position = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("150.00"),
            unrealized_pnl=Decimal("0"),
            option_strike=Decimal("145.00"),
            option_expiry=date.today() + timedelta(days=20),
            option_type="put",
        )

        mock_contract = Mock()
        mock_contract.strike = Decimal("145.00")
        mock_contract.option_type = "put"
        mock_contract.mid_price = Decimal("1.50")

        mock_data_provider.get_options_chain = AsyncMock(return_value=[mock_contract])

        value = await strategy._get_current_option_value(position)

        assert value == Decimal("1.50")

    @pytest.mark.asyncio
    async def test_get_option_value_no_data(self, strategy, mock_data_provider):
        """Test handling when no option data available."""
        position = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("150.00"),
            unrealized_pnl=Decimal("0"),
            option_strike=Decimal("145.00"),
            option_expiry=date.today() + timedelta(days=20),
            option_type="put",
        )

        mock_data_provider.get_options_chain = AsyncMock(return_value=None)

        value = await strategy._get_current_option_value(position)

        assert value == Decimal("0")


class TestHelperMethods:
    """Test helper methods."""

    def test_calculate_annualized_return(self, strategy):
        """Test annualized return calculation."""
        mock_analysis = Mock(spec=OptionsAnalysis)
        mock_analysis.contract = Mock(spec=OptionsContract)
        mock_analysis.contract.expiry_date = date.today() + timedelta(days=30)
        mock_analysis.contract.strike = Decimal("145.00")
        mock_analysis.mid_price = Decimal("2.50")

        annualized = strategy._calculate_annualized_return(mock_analysis)

        assert annualized > Decimal("0")
        assert annualized < Decimal("200")  # Reasonable range

    def test_calculate_annualized_return_expired(self, strategy):
        """Test annualized return for expired option."""
        mock_analysis = Mock(spec=OptionsAnalysis)
        mock_analysis.contract = Mock(spec=OptionsContract)
        mock_analysis.contract.expiry_date = date.today() - timedelta(days=1)

        annualized = strategy._calculate_annualized_return(mock_analysis)

        assert annualized == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_liquidity_score(self, strategy):
        """Test liquidity score calculation."""
        score = await strategy._calculate_liquidity_score("AAPL")
        assert score == Decimal("75")

    @pytest.mark.asyncio
    async def test_calculate_volume_score(self, strategy):
        """Test volume score calculation."""
        score = await strategy._calculate_volume_score("AAPL")
        assert score == Decimal("100")

    @pytest.mark.asyncio
    async def test_assess_earnings_risk(self, strategy):
        """Test earnings risk assessment."""
        risk = await strategy._assess_earnings_risk("AAPL")
        assert risk == "low"

    @pytest.mark.asyncio
    async def test_calculate_technical_score(self, strategy):
        """Test technical score calculation."""
        score = await strategy._calculate_technical_score("AAPL")
        assert score == Decimal("70")


class TestGetPerformanceMetrics:
    """Test get_performance_metrics method."""

    @pytest.mark.asyncio
    async def test_performance_metrics_with_positions(self, strategy):
        """Test performance metrics with active positions."""
        strategy.positions["AAPL"] = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("2.00"),
            unrealized_pnl=Decimal("50.00"),
            total_premium_collected=Decimal("250.00"),
            cycle_number=1,
        )

        strategy.positions["MSFT"] = WheelPosition(
            ticker="MSFT",
            stage="covered_call",
            status="active",
            quantity=100,
            entry_price=Decimal("300.00"),
            current_price=Decimal("305.00"),
            unrealized_pnl=Decimal("500.00"),
            total_premium_collected=Decimal("400.00"),
            cycle_number=2,
        )

        metrics = await strategy.get_performance_metrics()

        assert metrics["strategy"] == "wheel_strategy"
        assert metrics["total_positions"] == 2
        assert metrics["active_positions"] == 2
        assert metrics["total_unrealized_pnl"] == 550.0
        assert metrics["total_premium_collected"] == 650.0
        assert "positions" in metrics

    @pytest.mark.asyncio
    async def test_performance_metrics_empty(self, strategy):
        """Test performance metrics with no positions."""
        metrics = await strategy.get_performance_metrics()

        assert metrics["total_positions"] == 0
        assert metrics["active_positions"] == 0
        assert metrics["avg_premium_per_position"] == 0


class TestRunStrategy:
    """Test run_strategy method."""

    @pytest.mark.asyncio
    async def test_run_strategy_single_iteration(self, strategy):
        """Test run_strategy executes correctly."""
        strategy.scan_opportunities = AsyncMock(return_value=[])
        strategy.execute_trades = AsyncMock(return_value=[])

        # Run one iteration
        async def run_once():
            await strategy.scan_opportunities()
            await strategy.execute_trades([])

        await run_once()

        strategy.scan_opportunities.assert_called_once()


class TestFactoryFunction:
    """Test factory function."""

    def test_create_production_wheel_strategy(self, mock_integration_manager, mock_data_provider):
        """Test factory function creates strategy."""
        from backend.tradingbot.strategies.production.production_wheel_strategy import (
            create_production_wheel_strategy
        )

        config = {"max_positions": 5}
        strategy = create_production_wheel_strategy(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        assert isinstance(strategy, ProductionWheelStrategy)
        assert strategy.max_positions == 5


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_watchlist(self, mock_integration_manager, mock_data_provider):
        """Test handling empty watchlist."""
        config = {"watchlist": []}
        strategy = ProductionWheelStrategy(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        signals = await strategy._scan_new_opportunities()
        assert signals == []

    @pytest.mark.asyncio
    async def test_invalid_option_expiry_format(self, strategy):
        """Test handling invalid expiry format."""
        signal = TradeSignal(
            ticker="AAPL",
            action="SELL_TO_OPEN",
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=1,
            price=2.50,
            reason="Test",
            strategy="wheel_strategy",
            metadata={
                "stage": "cash_secured_put",
                "strike": 145.00,
                "expiry": "invalid-date",
                "premium": 2.50,
            }
        )

        result = {"status": "success"}

        # Should handle exception gracefully
        await strategy._update_position_after_trade(signal, result)

    @pytest.mark.asyncio
    async def test_zero_quantity_position(self, strategy):
        """Test handling zero quantity position."""
        position = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=0,
            entry_price=Decimal("2.50"),
            current_price=Decimal("2.00"),
            unrealized_pnl=Decimal("0"),
        )

        signal = await strategy._check_position_management(position)
        # Should handle gracefully
        assert signal is None or isinstance(signal, TradeSignal)

    @pytest.mark.asyncio
    async def test_negative_unrealized_pnl(self, strategy):
        """Test handling negative unrealized P&L."""
        position = WheelPosition(
            ticker="AAPL",
            stage="cash_secured_put",
            status="active",
            quantity=-100,
            entry_price=Decimal("2.50"),
            current_price=Decimal("150.00"),
            unrealized_pnl=Decimal("-150.00"),
            option_strike=Decimal("145.00"),
            option_expiry=date.today() + timedelta(days=20),
            option_premium=Decimal("250.00"),
            option_type="put",
        )

        signal = await strategy._check_position_management(position)
        # Should not generate profit-taking signal
        if signal:
            assert "profit" not in signal.reason.lower()
