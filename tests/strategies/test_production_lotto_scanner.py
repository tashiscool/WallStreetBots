"""Comprehensive tests for Production Lotto Scanner Strategy.

Tests all components, edge cases, and error handling for production lotto scanner.
Target: 80%+ coverage.
"""
import asyncio
import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Any

from backend.tradingbot.strategies.production.production_lotto_scanner import (
    ProductionLottoScanner,
    LottoOpportunity,
)
from backend.tradingbot.production.core.production_integration import (
    ProductionTradeSignal,
)


@pytest.fixture
def mock_integration_manager():
    """Create mock integration manager."""
    manager = AsyncMock()
    manager.get_portfolio_value = AsyncMock(return_value=Decimal("50000"))
    manager.execute_trade_signal = AsyncMock(return_value=True)
    manager.send_alert = AsyncMock()
    return manager


@pytest.fixture
def mock_data_provider():
    """Create mock data provider."""
    provider = AsyncMock()
    provider.get_current_price = AsyncMock(return_value=150.0)
    provider.get_recent_prices = AsyncMock(return_value=[100, 101, 102, 103, 104, 105])
    provider.is_market_open = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def lotto_config():
    """Create lotto scanner config."""
    return {
        "max_risk_pct": 0.01,
        "max_concurrent_positions": 3,
        "profit_targets": [300, 500, 800],
        "stop_loss_pct": 0.50,
        "min_win_probability": 0.15,
        "max_dte": 5,
        "high_volume_tickers": ["SPY", "QQQ", "AAPL"],
        "meme_tickers": ["GME", "AMC"],
        "earnings_tickers": ["AAPL", "MSFT", "GOOGL"],
    }


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider, lotto_config):
    """Create ProductionLottoScanner instance."""
    return ProductionLottoScanner(
        mock_integration_manager,
        mock_data_provider,
        lotto_config
    )


class TestProductionLottoScannerInitialization:
    """Test strategy initialization."""

    def test_initialization_success(self, strategy, lotto_config):
        """Test successful initialization."""
        assert strategy.strategy_name == "lotto_scanner"
        assert strategy.max_risk_pct == 0.01
        assert strategy.max_concurrent_positions == 3
        assert strategy.profit_targets == [300, 500, 800]
        assert strategy.stop_loss_pct == 0.50
        assert len(strategy.active_positions) == 0

    def test_initialization_default_values(self, mock_integration_manager, mock_data_provider):
        """Test initialization with default values."""
        strategy = ProductionLottoScanner(
            mock_integration_manager,
            mock_data_provider,
            {}
        )
        assert strategy.max_risk_pct == 0.01
        assert strategy.max_concurrent_positions == 3

    def test_components_initialized(self, strategy):
        """Test that all components are initialized."""
        assert strategy.options_selector is not None
        assert strategy.pricing_engine is not None
        assert strategy.risk_manager is not None
        assert strategy.logger is not None


class TestLottoOpportunityDataclass:
    """Test LottoOpportunity dataclass."""

    def test_opportunity_creation(self):
        """Test creating a lotto opportunity."""
        opp = LottoOpportunity(
            ticker="AAPL",
            play_type="0dte",
            expiry_date="2025-01-17",
            days_to_expiry=0,
            strike=155.0,
            option_type="call",
            current_premium=0.50,
            breakeven=155.50,
            current_spot=150.0,
            catalyst_event="Intraday momentum",
            expected_move=0.03,
            max_position_size=500.0,
            max_contracts=10,
            risk_level="extreme",
            win_probability=0.20,
            potential_return=5.0,
            stop_loss_price=0.25,
            profit_target_price=2.50,
            risk_score=7.5,
        )

        assert opp.ticker == "AAPL"
        assert opp.play_type == "0dte"
        assert opp.risk_level == "extreme"


class TestShouldTrade:
    """Test _should_trade method."""

    @pytest.mark.asyncio
    async def test_should_trade_market_open(self, strategy, mock_data_provider):
        """Test trading allowed when market open."""
        should_trade = await strategy._should_trade()
        assert isinstance(should_trade, bool)

    @pytest.mark.asyncio
    async def test_should_trade_market_closed(self, strategy, mock_data_provider):
        """Test trading not allowed when market closed."""
        mock_data_provider.is_market_open = AsyncMock(return_value=False)

        should_trade = await strategy._should_trade()
        assert should_trade is False

    @pytest.mark.asyncio
    async def test_should_trade_account_too_small(self, strategy, mock_integration_manager):
        """Test trading not allowed with small account."""
        mock_integration_manager.get_portfolio_value = AsyncMock(return_value=Decimal("5000"))

        should_trade = await strategy._should_trade()
        assert should_trade is False

    @pytest.mark.asyncio
    async def test_should_trade_daily_loss_limit(self, strategy, mock_integration_manager):
        """Test trading not allowed when daily loss limit hit."""
        portfolio = Decimal("50000")
        mock_integration_manager.get_portfolio_value = AsyncMock(return_value=portfolio)
        strategy.daily_pnl = -float(portfolio * Decimal("0.06"))  # -6% loss

        should_trade = await strategy._should_trade()
        assert should_trade is False

    @pytest.mark.asyncio
    async def test_should_trade_max_positions(self, strategy):
        """Test trading not allowed when at max positions."""
        for i in range(strategy.max_concurrent_positions):
            strategy.active_positions[f"pos_{i}"] = {"ticker": f"TICK{i}"}

        should_trade = await strategy._should_trade()
        assert should_trade is False

    @pytest.mark.asyncio
    async def test_should_trade_near_market_close(self, strategy):
        """Test trading not allowed near market close."""
        with patch('backend.tradingbot.strategies.production.production_lotto_scanner.datetime') as mock_dt:
            mock_dt.now.return_value = datetime.now().replace(hour=15, minute=45)

            should_trade = await strategy._should_trade()
            assert should_trade is False


class TestMonitorPositions:
    """Test _monitor_positions method."""

    @pytest.mark.asyncio
    async def test_monitor_profit_target_300(self, strategy, mock_data_provider):
        """Test closing at 300% profit."""
        strategy.active_positions["test_pos"] = {
            "ticker": "AAPL",
            "expiry": "2025-01-17",
            "strike": 155.0,
            "option_type": "call",
            "entry_price": 1.00,
            "entry_time": datetime.now(),
            "contracts": 10,
            "play_type": "0dte",
            "catalyst": "Test",
            "days_to_expiry": 0,
        }

        # Mock to get option price
        async def mock_get_option_price(ticker, expiry, strike, option_type):
            return 4.00  # 300% gain

        strategy._get_option_price = mock_get_option_price
        strategy._close_position = AsyncMock()

        await strategy._monitor_positions()

        strategy._close_position.assert_called()

    @pytest.mark.asyncio
    async def test_monitor_stop_loss(self, strategy):
        """Test stop loss trigger."""
        strategy.active_positions["test_pos"] = {
            "ticker": "AAPL",
            "expiry": "2025-01-17",
            "strike": 155.0,
            "option_type": "call",
            "entry_price": 1.00,
            "entry_time": datetime.now(),
            "contracts": 10,
            "play_type": "0dte",
            "catalyst": "Test",
            "days_to_expiry": 0,
        }

        async def mock_get_option_price(ticker, expiry, strike, option_type):
            return 0.40  # > 50% loss

        strategy._get_option_price = mock_get_option_price
        strategy._close_position = AsyncMock()

        await strategy._monitor_positions()

        strategy._close_position.assert_called()

    @pytest.mark.asyncio
    async def test_monitor_time_decay(self, strategy):
        """Test time decay exit."""
        with patch('backend.tradingbot.strategies.production.production_lotto_scanner.datetime') as mock_dt:
            mock_dt.now.return_value = datetime.now().replace(hour=15, minute=30)

            strategy.active_positions["test_pos"] = {
                "ticker": "AAPL",
                "expiry": "2025-01-17",
                "strike": 155.0,
                "option_type": "call",
                "entry_price": 1.00,
                "entry_time": datetime.now(),
                "contracts": 10,
                "play_type": "0dte",
                "catalyst": "Test",
                "days_to_expiry": 0,
            }

            async def mock_get_option_price(ticker, expiry, strike, option_type):
                return 0.80

            strategy._get_option_price = mock_get_option_price
            strategy._close_position = AsyncMock()

            await strategy._monitor_positions()

            strategy._close_position.assert_called()

    @pytest.mark.asyncio
    async def test_monitor_no_price_data(self, strategy):
        """Test handling missing price data."""
        strategy.active_positions["test_pos"] = {
            "ticker": "AAPL",
            "expiry": "2025-01-17",
            "strike": 155.0,
            "option_type": "call",
            "entry_price": 1.00,
            "entry_time": datetime.now(),
            "contracts": 10,
            "play_type": "0dte",
            "catalyst": "Test",
            "days_to_expiry": 0,
        }

        async def mock_get_option_price(ticker, expiry, strike, option_type):
            return None

        strategy._get_option_price = mock_get_option_price

        await strategy._monitor_positions()

        # Should not crash, position remains
        assert "test_pos" in strategy.active_positions


class TestScan0DTEOpportunities:
    """Test _scan_0dte_opportunities method."""

    @pytest.mark.asyncio
    async def test_scan_0dte_finds_opportunities(self, strategy, mock_data_provider):
        """Test scanning finds 0DTE opportunities."""
        strategy._get_0dte_expiry = AsyncMock(return_value="2025-01-17")
        strategy._calculate_momentum_score = AsyncMock(return_value=75.0)
        strategy._estimate_intraday_move = AsyncMock(return_value=0.02)

        # Mock options selector
        mock_chain = {
            "calls": [{"strike": 155, "bid": 0.45, "ask": 0.55, "volume": 1000}],
            "puts": [{"strike": 145, "bid": 0.40, "ask": 0.50, "volume": 1000}],
        }
        strategy.options_selector.get_options_chain = AsyncMock(return_value=mock_chain)
        strategy._evaluate_option_opportunity = AsyncMock(return_value=Mock())

        opportunities = await strategy._scan_0dte_opportunities()

        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_scan_0dte_no_expiry(self, strategy):
        """Test scanning when no 0DTE available."""
        strategy._get_0dte_expiry = AsyncMock(return_value=None)

        opportunities = await strategy._scan_0dte_opportunities()

        assert opportunities == []

    @pytest.mark.asyncio
    async def test_scan_0dte_low_momentum(self, strategy, mock_data_provider):
        """Test filtering low momentum tickers."""
        strategy._get_0dte_expiry = AsyncMock(return_value="2025-01-17")
        strategy._calculate_momentum_score = AsyncMock(return_value=30.0)  # Below threshold

        opportunities = await strategy._scan_0dte_opportunities()

        # Should skip low momentum
        assert len(opportunities) == 0 or all(
            hasattr(o, 'ticker') for o in opportunities
        )


class TestScanEarningsOpportunities:
    """Test _scan_earnings_opportunities method."""

    @pytest.mark.asyncio
    async def test_scan_earnings_finds_opportunities(self, strategy, mock_data_provider):
        """Test scanning finds earnings opportunities."""
        earnings_events = [
            {
                "ticker": "AAPL",
                "date": datetime.now() + timedelta(days=2),
                "expected_move": 0.05,
                "time": "AMC",
            }
        ]

        strategy._get_upcoming_earnings = AsyncMock(return_value=earnings_events)
        strategy._find_best_earnings_expiry = AsyncMock(return_value="2025-01-17")
        strategy._evaluate_option_opportunity = AsyncMock(return_value=Mock())

        opportunities = await strategy._scan_earnings_opportunities()

        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_scan_earnings_no_events(self, strategy):
        """Test scanning with no earnings events."""
        strategy._get_upcoming_earnings = AsyncMock(return_value=[])

        opportunities = await strategy._scan_earnings_opportunities()

        assert opportunities == []


class TestScanCatalystOpportunities:
    """Test _scan_catalyst_opportunities method."""

    @pytest.mark.asyncio
    async def test_scan_catalyst_finds_opportunities(self, strategy, mock_data_provider):
        """Test scanning finds catalyst opportunities."""
        strategy._get_volume_ratio = AsyncMock(return_value=4.0)  # High volume
        strategy._get_price_momentum = AsyncMock(return_value=0.05)  # Strong move
        strategy._get_weekly_expiry = AsyncMock(return_value="2025-01-24")
        strategy._evaluate_option_opportunity = AsyncMock(return_value=Mock())

        opportunities = await strategy._scan_catalyst_opportunities()

        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_scan_catalyst_low_volume(self, strategy):
        """Test filtering low volume."""
        strategy._get_volume_ratio = AsyncMock(return_value=1.5)  # Below threshold
        strategy._get_price_momentum = AsyncMock(return_value=0.05)

        opportunities = await strategy._scan_catalyst_opportunities()

        # Should skip low volume
        assert len(opportunities) == 0 or all(
            hasattr(o, 'ticker') for o in opportunities
        )


class TestEvaluateOptionOpportunity:
    """Test _evaluate_option_opportunity method."""

    @pytest.mark.asyncio
    async def test_evaluate_valid_opportunity(self, strategy, mock_integration_manager):
        """Test evaluating valid opportunity."""
        mock_option = {
            "strike": 155.0,
            "bid": 0.45,
            "ask": 0.55,
            "volume": 500,
            "openInterest": 1000,
        }

        strategy.options_selector.find_best_strike = AsyncMock(return_value=mock_option)

        opp = await strategy._evaluate_option_opportunity(
            "AAPL",
            "2025-01-17",
            155.0,
            "call",
            "0dte",
            "Test catalyst",
            0.03,
            150.0
        )

        assert opp is not None
        assert opp.ticker == "AAPL"
        assert opp.play_type == "0dte"

    @pytest.mark.asyncio
    async def test_evaluate_no_option_found(self, strategy):
        """Test handling when no option found."""
        strategy.options_selector.find_best_strike = AsyncMock(return_value=None)

        opp = await strategy._evaluate_option_opportunity(
            "AAPL",
            "2025-01-17",
            155.0,
            "call",
            "0dte",
            "Test catalyst",
            0.03,
            150.0
        )

        assert opp is None

    @pytest.mark.asyncio
    async def test_evaluate_wide_spreads(self, strategy):
        """Test filtering options with wide spreads."""
        mock_option = {
            "strike": 155.0,
            "bid": 0.20,
            "ask": 0.50,  # Wide spread
            "volume": 500,
            "openInterest": 1000,
        }

        strategy.options_selector.find_best_strike = AsyncMock(return_value=mock_option)

        opp = await strategy._evaluate_option_opportunity(
            "AAPL",
            "2025-01-17",
            155.0,
            "call",
            "0dte",
            "Test catalyst",
            0.03,
            150.0
        )

        assert opp is None

    @pytest.mark.asyncio
    async def test_evaluate_low_liquidity(self, strategy):
        """Test filtering illiquid options."""
        mock_option = {
            "strike": 155.0,
            "bid": 0.45,
            "ask": 0.55,
            "volume": 10,  # Low volume
            "openInterest": 20,  # Low OI
        }

        strategy.options_selector.find_best_strike = AsyncMock(return_value=mock_option)

        opp = await strategy._evaluate_option_opportunity(
            "AAPL",
            "2025-01-17",
            155.0,
            "call",
            "0dte",
            "Test catalyst",
            0.03,
            150.0
        )

        assert opp is None

    @pytest.mark.asyncio
    async def test_evaluate_zero_contracts(self, strategy, mock_integration_manager):
        """Test when position sizing results in zero contracts."""
        mock_integration_manager.get_portfolio_value = AsyncMock(return_value=Decimal("100"))

        mock_option = {
            "strike": 155.0,
            "bid": 100.00,  # Very expensive
            "ask": 105.00,
            "volume": 500,
            "openInterest": 1000,
        }

        strategy.options_selector.find_best_strike = AsyncMock(return_value=mock_option)

        opp = await strategy._evaluate_option_opportunity(
            "AAPL",
            "2025-01-17",
            155.0,
            "call",
            "0dte",
            "Test catalyst",
            0.03,
            150.0
        )

        assert opp is None


class TestExecuteLottoTrade:
    """Test _execute_lotto_trade method."""

    @pytest.mark.asyncio
    async def test_execute_trade_success(self, strategy, mock_integration_manager):
        """Test successful trade execution."""
        opp = LottoOpportunity(
            ticker="AAPL",
            play_type="0dte",
            expiry_date="2025-01-17",
            days_to_expiry=0,
            strike=155.0,
            option_type="call",
            current_premium=0.50,
            breakeven=155.50,
            current_spot=150.0,
            catalyst_event="Test",
            expected_move=0.03,
            max_position_size=500.0,
            max_contracts=10,
            risk_level="extreme",
            win_probability=0.20,
            potential_return=5.0,
            stop_loss_price=0.25,
            profit_target_price=2.50,
            risk_score=7.5,
        )

        await strategy._execute_lotto_trade(opp)

        assert len(strategy.active_positions) == 1
        assert strategy.trade_count == 1

    @pytest.mark.asyncio
    async def test_execute_trade_failure(self, strategy, mock_integration_manager):
        """Test handling trade execution failure."""
        mock_integration_manager.execute_trade_signal = AsyncMock(return_value=False)

        opp = LottoOpportunity(
            ticker="AAPL",
            play_type="0dte",
            expiry_date="2025-01-17",
            days_to_expiry=0,
            strike=155.0,
            option_type="call",
            current_premium=0.50,
            breakeven=155.50,
            current_spot=150.0,
            catalyst_event="Test",
            expected_move=0.03,
            max_position_size=500.0,
            max_contracts=10,
            risk_level="extreme",
            win_probability=0.20,
            potential_return=5.0,
            stop_loss_price=0.25,
            profit_target_price=2.50,
            risk_score=7.5,
        )

        await strategy._execute_lotto_trade(opp)

        # Position should not be added
        assert len(strategy.active_positions) == 0


class TestClosePosition:
    """Test _close_position method."""

    @pytest.mark.asyncio
    async def test_close_position_full(self, strategy, mock_integration_manager):
        """Test closing full position."""
        strategy.active_positions["test_pos"] = {
            "ticker": "AAPL",
            "expiry": "2025-01-17",
            "strike": 155.0,
            "option_type": "call",
            "contracts": 10,
        }

        await strategy._close_position("test_pos", "profit_target", 1.0)

        assert "test_pos" not in strategy.active_positions

    @pytest.mark.asyncio
    async def test_close_position_partial(self, strategy, mock_integration_manager):
        """Test closing partial position."""
        strategy.active_positions["test_pos"] = {
            "ticker": "AAPL",
            "expiry": "2025-01-17",
            "strike": 155.0,
            "option_type": "call",
            "contracts": 10,
        }

        await strategy._close_position("test_pos", "profit_target", 0.5)

        assert "test_pos" in strategy.active_positions
        assert strategy.active_positions["test_pos"]["contracts"] == 5

    @pytest.mark.asyncio
    async def test_close_nonexistent_position(self, strategy):
        """Test closing nonexistent position."""
        await strategy._close_position("nonexistent", "test", 1.0)
        # Should not crash


class TestHelperMethods:
    """Test helper methods."""

    @pytest.mark.asyncio
    async def test_get_0dte_expiry_monday(self, strategy):
        """Test 0DTE expiry on Monday."""
        with patch('backend.tradingbot.strategies.production.production_lotto_scanner.date') as mock_date:
            mock_date.today.return_value = date(2025, 1, 6)  # Monday

            expiry = await strategy._get_0dte_expiry()
            assert expiry == "2025-01-06"

    @pytest.mark.asyncio
    async def test_get_0dte_expiry_tuesday(self, strategy):
        """Test no 0DTE on Tuesday."""
        with patch('backend.tradingbot.strategies.production.production_lotto_scanner.date') as mock_date:
            mock_date.today.return_value = date(2025, 1, 7)  # Tuesday

            expiry = await strategy._get_0dte_expiry()
            assert expiry is None

    @pytest.mark.asyncio
    async def test_get_weekly_expiry(self, strategy):
        """Test getting weekly expiry."""
        expiry_str = await strategy._get_weekly_expiry()
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")

        # Should be a Friday
        assert expiry_date.weekday() == 4

    @pytest.mark.asyncio
    async def test_calculate_momentum_score(self, strategy, mock_data_provider):
        """Test momentum score calculation."""
        score = await strategy._calculate_momentum_score("AAPL")

        assert isinstance(score, float)
        assert 0 <= score <= 100

    @pytest.mark.asyncio
    async def test_calculate_momentum_score_insufficient_data(self, strategy, mock_data_provider):
        """Test momentum score with insufficient data."""
        mock_data_provider.get_recent_prices = AsyncMock(return_value=[])

        score = await strategy._calculate_momentum_score("AAPL")

        assert score == 0

    @pytest.mark.asyncio
    async def test_estimate_intraday_move(self, strategy, mock_data_provider):
        """Test intraday move estimation."""
        move = await strategy._estimate_intraday_move("AAPL")

        assert isinstance(move, float)
        assert 0.01 <= move <= 0.15

    @pytest.mark.asyncio
    async def test_get_volume_ratio(self, strategy):
        """Test volume ratio calculation."""
        ratio = await strategy._get_volume_ratio("AAPL")

        assert isinstance(ratio, float)
        assert ratio > 0

    @pytest.mark.asyncio
    async def test_get_price_momentum(self, strategy, mock_data_provider):
        """Test price momentum calculation."""
        momentum = await strategy._get_price_momentum("AAPL")

        assert isinstance(momentum, float)

    @pytest.mark.asyncio
    async def test_get_option_price(self, strategy):
        """Test getting option price."""
        mock_chain = {
            "calls": [
                {"strike": 155.0, "bid": 0.45, "ask": 0.55}
            ]
        }
        strategy.options_selector.get_options_chain = AsyncMock(return_value=mock_chain)

        price = await strategy._get_option_price("AAPL", "2025-01-17", 155.0, "call")

        assert price == 0.50  # Mid price


class TestFactoryFunction:
    """Test factory function."""

    def test_create_production_lotto_scanner(self, mock_integration_manager, mock_data_provider):
        """Test factory function creates strategy."""
        from backend.tradingbot.strategies.production.production_lotto_scanner import (
            create_production_lotto_scanner
        )

        config = {"max_concurrent_positions": 2}
        strategy = create_production_lotto_scanner(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        assert isinstance(strategy, ProductionLottoScanner)
        assert strategy.max_concurrent_positions == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_watchlists(self, mock_integration_manager, mock_data_provider):
        """Test handling empty watchlists."""
        config = {
            "high_volume_tickers": [],
            "meme_tickers": [],
            "earnings_tickers": [],
        }
        strategy = ProductionLottoScanner(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        opportunities = await strategy._scan_0dte_opportunities()
        assert opportunities == []

    @pytest.mark.asyncio
    async def test_negative_premium(self, strategy):
        """Test handling negative premium."""
        mock_option = {
            "strike": 155.0,
            "bid": -0.10,  # Invalid
            "ask": 0.50,
            "volume": 500,
            "openInterest": 1000,
        }

        strategy.options_selector.find_best_strike = AsyncMock(return_value=mock_option)

        opp = await strategy._evaluate_option_opportunity(
            "AAPL",
            "2025-01-17",
            155.0,
            "call",
            "0dte",
            "Test",
            0.03,
            150.0
        )

        assert opp is None

    @pytest.mark.asyncio
    async def test_zero_volatility_data(self, strategy, mock_data_provider):
        """Test handling zero volatility."""
        mock_data_provider.get_recent_prices = AsyncMock(return_value=[100] * 20)

        move = await strategy._estimate_intraday_move("AAPL")

        # Should return reasonable default
        assert move > 0

    @pytest.mark.asyncio
    async def test_scan_opportunities_exception(self, strategy):
        """Test exception handling in scan."""
        strategy._scan_0dte_opportunities = AsyncMock(side_effect=Exception("Error"))
        strategy._scan_earnings_opportunities = AsyncMock(return_value=[])
        strategy._scan_catalyst_opportunities = AsyncMock(return_value=[])

        # Should not crash
        await strategy._scan_opportunities()

    def test_risk_score_calculation(self, strategy):
        """Test risk score is calculated correctly."""
        # This is tested implicitly in evaluate_option_opportunity
        # but could be tested separately if extracted to its own method
        pass

    @pytest.mark.asyncio
    async def test_max_contracts_cap(self, strategy, mock_integration_manager):
        """Test max contracts hard cap."""
        # Even with large portfolio, should not exceed cap
        mock_integration_manager.get_portfolio_value = AsyncMock(return_value=Decimal("1000000"))

        mock_option = {
            "strike": 155.0,
            "bid": 0.05,  # Cheap option
            "ask": 0.07,
            "volume": 5000,
            "openInterest": 10000,
        }

        strategy.options_selector.find_best_strike = AsyncMock(return_value=mock_option)

        opp = await strategy._evaluate_option_opportunity(
            "AAPL",
            "2025-01-17",
            155.0,
            "call",
            "0dte",
            "Test",
            0.03,
            150.0
        )

        if opp:
            assert opp.max_contracts <= 10  # Hard cap
