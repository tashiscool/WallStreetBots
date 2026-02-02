#!/usr/bin/env python3
"""Comprehensive tests for Production SPX Credit Spreads strategy.

Tests all strategy methods to achieve maximum coverage including:
- Credit spread opportunity scanning
- Strike selection and delta targeting
- Risk/reward calculations
- Position management
- Exit conditions
- Black-Scholes pricing
- Error handling
"""

import pytest
import asyncio
import math
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from backend.tradingbot.strategies.production.production_spx_credit_spreads import (
    ProductionSPXCreditSpreads,
    CreditSpreadOpportunity,
)
from backend.tradingbot.production.core.production_integration import (
    ProductionTradeSignal,
    TradeResult,
    TradeStatus,
)
from backend.tradingbot.production.data.production_data_integration import ReliableDataProvider


@pytest.fixture
def mock_data_provider():
    """Create mock data provider."""
    provider = AsyncMock(spec=ReliableDataProvider)

    # Mock market open
    provider.is_market_open = AsyncMock(return_value=True)

    # Mock current price
    provider.get_current_price = AsyncMock(return_value=450.0)

    # Mock historical data
    import pandas as pd
    hist_data = pd.DataFrame({
        'close': [440 + i for i in range(20)],
        'volume': [1000000] * 20
    })
    provider.get_historical_data = AsyncMock(return_value=hist_data)

    # Mock option expiries
    today = date.today()
    expiries = [
        (today + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in [1, 2, 3]
    ]
    provider.get_option_expiries = AsyncMock(return_value=expiries)

    # Mock options chain
    provider.get_options_chain = AsyncMock(return_value={
        "puts": [
            {
                "strike": 440.0,
                "bid": 2.50,
                "ask": 2.60,
                "volume": 100,
                "open_interest": 500,
                "delta": -0.30
            },
            {
                "strike": 430.0,
                "bid": 1.20,
                "ask": 1.30,
                "volume": 80,
                "open_interest": 400,
                "delta": -0.15
            }
        ],
        "calls": [
            {
                "strike": 460.0,
                "bid": 2.40,
                "ask": 2.50,
                "volume": 90,
                "open_interest": 450,
                "delta": 0.30
            },
            {
                "strike": 470.0,
                "bid": 1.10,
                "ask": 1.20,
                "volume": 70,
                "open_interest": 350,
                "delta": 0.15
            }
        ]
    })

    # Mock option price
    provider.get_option_price = AsyncMock(return_value=2.00)

    return provider


@pytest.fixture
def mock_integration_manager():
    """Create mock integration manager."""
    manager = AsyncMock()

    manager.get_portfolio_value = AsyncMock(return_value=100000.0)

    # Create proper TradeResult with required fields
    mock_signal = Mock()
    mock_signal.strategy_name = "spx-credit-spreads"
    mock_signal.ticker = "SPY"
    trade_result = TradeResult(
        trade_id="test_123",
        signal=mock_signal,
        status=TradeStatus.FILLED,
        filled_quantity=1,
        filled_price=2.00,
        commission=0.0,
        error_message=None
    )
    manager.execute_trade = AsyncMock(return_value=trade_result)
    manager.execute_trade_signal = AsyncMock(return_value=True)

    manager.alert_system = AsyncMock()
    manager.alert_system.send_alert = AsyncMock()

    return manager


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "watchlist": ["SPY", "QQQ"],
        "max_positions": 3,
        "max_position_size": 0.05,
        "target_short_delta": 0.30,
        "max_dte": 3,
        "min_net_credit": 0.10,
        "min_spread_width": 5.0,
        "max_spread_width": 20.0,
        "min_prob_profit": 60.0,
        "profit_target_pct": 0.25,
        "stop_loss_pct": 0.75,
        "time_exit_minutes": 60,
    }


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider, config):
    """Create strategy instance."""
    return ProductionSPXCreditSpreads(
        mock_integration_manager,
        mock_data_provider,
        config
    )


class TestProductionSPXCreditSpreads:
    """Test ProductionSPXCreditSpreads class."""

    def test_initialization(self, strategy, config):
        """Test strategy initialization."""
        assert strategy.strategy_name == "spx_credit_spreads"
        assert strategy.max_positions == config["max_positions"]
        assert strategy.max_position_size == config["max_position_size"]
        assert strategy.target_short_delta == config["target_short_delta"]
        assert len(strategy.active_positions) == 0

    def test_norm_cdf(self, strategy):
        """Test normal CDF calculation."""
        # Test known values
        assert abs(strategy.norm_cdf(0) - 0.5) < 0.001
        assert strategy.norm_cdf(-3) < 0.01
        assert strategy.norm_cdf(3) > 0.99

    def test_black_scholes_put(self, strategy):
        """Test Black-Scholes put pricing."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

        price, delta = strategy.black_scholes_put(S, K, T, r, sigma)

        assert price > 0
        assert -1.0 <= delta <= 0.0

    def test_black_scholes_put_zero_time(self, strategy):
        """Test Black-Scholes put with zero time to expiry."""
        S, K, T, r, sigma = 95.0, 100.0, 0.0, 0.05, 0.20

        price, delta = strategy.black_scholes_put(S, K, T, r, sigma)

        # Should return intrinsic value
        assert price == max(K - S, 0)
        assert delta == -1.0

    def test_black_scholes_call(self, strategy):
        """Test Black-Scholes call pricing."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

        price, delta = strategy.black_scholes_call(S, K, T, r, sigma)

        assert price > 0
        assert 0.0 <= delta <= 1.0

    def test_black_scholes_call_zero_time(self, strategy):
        """Test Black-Scholes call with zero time to expiry."""
        S, K, T, r, sigma = 105.0, 100.0, 0.0, 0.05, 0.20

        price, delta = strategy.black_scholes_call(S, K, T, r, sigma)

        # Should return intrinsic value
        assert price == max(S - K, 0)
        assert delta == 1.0

    @pytest.mark.asyncio
    async def test_get_available_expiries(self, strategy, mock_data_provider):
        """Test getting available expiries."""
        expiries = await strategy.get_available_expiries("SPY")

        assert len(expiries) > 0
        for exp_str, dte in expiries:
            assert isinstance(exp_str, str)
            assert 0 <= dte <= strategy.max_dte

    @pytest.mark.asyncio
    async def test_get_available_expiries_no_data(self, strategy, mock_data_provider):
        """Test getting expiries when no data available."""
        mock_data_provider.get_option_expiries = AsyncMock(return_value=None)

        expiries = await strategy.get_available_expiries("SPY")

        assert expiries == []

    @pytest.mark.asyncio
    async def test_get_expected_move(self, strategy, mock_data_provider):
        """Test expected move calculation."""
        move = await strategy.get_expected_move("SPY")

        assert 0.01 <= move <= 0.05

    @pytest.mark.asyncio
    async def test_get_expected_move_insufficient_data(self, strategy, mock_data_provider):
        """Test expected move with insufficient data."""
        import pandas as pd
        mock_data_provider.get_historical_data = AsyncMock(return_value=pd.DataFrame())

        move = await strategy.get_expected_move("SPY")

        assert move == 0.015  # Default value

    @pytest.mark.asyncio
    async def test_find_target_delta_strike_no_data(self, strategy, mock_data_provider):
        """Test finding target delta strike with no options data."""
        mock_data_provider.get_options_chain = AsyncMock(return_value=None)

        strike, delta, premium = await strategy.find_target_delta_strike(
            "SPY", "2024-12-31", "put", 0.30, 450.0
        )

        assert strike is None
        assert delta == 0.0
        assert premium == 0.0

    @pytest.mark.asyncio
    async def test_find_target_delta_strike_put(self, strategy, mock_data_provider):
        """Test finding target delta strike for puts."""
        strike, delta, premium = await strategy.find_target_delta_strike(
            "SPY", "2024-12-31", "put", 0.30, 450.0
        )

        if strike:
            assert strike > 0
            assert delta >= 0
            assert premium >= 0

    @pytest.mark.asyncio
    async def test_find_target_delta_strike_call(self, strategy, mock_data_provider):
        """Test finding target delta strike for calls."""
        strike, delta, premium = await strategy.find_target_delta_strike(
            "SPY", "2024-12-31", "call", 0.30, 450.0
        )

        if strike:
            assert strike > 0
            assert delta >= 0
            assert premium >= 0

    @pytest.mark.asyncio
    async def test_find_target_delta_strike_no_liquid_options(self, strategy, mock_data_provider):
        """Test finding strike when no liquid options available."""
        mock_data_provider.get_options_chain = AsyncMock(return_value={
            "puts": [
                {
                    "strike": 440.0,
                    "bid": 0.01,  # Below minimum
                    "ask": 0.02,
                    "volume": 0,
                    "open_interest": 0,
                }
            ]
        })

        strike, delta, premium = await strategy.find_target_delta_strike(
            "SPY", "2024-12-31", "put", 0.30, 450.0
        )

        assert strike is None

    def test_calculate_spread_metrics(self, strategy):
        """Test spread metrics calculation."""
        net_credit, max_profit, max_loss = strategy.calculate_spread_metrics(
            short_strike=440.0,
            long_strike=430.0,
            short_premium=2.50,
            long_premium=1.20
        )

        assert net_credit == 1.30
        assert max_profit == 1.30
        assert max_loss == 8.70  # 10 - 1.30

    @pytest.mark.asyncio
    async def test_scan_credit_spread_opportunities(self, strategy, mock_data_provider):
        """Test scanning for credit spread opportunities."""
        opportunities = await strategy.scan_credit_spread_opportunities()

        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_scan_credit_spread_opportunities_skip_existing_position(self, strategy):
        """Test scan skips tickers with existing positions."""
        # Add existing position
        strategy.active_positions.append({"ticker": "SPY"})

        opportunities = await strategy.scan_credit_spread_opportunities()

        # Should not include SPY
        spy_opportunities = [opp for opp in opportunities if opp.ticker == "SPY"]
        assert len(spy_opportunities) == 0

    @pytest.mark.asyncio
    async def test_execute_credit_spread_max_positions(self, strategy):
        """Test execute credit spread when at max positions."""
        # Fill up positions
        for i in range(strategy.max_positions):
            strategy.active_positions.append({"ticker": f"TEST{i}"})

        opportunity = CreditSpreadOpportunity(
            ticker="SPY",
            strategy_type="put_credit_spread",
            expiry_date="2024-12-31",
            dte=1,
            short_strike=440.0,
            long_strike=430.0,
            spread_width=10.0,
            net_credit=1.30,
            max_profit=1.30,
            max_loss=8.70,
            short_delta=0.30,
            prob_profit=70.0,
            profit_target=0.325,
            break_even_lower=438.70,
            break_even_upper=float('inf'),
            underlying_price=450.0,
            expected_move=0.015,
            volume_score=70.0
        )

        result = await strategy.execute_credit_spread(opportunity)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_credit_spread_success(self, strategy, mock_integration_manager):
        """Test successful credit spread execution."""
        opportunity = CreditSpreadOpportunity(
            ticker="SPY",
            strategy_type="put_credit_spread",
            expiry_date="2024-12-31",
            dte=1,
            short_strike=440.0,
            long_strike=430.0,
            spread_width=10.0,
            net_credit=1.30,
            max_profit=1.30,
            max_loss=8.70,
            short_delta=0.30,
            prob_profit=70.0,
            profit_target=0.325,
            break_even_lower=438.70,
            break_even_upper=float('inf'),
            underlying_price=450.0,
            expected_move=0.015,
            volume_score=70.0
        )

        result = await strategy.execute_credit_spread(opportunity)

        assert result is True
        assert len(strategy.active_positions) == 1

    @pytest.mark.asyncio
    async def test_execute_credit_spread_call_spread(self, strategy, mock_integration_manager):
        """Test executing call credit spread."""
        opportunity = CreditSpreadOpportunity(
            ticker="SPY",
            strategy_type="call_credit_spread",
            expiry_date="2024-12-31",
            dte=1,
            short_strike=460.0,
            long_strike=470.0,
            spread_width=10.0,
            net_credit=1.30,
            max_profit=1.30,
            max_loss=8.70,
            short_delta=0.30,
            prob_profit=70.0,
            profit_target=0.325,
            break_even_lower=0,
            break_even_upper=461.30,
            underlying_price=450.0,
            expected_move=0.015,
            volume_score=70.0
        )

        result = await strategy.execute_credit_spread(opportunity)

        assert result is True

    @pytest.mark.asyncio
    async def test_manage_positions_profit_target(self, strategy, mock_data_provider):
        """Test position management when profit target is reached."""
        # Create position
        position = {
            "ticker": "SPY",
            "strategy_type": "put_credit_spread",
            "spread_id": "SPY_2024-12-31_440_430",
            "short_signal": Mock(
                strike_price=Decimal("440"),
                expiration_date=date.today() + timedelta(days=1),
                option_type="PUT"
            ),
            "long_signal": Mock(
                strike_price=Decimal("430"),
                expiration_date=date.today() + timedelta(days=1),
                option_type="PUT"
            ),
            "contracts": 1,
            "net_credit": 1.30,
            "profit_target": 0.325,
            "cost_basis": -130,
            "expiry_date": date.today() + timedelta(days=1),
            "dte": 1,
            "prob_profit": 70.0,
        }
        strategy.active_positions.append(position)

        # Mock prices showing profit
        mock_data_provider.get_option_price = AsyncMock(side_effect=[0.50, 0.20])

        await strategy.manage_positions()

        # Position should be closed
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_stop_loss(self, strategy, mock_data_provider):
        """Test position management when stop loss is hit."""
        position = {
            "ticker": "SPY",
            "strategy_type": "put_credit_spread",
            "spread_id": "SPY_2024-12-31_440_430",
            "short_signal": Mock(
                strike_price=Decimal("440"),
                expiration_date=date.today() + timedelta(days=1),
                option_type="PUT"
            ),
            "long_signal": Mock(
                strike_price=Decimal("430"),
                expiration_date=date.today() + timedelta(days=1),
                option_type="PUT"
            ),
            "contracts": 1,
            "net_credit": 1.30,
            "profit_target": 0.325,
            "cost_basis": -130,
            "expiry_date": date.today() + timedelta(days=1),
            "dte": 1,
            "prob_profit": 70.0,
        }
        strategy.active_positions.append(position)

        # Mock prices showing loss
        mock_data_provider.get_option_price = AsyncMock(side_effect=[5.00, 2.50])

        await strategy.manage_positions()

        # Position should be closed
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_time_exit(self, strategy, mock_data_provider):
        """Test position management with time-based exit."""
        position = {
            "ticker": "SPY",
            "strategy_type": "put_credit_spread",
            "spread_id": "SPY_2024-12-31_440_430",
            "short_signal": Mock(
                strike_price=Decimal("440"),
                expiration_date=date.today(),
                option_type="PUT"
            ),
            "long_signal": Mock(
                strike_price=Decimal("430"),
                expiration_date=date.today(),
                option_type="PUT"
            ),
            "contracts": 1,
            "net_credit": 1.30,
            "profit_target": 0.325,
            "cost_basis": -130,
            "expiry_date": date.today(),
            "dte": 0,
            "prob_profit": 70.0,
        }
        strategy.active_positions.append(position)

        mock_data_provider.get_option_price = AsyncMock(side_effect=[1.00, 0.50])

        await strategy.manage_positions()

        # Position should be closed due to time
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_no_price_data(self, strategy, mock_data_provider):
        """Test position management when price data unavailable."""
        position = {
            "ticker": "SPY",
            "strategy_type": "put_credit_spread",
            "spread_id": "SPY_2024-12-31_440_430",
            "short_signal": Mock(
                strike_price=Decimal("440"),
                expiration_date=date.today() + timedelta(days=1),
                option_type="PUT"
            ),
            "long_signal": Mock(
                strike_price=Decimal("430"),
                expiration_date=date.today() + timedelta(days=1),
                option_type="PUT"
            ),
            "contracts": 1,
            "net_credit": 1.30,
            "profit_target": 0.325,
            "cost_basis": -130,
            "expiry_date": date.today() + timedelta(days=1),
            "dte": 1,
            "prob_profit": 70.0,
        }
        strategy.active_positions.append(position)

        mock_data_provider.get_option_price = AsyncMock(return_value=None)

        await strategy.manage_positions()

        # Position should remain
        assert len(strategy.active_positions) == 1

    @pytest.mark.asyncio
    async def test_scan_opportunities_market_closed(self, strategy, mock_data_provider):
        """Test scan opportunities when market is closed."""
        mock_data_provider.is_market_open = AsyncMock(return_value=False)

        signals = await strategy.scan_opportunities()

        assert signals == []

    @pytest.mark.asyncio
    async def test_scan_opportunities_max_positions(self, strategy):
        """Test scan opportunities when at max positions."""
        for i in range(strategy.max_positions):
            strategy.active_positions.append({"ticker": f"TEST{i}"})

        signals = await strategy.scan_opportunities()

        assert signals == []

    @pytest.mark.asyncio
    async def test_scan_opportunities_late_in_day(self, strategy, mock_data_provider):
        """Test scan opportunities late in trading day."""
        with patch('backend.tradingbot.strategies.production.production_spx_credit_spreads.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 15, 30)

            signals = await strategy.scan_opportunities()

            assert signals == []

    def test_get_strategy_status(self, strategy):
        """Test strategy status reporting."""
        # Add a position
        position = {
            "ticker": "SPY",
            "strategy_type": "put_credit_spread",
            "spread_id": "SPY_2024-12-31_440_430",
            "short_signal": Mock(strike_price=Decimal("440")),
            "long_signal": Mock(strike_price=Decimal("430")),
            "contracts": 1,
            "net_credit": 1.30,
            "max_profit": 1.30,
            "max_loss": 8.70,
            "profit_target": 0.325,
            "prob_profit": 70.0,
            "dte": 1,
            "entry_time": datetime.now(),
            "expiry_date": date.today() + timedelta(days=1),
            "cost_basis": -130,
        }
        strategy.active_positions.append(position)

        status = strategy.get_strategy_status()

        assert status["strategy_name"] == "spx_credit_spreads"
        assert status["active_positions"] == 1
        assert len(status["position_details"]) == 1
        assert "config" in status

    def test_get_strategy_status_error_handling(self, strategy):
        """Test strategy status with error."""
        # Add a malformed position
        strategy.active_positions.append({"bad": "data"})

        status = strategy.get_strategy_status()

        assert "error" in status or "strategy_name" in status

    @pytest.mark.asyncio
    async def test_run_strategy_loop(self, strategy):
        """Test main strategy loop (partial execution)."""
        # Mock scan_opportunities to avoid infinite loop
        call_count = 0

        async def mock_scan():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt("Test stop")
            return []

        strategy.scan_opportunities = mock_scan

        with pytest.raises(KeyboardInterrupt):
            await strategy.run_strategy()

    @pytest.mark.asyncio
    async def test_error_handling_in_scan(self, strategy, mock_data_provider):
        """Test error handling during credit spread scan."""
        mock_data_provider.get_current_price = AsyncMock(side_effect=Exception("API Error"))

        opportunities = await strategy.scan_credit_spread_opportunities()

        # Should return empty list, not crash
        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_0dte_eod_exit(self, strategy, mock_data_provider):
        """Test 0DTE end-of-day exit logic."""
        with patch('backend.tradingbot.strategies.production.production_spx_credit_spreads.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 1, 15, 30)
            mock_datetime.now.return_value = mock_now

            position = {
                "ticker": "SPY",
                "strategy_type": "put_credit_spread",
                "spread_id": "SPY_2024-01-01_440_430",
                "short_signal": Mock(
                    strike_price=Decimal("440"),
                    expiration_date=date(2024, 1, 1),
                    option_type="PUT"
                ),
                "long_signal": Mock(
                    strike_price=Decimal("430"),
                    expiration_date=date(2024, 1, 1),
                    option_type="PUT"
                ),
                "contracts": 1,
                "net_credit": 1.30,
                "profit_target": 0.325,
                "cost_basis": -130,
                "expiry_date": date(2024, 1, 1),
                "dte": 0,
                "prob_profit": 70.0,
            }
            strategy.active_positions.append(position)

            mock_data_provider.get_option_price = AsyncMock(side_effect=[1.00, 0.50])

            await strategy.manage_positions()

            # Should exit on 0DTE at 3pm
            assert len(strategy.active_positions) == 0

    def test_credit_spread_opportunity_dataclass(self):
        """Test CreditSpreadOpportunity dataclass."""
        opportunity = CreditSpreadOpportunity(
            ticker="SPY",
            strategy_type="put_credit_spread",
            expiry_date="2024-12-31",
            dte=1,
            short_strike=440.0,
            long_strike=430.0,
            spread_width=10.0,
            net_credit=1.30,
            max_profit=1.30,
            max_loss=8.70,
            short_delta=0.30,
            prob_profit=70.0,
            profit_target=0.325,
            break_even_lower=438.70,
            break_even_upper=float('inf'),
            underlying_price=450.0,
            expected_move=0.015,
            volume_score=70.0
        )

        assert opportunity.ticker == "SPY"
        assert opportunity.strategy_type == "put_credit_spread"
        assert opportunity.net_credit == 1.30
        assert opportunity.prob_profit == 70.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
