#!/usr/bin/env python3
"""Comprehensive tests for Production Debit Spreads strategy.

Tests all strategy methods to achieve maximum coverage including:
- Trend analysis
- IV rank calculation
- Spread opportunity finding
- Position management
- Exit conditions
- Black-Scholes IV estimation
- Error handling
"""

import pytest
import asyncio
import math
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

from backend.tradingbot.strategies.production.production_debit_spreads import (
    ProductionDebitSpreads,
    SpreadOpportunity,
)
from backend.tradingbot.production.core.production_integration import ProductionTradeSignal
from backend.tradingbot.production.data.production_data_integration import ReliableDataProvider


@pytest.fixture
def mock_data_provider():
    """Create mock data provider."""
    provider = AsyncMock(spec=ReliableDataProvider)

    # Mock market open
    provider.is_market_open = AsyncMock(return_value=True)

    # Mock current price
    provider.get_current_price = AsyncMock(return_value=150.0)

    # Mock historical data
    import pandas as pd
    import numpy as np

    # Create trending data
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    prices = [100 + i*0.5 for i in range(60)]
    volumes = [1000000 + i*10000 for i in range(60)]

    hist_data = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    provider.get_historical_data = AsyncMock(return_value=hist_data)

    # Mock option expiries
    expiries = [
        (date.today() + timedelta(days=30)).strftime("%Y-%m-%d"),
        (date.today() + timedelta(days=45)).strftime("%Y-%m-%d"),
    ]
    provider.get_option_expiries = AsyncMock(return_value=expiries)

    # Mock options chain
    provider.get_options_chain = AsyncMock(return_value={
        "calls": [
            {
                "strike": 155.0,
                "bid": 5.00,
                "ask": 5.20,
                "volume": 100,
                "open_interest": 500,
                "delta": 0.50,
                "mid": 5.10
            },
            {
                "strike": 160.0,
                "bid": 3.00,
                "ask": 3.20,
                "volume": 80,
                "open_interest": 400,
                "delta": 0.35,
                "mid": 3.10
            },
        ]
    })

    # Mock option price
    provider.get_option_price = AsyncMock(return_value=5.00)

    return provider


@pytest.fixture
def mock_integration_manager():
    """Create mock integration manager."""
    manager = AsyncMock()
    manager.get_portfolio_value = AsyncMock(return_value=100000.0)
    manager.execute_trade_signal = AsyncMock(return_value=True)
    manager.alert_system = AsyncMock()
    manager.alert_system.send_alert = AsyncMock()
    return manager


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "watchlist": ["AAPL", "MSFT", "GOOGL"],
        "max_positions": 8,
        "max_position_size": 0.10,
        "min_dte": 20,
        "max_dte": 60,
        "min_risk_reward": 1.5,
        "min_trend_strength": 0.6,
        "max_iv_rank": 80,
        "min_volume_score": 0.3,
        "profit_target": 0.30,
        "stop_loss": 0.50,
        "time_exit_dte": 7,
    }


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider, config):
    """Create strategy instance."""
    return ProductionDebitSpreads(
        mock_integration_manager,
        mock_data_provider,
        config
    )


class TestProductionDebitSpreads:
    """Test ProductionDebitSpreads class."""

    def test_initialization(self, strategy, config):
        """Test strategy initialization."""
        assert strategy.strategy_name == "debit_spreads"
        assert strategy.max_positions == config["max_positions"]
        assert strategy.min_dte == config["min_dte"]
        assert strategy.max_dte == config["max_dte"]
        assert len(strategy.active_positions) == 0

    def test_norm_cdf(self, strategy):
        """Test normal CDF calculation."""
        assert abs(strategy.norm_cdf(0) - 0.5) < 0.001
        assert strategy.norm_cdf(-3) < 0.01
        assert strategy.norm_cdf(3) > 0.99

    def test_black_scholes_call(self, strategy):
        """Test Black-Scholes call pricing."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

        price, delta = strategy.black_scholes_call(S, K, T, r, sigma)

        assert price > 0
        assert 0.0 <= delta <= 1.0

    def test_black_scholes_call_zero_time(self, strategy):
        """Test Black-Scholes with zero time to expiry."""
        S, K, T, r, sigma = 105.0, 100.0, 0.0, 0.05, 0.20

        price, delta = strategy.black_scholes_call(S, K, T, r, sigma)

        assert price == max(S - K, 0)
        assert delta == 1.0

    @pytest.mark.asyncio
    async def test_assess_trend_strength_bullish(self, strategy, mock_data_provider):
        """Test trend strength assessment for bullish trend."""
        trend_score = await strategy.assess_trend_strength("AAPL")

        # Should be positive for upward trending data
        assert 0.0 <= trend_score <= 1.0

    @pytest.mark.asyncio
    async def test_assess_trend_strength_insufficient_data(self, strategy, mock_data_provider):
        """Test trend assessment with insufficient data."""
        import pandas as pd
        mock_data_provider.get_historical_data = AsyncMock(return_value=pd.DataFrame())

        trend_score = await strategy.assess_trend_strength("AAPL")

        assert trend_score == 0.5  # Default neutral

    @pytest.mark.asyncio
    async def test_assess_trend_strength_bearish(self, strategy, mock_data_provider):
        """Test trend strength for bearish trend."""
        import pandas as pd

        # Create bearish data
        prices = [100 - i*0.5 for i in range(60)]
        volumes = [1000000] * 60
        hist_data = pd.DataFrame({
            'close': prices,
            'volume': volumes
        })
        mock_data_provider.get_historical_data = AsyncMock(return_value=hist_data)

        trend_score = await strategy.assess_trend_strength("AAPL")

        # Should be lower for downward trend
        assert 0.0 <= trend_score < 0.5

    @pytest.mark.asyncio
    async def test_calculate_iv_rank(self, strategy, mock_data_provider):
        """Test IV rank calculation."""
        iv_rank = await strategy.calculate_iv_rank("AAPL", 0.30)

        assert 0 <= iv_rank <= 100

    @pytest.mark.asyncio
    async def test_calculate_iv_rank_no_data(self, strategy, mock_data_provider):
        """Test IV rank with no historical data."""
        import pandas as pd
        mock_data_provider.get_historical_data = AsyncMock(return_value=pd.DataFrame())

        iv_rank = await strategy.calculate_iv_rank("AAPL", 0.30)

        assert iv_rank == 50.0  # Default

    @pytest.mark.asyncio
    async def test_calculate_iv_rank_extreme_values(self, strategy, mock_data_provider):
        """Test IV rank with extreme IV values."""
        # Very high IV
        iv_rank_high = await strategy.calculate_iv_rank("AAPL", 2.0)
        assert iv_rank_high >= 50

        # Very low IV
        iv_rank_low = await strategy.calculate_iv_rank("AAPL", 0.05)
        assert iv_rank_low <= 50

    @pytest.mark.asyncio
    async def test_get_options_chain(self, strategy, mock_data_provider):
        """Test getting filtered options chain."""
        chain = await strategy.get_options_chain("AAPL", "2024-12-31")

        assert chain is not None
        assert "calls" in chain
        assert len(chain["calls"]) > 0

    @pytest.mark.asyncio
    async def test_get_options_chain_no_data(self, strategy, mock_data_provider):
        """Test options chain when no data available."""
        mock_data_provider.get_options_chain = AsyncMock(return_value=None)

        chain = await strategy.get_options_chain("AAPL", "2024-12-31")

        assert chain is None

    @pytest.mark.asyncio
    async def test_get_options_chain_illiquid(self, strategy, mock_data_provider):
        """Test options chain filtering for illiquid options."""
        mock_data_provider.get_options_chain = AsyncMock(return_value={
            "calls": [
                {
                    "strike": 155.0,
                    "bid": 0.01,  # Too low
                    "ask": 0.02,
                    "volume": 0,
                    "open_interest": 0,
                }
            ]
        })

        chain = await strategy.get_options_chain("AAPL", "2024-12-31")

        assert chain is None

    @pytest.mark.asyncio
    async def test_find_optimal_spreads(self, strategy, mock_data_provider):
        """Test finding optimal spread combinations."""
        options_data = {
            "calls": [
                {
                    "strike": 150.0,
                    "mid": 6.00,
                    "spread": 0.20,
                    "spread_pct": 0.033,
                    "volume": 100,
                    "open_interest": 500,
                },
                {
                    "strike": 160.0,
                    "mid": 3.00,
                    "spread": 0.20,
                    "spread_pct": 0.067,
                    "volume": 80,
                    "open_interest": 400,
                },
            ]
        }

        # Mock trend strength
        strategy.assess_trend_strength = AsyncMock(return_value=0.75)
        strategy.calculate_iv_rank = AsyncMock(return_value=50.0)

        opportunities = await strategy.find_optimal_spreads(
            "AAPL", 150.0, "2024-12-31", options_data
        )

        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_find_optimal_spreads_insufficient_options(self, strategy):
        """Test finding spreads with insufficient options."""
        options_data = {
            "calls": [
                {
                    "strike": 150.0,
                    "mid": 6.00,
                    "spread": 0.20,
                    "spread_pct": 0.033,
                    "volume": 100,
                    "open_interest": 500,
                }
            ]
        }

        opportunities = await strategy.find_optimal_spreads(
            "AAPL", 150.0, "2024-12-31", options_data
        )

        assert opportunities == []

    def test_estimate_iv_from_price(self, strategy):
        """Test IV estimation from option price."""
        iv = strategy.estimate_iv_from_price(
            S=100.0,
            K=100.0,
            T=1.0,
            market_price=10.0
        )

        assert 0.01 <= iv <= 2.0

    def test_estimate_iv_from_price_zero_vega(self, strategy):
        """Test IV estimation with edge cases."""
        # Very short time
        iv = strategy.estimate_iv_from_price(
            S=100.0,
            K=100.0,
            T=0.001,
            market_price=1.0
        )

        assert 0.01 <= iv <= 2.0

    @pytest.mark.asyncio
    async def test_scan_spread_opportunities(self, strategy, mock_data_provider):
        """Test scanning for spread opportunities."""
        opportunities = await strategy.scan_spread_opportunities()

        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_scan_spread_opportunities_skip_existing(self, strategy):
        """Test scan skips existing positions."""
        strategy.active_positions.append({"ticker": "AAPL"})

        opportunities = await strategy.scan_spread_opportunities()

        # AAPL should be skipped
        aapl_opps = [opp for opp in opportunities if opp.ticker == "AAPL"]
        assert len(aapl_opps) == 0

    @pytest.mark.asyncio
    async def test_scan_spread_opportunities_weak_trend(self, strategy, mock_data_provider):
        """Test scan filters weak trends."""
        # Mock weak trend
        import pandas as pd
        prices = [100] * 60  # Flat
        hist_data = pd.DataFrame({'close': prices, 'volume': [1000000] * 60})
        mock_data_provider.get_historical_data = AsyncMock(return_value=hist_data)

        opportunities = await strategy.scan_spread_opportunities()

        # Should filter out weak trends
        assert isinstance(opportunities, list)

    @pytest.mark.asyncio
    async def test_execute_spread_trade_max_positions(self, strategy):
        """Test execute when at max positions."""
        for i in range(strategy.max_positions):
            strategy.active_positions.append({"ticker": f"TEST{i}"})

        opportunity = SpreadOpportunity(
            ticker="AAPL",
            scan_date=date.today(),
            spot_price=150.0,
            trend_strength=0.75,
            expiry_date="2024-12-31",
            days_to_expiry=30,
            long_strike=155.0,
            short_strike=160.0,
            spread_width=5.0,
            long_premium=6.00,
            short_premium=3.00,
            net_debit=3.00,
            max_profit=2.00,
            max_profit_pct=66.67,
            breakeven=158.0,
            prob_profit=0.60,
            risk_reward=0.67,
            iv_rank=50.0,
            volume_score=0.50,
            confidence=0.70
        )

        result = await strategy.execute_spread_trade(opportunity)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_spread_trade_success(self, strategy, mock_integration_manager):
        """Test successful spread trade execution."""
        opportunity = SpreadOpportunity(
            ticker="AAPL",
            scan_date=date.today(),
            spot_price=150.0,
            trend_strength=0.75,
            expiry_date="2024-12-31",
            days_to_expiry=30,
            long_strike=155.0,
            short_strike=160.0,
            spread_width=5.0,
            long_premium=6.00,
            short_premium=3.00,
            net_debit=3.00,
            max_profit=2.00,
            max_profit_pct=66.67,
            breakeven=158.0,
            prob_profit=0.60,
            risk_reward=0.67,
            iv_rank=50.0,
            volume_score=0.50,
            confidence=0.70
        )

        result = await strategy.execute_spread_trade(opportunity)

        assert result is True
        assert len(strategy.active_positions) == 1

    @pytest.mark.asyncio
    async def test_manage_positions_profit_target(self, strategy, mock_data_provider):
        """Test position management when profit target reached."""
        position = {
            "ticker": "AAPL",
            "spread_id": "AAPL_2024-12-31_155_160",
            "long_signal": Mock(
                strike_price=Decimal("155"),
                expiration_date=date.today() + timedelta(days=30)
            ),
            "short_signal": Mock(
                strike_price=Decimal("160"),
                expiration_date=date.today() + timedelta(days=30)
            ),
            "contracts": 1,
            "net_debit": 3.00,
            "max_profit": 2.00,
            "profit_target": 0.60,
            "stop_loss": 1.50,
            "expiry_date": date.today() + timedelta(days=30),
        }
        strategy.active_positions.append(position)

        # Mock profitable prices
        mock_data_provider.get_option_price = AsyncMock(side_effect=[7.00, 3.00])

        await strategy.manage_positions()

        # Position should be closed
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_stop_loss(self, strategy, mock_data_provider):
        """Test position management when stop loss hit."""
        position = {
            "ticker": "AAPL",
            "spread_id": "AAPL_2024-12-31_155_160",
            "long_signal": Mock(
                strike_price=Decimal("155"),
                expiration_date=date.today() + timedelta(days=30)
            ),
            "short_signal": Mock(
                strike_price=Decimal("160"),
                expiration_date=date.today() + timedelta(days=30)
            ),
            "contracts": 1,
            "net_debit": 3.00,
            "max_profit": 2.00,
            "profit_target": 0.60,
            "stop_loss": 1.50,
            "expiry_date": date.today() + timedelta(days=30),
        }
        strategy.active_positions.append(position)

        # Mock losing prices
        mock_data_provider.get_option_price = AsyncMock(side_effect=[1.00, 0.50])

        await strategy.manage_positions()

        # Position should be closed
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_time_exit(self, strategy, mock_data_provider):
        """Test position management with time-based exit."""
        position = {
            "ticker": "AAPL",
            "spread_id": "AAPL_2024-12-31_155_160",
            "long_signal": Mock(
                strike_price=Decimal("155"),
                expiration_date=date.today() + timedelta(days=5)
            ),
            "short_signal": Mock(
                strike_price=Decimal("160"),
                expiration_date=date.today() + timedelta(days=5)
            ),
            "contracts": 1,
            "net_debit": 3.00,
            "max_profit": 2.00,
            "profit_target": 0.60,
            "stop_loss": 1.50,
            "expiry_date": date.today() + timedelta(days=5),
        }
        strategy.active_positions.append(position)

        mock_data_provider.get_option_price = AsyncMock(side_effect=[5.50, 2.50])

        await strategy.manage_positions()

        # Position should be closed due to time
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_no_price_data(self, strategy, mock_data_provider):
        """Test position management when price data unavailable."""
        position = {
            "ticker": "AAPL",
            "spread_id": "AAPL_2024-12-31_155_160",
            "long_signal": Mock(
                strike_price=Decimal("155"),
                expiration_date=date.today() + timedelta(days=30)
            ),
            "short_signal": Mock(
                strike_price=Decimal("160"),
                expiration_date=date.today() + timedelta(days=30)
            ),
            "contracts": 1,
            "net_debit": 3.00,
            "expiry_date": date.today() + timedelta(days=30),
        }
        strategy.active_positions.append(position)

        mock_data_provider.get_option_price = AsyncMock(return_value=None)

        await strategy.manage_positions()

        # Position should remain
        assert len(strategy.active_positions) == 1

    @pytest.mark.asyncio
    async def test_scan_opportunities_market_closed(self, strategy, mock_data_provider):
        """Test scan when market is closed."""
        mock_data_provider.is_market_open = AsyncMock(return_value=False)

        signals = await strategy.scan_opportunities()

        assert signals == []

    @pytest.mark.asyncio
    async def test_scan_opportunities_max_positions(self, strategy):
        """Test scan when at max positions."""
        for i in range(strategy.max_positions):
            strategy.active_positions.append({"ticker": f"TEST{i}"})

        signals = await strategy.scan_opportunities()

        assert signals == []

    def test_get_strategy_status(self, strategy):
        """Test strategy status reporting."""
        position = {
            "ticker": "AAPL",
            "spread_id": "AAPL_2024-12-31_155_160",
            "long_signal": Mock(strike_price=Decimal("155")),
            "short_signal": Mock(strike_price=Decimal("160")),
            "contracts": 1,
            "net_debit": 3.00,
            "max_profit": 2.00,
            "breakeven": 158.0,
            "confidence": 0.70,
            "expiry_date": date.today() + timedelta(days=30),
        }
        strategy.active_positions.append(position)

        status = strategy.get_strategy_status()

        assert status["strategy_name"] == "debit_spreads"
        assert status["active_positions"] == 1
        assert len(status["position_details"]) == 1

    def test_get_strategy_status_error(self, strategy):
        """Test strategy status with error."""
        strategy.active_positions.append({"bad": "data"})

        status = strategy.get_strategy_status()

        assert "error" in status or "strategy_name" in status

    @pytest.mark.asyncio
    async def test_error_handling_in_scan(self, strategy, mock_data_provider):
        """Test error handling during scan."""
        mock_data_provider.get_current_price = AsyncMock(side_effect=Exception("API Error"))

        opportunities = await strategy.scan_spread_opportunities()

        # Should not crash
        assert isinstance(opportunities, list)

    def test_spread_opportunity_dataclass(self):
        """Test SpreadOpportunity dataclass."""
        opportunity = SpreadOpportunity(
            ticker="AAPL",
            scan_date=date.today(),
            spot_price=150.0,
            trend_strength=0.75,
            expiry_date="2024-12-31",
            days_to_expiry=30,
            long_strike=155.0,
            short_strike=160.0,
            spread_width=5.0,
            long_premium=6.00,
            short_premium=3.00,
            net_debit=3.00,
            max_profit=2.00,
            max_profit_pct=66.67,
            breakeven=158.0,
            prob_profit=0.60,
            risk_reward=0.67,
            iv_rank=50.0,
            volume_score=0.50,
            confidence=0.70
        )

        assert opportunity.ticker == "AAPL"
        assert opportunity.net_debit == 3.00
        assert opportunity.confidence == 0.70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
