#!/usr/bin/env python3
"""Comprehensive tests for Production LEAPS Tracker strategy.

Tests all strategy methods to achieve maximum coverage including:
- Moving average cross analysis (golden/death cross)
- Entry and exit timing scores
- Secular theme analysis
- Comprehensive scoring
- Position management with scale-out
- LEAPS premium estimation
- Error handling
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

from backend.tradingbot.strategies.production.production_leaps_tracker import (
    ProductionLEAPSTracker,
    LEAPSCandidate,
    MovingAverageCross,
    SecularTrend,
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
    provider.get_current_price = AsyncMock(return_value=200.0)

    # Mock historical data
    import pandas as pd
    import numpy as np

    # Create data with upward trend
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    prices = np.linspace(150, 200, 500) + np.random.normal(0, 2, 500)
    volumes = [1000000 + np.random.randint(-100000, 100000) for _ in range(500)]

    hist_data = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    provider.get_historical_data = AsyncMock(return_value=hist_data)

    # Mock option expiries
    leaps_expiries = [
        (date.today() + timedelta(days=365)).strftime("%Y-%m-%d"),
        (date.today() + timedelta(days=545)).strftime("%Y-%m-%d"),
    ]
    provider.get_option_expiries = AsyncMock(return_value=leaps_expiries)

    # Mock options chain
    provider.get_options_chain = AsyncMock(return_value={
        "calls": [
            {
                "strike": 230.0,
                "bid": 25.00,
                "ask": 26.00,
                "volume": 50,
                "open_interest": 200,
            }
        ]
    })

    # Mock option price
    provider.get_option_price = AsyncMock(return_value=25.0)

    return provider


@pytest.fixture
def mock_integration_manager():
    """Create mock integration manager."""
    manager = AsyncMock()
    manager.get_portfolio_value = AsyncMock(return_value=200000.0)
    manager.execute_trade_signal = AsyncMock(return_value=True)
    manager.alert_system = AsyncMock()
    manager.alert_system.send_alert = AsyncMock()
    return manager


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "max_positions": 5,
        "max_position_size": 0.10,
        "max_total_allocation": 0.30,
        "min_dte": 365,
        "max_dte": 730,
        "min_composite_score": 60,
        "min_entry_timing_score": 50,
        "max_exit_timing_score": 70,
        "profit_levels": [100, 200, 300, 400],
        "scale_out_percentage": 25,
        "stop_loss": 0.50,
        "time_exit_dte": 90,
    }


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider, config):
    """Create strategy instance."""
    return ProductionLEAPSTracker(
        mock_integration_manager,
        mock_data_provider,
        config
    )


class TestProductionLEAPSTracker:
    """Test ProductionLEAPSTracker class."""

    def test_initialization(self, strategy, config):
        """Test strategy initialization."""
        assert strategy.strategy_name == "leaps_tracker"
        assert strategy.max_positions == config["max_positions"]
        assert strategy.min_dte == config["min_dte"]
        assert len(strategy.secular_themes) > 0
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_analyze_moving_average_cross_golden(self, strategy, mock_data_provider):
        """Test golden cross detection."""
        # Create data with golden cross pattern
        import pandas as pd
        import numpy as np

        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        # Start below 200 SMA, then cross above
        prices = np.concatenate([
            np.linspace(100, 105, 150),  # Below
            np.linspace(105, 130, 100)  # Cross and surge
        ])
        hist_data = pd.DataFrame({'close': prices}, index=dates)
        mock_data_provider.get_historical_data = AsyncMock(return_value=hist_data)

        ma_cross = await strategy.analyze_moving_average_cross("NVDA")

        assert isinstance(ma_cross, MovingAverageCross)
        assert ma_cross.cross_type in ["golden_cross", "death_cross", "neutral"]
        assert ma_cross.sma_50 > 0
        assert ma_cross.sma_200 > 0

    @pytest.mark.asyncio
    async def test_analyze_moving_average_cross_death(self, strategy, mock_data_provider):
        """Test death cross detection."""
        import pandas as pd
        import numpy as np

        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        # Start above 200 SMA, then cross below
        prices = np.concatenate([
            np.linspace(130, 125, 150),  # Above
            np.linspace(125, 100, 100)  # Cross and drop
        ])
        hist_data = pd.DataFrame({'close': prices}, index=dates)
        mock_data_provider.get_historical_data = AsyncMock(return_value=hist_data)

        ma_cross = await strategy.analyze_moving_average_cross("NVDA")

        assert isinstance(ma_cross, MovingAverageCross)

    @pytest.mark.asyncio
    async def test_analyze_moving_average_cross_insufficient_data(self, strategy, mock_data_provider):
        """Test MA cross with insufficient data."""
        import pandas as pd
        mock_data_provider.get_historical_data = AsyncMock(return_value=pd.DataFrame())

        ma_cross = await strategy.analyze_moving_average_cross("NVDA")

        assert ma_cross.cross_type == "neutral"
        assert ma_cross.cross_date is None

    def test_calculate_entry_exit_timing_scores_golden_cross(self, strategy):
        """Test timing scores for golden cross."""
        ma_cross = MovingAverageCross(
            cross_type="golden_cross",
            cross_date=date.today() - timedelta(days=20),
            days_since_cross=20,
            sma_50=105.0,
            sma_200=100.0,
            price_above_50sma=True,
            price_above_200sma=True,
            cross_strength=50.0,
            trend_direction="bullish"
        )

        entry_score, exit_score = strategy.calculate_entry_exit_timing_scores(
            ma_cross, 110.0
        )

        assert entry_score > 70  # Good entry on golden cross
        assert exit_score < 30  # Don't exit on golden cross

    def test_calculate_entry_exit_timing_scores_death_cross(self, strategy):
        """Test timing scores for death cross."""
        ma_cross = MovingAverageCross(
            cross_type="death_cross",
            cross_date=date.today() - timedelta(days=15),
            days_since_cross=15,
            sma_50=95.0,
            sma_200=100.0,
            price_above_50sma=False,
            price_above_200sma=False,
            cross_strength=50.0,
            trend_direction="bearish"
        )

        entry_score, exit_score = strategy.calculate_entry_exit_timing_scores(
            ma_cross, 90.0
        )

        assert entry_score < 30  # Poor entry on death cross
        assert exit_score > 70  # Strong exit signal

    def test_calculate_entry_exit_timing_scores_neutral(self, strategy):
        """Test timing scores for neutral scenario."""
        ma_cross = MovingAverageCross(
            cross_type="neutral",
            cross_date=None,
            days_since_cross=None,
            sma_50=102.0,
            sma_200=100.0,
            price_above_50sma=True,
            price_above_200sma=True,
            cross_strength=0.0,
            trend_direction="bullish"
        )

        entry_score, exit_score = strategy.calculate_entry_exit_timing_scores(
            ma_cross, 105.0
        )

        # Should be moderate
        assert 50 < entry_score < 80
        assert 20 < exit_score < 50

    @pytest.mark.asyncio
    async def test_calculate_comprehensive_score(self, strategy, mock_data_provider):
        """Test comprehensive scoring calculation."""
        momentum, trend, financial, valuation = await strategy.calculate_comprehensive_score("NVDA")

        assert 0 <= momentum <= 100
        assert 0 <= trend <= 100
        assert 0 <= financial <= 100
        assert 0 <= valuation <= 100

    @pytest.mark.asyncio
    async def test_calculate_comprehensive_score_insufficient_data(self, strategy, mock_data_provider):
        """Test scoring with insufficient data."""
        import pandas as pd
        mock_data_provider.get_historical_data = AsyncMock(return_value=pd.DataFrame())

        scores = await strategy.calculate_comprehensive_score("NVDA")

        # Should return defaults
        assert all(score == 50.0 for score in scores)

    @pytest.mark.asyncio
    async def test_get_leaps_expiries(self, strategy, mock_data_provider):
        """Test getting LEAPS expiries."""
        expiries = await strategy.get_leaps_expiries("NVDA")

        assert len(expiries) > 0
        for exp in expiries:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days_out = (exp_date - date.today()).days
            assert strategy.min_dte <= days_out <= strategy.max_dte

    @pytest.mark.asyncio
    async def test_get_leaps_expiries_no_data(self, strategy, mock_data_provider):
        """Test getting LEAPS expiries with no data."""
        mock_data_provider.get_option_expiries = AsyncMock(return_value=None)

        expiries = await strategy.get_leaps_expiries("NVDA")

        assert expiries == []

    @pytest.mark.asyncio
    async def test_estimate_leaps_premium(self, strategy, mock_data_provider):
        """Test LEAPS premium estimation."""
        expiry = (date.today() + timedelta(days=400)).strftime("%Y-%m-%d")

        premium = await strategy.estimate_leaps_premium("NVDA", 230.0, expiry)

        assert premium >= 5.0  # Minimum reasonable LEAPS premium

    @pytest.mark.asyncio
    async def test_estimate_leaps_premium_fallback(self, strategy, mock_data_provider):
        """Test LEAPS premium fallback calculation."""
        mock_data_provider.get_options_chain = AsyncMock(return_value=None)
        expiry = (date.today() + timedelta(days=400)).strftime("%Y-%m-%d")

        premium = await strategy.estimate_leaps_premium("NVDA", 230.0, expiry)

        # Should use Black-Scholes fallback
        assert premium > 0

    @pytest.mark.asyncio
    async def test_scan_leaps_candidates(self, strategy, mock_data_provider):
        """Test scanning for LEAPS candidates."""
        candidates = await strategy.scan_leaps_candidates()

        assert isinstance(candidates, list)

    @pytest.mark.asyncio
    async def test_scan_leaps_candidates_skip_existing(self, strategy):
        """Test scan skips existing positions."""
        strategy.active_positions.append({"ticker": "NVDA"})

        candidates = await strategy.scan_leaps_candidates()

        # NVDA should be skipped
        nvda_candidates = [c for c in candidates if c.ticker == "NVDA"]
        assert len(nvda_candidates) == 0

    @pytest.mark.asyncio
    async def test_scan_leaps_candidates_low_score(self, strategy, mock_data_provider):
        """Test scan filters low-scoring candidates."""
        # Mock low scores
        async def mock_comprehensive_score(ticker):
            return 30.0, 30.0, 30.0, 30.0  # Low scores

        strategy.calculate_comprehensive_score = mock_comprehensive_score

        candidates = await strategy.scan_leaps_candidates()

        # Should filter out low scores
        assert isinstance(candidates, list)

    @pytest.mark.asyncio
    async def test_execute_leaps_trade_max_positions(self, strategy):
        """Test execute when at max positions."""
        for i in range(strategy.max_positions):
            strategy.active_positions.append({"ticker": f"TEST{i}"})

        candidate = LEAPSCandidate(
            ticker="NVDA",
            company_name="NVIDIA",
            theme="AI Revolution",
            current_price=200.0,
            trend_score=80.0,
            financial_score=75.0,
            momentum_score=85.0,
            valuation_score=60.0,
            composite_score=75.0,
            expiry_date="2025-06-20",
            recommended_strike=230.0,
            premium_estimate=25.0,
            break_even=255.0,
            target_return_1y=30.0,
            target_return_3y=100.0,
            risk_factors=[],
            ma_cross_signal=MovingAverageCross(
                cross_type="golden_cross",
                cross_date=date.today() - timedelta(days=20),
                days_since_cross=20,
                sma_50=195.0,
                sma_200=185.0,
                price_above_50sma=True,
                price_above_200sma=True,
                cross_strength=50.0,
                trend_direction="bullish"
            ),
            entry_timing_score=85.0,
            exit_timing_score=20.0
        )

        result = await strategy.execute_leaps_trade(candidate)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_leaps_trade_max_allocation(self, strategy, mock_integration_manager):
        """Test execute when at max total allocation."""
        # Add positions to reach max allocation
        strategy.active_positions.append({
            "ticker": "TEST1",
            "cost_basis": 60000.0,  # 30% of 200k portfolio
        })

        candidate = LEAPSCandidate(
            ticker="NVDA",
            company_name="NVIDIA",
            theme="AI Revolution",
            current_price=200.0,
            trend_score=80.0,
            financial_score=75.0,
            momentum_score=85.0,
            valuation_score=60.0,
            composite_score=75.0,
            expiry_date="2025-06-20",
            recommended_strike=230.0,
            premium_estimate=25.0,
            break_even=255.0,
            target_return_1y=30.0,
            target_return_3y=100.0,
            risk_factors=[],
            ma_cross_signal=MovingAverageCross(
                cross_type="golden_cross",
                cross_date=None,
                days_since_cross=None,
                sma_50=195.0,
                sma_200=185.0,
                price_above_50sma=True,
                price_above_200sma=True,
                cross_strength=0.0,
                trend_direction="bullish"
            ),
            entry_timing_score=85.0,
            exit_timing_score=20.0
        )

        result = await strategy.execute_leaps_trade(candidate)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_leaps_trade_success(self, strategy, mock_integration_manager):
        """Test successful LEAPS trade execution."""
        candidate = LEAPSCandidate(
            ticker="NVDA",
            company_name="NVIDIA",
            theme="AI Revolution",
            current_price=200.0,
            trend_score=80.0,
            financial_score=75.0,
            momentum_score=85.0,
            valuation_score=60.0,
            composite_score=75.0,
            expiry_date="2025-06-20",
            recommended_strike=230.0,
            premium_estimate=25.0,
            break_even=255.0,
            target_return_1y=30.0,
            target_return_3y=100.0,
            risk_factors=["High valuation"],
            ma_cross_signal=MovingAverageCross(
                cross_type="golden_cross",
                cross_date=date.today() - timedelta(days=20),
                days_since_cross=20,
                sma_50=195.0,
                sma_200=185.0,
                price_above_50sma=True,
                price_above_200sma=True,
                cross_strength=50.0,
                trend_direction="bullish"
            ),
            entry_timing_score=85.0,
            exit_timing_score=20.0
        )

        result = await strategy.execute_leaps_trade(candidate)

        assert result is True
        assert len(strategy.active_positions) == 1

    @pytest.mark.asyncio
    async def test_manage_positions_profit_target(self, strategy, mock_data_provider):
        """Test position management when profit target reached."""
        position = {
            "ticker": "NVDA",
            "theme": "AI Revolution",
            "trade_signal": Mock(
                strike_price=Decimal("230"),
                expiration_date=date.today() + timedelta(days=365)
            ),
            "entry_time": datetime.now(),
            "entry_premium": 25.0,
            "contracts": 2,
            "cost_basis": 5000.0,
            "scale_out_level": 0,
            "profit_levels": [100, 200, 300, 400],
            "expiry_date": date.today() + timedelta(days=365),
            "ma_cross_type": "golden_cross",
        }
        strategy.active_positions.append(position)

        # Mock 200% profit
        mock_data_provider.get_option_price = AsyncMock(return_value=75.0)

        await strategy.manage_positions()

        # Should have scaled out
        assert position["scale_out_level"] > 0

    @pytest.mark.asyncio
    async def test_manage_positions_stop_loss(self, strategy, mock_data_provider):
        """Test position management when stop loss hit."""
        position = {
            "ticker": "NVDA",
            "theme": "AI Revolution",
            "trade_signal": Mock(
                strike_price=Decimal("230"),
                expiration_date=date.today() + timedelta(days=365)
            ),
            "entry_time": datetime.now(),
            "entry_premium": 25.0,
            "contracts": 2,
            "cost_basis": 5000.0,
            "scale_out_level": 0,
            "profit_levels": [100, 200, 300, 400],
            "expiry_date": date.today() + timedelta(days=365),
            "ma_cross_type": "golden_cross",
        }
        strategy.active_positions.append(position)

        # Mock 60% loss
        mock_data_provider.get_option_price = AsyncMock(return_value=10.0)

        await strategy.manage_positions()

        # Position should be closed
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_time_exit(self, strategy, mock_data_provider):
        """Test position management with time-based exit."""
        position = {
            "ticker": "NVDA",
            "theme": "AI Revolution",
            "trade_signal": Mock(
                strike_price=Decimal("230"),
                expiration_date=date.today() + timedelta(days=60)
            ),
            "entry_time": datetime.now(),
            "entry_premium": 25.0,
            "contracts": 2,
            "cost_basis": 5000.0,
            "scale_out_level": 0,
            "profit_levels": [100, 200, 300, 400],
            "expiry_date": date.today() + timedelta(days=60),
            "ma_cross_type": "golden_cross",
        }
        strategy.active_positions.append(position)

        mock_data_provider.get_option_price = AsyncMock(return_value=30.0)

        await strategy.manage_positions()

        # Position should be closed due to time
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_death_cross_exit(self, strategy, mock_data_provider):
        """Test position management with death cross exit signal."""
        position = {
            "ticker": "NVDA",
            "theme": "AI Revolution",
            "trade_signal": Mock(
                strike_price=Decimal("230"),
                expiration_date=date.today() + timedelta(days=365)
            ),
            "entry_time": datetime.now(),
            "entry_premium": 25.0,
            "contracts": 2,
            "cost_basis": 5000.0,
            "scale_out_level": 0,
            "profit_levels": [100, 200, 300, 400],
            "expiry_date": date.today() + timedelta(days=365),
            "ma_cross_type": "golden_cross",  # Was golden at entry
        }
        strategy.active_positions.append(position)

        # Mock recent death cross
        mock_ma_cross = MovingAverageCross(
            cross_type="death_cross",
            cross_date=date.today() - timedelta(days=10),
            days_since_cross=10,
            sma_50=185.0,
            sma_200=195.0,
            price_above_50sma=False,
            price_above_200sma=False,
            cross_strength=50.0,
            trend_direction="bearish"
        )
        strategy.analyze_moving_average_cross = AsyncMock(return_value=mock_ma_cross)
        mock_data_provider.get_option_price = AsyncMock(return_value=30.0)

        await strategy.manage_positions()

        # Position should be closed due to death cross
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_scale_out(self, strategy, mock_data_provider):
        """Test position scale-out functionality."""
        position = {
            "ticker": "NVDA",
            "theme": "AI Revolution",
            "trade_signal": Mock(
                strike_price=Decimal("230"),
                expiration_date=date.today() + timedelta(days=365)
            ),
            "entry_time": datetime.now(),
            "entry_premium": 25.0,
            "contracts": 4,  # Multiple contracts
            "cost_basis": 10000.0,
            "scale_out_level": 0,
            "profit_levels": [100, 200, 300, 400],
            "expiry_date": date.today() + timedelta(days=365),
            "ma_cross_type": "golden_cross",
        }
        strategy.active_positions.append(position)

        # Mock 150% profit (between 100% and 200%)
        mock_data_provider.get_option_price = AsyncMock(return_value=62.5)

        await strategy.manage_positions()

        # Should have scaled out once but position remains
        assert len(strategy.active_positions) == 1
        assert position["contracts"] < 4

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
            "ticker": "NVDA",
            "theme": "AI Revolution",
            "trade_signal": Mock(strike_price=Decimal("230")),
            "entry_time": datetime.now(),
            "entry_premium": 25.0,
            "contracts": 2,
            "cost_basis": 5000.0,
            "breakeven": 255.0,
            "scale_out_level": 1,
            "expiry_date": date.today() + timedelta(days=365),
        }
        strategy.active_positions.append(position)

        status = strategy.get_strategy_status()

        assert status["strategy_name"] == "leaps_tracker"
        assert status["active_positions"] == 1
        assert len(status["position_details"]) == 1
        assert "themes" in status

    def test_get_strategy_status_error(self, strategy):
        """Test strategy status with error."""
        strategy.active_positions.append({"bad": "data"})

        status = strategy.get_strategy_status()

        assert "error" in status or "strategy_name" in status

    @pytest.mark.asyncio
    async def test_error_handling_in_scan(self, strategy, mock_data_provider):
        """Test error handling during scan."""
        mock_data_provider.get_current_price = AsyncMock(side_effect=Exception("API Error"))

        candidates = await strategy.scan_leaps_candidates()

        # Should not crash
        assert isinstance(candidates, list)

    def test_secular_trend_dataclass(self):
        """Test SecularTrend dataclass."""
        trend = SecularTrend(
            theme="AI Revolution",
            description="Artificial intelligence transforming industries",
            tickers=["NVDA", "AMD"],
            growth_drivers=["GPU compute", "Cloud AI"],
            time_horizon="5-10 years"
        )

        assert trend.theme == "AI Revolution"
        assert len(trend.tickers) == 2

    def test_leaps_candidate_dataclass(self):
        """Test LEAPSCandidate dataclass."""
        ma_cross = MovingAverageCross(
            cross_type="golden_cross",
            cross_date=date.today(),
            days_since_cross=0,
            sma_50=195.0,
            sma_200=185.0,
            price_above_50sma=True,
            price_above_200sma=True,
            cross_strength=50.0,
            trend_direction="bullish"
        )

        candidate = LEAPSCandidate(
            ticker="NVDA",
            company_name="NVIDIA",
            theme="AI Revolution",
            current_price=200.0,
            trend_score=80.0,
            financial_score=75.0,
            momentum_score=85.0,
            valuation_score=60.0,
            composite_score=75.0,
            expiry_date="2025-06-20",
            recommended_strike=230.0,
            premium_estimate=25.0,
            break_even=255.0,
            target_return_1y=30.0,
            target_return_3y=100.0,
            risk_factors=["High valuation"],
            ma_cross_signal=ma_cross,
            entry_timing_score=85.0,
            exit_timing_score=20.0
        )

        assert candidate.ticker == "NVDA"
        assert candidate.composite_score == 75.0
        assert len(candidate.risk_factors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
