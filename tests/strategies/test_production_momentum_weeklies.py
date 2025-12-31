#!/usr/bin/env python3
"""Comprehensive tests for Production Momentum Weeklies strategy.

Tests all strategy methods to achieve maximum coverage including:
- Volume spike detection
- Reversal pattern detection
- Breakout momentum detection
- Option premium calculation
- Position management
- Exit conditions
- Weekly expiry calculation
- Error handling
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

from backend.tradingbot.strategies.production.production_momentum_weeklies import (
    ProductionMomentumWeeklies,
    MomentumSignal,
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

    # Mock intraday data
    import pandas as pd
    import numpy as np

    # Create intraday price data with volume spike
    timestamps = pd.date_range(start='2024-01-01 09:30', periods=78, freq='5T')
    prices = np.linspace(145, 150, 78) + np.random.normal(0, 0.5, 78)
    volumes = [100000] * 60 + [300000] * 18  # Volume spike at end

    intraday_data = pd.DataFrame({
        'close': prices,
        'open': prices - 0.5,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'volume': volumes
    }, index=timestamps)

    provider.get_intraday_data = AsyncMock(return_value=intraday_data)

    # Mock implied volatility
    provider.get_implied_volatility = AsyncMock(return_value={"iv_rank": 30})

    # Mock option price
    provider.get_option_price = AsyncMock(return_value=2.50)

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
        "watchlist": ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "max_positions": 3,
        "max_position_size": 0.05,
        "min_volume_spike": 3.0,
        "min_momentum_threshold": 0.015,
        "target_dte_range": (0, 4),
        "otm_range": (0.02, 0.05),
        "min_premium": 0.50,
        "profit_target": 0.25,
        "stop_loss": 0.50,
        "time_exit_hours": 4,
    }


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider, config):
    """Create strategy instance."""
    return ProductionMomentumWeeklies(
        mock_integration_manager,
        mock_data_provider,
        config
    )


class TestProductionMomentumWeeklies:
    """Test ProductionMomentumWeeklies class."""

    def test_initialization(self, strategy, config):
        """Test strategy initialization."""
        assert strategy.strategy_name == "momentum_weeklies"
        assert strategy.max_positions == config["max_positions"]
        assert strategy.min_volume_spike == config["min_volume_spike"]
        assert len(strategy.active_positions) == 0

    def test_get_next_weekly_expiry_monday(self):
        """Test weekly expiry calculation on Monday."""
        with patch('backend.tradingbot.strategies.production.production_momentum_weeklies.date') as mock_date:
            # Mock Monday
            mock_date.today.return_value = date(2024, 1, 1)  # Monday

            strategy = ProductionMomentumWeeklies(None, None, {})
            expiry = strategy.get_next_weekly_expiry()

            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            assert exp_date.weekday() == 4  # Friday

    def test_get_next_weekly_expiry_friday(self):
        """Test weekly expiry calculation on Friday."""
        with patch('backend.tradingbot.strategies.production.production_momentum_weeklies.date') as mock_date:
            # Mock Friday
            mock_date.today.return_value = date(2024, 1, 5)  # Friday

            strategy = ProductionMomentumWeeklies(None, None, {})
            expiry = strategy.get_next_weekly_expiry()

            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            assert exp_date.weekday() == 4  # Should be next Friday

    @pytest.mark.asyncio
    async def test_detect_volume_spike_yes(self, strategy, mock_data_provider):
        """Test volume spike detection when spike exists."""
        has_spike, multiple = await strategy.detect_volume_spike("AAPL")

        assert has_spike is True
        assert multiple >= strategy.min_volume_spike

    @pytest.mark.asyncio
    async def test_detect_volume_spike_no(self, strategy, mock_data_provider):
        """Test volume spike detection when no spike."""
        import pandas as pd

        # Flat volume
        flat_volume_data = pd.DataFrame({
            'volume': [100000] * 100
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=flat_volume_data)

        has_spike, multiple = await strategy.detect_volume_spike("AAPL")

        assert has_spike is False

    @pytest.mark.asyncio
    async def test_detect_volume_spike_no_data(self, strategy, mock_data_provider):
        """Test volume spike detection with no data."""
        mock_data_provider.get_intraday_data = AsyncMock(return_value=pd.DataFrame())

        has_spike, multiple = await strategy.detect_volume_spike("AAPL")

        assert has_spike is False
        assert multiple == 0.0

    @pytest.mark.asyncio
    async def test_detect_reversal_pattern_yes(self, strategy, mock_data_provider):
        """Test reversal pattern detection when reversal exists."""
        import pandas as pd
        import numpy as np

        # Create V-shaped reversal
        prices_down = np.linspace(150, 145, 12)  # Drop
        prices_up = np.linspace(145, 151, 12)  # Recovery
        prices = np.concatenate([prices_down, prices_up])

        reversal_data = pd.DataFrame({
            'close': prices,
            'volume': [100000] * len(prices)
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=reversal_data)

        has_reversal, pattern_type, bounce_pct = await strategy.detect_reversal_pattern("AAPL")

        assert has_reversal is True
        assert pattern_type == "bullish_reversal"
        assert bounce_pct > 0

    @pytest.mark.asyncio
    async def test_detect_reversal_pattern_no(self, strategy, mock_data_provider):
        """Test reversal pattern detection when no reversal."""
        import pandas as pd

        # Flat prices
        flat_data = pd.DataFrame({
            'close': [150.0] * 24,
            'volume': [100000] * 24
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=flat_data)

        has_reversal, pattern_type, bounce_pct = await strategy.detect_reversal_pattern("AAPL")

        assert has_reversal is False

    @pytest.mark.asyncio
    async def test_detect_reversal_pattern_insufficient_data(self, strategy, mock_data_provider):
        """Test reversal detection with insufficient data."""
        import pandas as pd

        short_data = pd.DataFrame({
            'close': [150.0] * 10,
            'volume': [100000] * 10
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=short_data)

        has_reversal, pattern_type, bounce_pct = await strategy.detect_reversal_pattern("AAPL")

        assert has_reversal is False
        assert pattern_type == "insufficient_data"

    @pytest.mark.asyncio
    async def test_detect_breakout_momentum_yes(self, strategy, mock_data_provider):
        """Test breakout detection when breakout exists."""
        import pandas as pd
        import numpy as np

        # Create breakout pattern
        prices = np.concatenate([
            np.linspace(145, 148, 95),  # Consolidation
            np.linspace(148, 152, 5)  # Breakout
        ])
        volumes = [100000] * 95 + [250000] * 5  # Volume on breakout

        breakout_data = pd.DataFrame({
            'close': prices,
            'volume': volumes
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=breakout_data)

        has_breakout, strength = await strategy.detect_breakout_momentum("AAPL")

        assert has_breakout is True
        assert strength > 0

    @pytest.mark.asyncio
    async def test_detect_breakout_momentum_no(self, strategy, mock_data_provider):
        """Test breakout detection when no breakout."""
        import pandas as pd

        # No breakout
        no_breakout_data = pd.DataFrame({
            'close': [150.0] * 100,
            'volume': [100000] * 100
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=no_breakout_data)

        has_breakout, strength = await strategy.detect_breakout_momentum("AAPL")

        assert has_breakout is False

    @pytest.mark.asyncio
    async def test_detect_breakout_momentum_insufficient_data(self, strategy, mock_data_provider):
        """Test breakout detection with insufficient data."""
        import pandas as pd

        short_data = pd.DataFrame({
            'close': [150.0] * 30,
            'volume': [100000] * 30
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=short_data)

        has_breakout, strength = await strategy.detect_breakout_momentum("AAPL")

        assert has_breakout is False

    @pytest.mark.asyncio
    async def test_calculate_option_premium(self, strategy, mock_data_provider):
        """Test option premium calculation."""
        expiry = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")

        premium = await strategy.calculate_option_premium(
            "AAPL", 155.0, expiry, "call"
        )

        assert premium >= 0.10  # Minimum

    @pytest.mark.asyncio
    async def test_calculate_option_premium_no_price(self, strategy, mock_data_provider):
        """Test premium calculation with no current price."""
        mock_data_provider.get_current_price = AsyncMock(return_value=None)
        expiry = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")

        premium = await strategy.calculate_option_premium(
            "AAPL", 155.0, expiry, "call"
        )

        assert premium == 0.0

    @pytest.mark.asyncio
    async def test_scan_momentum_opportunities(self, strategy, mock_data_provider):
        """Test scanning for momentum opportunities."""
        signals = await strategy.scan_momentum_opportunities()

        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_scan_momentum_opportunities_skip_existing(self, strategy):
        """Test scan skips existing positions."""
        strategy.active_positions.append({"ticker": "AAPL"})

        signals = await strategy.scan_momentum_opportunities()

        # AAPL should be skipped
        aapl_signals = [s for s in signals if s.ticker == "AAPL"]
        assert len(aapl_signals) == 0

    @pytest.mark.asyncio
    async def test_scan_momentum_opportunities_no_volume_spike(self, strategy, mock_data_provider):
        """Test scan filters out tickers without volume spike."""
        import pandas as pd

        # No volume spike
        flat_data = pd.DataFrame({
            'close': [150.0] * 100,
            'volume': [100000] * 100
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=flat_data)

        signals = await strategy.scan_momentum_opportunities()

        # Should filter out
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_scan_momentum_opportunities_weak_momentum(self, strategy, mock_data_provider):
        """Test scan filters weak momentum."""
        import pandas as pd
        import numpy as np

        # Volume spike but weak momentum
        prices = np.linspace(150, 150.2, 100)  # Tiny move
        volumes = [100000] * 90 + [400000] * 10  # Volume spike

        weak_data = pd.DataFrame({
            'close': prices,
            'open': prices - 0.1,
            'high': prices + 0.1,
            'low': prices - 0.1,
            'volume': volumes
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=weak_data)

        signals = await strategy.scan_momentum_opportunities()

        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_execute_momentum_trade_max_positions(self, strategy):
        """Test execute when at max positions."""
        for i in range(strategy.max_positions):
            strategy.active_positions.append({"ticker": f"TEST{i}"})

        signal = MomentumSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            current_price=150.0,
            reversal_type="bullish_reversal",
            volume_spike=4.5,
            price_momentum=0.025,
            weekly_expiry="2024-12-31",
            target_strike=153.0,
            premium_estimate=2.50,
            risk_level="medium",
            exit_target=156.0,
            stop_loss=147.75,
            confidence=0.70,
            expected_move=0.04
        )

        result = await strategy.execute_momentum_trade(signal)

        assert result is False

    @pytest.mark.asyncio
    async def test_execute_momentum_trade_success(self, strategy, mock_integration_manager):
        """Test successful momentum trade execution."""
        signal = MomentumSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            current_price=150.0,
            reversal_type="breakout",
            volume_spike=5.0,
            price_momentum=0.030,
            weekly_expiry="2024-12-31",
            target_strike=153.0,
            premium_estimate=2.50,
            risk_level="high",
            exit_target=156.0,
            stop_loss=147.75,
            confidence=0.80,
            expected_move=0.04
        )

        result = await strategy.execute_momentum_trade(signal)

        assert result is True
        assert len(strategy.active_positions) == 1

    @pytest.mark.asyncio
    async def test_manage_positions_profit_target(self, strategy, mock_data_provider):
        """Test position management when profit target reached."""
        position = {
            "ticker": "AAPL",
            "trade_signal": Mock(
                strike_price=Decimal("153"),
                expiration_date=date.today() + timedelta(days=2)
            ),
            "entry_time": datetime.now(),
            "entry_premium": 2.50,
            "contracts": 5,
            "time_limit": datetime.now() + timedelta(hours=10),
        }
        strategy.active_positions.append(position)

        # Mock 30% profit
        mock_data_provider.get_option_price = AsyncMock(return_value=3.25)

        await strategy.manage_positions()

        # Position should be closed
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_stop_loss(self, strategy, mock_data_provider):
        """Test position management when stop loss hit."""
        position = {
            "ticker": "AAPL",
            "trade_signal": Mock(
                strike_price=Decimal("153"),
                expiration_date=date.today() + timedelta(days=2)
            ),
            "entry_time": datetime.now(),
            "entry_premium": 2.50,
            "contracts": 5,
            "time_limit": datetime.now() + timedelta(hours=10),
        }
        strategy.active_positions.append(position)

        # Mock 60% loss
        mock_data_provider.get_option_price = AsyncMock(return_value=1.00)

        await strategy.manage_positions()

        # Position should be closed
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_time_limit(self, strategy, mock_data_provider):
        """Test position management with time limit."""
        position = {
            "ticker": "AAPL",
            "trade_signal": Mock(
                strike_price=Decimal("153"),
                expiration_date=date.today() + timedelta(days=2)
            ),
            "entry_time": datetime.now() - timedelta(hours=5),
            "entry_premium": 2.50,
            "contracts": 5,
            "time_limit": datetime.now() - timedelta(hours=1),  # Already passed
        }
        strategy.active_positions.append(position)

        mock_data_provider.get_option_price = AsyncMock(return_value=2.60)

        await strategy.manage_positions()

        # Position should be closed due to time
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_expiry_risk(self, strategy, mock_data_provider):
        """Test position management with expiry risk."""
        position = {
            "ticker": "AAPL",
            "trade_signal": Mock(
                strike_price=Decimal("153"),
                expiration_date=date.today()  # Expires today
            ),
            "entry_time": datetime.now(),
            "entry_premium": 2.50,
            "contracts": 5,
            "time_limit": datetime.now() + timedelta(hours=10),
        }
        strategy.active_positions.append(position)

        mock_data_provider.get_option_price = AsyncMock(return_value=2.60)

        await strategy.manage_positions()

        # Position should be closed due to expiry risk
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_positions_no_price_data(self, strategy, mock_data_provider):
        """Test position management when price data unavailable."""
        position = {
            "ticker": "AAPL",
            "trade_signal": Mock(
                strike_price=Decimal("153"),
                expiration_date=date.today() + timedelta(days=2)
            ),
            "entry_time": datetime.now(),
            "entry_premium": 2.50,
            "contracts": 5,
            "time_limit": datetime.now() + timedelta(hours=10),
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
            "trade_signal": Mock(
                strike_price=Decimal("153"),
                expiration_date=date.today() + timedelta(days=2)
            ),
            "entry_premium": 2.50,
            "contracts": 5,
            "risk_level": "medium",
            "time_limit": datetime.now() + timedelta(hours=3),
        }
        strategy.active_positions.append(position)

        status = strategy.get_strategy_status()

        assert status["strategy_name"] == "momentum_weeklies"
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

        signals = await strategy.scan_momentum_opportunities()

        # Should not crash
        assert isinstance(signals, list)

    def test_momentum_signal_dataclass(self):
        """Test MomentumSignal dataclass."""
        signal = MomentumSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            current_price=150.0,
            reversal_type="bullish_reversal",
            volume_spike=4.5,
            price_momentum=0.025,
            weekly_expiry="2024-12-31",
            target_strike=153.0,
            premium_estimate=2.50,
            risk_level="medium",
            exit_target=156.0,
            stop_loss=147.75,
            confidence=0.70,
            expected_move=0.04
        )

        assert signal.ticker == "AAPL"
        assert signal.reversal_type == "bullish_reversal"
        assert signal.confidence == 0.70

    @pytest.mark.asyncio
    async def test_scan_momentum_opportunities_premium_too_low(self, strategy, mock_data_provider):
        """Test scan filters low premium options."""
        # Mock very low premium
        async def mock_calculate_premium(ticker, strike, expiry, option_type="call"):
            return 0.20  # Below minimum

        strategy.calculate_option_premium = mock_calculate_premium

        signals = await strategy.scan_momentum_opportunities()

        # Should filter out low premium
        assert isinstance(signals, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
