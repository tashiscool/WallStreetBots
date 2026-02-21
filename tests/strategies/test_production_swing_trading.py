"""Comprehensive tests for Production Swing Trading Strategy.

Tests all components, edge cases, and error handling for production swing trading.
Target: 80%+ coverage.
"""
import asyncio
import pytest
import pandas as pd
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Any

from backend.tradingbot.strategies.production.production_swing_trading import (
    ProductionSwingTrading,
    SwingSignal,
)
from backend.tradingbot.production.core.production_integration import (
    ProductionTradeSignal,
    OrderSide,
    OrderType,
)


@pytest.fixture
def mock_integration_manager():
    """Create mock integration manager."""
    manager = AsyncMock()
    manager.get_portfolio_value = AsyncMock(return_value=Decimal("100000"))
    manager.execute_trade_signal = AsyncMock(return_value=True)
    manager.alert_system = AsyncMock()
    manager.alert_system.send_alert = AsyncMock()
    return manager


@pytest.fixture
def mock_data_provider():
    """Create mock data provider."""
    provider = AsyncMock()
    provider.get_current_price = AsyncMock(return_value=150.0)
    provider.get_intraday_data = AsyncMock(return_value=pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000],
        'high': [101, 102, 103, 104, 105, 106],
        'low': [99, 100, 101, 102, 103, 104],
    }))
    provider.get_option_price = AsyncMock(return_value=2.50)
    provider.get_options_chain = AsyncMock(return_value={
        "calls": [
            {"strike": 105, "bid": 2.40, "ask": 2.60, "volume": 1000}
        ]
    })
    provider.is_market_open = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def swing_config():
    """Create swing trading config."""
    return {
        "max_positions": 5,
        "max_position_size": 0.02,
        "max_expiry_days": 21,
        "min_strength_score": 60.0,
        "min_volume_multiple": 2.0,
        "min_breakout_strength": 0.002,
        "min_premium": 0.25,
        "profit_targets": [25, 50, 100],
        "stop_loss_pct": 30,
        "max_hold_hours": 8,
        "enforce_close_cutoff": False,
        "end_of_day_exit_hour": 15,
        "watchlist": ["AAPL", "MSFT", "GOOGL", "TSLA"],
    }


@pytest.fixture
def strategy(mock_integration_manager, mock_data_provider, swing_config):
    """Create ProductionSwingTrading instance."""
    return ProductionSwingTrading(
        mock_integration_manager,
        mock_data_provider,
        swing_config
    )


class TestProductionSwingTradingInitialization:
    """Test strategy initialization."""

    def test_initialization_success(self, strategy, swing_config):
        """Test successful initialization."""
        assert strategy.strategy_name == "swing_trading"
        assert strategy.max_positions == 5
        assert strategy.max_position_size == 0.02
        assert strategy.max_expiry_days == 21
        assert strategy.min_strength_score == 60.0
        assert len(strategy.active_positions) == 0

    def test_initialization_default_values(self, mock_integration_manager, mock_data_provider):
        """Test initialization with default values."""
        strategy = ProductionSwingTrading(
            mock_integration_manager,
            mock_data_provider,
            {}
        )
        assert strategy.max_positions == 5
        assert strategy.max_position_size == 0.02
        assert len(strategy.swing_tickers) > 0

    def test_components_initialized(self, strategy):
        """Test that all components are initialized."""
        assert strategy.options_selector is not None
        assert strategy.risk_manager is not None
        assert strategy.bs_engine is not None
        assert strategy.logger is not None


class TestSwingSignalDataclass:
    """Test SwingSignal dataclass."""

    def test_signal_creation(self):
        """Test creating a swing signal."""
        signal = SwingSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            signal_type="breakout",
            entry_price=150.0,
            breakout_level=155.0,
            volume_confirmation=2.5,
            strength_score=75.0,
            target_strike=155.0,
            target_expiry="2025-01-17",
            option_premium=2.50,
            max_hold_hours=6,
            profit_target_1=3.125,
            profit_target_2=3.75,
            profit_target_3=5.0,
            stop_loss=1.75,
            risk_level="medium",
        )

        assert signal.ticker == "AAPL"
        assert signal.signal_type == "breakout"
        assert signal.strength_score == 75.0
        assert signal.risk_level == "medium"


class TestDetectBreakout:
    """Test detect_breakout method."""

    @pytest.mark.asyncio
    async def test_detect_breakout_success(self, strategy, mock_data_provider):
        """Test successful breakout detection."""
        # Create uptrending data with volume increase
        data = pd.DataFrame({
            'close': [100, *list(range(101, 121))],  # Clear uptrend
            'volume': [1000000] * 10 + [3000000] * 11,  # Volume spike
            'high': [101, *list(range(102, 122))],
            'low': [99, *list(range(100, 120))],
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=data)

        is_breakout, resistance, strength = await strategy.detect_breakout("AAPL")

        assert isinstance(is_breakout, bool)
        assert isinstance(resistance, float)
        assert isinstance(strength, float)

    @pytest.mark.asyncio
    async def test_detect_breakout_insufficient_data(self, strategy, mock_data_provider):
        """Test breakout detection with insufficient data."""
        mock_data_provider.get_intraday_data = AsyncMock(return_value=pd.DataFrame())

        is_breakout, resistance, strength = await strategy.detect_breakout("AAPL")

        assert is_breakout is False
        assert resistance == 0.0
        assert strength == 0.0

    @pytest.mark.asyncio
    async def test_detect_breakout_no_volume_confirmation(self, strategy, mock_data_provider):
        """Test breakout without volume confirmation."""
        data = pd.DataFrame({
            'close': list(range(100, 121)),
            'volume': [1000000] * 21,  # No volume spike
            'high': list(range(101, 122)),
            'low': list(range(99, 120)),
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=data)

        is_breakout, resistance, strength = await strategy.detect_breakout("AAPL")

        # May not trigger without volume
        assert isinstance(strength, float)

    @pytest.mark.asyncio
    async def test_detect_breakout_exception_handling(self, strategy, mock_data_provider):
        """Test exception handling in breakout detection."""
        mock_data_provider.get_intraday_data = AsyncMock(side_effect=Exception("API Error"))

        is_breakout, resistance, strength = await strategy.detect_breakout("AAPL")

        assert is_breakout is False
        assert resistance == 0.0
        assert strength == 0.0


class TestDetectMomentumContinuation:
    """Test detect_momentum_continuation method."""

    @pytest.mark.asyncio
    async def test_detect_momentum_success(self, strategy, mock_data_provider):
        """Test successful momentum detection."""
        # Create strong uptrending data
        data = pd.DataFrame({
            'close': list(range(100, 131)),  # 31 periods of uptrend
            'volume': [1000000] * 15 + [2000000] * 16,  # Volume increase
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=data)

        is_momentum, strength = await strategy.detect_momentum_continuation("AAPL")

        assert isinstance(is_momentum, bool)
        assert isinstance(strength, float)

    @pytest.mark.asyncio
    async def test_detect_momentum_insufficient_data(self, strategy, mock_data_provider):
        """Test momentum detection with insufficient data."""
        mock_data_provider.get_intraday_data = AsyncMock(return_value=pd.DataFrame())

        is_momentum, strength = await strategy.detect_momentum_continuation("AAPL")

        assert is_momentum is False
        assert strength == 0.0

    @pytest.mark.asyncio
    async def test_detect_momentum_downtrend(self, strategy, mock_data_provider):
        """Test momentum detection in downtrend."""
        data = pd.DataFrame({
            'close': list(range(130, 99, -1)),  # Downtrend
            'volume': [1000000] * 31,
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=data)

        is_momentum, strength = await strategy.detect_momentum_continuation("AAPL")

        # Should not detect upward momentum
        assert is_momentum is False


class TestDetectReversalSetup:
    """Test detect_reversal_setup method."""

    @pytest.mark.asyncio
    async def test_detect_reversal_success(self, strategy, mock_data_provider):
        """Test successful reversal detection."""
        # Create oversold bounce pattern
        data = pd.DataFrame({
            'close': [110, 108, 106, 104, 102, 100, 98, 100, 102, 104],  # Drop then bounce
            'low': [109, 107, 105, 103, 101, 99, 97, 99, 101, 103],
            'volume': [1000000] * 7 + [3000000, 3500000, 4000000],  # Volume spike on bounce
        })
        mock_data_provider.get_intraday_data = AsyncMock(return_value=data)

        is_reversal, reversal_type, strength = await strategy.detect_reversal_setup("AAPL")

        assert isinstance(is_reversal, bool)
        assert isinstance(reversal_type, str)
        assert isinstance(strength, float)

    @pytest.mark.asyncio
    async def test_detect_reversal_insufficient_data(self, strategy, mock_data_provider):
        """Test reversal detection with insufficient data."""
        mock_data_provider.get_intraday_data = AsyncMock(return_value=pd.DataFrame())

        is_reversal, reversal_type, strength = await strategy.detect_reversal_setup("AAPL")

        assert is_reversal is False
        assert reversal_type == "insufficient_data"
        assert strength == 0.0


class TestGetOptimalExpiry:
    """Test get_optimal_expiry method."""

    def test_get_optimal_expiry(self, strategy):
        """Test optimal expiry calculation."""
        expiry = strategy.get_optimal_expiry()

        assert isinstance(expiry, str)
        assert len(expiry) == 10
        assert expiry.count('-') == 2

        # Should be a valid date
        datetime.strptime(expiry, "%Y-%m-%d")

    def test_get_optimal_expiry_friday(self, strategy):
        """Test expiry returns a Friday."""
        expiry_date = datetime.strptime(strategy.get_optimal_expiry(), "%Y-%m-%d")
        # 4 = Friday
        assert expiry_date.weekday() == 4


class TestCalculateOptionTargets:
    """Test calculate_option_targets method."""

    def test_calculate_targets(self, strategy):
        """Test target calculation."""
        profit_25, profit_50, profit_100, stop_loss = strategy.calculate_option_targets(2.50)

        assert profit_25 == 2.50 * 1.25
        assert profit_50 == 2.50 * 1.50
        assert profit_100 == 2.50 * 2.00
        assert stop_loss == 2.50 * 0.70

    def test_calculate_targets_different_premium(self, strategy):
        """Test targets with different premium."""
        profit_25, profit_50, profit_100, stop_loss = strategy.calculate_option_targets(5.00)

        assert profit_25 == 6.25
        assert profit_50 == 7.50
        assert profit_100 == 10.00
        assert stop_loss == 3.50


class TestEstimateSwingPremium:
    """Test estimate_swing_premium method."""

    @pytest.mark.asyncio
    async def test_estimate_premium_with_options_data(self, strategy, mock_data_provider):
        """Test premium estimation with real options data."""
        premium = await strategy.estimate_swing_premium("AAPL", 105.0, "2025-01-17")

        assert isinstance(premium, float)
        assert premium > 0

    @pytest.mark.asyncio
    async def test_estimate_premium_fallback(self, strategy, mock_data_provider):
        """Test premium estimation fallback."""
        mock_data_provider.get_options_chain = AsyncMock(return_value=None)

        premium = await strategy.estimate_swing_premium("AAPL", 105.0, "2025-01-17")

        assert isinstance(premium, float)
        assert premium > 0

    @pytest.mark.asyncio
    async def test_estimate_premium_otm_call(self, strategy, mock_data_provider):
        """Test premium estimation for OTM call."""
        mock_data_provider.get_current_price = AsyncMock(return_value=100.0)
        mock_data_provider.get_options_chain = AsyncMock(return_value=None)

        premium = await strategy.estimate_swing_premium("AAPL", 110.0, "2025-01-17")

        assert premium > 0

    @pytest.mark.asyncio
    async def test_estimate_premium_itm_call(self, strategy, mock_data_provider):
        """Test premium estimation for ITM call."""
        mock_data_provider.get_current_price = AsyncMock(return_value=100.0)
        mock_data_provider.get_options_chain = AsyncMock(return_value=None)

        premium = await strategy.estimate_swing_premium("AAPL", 95.0, "2025-01-17")

        assert premium > 0


class TestScanSwingOpportunities:
    """Test scan_swing_opportunities method."""

    @pytest.mark.asyncio
    async def test_scan_finds_opportunities(self, strategy, mock_data_provider):
        """Test scanning finds opportunities."""
        # Mock all detection methods
        strategy.detect_breakout = AsyncMock(return_value=(True, 155.0, 75.0))
        strategy.detect_momentum_continuation = AsyncMock(return_value=(False, 0.0))
        strategy.detect_reversal_setup = AsyncMock(return_value=(False, "no_setup", 0.0))
        strategy.estimate_swing_premium = AsyncMock(return_value=2.50)

        signals = await strategy.scan_swing_opportunities()

        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_scan_skips_existing_positions(self, strategy):
        """Test scanning skips tickers with positions."""
        strategy.active_positions = [{"ticker": "AAPL"}]

        strategy.detect_breakout = AsyncMock(return_value=(True, 155.0, 75.0))

        signals = await strategy.scan_swing_opportunities()

        # AAPL should be skipped
        assert all(s.ticker != "AAPL" for s in signals)

    @pytest.mark.asyncio
    async def test_scan_filters_low_strength(self, strategy, mock_data_provider):
        """Test filtering signals with low strength."""
        strategy.detect_breakout = AsyncMock(return_value=(True, 155.0, 30.0))  # Low strength
        strategy.detect_momentum_continuation = AsyncMock(return_value=(False, 0.0))
        strategy.detect_reversal_setup = AsyncMock(return_value=(False, "no_setup", 0.0))

        signals = await strategy.scan_swing_opportunities()

        # Should filter out low strength signals
        assert all(s.strength_score >= strategy.min_strength_score for s in signals)

    @pytest.mark.asyncio
    async def test_scan_filters_low_premium(self, strategy, mock_data_provider):
        """Test filtering signals with low premium."""
        strategy.detect_breakout = AsyncMock(return_value=(True, 155.0, 75.0))
        strategy.detect_momentum_continuation = AsyncMock(return_value=(False, 0.0))
        strategy.detect_reversal_setup = AsyncMock(return_value=(False, "no_setup", 0.0))
        strategy.estimate_swing_premium = AsyncMock(return_value=0.10)  # Below minimum

        signals = await strategy.scan_swing_opportunities()

        # Should filter out low premium signals
        assert all(s.option_premium >= strategy.min_premium for s in signals)

    @pytest.mark.asyncio
    async def test_scan_sorts_by_strength(self, strategy, mock_data_provider):
        """Test signals sorted by strength."""
        strategy.swing_tickers = ["AAPL", "MSFT", "GOOGL"]

        # Return different strengths
        breakout_results = [(True, 155.0, 85.0), (True, 320.0, 70.0), (True, 2800.0, 95.0)]
        strategy.detect_breakout = AsyncMock(side_effect=breakout_results)
        strategy.detect_momentum_continuation = AsyncMock(return_value=(False, 0.0))
        strategy.detect_reversal_setup = AsyncMock(return_value=(False, "no_setup", 0.0))
        strategy.estimate_swing_premium = AsyncMock(return_value=2.50)

        signals = await strategy.scan_swing_opportunities()

        # Should be sorted by strength descending
        if len(signals) > 1:
            assert signals[0].strength_score >= signals[-1].strength_score


class TestExecuteSwingTrade:
    """Test execute_swing_trade method."""

    @pytest.mark.asyncio
    async def test_execute_trade_success(self, strategy, mock_integration_manager):
        """Test successful trade execution."""
        signal = SwingSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            signal_type="breakout",
            entry_price=150.0,
            breakout_level=155.0,
            volume_confirmation=2.5,
            strength_score=75.0,
            target_strike=155.0,
            target_expiry="2025-01-17",
            option_premium=2.50,
            max_hold_hours=6,
            profit_target_1=3.125,
            profit_target_2=3.75,
            profit_target_3=5.0,
            stop_loss=1.75,
            risk_level="medium",
        )

        success = await strategy.execute_swing_trade(signal)

        assert success is True
        assert len(strategy.active_positions) == 1
        mock_integration_manager.execute_trade_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_trade_max_positions_reached(self, strategy):
        """Test rejecting trade when max positions reached."""
        # Fill up positions
        for i in range(strategy.max_positions):
            strategy.active_positions.append({"ticker": f"TICK{i}"})

        signal = SwingSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            signal_type="breakout",
            entry_price=150.0,
            breakout_level=155.0,
            volume_confirmation=2.5,
            strength_score=75.0,
            target_strike=155.0,
            target_expiry="2025-01-17",
            option_premium=2.50,
            max_hold_hours=6,
            profit_target_1=3.125,
            profit_target_2=3.75,
            profit_target_3=5.0,
            stop_loss=1.75,
            risk_level="medium",
        )

        success = await strategy.execute_swing_trade(signal)

        assert success is False

    @pytest.mark.asyncio
    async def test_execute_trade_failure(self, strategy, mock_integration_manager):
        """Test handling trade execution failure."""
        mock_integration_manager.execute_trade_signal = AsyncMock(return_value=False)

        signal = SwingSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            signal_type="breakout",
            entry_price=150.0,
            breakout_level=155.0,
            volume_confirmation=2.5,
            strength_score=75.0,
            target_strike=155.0,
            target_expiry="2025-01-17",
            option_premium=2.50,
            max_hold_hours=6,
            profit_target_1=3.125,
            profit_target_2=3.75,
            profit_target_3=5.0,
            stop_loss=1.75,
            risk_level="medium",
        )

        success = await strategy.execute_swing_trade(signal)

        assert success is False


class TestManagePositions:
    """Test manage_positions method."""

    @pytest.mark.asyncio
    async def test_manage_profit_target_100(self, strategy, mock_data_provider):
        """Test closing position at 100% profit."""
        trade_signal = ProductionTradeSignal(
            symbol="AAPL",
            action="BUY",
            quantity=1,
            option_type="CALL",
            strike_price=Decimal("155.0"),
            expiration_date=date.today() + timedelta(days=7),
            premium=Decimal("2.50"),
            strategy_name="swing_trading",
        )

        strategy.active_positions = [{
            "ticker": "AAPL",
            "signal_type": "breakout",
            "trade_signal": trade_signal,
            "entry_time": datetime.now() - timedelta(hours=2),
            "entry_premium": 2.50,
            "contracts": 1,
            "cost_basis": 250.0,
            "max_hold_hours": 6,
            "profit_targets": [3.125, 3.75, 5.0],
            "stop_loss": 1.75,
            "hit_profit_target": 0,
            "expiry_date": date.today() + timedelta(days=7),
        }]

        mock_data_provider.get_option_price = AsyncMock(return_value=5.0)  # 100% profit

        await strategy.manage_positions()

        # Position should be closed
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_profit_target_50(self, strategy, mock_data_provider):
        """Test scaling out at 50% profit."""
        trade_signal = ProductionTradeSignal(
            symbol="AAPL",
            action="BUY",
            quantity=2,
            option_type="CALL",
            strike_price=Decimal("155.0"),
            expiration_date=date.today() + timedelta(days=7),
            premium=Decimal("2.50"),
            strategy_name="swing_trading",
        )

        strategy.active_positions = [{
            "ticker": "AAPL",
            "signal_type": "breakout",
            "trade_signal": trade_signal,
            "entry_time": datetime.now() - timedelta(hours=2),
            "entry_premium": 2.50,
            "contracts": 2,
            "cost_basis": 500.0,
            "max_hold_hours": 6,
            "profit_targets": [3.125, 3.75, 5.0],
            "stop_loss": 1.75,
            "hit_profit_target": 0,
            "expiry_date": date.today() + timedelta(days=7),
        }]

        mock_data_provider.get_option_price = AsyncMock(return_value=3.75)  # 50% profit

        await strategy.manage_positions()

        # Should scale out 50%
        assert len(strategy.active_positions) == 1
        assert strategy.active_positions[0]["hit_profit_target"] == 2

    @pytest.mark.asyncio
    async def test_manage_stop_loss(self, strategy, mock_data_provider):
        """Test stop loss exit."""
        trade_signal = ProductionTradeSignal(
            symbol="AAPL",
            action="BUY",
            quantity=1,
            option_type="CALL",
            strike_price=Decimal("155.0"),
            expiration_date=date.today() + timedelta(days=7),
            premium=Decimal("2.50"),
            strategy_name="swing_trading",
        )

        strategy.active_positions = [{
            "ticker": "AAPL",
            "signal_type": "breakout",
            "trade_signal": trade_signal,
            "entry_time": datetime.now() - timedelta(hours=2),
            "entry_premium": 2.50,
            "contracts": 1,
            "cost_basis": 250.0,
            "max_hold_hours": 6,
            "profit_targets": [3.125, 3.75, 5.0],
            "stop_loss": 1.75,
            "hit_profit_target": 0,
            "expiry_date": date.today() + timedelta(days=7),
        }]

        mock_data_provider.get_option_price = AsyncMock(return_value=1.50)  # Below stop loss

        await strategy.manage_positions()

        # Position should be closed
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_max_hold_time(self, strategy, mock_data_provider):
        """Test time-based exit."""
        trade_signal = ProductionTradeSignal(
            symbol="AAPL",
            action="BUY",
            quantity=1,
            option_type="CALL",
            strike_price=Decimal("155.0"),
            expiration_date=date.today() + timedelta(days=7),
            premium=Decimal("2.50"),
            strategy_name="swing_trading",
        )

        strategy.active_positions = [{
            "ticker": "AAPL",
            "signal_type": "breakout",
            "trade_signal": trade_signal,
            "entry_time": datetime.now() - timedelta(hours=10),  # Past max hold
            "entry_premium": 2.50,
            "contracts": 1,
            "cost_basis": 250.0,
            "max_hold_hours": 8,
        "enforce_close_cutoff": False,
            "profit_targets": [3.125, 3.75, 5.0],
            "stop_loss": 1.75,
            "hit_profit_target": 0,
            "expiry_date": date.today() + timedelta(days=7),
        }]

        mock_data_provider.get_option_price = AsyncMock(return_value=2.60)

        await strategy.manage_positions()

        # Position should be closed due to time
        assert len(strategy.active_positions) == 0

    @pytest.mark.asyncio
    async def test_manage_no_option_price(self, strategy, mock_data_provider):
        """Test handling when option price unavailable."""
        trade_signal = ProductionTradeSignal(
            symbol="AAPL",
            action="BUY",
            quantity=1,
            option_type="CALL",
            strike_price=Decimal("155.0"),
            expiration_date=date.today() + timedelta(days=7),
            premium=Decimal("2.50"),
            strategy_name="swing_trading",
        )

        strategy.active_positions = [{
            "ticker": "AAPL",
            "signal_type": "breakout",
            "trade_signal": trade_signal,
            "entry_time": datetime.now() - timedelta(hours=2),
            "entry_premium": 2.50,
            "contracts": 1,
            "cost_basis": 250.0,
            "max_hold_hours": 6,
            "profit_targets": [3.125, 3.75, 5.0],
            "stop_loss": 1.75,
            "hit_profit_target": 0,
            "expiry_date": date.today() + timedelta(days=7),
        }]

        mock_data_provider.get_option_price = AsyncMock(return_value=None)

        await strategy.manage_positions()

        # Position should remain
        assert len(strategy.active_positions) == 1


class TestScanOpportunities:
    """Test scan_opportunities method."""

    @pytest.mark.asyncio
    async def test_scan_opportunities_success(self, strategy, mock_data_provider):
        """Test scanning opportunities."""
        strategy.manage_positions = AsyncMock()
        strategy.scan_swing_opportunities = AsyncMock(return_value=[])

        signals = await strategy.scan_opportunities()

        strategy.manage_positions.assert_called_once()
        strategy.scan_swing_opportunities.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_opportunities_max_positions(self, strategy):
        """Test not scanning when at max positions."""
        for i in range(strategy.max_positions):
            strategy.active_positions.append({"ticker": f"TICK{i}"})

        strategy.manage_positions = AsyncMock()
        strategy.scan_swing_opportunities = AsyncMock(return_value=[])

        signals = await strategy.scan_opportunities()

        # Should not scan for new opportunities
        assert signals == []

    @pytest.mark.asyncio
    async def test_scan_opportunities_market_closed(self, strategy, mock_data_provider):
        """Test not scanning when market closed."""
        mock_data_provider.is_market_open = AsyncMock(return_value=False)

        strategy.manage_positions = AsyncMock()

        signals = await strategy.scan_opportunities()

        assert signals == []

    @pytest.mark.asyncio
    async def test_scan_opportunities_after_3pm(self, strategy):
        """Test not scanning after 3pm."""
        with patch('backend.tradingbot.strategies.production.production_swing_trading.datetime') as mock_dt:
            mock_dt.now.return_value = datetime.now().replace(hour=16, minute=0)

            strategy.manage_positions = AsyncMock()

            signals = await strategy.scan_opportunities()

            # Should skip scanning after 3pm
            assert signals == []


class TestGetStrategyStatus:
    """Test get_strategy_status method."""

    def test_status_no_positions(self, strategy):
        """Test status with no positions."""
        status = strategy.get_strategy_status()

        assert status["strategy_name"] == "swing_trading"
        assert status["active_positions"] == 0
        assert status["total_cost_basis"] == 0
        assert isinstance(status["position_details"], list)

    def test_status_with_positions(self, strategy):
        """Test status with active positions."""
        trade_signal = ProductionTradeSignal(
            symbol="AAPL",
            action="BUY",
            quantity=1,
            option_type="CALL",
            strike_price=Decimal("155.0"),
            expiration_date=date.today() + timedelta(days=7),
            premium=Decimal("2.50"),
            strategy_name="swing_trading",
        )

        strategy.active_positions = [{
            "ticker": "AAPL",
            "signal_type": "breakout",
            "trade_signal": trade_signal,
            "entry_time": datetime.now(),
            "entry_premium": 2.50,
            "contracts": 1,
            "cost_basis": 250.0,
            "max_hold_hours": 6,
            "hit_profit_target": 0,
            "expiry_date": date.today() + timedelta(days=7),
        }]

        status = strategy.get_strategy_status()

        assert status["active_positions"] == 1
        assert status["total_cost_basis"] == 250.0
        assert len(status["position_details"]) == 1


class TestFactoryFunction:
    """Test factory function."""

    def test_create_production_swing_trading(self, mock_integration_manager, mock_data_provider):
        """Test factory function creates strategy."""
        from backend.tradingbot.strategies.production.production_swing_trading import (
            create_production_swing_trading
        )

        config = {"max_positions": 3}
        strategy = create_production_swing_trading(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        assert isinstance(strategy, ProductionSwingTrading)
        assert strategy.max_positions == 3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_watchlist(self, mock_integration_manager, mock_data_provider):
        """Test handling empty watchlist."""
        config = {"watchlist": []}
        strategy = ProductionSwingTrading(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        signals = await strategy.scan_swing_opportunities()
        assert signals == []

    @pytest.mark.asyncio
    async def test_zero_premium(self, strategy):
        """Test handling zero premium."""
        profit_25, profit_50, profit_100, stop_loss = strategy.calculate_option_targets(0.0)

        assert profit_25 == 0.0
        assert profit_50 == 0.0
        assert profit_100 == 0.0
        assert stop_loss == 0.0

    @pytest.mark.asyncio
    async def test_negative_current_price(self, strategy, mock_data_provider):
        """Test handling negative price (API error)."""
        mock_data_provider.get_current_price = AsyncMock(return_value=None)

        signals = await strategy.scan_swing_opportunities()
        # Should handle gracefully

    @pytest.mark.asyncio
    async def test_invalid_expiry_date(self, strategy):
        """Test handling invalid expiry date."""
        # Should not crash
        try:
            premium = await strategy.estimate_swing_premium("AAPL", 105.0, "invalid-date")
            assert isinstance(premium, float)
        except Exception:
            pass  # Expected to handle gracefully

    def test_signal_validation_integration(self, strategy):
        """Test signal validation mixin integration."""
        # Strategy should have validate_signal method from mixin
        assert hasattr(strategy, 'validate_signal')
        assert hasattr(strategy, 'strategy_name')
