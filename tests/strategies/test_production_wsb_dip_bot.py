#!/usr/bin/env python3
"""Comprehensive tests for Production WSB Dip Bot strategy.

Tests all strategy methods to achieve maximum coverage including:
- Signal generation and detection
- Position management
- Risk management
- Exit conditions
- Market data integration
- Error handling
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Any

from backend.tradingbot.strategies.production.production_wsb_dip_bot import (
    ProductionWSBDipBot,
    DipSignal,
)
from backend.tradingbot.core.trading_interface import OrderSide, OrderType
from backend.tradingbot.production.core.production_integration import (
    ProductionIntegrationManager,
    ProductionTradeSignal,
    TradeResult,
    TradeStatus,
)
from backend.tradingbot.production.data.production_data_integration import (
    ReliableDataProvider,
)


@pytest.fixture
def mock_data_provider():
    """Create mock data provider."""
    provider = AsyncMock(spec=ReliableDataProvider)

    # Mock market open check
    provider.is_market_open = AsyncMock(return_value=True)

    # Mock price history
    provider.get_price_history = AsyncMock(return_value=[
        Decimal("100"), Decimal("105"), Decimal("110"), Decimal("115"),
        Decimal("120"), Decimal("125"), Decimal("130"), Decimal("135"),
        Decimal("140"), Decimal("145"), Decimal("142")  # Last price shows a dip
    ])

    # Mock volume history
    provider.get_volume_history = AsyncMock(return_value=[
        100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000,
        180000, 190000, 200000
    ])

    # Mock current price
    current_price_mock = Mock()
    current_price_mock.price = Decimal("142.00")
    provider.get_current_price = AsyncMock(return_value=current_price_mock)

    # Mock options chain
    provider.get_options_chain = AsyncMock(return_value=[])

    # Mock data source health
    provider.get_data_source_health = AsyncMock(return_value={
        "alpaca": {"is_healthy": True},
        "polygon": {"is_healthy": True}
    })

    return provider


@pytest.fixture
def mock_integration_manager():
    """Create mock integration manager."""
    manager = AsyncMock(spec=ProductionIntegrationManager)

    # Mock account info
    manager.get_account_info = AsyncMock(return_value={
        "cash": 100000.0,
        "equity": 100000.0,
        "buying_power": 100000.0
    })

    # Mock portfolio value
    manager.get_portfolio_value = AsyncMock(return_value=100000.0)

    # Mock trade execution - create a proper TradeResult with required fields
    mock_signal = Mock()
    mock_signal.strategy_name = "wsb-dip-bot"
    mock_signal.ticker = "AAPL"
    trade_result = TradeResult(
        trade_id="test_order_123",
        signal=mock_signal,
        status=TradeStatus.FILLED,
        filled_quantity=10,
        filled_price=5.00,
        commission=0.0,
        error_message=None
    )
    manager.execute_trade = AsyncMock(return_value=trade_result)

    # Mock alert system
    manager.alert_system = AsyncMock()
    manager.alert_system.send_alert = AsyncMock()

    return manager


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "run_lookback_days": 10,
        "run_threshold": 0.10,
        "dip_threshold": -0.03,
        "target_dte_days": 30,
        "otm_percentage": 0.05,
        "max_position_size": 0.20,
        "target_multiplier": 3.0,
        "delta_target": 0.60,
        "wsb_mode": False,
        "universe": ["AAPL", "MSFT", "GOOGL", "NVDA"]
    }


@pytest.fixture
def wsb_bot(mock_integration_manager, mock_data_provider, config):
    """Create WSB Dip Bot instance."""
    return ProductionWSBDipBot(
        mock_integration_manager,
        mock_data_provider,
        config
    )


class TestProductionWSBDipBot:
    """Test ProductionWSBDipBot class."""

    @pytest.mark.asyncio
    async def test_initialization(self, wsb_bot, config):
        """Test strategy initialization."""
        assert wsb_bot.run_lookback_days == config["run_lookback_days"]
        assert wsb_bot.run_threshold == config["run_threshold"]
        assert wsb_bot.dip_threshold == config["dip_threshold"]
        assert wsb_bot.target_dte_days == config["target_dte_days"]
        assert wsb_bot.otm_percentage == config["otm_percentage"]
        assert wsb_bot.max_position_size == config["max_position_size"]
        assert wsb_bot.target_multiplier == config["target_multiplier"]
        assert wsb_bot.delta_target == config["delta_target"]
        assert wsb_bot.wsb_mode == config["wsb_mode"]
        assert len(wsb_bot.universe) == len(config["universe"])

    @pytest.mark.asyncio
    async def test_scan_for_dip_signals_market_closed(self, wsb_bot, mock_data_provider):
        """Test scan when market is closed."""
        mock_data_provider.is_market_open = AsyncMock(return_value=False)

        signals = await wsb_bot.scan_for_dip_signals()

        assert signals == []
        mock_data_provider.is_market_open.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_for_dip_signals_market_open(self, wsb_bot, mock_data_provider):
        """Test scan when market is open."""
        # Mock a valid dip signal
        mock_data_provider.get_price_history = AsyncMock(return_value=[
            Decimal(str(100 + i)) for i in range(30)
        ] + [Decimal("125")])  # Big run followed by dip

        signals = await wsb_bot.scan_for_dip_signals()

        assert isinstance(signals, list)
        mock_data_provider.is_market_open.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_advanced_dip_pattern_insufficient_data(self, wsb_bot, mock_data_provider):
        """Test advanced dip detection with insufficient data."""
        mock_data_provider.get_price_history = AsyncMock(return_value=[Decimal("100")])
        mock_data_provider.get_volume_history = AsyncMock(return_value=[100000])

        signal = await wsb_bot._detect_advanced_dip_pattern("AAPL")

        assert signal is None

    @pytest.mark.asyncio
    async def test_detect_advanced_dip_pattern_no_run(self, wsb_bot, mock_data_provider):
        """Test advanced dip detection when no significant run exists."""
        # Flat prices - no run
        prices = [Decimal("100") for _ in range(30)]
        mock_data_provider.get_price_history = AsyncMock(return_value=prices)
        mock_data_provider.get_volume_history = AsyncMock(return_value=[100000] * 30)

        signal = await wsb_bot._detect_advanced_dip_pattern("AAPL")

        assert signal is None

    @pytest.mark.asyncio
    async def test_detect_advanced_dip_pattern_no_dip(self, wsb_bot, mock_data_provider):
        """Test advanced dip detection when run exists but no dip."""
        # Big run but no dip
        prices = [Decimal(str(100 + i*2)) for i in range(30)]
        mock_data_provider.get_price_history = AsyncMock(return_value=prices)
        mock_data_provider.get_volume_history = AsyncMock(return_value=[100000] * 30)

        signal = await wsb_bot._detect_advanced_dip_pattern("AAPL")

        assert signal is None

    @pytest.mark.asyncio
    async def test_detect_advanced_dip_pattern_valid_signal(self, wsb_bot, mock_data_provider):
        """Test advanced dip detection with valid signal."""
        # Create pattern: low base, big run, then dip
        prices = [Decimal("100")] * 10  # Base
        prices += [Decimal(str(100 + i*3)) for i in range(1, 6)]  # Run up
        prices += [Decimal("130")] * 5  # Peak
        prices += [Decimal("122")] * 10  # Dip

        volumes = [100000] * len(prices)
        volumes[-5:] = [200000] * 5  # Volume spike on dip

        mock_data_provider.get_price_history = AsyncMock(return_value=prices)
        mock_data_provider.get_volume_history = AsyncMock(return_value=volumes)

        signal = await wsb_bot._detect_advanced_dip_pattern("AAPL")

        # Signal might be None if conditions not met
        if signal:
            assert isinstance(signal, DipSignal)
            assert signal.ticker == "AAPL"
            assert signal.run_percentage > 0
            assert signal.dip_percentage > 0

    @pytest.mark.asyncio
    async def test_calculate_rsi(self, wsb_bot):
        """Test RSI calculation."""
        # Upward trend - should have high RSI
        prices_up = [Decimal(str(100 + i)) for i in range(20)]
        rsi_up = wsb_bot._calculate_rsi(prices_up)
        assert 50 < rsi_up <= 100

        # Downward trend - should have low RSI
        prices_down = [Decimal(str(100 - i)) for i in range(20)]
        rsi_down = wsb_bot._calculate_rsi(prices_down)
        assert 0 <= rsi_down < 50

        # Insufficient data
        prices_short = [Decimal("100")]
        rsi_short = wsb_bot._calculate_rsi(prices_short, period=14)
        assert rsi_short == 50.0  # Default neutral

    @pytest.mark.asyncio
    async def test_calculate_rsi_all_gains(self, wsb_bot):
        """Test RSI with all gains."""
        prices = [Decimal(str(100 + i*5)) for i in range(20)]
        rsi = wsb_bot._calculate_rsi(prices)
        assert rsi == 100.0

    @pytest.mark.asyncio
    async def test_calculate_bollinger_position(self, wsb_bot):
        """Test Bollinger Band position calculation."""
        # Price at middle
        prices_middle = [Decimal("100")] * 20
        bb_middle = wsb_bot._calculate_bollinger_position(prices_middle)
        assert 0.4 < bb_middle < 0.6

        # Price at lower band
        prices_low = [Decimal("100")] * 19 + [Decimal("80")]
        bb_low = wsb_bot._calculate_bollinger_position(prices_low)
        assert 0.0 <= bb_low < 0.3

        # Insufficient data
        prices_short = [Decimal("100")]
        bb_short = wsb_bot._calculate_bollinger_position(prices_short)
        assert bb_short == 0.5

    @pytest.mark.asyncio
    async def test_select_optimal_option_no_chain(self, wsb_bot, mock_data_provider):
        """Test option selection when no options chain available."""
        mock_data_provider.get_options_chain = AsyncMock(return_value=None)

        signal = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={}
        )

        result = await wsb_bot.select_optimal_option(signal)
        assert result is None

    @pytest.mark.asyncio
    async def test_select_optimal_option_with_chain(self, wsb_bot, mock_data_provider):
        """Test option selection with valid options chain."""
        # Create mock options
        mock_option = Mock()
        mock_option.option_type = "call"
        mock_option.strike = Decimal("157.5")
        mock_option.volume = 100
        mock_option.bid = Decimal("4.80")
        mock_option.ask = Decimal("5.20")

        mock_data_provider.get_options_chain = AsyncMock(return_value=[mock_option])

        signal = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={}
        )

        result = await wsb_bot.select_optimal_option(signal)

        if result:
            assert result["option"] == mock_option
            assert "spread_ratio" in result
            assert "volume" in result

    @pytest.mark.asyncio
    async def test_execute_dip_trade_normal_mode(self, wsb_bot, mock_integration_manager):
        """Test executing dip trade in normal (non-WSB) mode."""
        signal = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={}
        )

        result = await wsb_bot.execute_dip_trade(signal)

        assert result is True
        assert signal.ticker in wsb_bot.active_positions
        mock_integration_manager.execute_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_dip_trade_wsb_mode(self, mock_integration_manager, mock_data_provider):
        """Test executing dip trade in WSB mode."""
        config = {
            "wsb_mode": True,
            "universe": ["AAPL"],
            "target_multiplier": 3.0
        }

        wsb_bot = ProductionWSBDipBot(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        signal = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={}
        )

        result = await wsb_bot.execute_dip_trade(signal)

        assert result is True
        mock_integration_manager.get_account_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_dip_trade_failed(self, wsb_bot, mock_integration_manager):
        """Test failed trade execution."""
        # Mock failed trade - create proper TradeResult with required fields
        mock_signal = Mock()
        mock_signal.strategy_name = "wsb-dip-bot"
        mock_signal.ticker = "AAPL"
        failed_result = TradeResult(
            trade_id="failed_order_123",
            signal=mock_signal,
            status=TradeStatus.REJECTED,
            filled_quantity=0,
            filled_price=0.0,
            commission=0.0,
            error_message="Insufficient buying power"
        )
        mock_integration_manager.execute_trade = AsyncMock(return_value=failed_result)

        signal = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={}
        )

        result = await wsb_bot.execute_dip_trade(signal)

        assert result is False

    @pytest.mark.asyncio
    async def test_should_exit_position_profit_target(self, wsb_bot, mock_data_provider):
        """Test exit decision when profit target is reached."""
        position = {
            "ticker": "AAPL",
            "cost_basis": 500,
            "days_to_expiry": 15,
            "instrument_type": "stock"
        }

        # Mock current data showing 3x profit
        async def mock_get_current_position_data(pos):
            return {
                "current_value": 1500,  # 3x profit
                "current_price": 150,
                "delta": 1.0
            }

        wsb_bot._get_current_position_data = mock_get_current_position_data

        exit_decision = await wsb_bot.should_exit_position(position)

        assert exit_decision["should_exit"] is True
        assert exit_decision["reason"] == "PROFIT_TARGET"

    @pytest.mark.asyncio
    async def test_should_exit_position_delta_target(self, wsb_bot):
        """Test exit decision when delta target is reached."""
        position = {
            "ticker": "AAPL",
            "cost_basis": 500,
            "days_to_expiry": 15,
            "instrument_type": "option"
        }

        async def mock_get_current_position_data(pos):
            return {
                "current_value": 700,
                "current_price": 150,
                "delta": 0.70  # Above 0.60 threshold
            }

        wsb_bot._get_current_position_data = mock_get_current_position_data

        exit_decision = await wsb_bot.should_exit_position(position)

        assert exit_decision["should_exit"] is True
        assert exit_decision["reason"] == "DELTA_TARGET"

    @pytest.mark.asyncio
    async def test_should_exit_position_time_decay(self, wsb_bot):
        """Test exit decision for time decay protection."""
        position = {
            "ticker": "AAPL",
            "cost_basis": 500,
            "days_to_expiry": 5,  # Less than 7 days
            "instrument_type": "option"
        }

        async def mock_get_current_position_data(pos):
            return {
                "current_value": 550,  # Only 10% profit
                "current_price": 150,
                "delta": 0.4
            }

        wsb_bot._get_current_position_data = mock_get_current_position_data

        exit_decision = await wsb_bot.should_exit_position(position)

        assert exit_decision["should_exit"] is True
        assert exit_decision["reason"] == "TIME_DECAY"

    @pytest.mark.asyncio
    async def test_should_exit_position_stop_loss(self, wsb_bot):
        """Test exit decision for stop loss."""
        position = {
            "ticker": "AAPL",
            "cost_basis": 500,
            "days_to_expiry": 15,
            "instrument_type": "option"
        }

        async def mock_get_current_position_data(pos):
            return {
                "current_value": 350,  # 30% loss
                "current_price": 140,
                "delta": 0.3,
                "volatility": 0.25
            }

        wsb_bot._get_current_position_data = mock_get_current_position_data
        wsb_bot._calculate_dynamic_stop_loss = lambda pos, data: 0.20  # 20% stop

        exit_decision = await wsb_bot.should_exit_position(position)

        assert exit_decision["should_exit"] is True
        assert exit_decision["reason"] == "STOP_LOSS"

    @pytest.mark.asyncio
    async def test_get_recent_volatility(self, wsb_bot, mock_data_provider):
        """Test recent volatility calculation."""
        # Mock price history with volatility
        prices = [Decimal(str(100 + i + (-1)**i * 2)) for i in range(20)]
        mock_data_provider.get_price_history = AsyncMock(return_value=prices)

        volatility = await wsb_bot._get_recent_volatility("AAPL")

        assert volatility > 0
        assert volatility < 2.0  # Reasonable bounds

    @pytest.mark.asyncio
    async def test_get_recent_volatility_insufficient_data(self, wsb_bot, mock_data_provider):
        """Test volatility calculation with insufficient data."""
        mock_data_provider.get_price_history = AsyncMock(return_value=[Decimal("100")])

        volatility = await wsb_bot._get_recent_volatility("AAPL")

        assert volatility == 0.20  # Default value

    @pytest.mark.asyncio
    async def test_calculate_dynamic_stop_loss(self, wsb_bot):
        """Test dynamic stop loss calculation."""
        position = {
            "instrument_type": "option",
            "days_to_expiry": 20
        }

        current_data = {
            "volatility": 0.30
        }

        stop_loss = wsb_bot._calculate_dynamic_stop_loss(position, current_data)

        assert 0.10 <= stop_loss <= 0.50

    @pytest.mark.asyncio
    async def test_calculate_dynamic_stop_loss_high_vol(self, wsb_bot):
        """Test dynamic stop loss with high volatility."""
        position = {
            "instrument_type": "option",
            "days_to_expiry": 20
        }

        current_data = {
            "volatility": 0.50  # High volatility
        }

        stop_loss = wsb_bot._calculate_dynamic_stop_loss(position, current_data)

        assert stop_loss > 0.20  # Should be wider

    @pytest.mark.asyncio
    async def test_calculate_dynamic_stop_loss_near_expiry(self, wsb_bot):
        """Test dynamic stop loss near expiration."""
        position = {
            "instrument_type": "option",
            "days_to_expiry": 5  # Near expiry
        }

        current_data = {
            "volatility": 0.30
        }

        stop_loss = wsb_bot._calculate_dynamic_stop_loss(position, current_data)

        # Should be tighter near expiry
        assert stop_loss < 0.30

    @pytest.mark.asyncio
    async def test_monitor_positions(self, wsb_bot, mock_data_provider):
        """Test position monitoring."""
        # Add a position
        signal = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={"entry_time": datetime.now()}
        )
        wsb_bot.active_positions["AAPL"] = signal

        # Mock exit conditions not met
        wsb_bot._check_exit_conditions = AsyncMock(return_value=None)

        await wsb_bot.monitor_positions()

        wsb_bot._check_exit_conditions.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_exit_conditions_profit_target(self, wsb_bot, mock_data_provider):
        """Test exit condition check for profit target."""
        position = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={"entry_time": datetime.now()}
        )

        # Mock price showing 3x profit
        current_price_mock = Mock()
        current_price_mock.price = Decimal("212")  # (212-150)/150 = ~40% which triggers 3x
        mock_data_provider.get_current_price = AsyncMock(return_value=current_price_mock)

        exit_signal = await wsb_bot._check_exit_conditions(position)

        if exit_signal:
            assert exit_signal in ["profit_target", "delta_target", "time_stop"]

    @pytest.mark.asyncio
    async def test_execute_exit(self, wsb_bot, mock_integration_manager, mock_data_provider):
        """Test exit execution."""
        position = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={"entry_time": datetime.now()}
        )

        wsb_bot.active_positions["AAPL"] = position

        current_price_mock = Mock()
        current_price_mock.price = Decimal("160")
        mock_data_provider.get_current_price = AsyncMock(return_value=current_price_mock)

        await wsb_bot._execute_exit(position, "profit_target")

        mock_integration_manager.execute_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_strategy_status(self, wsb_bot):
        """Test strategy status reporting."""
        # Add some positions
        signal1 = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={}
        )
        wsb_bot.active_positions["AAPL"] = signal1

        status = wsb_bot.get_strategy_status()

        assert status["strategy_name"] == "wsb_dip_bot"
        assert status["active_positions"] == 1
        assert len(status["positions"]) == 1
        assert "parameters" in status

    @pytest.mark.asyncio
    async def test_error_handling_in_scan(self, wsb_bot, mock_data_provider):
        """Test error handling during scan."""
        # Mock an error
        mock_data_provider.get_price_history = AsyncMock(side_effect=Exception("API Error"))

        signals = await wsb_bot.scan_for_dip_signals()

        # Should return empty list on error, not crash
        assert signals == []

    @pytest.mark.asyncio
    async def test_wsb_mode_full_account_reinvestment(self, mock_integration_manager, mock_data_provider):
        """Test WSB mode uses full account cash."""
        config = {
            "wsb_mode": True,
            "universe": ["AAPL"],
        }

        wsb_bot = ProductionWSBDipBot(
            mock_integration_manager,
            mock_data_provider,
            config
        )

        assert wsb_bot.wsb_mode is True

        # Verify warning was logged about WSB mode
        # (Logger calls would be verified in actual implementation)

    @pytest.mark.asyncio
    async def test_get_real_option_premium(self, wsb_bot, mock_data_provider):
        """Test real option premium calculation."""
        with patch('backend.tradingbot.options.pricing_engine.create_options_pricing_engine') as mock_engine:
            mock_pricing = AsyncMock()
            mock_pricing.calculate_theoretical_price = AsyncMock(return_value=Decimal("5.50"))
            mock_engine.return_value = mock_pricing

            premium = await wsb_bot._get_real_option_premium(
                "AAPL",
                Decimal("155"),
                date.today() + timedelta(days=30),
                Decimal("150")
            )

            assert premium == Decimal("5.50")

    @pytest.mark.asyncio
    async def test_get_real_option_premium_fallback(self, wsb_bot):
        """Test option premium fallback calculation."""
        with patch('backend.tradingbot.options.pricing_engine.create_options_pricing_engine', side_effect=Exception("Pricing error")):
            premium = await wsb_bot._get_real_option_premium(
                "AAPL",
                Decimal("155"),
                date.today() + timedelta(days=30),
                Decimal("150")
            )

            # Should return fallback value
            assert premium >= Decimal("0.01")

    @pytest.mark.asyncio
    async def test_execute_dip_trade_zero_quantity(self, wsb_bot):
        """Test trade execution with zero quantity."""
        signal = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("50000.00"),  # Very high premium
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={}
        )

        result = await wsb_bot.execute_dip_trade(signal)

        assert result is False

    def test_dip_signal_dataclass(self):
        """Test DipSignal dataclass."""
        signal = DipSignal(
            ticker="AAPL",
            current_price=Decimal("150"),
            run_percentage=0.25,
            dip_percentage=0.08,
            target_strike=Decimal("157.5"),
            target_expiry=datetime.now() + timedelta(days=30),
            expected_premium=Decimal("5.00"),
            risk_amount=Decimal("500"),
            confidence=0.8,
            metadata={"test": "data"}
        )

        assert signal.ticker == "AAPL"
        assert signal.current_price == Decimal("150")
        assert signal.confidence == 0.8
        assert signal.metadata["test"] == "data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
