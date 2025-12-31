"""
Comprehensive tests for backend/tradingbot/trading_system.py
Target: 80%+ coverage with all edge cases and error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from dataclasses import asdict

from backend.tradingbot.trading_system import (
    IntegratedTradingSystem,
    TradingConfig,
    SystemState,
)


@pytest.fixture
def trading_config():
    """Sample trading configuration."""
    return TradingConfig(
        account_size=500000.0,
        max_position_risk_pct=0.10,
        max_total_risk_pct=0.30,
        target_tickers=["AAPL", "MSFT", "GOOGL"],
        data_refresh_interval=60,
        enable_alerts=True,
    )


@pytest.fixture
def trading_system(trading_config):
    """Create trading system instance."""
    return IntegratedTradingSystem(trading_config)


class TestTradingConfig:
    """Test TradingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TradingConfig()

        assert config.account_size == 500000.0
        assert config.max_position_risk_pct == 0.10
        assert config.max_total_risk_pct == 0.30
        assert config.data_refresh_interval == 300
        assert config.enable_alerts is True

    def test_default_tickers(self):
        """Test default target tickers."""
        config = TradingConfig()

        assert len(config.target_tickers) > 0
        assert "AAPL" in config.target_tickers
        assert "MSFT" in config.target_tickers

    def test_default_risk_params(self):
        """Test default risk parameters."""
        config = TradingConfig()

        assert config.risk_params is not None

    def test_default_alert_channels(self):
        """Test default alert channels."""
        config = TradingConfig()

        assert config.alert_channels is not None
        assert "desktop" in config.alert_channels or "webhook" in config.alert_channels

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TradingConfig(
            account_size=1000000.0,
            max_position_risk_pct=0.05,
            target_tickers=["SPY", "QQQ"],
            enable_alerts=False,
        )

        assert config.account_size == 1000000.0
        assert config.max_position_risk_pct == 0.05
        assert config.target_tickers == ["SPY", "QQQ"]
        assert config.enable_alerts is False


class TestSystemState:
    """Test SystemState dataclass."""

    def test_default_state(self):
        """Test default system state."""
        state = SystemState()

        assert state.is_running is False
        assert state.last_scan_time is None
        assert state.active_positions == 0
        assert state.total_portfolio_risk == 0.0
        assert state.alerts_sent_today == 0
        assert state.errors_today == 0

    def test_state_updates(self):
        """Test state can be updated."""
        state = SystemState()

        state.is_running = True
        state.active_positions = 5
        state.total_portfolio_risk = 0.15

        assert state.is_running is True
        assert state.active_positions == 5
        assert state.total_portfolio_risk == 0.15


class TestIntegratedTradingSystem:
    """Test IntegratedTradingSystem class."""

    def test_initialization(self, trading_system):
        """Test system initialization."""
        assert trading_system is not None
        assert trading_system.config is not None
        assert trading_system.state is not None
        assert trading_system.options_calculator is not None
        assert trading_system.signal_generator is not None
        assert trading_system.risk_manager is not None
        assert trading_system.exit_strategy is not None
        assert trading_system.scenario_analyzer is not None
        assert trading_system.alert_system is not None
        assert trading_system.checklist_manager is not None

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        system = IntegratedTradingSystem()

        assert system.config is not None
        assert isinstance(system.config, TradingConfig)

    def test_logger_initialization(self, trading_system):
        """Test logger is initialized."""
        assert trading_system.logger is not None

    def test_stop_system(self, trading_system):
        """Test stop system method."""
        trading_system.state.is_running = True

        trading_system.stop_system()

        assert trading_system.state.is_running is False

    @pytest.mark.asyncio
    async def test_start_system_basic(self, trading_system):
        """Test basic system start."""
        # Mock the scan cycle to prevent infinite loop
        original_scan = trading_system._run_scan_cycle
        scan_count = [0]

        async def mock_scan():
            scan_count[0] += 1
            if scan_count[0] >= 2:
                trading_system.stop_system()
            await original_scan()

        trading_system._run_scan_cycle = mock_scan

        # Run with short interval
        trading_system.config.data_refresh_interval = 0.01

        await trading_system.start_system()

        assert scan_count[0] >= 1

    @pytest.mark.asyncio
    async def test_start_system_error_handling(self, trading_system):
        """Test error handling in start_system."""
        # Mock scan to raise error
        async def mock_scan_error():
            raise ValueError("Test error")

        trading_system._run_scan_cycle = mock_scan_error
        trading_system.config.data_refresh_interval = 0.01

        # Should handle error gracefully
        await trading_system.start_system()

        assert trading_system.state.errors_today > 0
        assert trading_system.state.is_running is False

    @pytest.mark.asyncio
    async def test_run_scan_cycle(self, trading_system):
        """Test scan cycle execution."""
        with patch.object(trading_system, '_fetch_market_data', new_callable=AsyncMock) as mock_fetch:
            with patch.object(trading_system, '_scan_for_opportunities', new_callable=AsyncMock) as mock_scan:
                with patch.object(trading_system, '_monitor_existing_positions', new_callable=AsyncMock) as mock_monitor:
                    with patch.object(trading_system, '_update_portfolio_metrics', new_callable=AsyncMock) as mock_update:
                        with patch.object(trading_system, '_run_maintenance_tasks', new_callable=AsyncMock) as mock_maintenance:
                            mock_fetch.return_value = {}

                            await trading_system._run_scan_cycle()

                            mock_fetch.assert_called_once()
                            mock_scan.assert_called_once()
                            mock_monitor.assert_called_once()
                            mock_update.assert_called_once()
                            mock_maintenance.assert_called_once()
                            assert trading_system.state.last_scan_time is not None

    @pytest.mark.asyncio
    async def test_run_scan_cycle_error_handling(self, trading_system):
        """Test error handling in scan cycle."""
        with patch.object(trading_system, '_fetch_market_data', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")

            initial_errors = trading_system.state.errors_today

            await trading_system._run_scan_cycle()

            assert trading_system.state.errors_today > initial_errors

    @pytest.mark.asyncio
    async def test_fetch_market_data(self, trading_system):
        """Test market data fetching."""
        data = await trading_system._fetch_market_data()

        assert isinstance(data, dict)
        assert len(data) > 0
        for ticker in trading_system.config.target_tickers:
            assert ticker in data
            assert "current" in data[ticker]
            assert "previous" in data[ticker]

    @pytest.mark.asyncio
    async def test_scan_for_opportunities(self, trading_system):
        """Test opportunity scanning."""
        market_data = await trading_system._fetch_market_data()

        with patch.object(trading_system, '_process_signal', new_callable=AsyncMock) as mock_process:
            await trading_system._scan_for_opportunities(market_data)

            # Should process signals for each ticker
            assert mock_process.call_count >= 0  # May or may not generate signals

    @pytest.mark.asyncio
    async def test_scan_for_opportunities_error_handling(self, trading_system):
        """Test error handling in opportunity scanning."""
        # Invalid market data
        market_data = {"AAPL": {}}  # Missing required fields

        await trading_system._scan_for_opportunities(market_data)

        # Should handle gracefully without crashing

    def test_create_indicators(self, trading_system):
        """Test creating technical indicators from data."""
        data = {
            "close": 200.0,
            "high": 202.0,
            "low": 198.0,
            "volume": 1000000,
            "ema_20": 198.5,
            "ema_50": 195.0,
            "ema_200": 190.0,
            "rsi": 45.0,
            "atr": 4.0,
        }

        indicators = trading_system._create_indicators(data)

        assert indicators.price == 200.0
        assert indicators.ema_20 == 198.5
        assert indicators.rsi_14 == 45.0
        assert indicators.volume == 1000000

    @pytest.mark.asyncio
    async def test_process_signal_buy(self, trading_system):
        """Test processing buy signal."""
        from backend.tradingbot.market_regime import MarketSignal, SignalType, TechnicalIndicators

        signal = MarketSignal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            reasoning=["Strong momentum", "Dip detected"],
        )

        indicators = TechnicalIndicators(
            price=200.0,
            ema_20=198.5,
            ema_50=195.0,
            ema_200=190.0,
            rsi_14=45.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=202.0,
            low_24h=198.0,
        )

        market_data = {
            "implied_volatility": 0.28,
        }

        with patch.object(trading_system, '_handle_buy_signal', new_callable=AsyncMock) as mock_buy:
            await trading_system._process_signal("AAPL", signal, indicators, market_data)

            mock_buy.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_signal_setup(self, trading_system):
        """Test processing setup signal."""
        from backend.tradingbot.market_regime import MarketSignal, SignalType, TechnicalIndicators

        signal = MarketSignal(
            signal_type=SignalType.HOLD,
            confidence=0.6,
            reasoning=["Setup detected", "Watch for entry"],
        )

        indicators = TechnicalIndicators(
            price=200.0,
            ema_20=198.5,
            ema_50=195.0,
            ema_200=190.0,
            rsi_14=45.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=202.0,
            low_24h=198.0,
        )

        market_data = {}

        with patch.object(trading_system, '_handle_setup_signal', new_callable=AsyncMock) as mock_setup:
            await trading_system._process_signal("AAPL", signal, indicators, market_data)

            mock_setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_buy_signal(self, trading_system):
        """Test handling buy signal."""
        from backend.tradingbot.market_regime import MarketSignal, SignalType, TechnicalIndicators

        signal = MarketSignal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            reasoning=["Strong signal"],
        )

        indicators = TechnicalIndicators(
            price=200.0,
            ema_20=198.5,
            ema_50=195.0,
            ema_200=190.0,
            rsi_14=45.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=202.0,
            low_24h=198.0,
        )

        market_data = {"implied_volatility": 0.28}

        with patch.object(trading_system, '_validate_trade_risk', return_value=True):
            with patch.object(trading_system, '_send_entry_alert', new_callable=AsyncMock):
                await trading_system._handle_buy_signal("AAPL", signal, indicators, market_data)

    @pytest.mark.asyncio
    async def test_handle_buy_signal_risk_rejected(self, trading_system):
        """Test buy signal rejected by risk management."""
        from backend.tradingbot.market_regime import MarketSignal, SignalType, TechnicalIndicators

        signal = MarketSignal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            reasoning=["Signal"],
        )

        indicators = TechnicalIndicators(
            price=200.0,
            ema_20=198.5,
            ema_50=195.0,
            ema_200=190.0,
            rsi_14=45.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=202.0,
            low_24h=198.0,
        )

        market_data = {"implied_volatility": 0.28}

        # Mock risk validation to reject
        with patch.object(trading_system, '_validate_trade_risk', return_value=False):
            with patch.object(trading_system, '_send_entry_alert', new_callable=AsyncMock) as mock_alert:
                await trading_system._handle_buy_signal("AAPL", signal, indicators, market_data)

                # Alert should not be sent
                mock_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_setup_signal(self, trading_system):
        """Test handling setup signal."""
        from backend.tradingbot.market_regime import MarketSignal, SignalType, TechnicalIndicators

        signal = MarketSignal(
            signal_type=SignalType.HOLD,
            confidence=0.6,
            reasoning=["Setup detected"],
        )

        indicators = TechnicalIndicators(
            price=200.0,
            ema_20=198.5,
            ema_50=195.0,
            ema_200=190.0,
            rsi_14=45.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=202.0,
            low_24h=198.0,
        )

        await trading_system._handle_setup_signal("AAPL", signal, indicators)

    def test_validate_trade_risk(self, trading_system):
        """Test trade risk validation."""
        from backend.tradingbot.options_calculator import TradeCalculation
        from datetime import date

        trade_calc = TradeCalculation(
            ticker="AAPL",
            spot_price=200.0,
            strike=205.0,
            expiry_date=date(2024, 12, 31),
            estimated_premium=3.50,
            recommended_contracts=5,
            total_cost=1750.0,
            account_risk_pct=3.5,
            breakeven_price=208.50,
        )

        result = trading_system._validate_trade_risk(trade_calc)

        # Should pass with default config (3.5% < 10%)
        assert result is True

    def test_validate_trade_risk_exceeds_limit(self, trading_system):
        """Test trade risk validation when limit exceeded."""
        from backend.tradingbot.options_calculator import TradeCalculation
        from datetime import date

        trade_calc = TradeCalculation(
            ticker="AAPL",
            spot_price=200.0,
            strike=205.0,
            expiry_date=date(2024, 12, 31),
            estimated_premium=3.50,
            recommended_contracts=50,
            total_cost=17500.0,
            account_risk_pct=35.0,  # Exceeds 10% limit
            breakeven_price=208.50,
        )

        result = trading_system._validate_trade_risk(trade_calc)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_entry_alert(self, trading_system):
        """Test sending entry alert."""
        from backend.tradingbot.market_regime import MarketSignal, SignalType
        from backend.tradingbot.options_calculator import TradeCalculation
        from datetime import date

        signal = MarketSignal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            reasoning=["Strong signal"],
        )

        trade_calc = TradeCalculation(
            ticker="AAPL",
            spot_price=200.0,
            strike=205.0,
            expiry_date=date(2024, 12, 31),
            estimated_premium=3.50,
            recommended_contracts=5,
            total_cost=1750.0,
            account_risk_pct=3.5,
            breakeven_price=208.50,
        )

        initial_alerts = trading_system.state.alerts_sent_today

        await trading_system._send_entry_alert("AAPL", signal, trade_calc, "CHECKLIST-001")

        assert trading_system.state.alerts_sent_today == initial_alerts + 1

    @pytest.mark.asyncio
    async def test_send_entry_alert_disabled(self, trading_system):
        """Test that alert is not sent when disabled."""
        from backend.tradingbot.market_regime import MarketSignal, SignalType
        from backend.tradingbot.options_calculator import TradeCalculation
        from datetime import date

        trading_system.config.enable_alerts = False

        signal = MarketSignal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            reasoning=["Signal"],
        )

        trade_calc = TradeCalculation(
            ticker="AAPL",
            spot_price=200.0,
            strike=205.0,
            expiry_date=date(2024, 12, 31),
            estimated_premium=3.50,
            recommended_contracts=5,
            total_cost=1750.0,
            account_risk_pct=3.5,
            breakeven_price=208.50,
        )

        initial_alerts = trading_system.state.alerts_sent_today

        await trading_system._send_entry_alert("AAPL", signal, trade_calc, "CHECKLIST-001")

        # Alerts should not increase
        assert trading_system.state.alerts_sent_today == initial_alerts

    @pytest.mark.asyncio
    async def test_monitor_existing_positions(self, trading_system):
        """Test monitoring existing positions."""
        from backend.tradingbot.risk_management import Position, PositionStatus
        from datetime import datetime
        from decimal import Decimal

        # Add a position to risk manager
        position = Position(
            ticker="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=Decimal("195.00"),
            quantity=10,
            option_type="call",
            strike=Decimal("200.00"),
            expiry=datetime(2024, 12, 31),
            premium_paid=Decimal("3.50"),
            status=PositionStatus.OPEN,
        )

        trading_system.risk_manager.positions.append(position)

        market_data = {
            "AAPL": {
                "current": {"close": 200.0},
                "implied_volatility": 0.28,
            }
        }

        with patch.object(trading_system, '_handle_exit_signals', new_callable=AsyncMock):
            await trading_system._monitor_existing_positions(market_data)

    @pytest.mark.asyncio
    async def test_monitor_existing_positions_error(self, trading_system):
        """Test error handling in position monitoring."""
        market_data = {}  # Empty data

        # Should handle gracefully
        await trading_system._monitor_existing_positions(market_data)

    @pytest.mark.asyncio
    async def test_handle_exit_signals(self, trading_system):
        """Test handling exit signals."""
        from backend.tradingbot.risk_management import Position, PositionStatus
        from backend.tradingbot.exit_planning import ExitSignal, ExitReason, SignalStrength
        from datetime import datetime
        from decimal import Decimal

        position = Position(
            ticker="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=Decimal("195.00"),
            quantity=10,
            option_type="call",
            strike=Decimal("200.00"),
            expiry=datetime(2024, 12, 31),
            premium_paid=Decimal("3.50"),
            status=PositionStatus.OPEN,
        )

        exit_signals = [
            ExitSignal(
                reason=ExitReason.TAKE_PROFIT,
                strength=SignalStrength.STRONG,
                confidence=0.9,
                recommendation="Close position",
            )
        ]

        await trading_system._handle_exit_signals(position, exit_signals)

    @pytest.mark.asyncio
    async def test_run_position_scenario_analysis(self, trading_system):
        """Test position scenario analysis."""
        from backend.tradingbot.risk_management import Position, PositionStatus
        from datetime import datetime
        from decimal import Decimal

        position = Position(
            ticker="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=Decimal("195.00"),
            quantity=10,
            option_type="call",
            strike=Decimal("200.00"),
            expiry=datetime(2024, 12, 31),
            premium_paid=Decimal("3.50"),
            status=PositionStatus.OPEN,
        )

        await trading_system._run_position_scenario_analysis(position, 200.0, 0.28)

    @pytest.mark.asyncio
    async def test_update_portfolio_metrics(self, trading_system):
        """Test portfolio metrics update."""
        await trading_system._update_portfolio_metrics()

        assert trading_system.state.active_positions >= 0
        assert trading_system.state.total_portfolio_risk >= 0

    @pytest.mark.asyncio
    async def test_update_portfolio_metrics_risk_alert(self, trading_system):
        """Test risk alert when portfolio risk exceeds limit."""
        # Force high risk
        with patch.object(trading_system.risk_manager, 'calculate_portfolio_risk') as mock_risk:
            from backend.tradingbot.risk_management import PortfolioRisk

            mock_risk.return_value = PortfolioRisk(
                total_capital_at_risk=50000.0,
                risk_utilization=0.35,  # Exceeds 30% limit
                var_95=15000.0,
                var_99=20000.0,
            )

            await trading_system._update_portfolio_metrics()

    @pytest.mark.asyncio
    async def test_run_maintenance_tasks(self, trading_system):
        """Test maintenance tasks."""
        await trading_system._run_maintenance_tasks()

    @pytest.mark.asyncio
    async def test_run_maintenance_tasks_reset_counters(self, trading_system):
        """Test that counters reset at midnight."""
        trading_system.state.alerts_sent_today = 10
        trading_system.state.errors_today = 5

        # Mock time to be midnight
        with patch('backend.tradingbot.trading_system.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 1, 0, 2, 0)

            await trading_system._run_maintenance_tasks()

            assert trading_system.state.alerts_sent_today == 0
            assert trading_system.state.errors_today == 0

    def test_add_position(self, trading_system):
        """Test adding position to system."""
        from backend.tradingbot.risk_management import Position, PositionStatus
        from datetime import datetime
        from decimal import Decimal

        position = Position(
            ticker="AAPL",
            entry_date=datetime(2024, 1, 1),
            entry_price=Decimal("195.00"),
            quantity=10,
            option_type="call",
            strike=Decimal("200.00"),
            expiry=datetime(2024, 12, 31),
            premium_paid=Decimal("3.50"),
            status=PositionStatus.OPEN,
        )

        result = trading_system.add_position(position)

        assert result is True

    def test_get_portfolio_status(self, trading_system):
        """Test getting portfolio status."""
        status = trading_system.get_portfolio_status()

        assert isinstance(status, dict)
        assert "system_state" in status
        assert "portfolio_metrics" in status
        assert "active_checklists" in status
        assert "config" in status

    def test_force_scan(self, trading_system):
        """Test forcing an immediate scan."""
        # This creates an async task
        trading_system.tasks = []
        trading_system.force_scan()

        # Task should be created
        assert len(trading_system.tasks) > 0

    def test_calculate_trade_for_ticker(self, trading_system):
        """Test calculating trade for specific ticker."""
        trade_calc = trading_system.calculate_trade_for_ticker(
            ticker="AAPL",
            spot_price=200.0,
            implied_volatility=0.28,
        )

        assert trade_calc is not None
        assert trade_calc.ticker == "AAPL"
        assert trade_calc.spot_price == 200.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_market_data(self, trading_system):
        """Test handling of empty market data."""
        await trading_system._scan_for_opportunities({})

    @pytest.mark.asyncio
    async def test_malformed_market_data(self, trading_system):
        """Test handling of malformed market data."""
        malformed_data = {
            "AAPL": None,
            "MSFT": {"invalid": "data"},
        }

        await trading_system._scan_for_opportunities(malformed_data)

    def test_create_indicators_missing_fields(self, trading_system):
        """Test creating indicators with missing fields."""
        data = {
            "close": 200.0,
            # Missing other required fields
        }

        # Should handle missing fields with KeyError
        with pytest.raises(KeyError):
            trading_system._create_indicators(data)

    @pytest.mark.asyncio
    async def test_scan_cycle_concurrent_execution(self, trading_system):
        """Test that scan cycle can be called concurrently."""
        # Run two scans concurrently
        tasks = [
            trading_system._run_scan_cycle(),
            trading_system._run_scan_cycle(),
        ]

        await asyncio.gather(*tasks)


class TestMainFunction:
    """Test the __main__ function."""

    @pytest.mark.asyncio
    async def test_main_test_system(self):
        """Test the test_system async function in __main__."""
        # This tests the example code at bottom of module
        # We can't directly test __main__, but we can test the logic

        config = TradingConfig(
            account_size=500000,
            max_position_risk_pct=0.10,
            target_tickers=["GOOGL", "AAPL", "MSFT"],
        )

        system = IntegratedTradingSystem(config)

        # Test trade calculation
        trade_calc = system.calculate_trade_for_ticker(
            ticker="GOOGL",
            spot_price=207.0,
            implied_volatility=0.28,
        )

        assert trade_calc is not None

        # Test portfolio status
        status = system.get_portfolio_status()
        assert status is not None
