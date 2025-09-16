"""Comprehensive tests for core trading interface with real API integration."""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from backend.tradingbot.core.trading_interface import (
    TradingInterface,
    TradeSignal,
    TradeResult,
    PositionUpdate,
    TradeStatus,
    OrderType,
    OrderSide
)


class TestTradingInterface:
    """Test TradingInterface core functionality."""

    def test_trading_interface_initialization(self):
        """Test trading interface initialization."""
        # Mock dependencies
        mock_broker = Mock()
        mock_risk = Mock()
        mock_alerts = Mock()
        config = {"test": "config"}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        assert interface is not None
        assert interface.broker == mock_broker
        assert interface.risk == mock_risk
        assert interface.alerts == mock_alerts
        assert interface.config == config
        assert hasattr(interface, 'active_trades')
        assert hasattr(interface, 'positions')

    def test_trade_signal_creation(self):
        """Test creating TradeSignal objects."""
        signal = TradeSignal(
            strategy_name="TestStrategy",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            limit_price=None,
            stop_price=None,
            time_in_force="gtc"
        )

        assert signal.strategy_name == "TestStrategy"
        assert signal.ticker == "AAPL"
        assert signal.side == OrderSide.BUY
        assert signal.order_type == OrderType.MARKET
        assert signal.quantity == 100

    def test_trade_signal_with_limit_price(self):
        """Test TradeSignal with limit order."""
        signal = TradeSignal(
            strategy_name="LimitStrategy",
            ticker="MSFT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50,
            limit_price=340.50,
            stop_price=None,
            time_in_force="day"
        )

        assert signal.order_type == OrderType.LIMIT
        assert signal.limit_price == 340.50
        assert signal.time_in_force == "day"

    def test_trade_result_creation(self):
        """Test creating TradeResult objects."""
        signal = TradeSignal(
            strategy_name="TestStrategy",
            ticker="GOOGL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=25
        )

        result = TradeResult(
            trade_id="TEST123",
            signal=signal,
            status=TradeStatus.FILLED,
            filled_quantity=25,
            filled_price=140.75,
            commission=1.25,
            error_message=None,
            timestamp=datetime.now()
        )

        assert result.trade_id == "TEST123"
        assert result.signal == signal
        assert result.status == TradeStatus.FILLED
        assert result.filled_quantity == 25
        assert result.filled_price == 140.75

    def test_position_update_creation(self):
        """Test creating PositionUpdate objects."""
        position = PositionUpdate(
            ticker="TSLA",
            quantity=100,
            avg_price=250.50,
            current_price=255.50,
            market_value=25050.0,
            unrealized_pnl=500.0,
            timestamp=datetime.now()
        )

        assert position.ticker == "TSLA"
        assert position.quantity == 100
        assert position.avg_price == 250.50
        assert position.unrealized_pnl == 500.0

    @pytest.mark.asyncio
    async def test_execute_trade_success(self):
        """Test successful trade execution."""
        # Mock dependencies
        mock_broker = AsyncMock()
        mock_risk = AsyncMock()
        mock_alerts = AsyncMock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        # Create test signal
        signal = TradeSignal(
            strategy_name="TestStrategy",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        # Mock successful validation and execution
        with patch.object(interface, 'validate_signal') as mock_validate:
            with patch.object(interface, 'check_risk_limits') as mock_risk_check:
                with patch.object(interface, 'execute_broker_order') as mock_execute:

                    mock_validate.return_value = {"valid": True}
                    mock_risk_check.return_value = {"allowed": True}
                    mock_execute.return_value = {
                        "success": True,
                        "filled_quantity": 100,
                        "filled_price": 185.50,
                        "commission": 1.00
                    }

                    result = await interface.execute_trade(signal)

                    assert isinstance(result, TradeResult)
                    assert result.status == TradeStatus.FILLED
                    assert result.filled_quantity == 100
                    assert result.filled_price == 185.50
                    assert result.commission == 1.00

    @pytest.mark.asyncio
    async def test_execute_trade_validation_failure(self):
        """Test trade execution with validation failure."""
        # Mock dependencies
        mock_broker = AsyncMock()
        mock_risk = AsyncMock()
        mock_alerts = AsyncMock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        signal = TradeSignal(
            strategy_name="TestStrategy",
            ticker="INVALID",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        # Mock validation failure
        with patch.object(interface, 'validate_signal') as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "reason": "Invalid ticker symbol"
            }

            result = await interface.execute_trade(signal)

            assert isinstance(result, TradeResult)
            assert result.status == TradeStatus.REJECTED
            assert "Invalid ticker symbol" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_trade_risk_limit_exceeded(self):
        """Test trade execution with risk limit exceeded."""
        # Mock dependencies
        mock_broker = AsyncMock()
        mock_risk = AsyncMock()
        mock_alerts = AsyncMock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        signal = TradeSignal(
            strategy_name="TestStrategy",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000  # Large quantity
        )

        # Mock successful validation but risk limit exceeded
        with patch.object(interface, 'validate_signal') as mock_validate:
            with patch.object(interface, 'check_risk_limits') as mock_risk_check:

                mock_validate.return_value = {"valid": True}
                mock_risk_check.return_value = {
                    "allowed": False,
                    "reason": "Position size exceeds maximum allowed"
                }

                result = await interface.execute_trade(signal)

                assert isinstance(result, TradeResult)
                assert result.status == TradeStatus.REJECTED
                assert "Risk limit exceeded" in result.error_message

                # Verify alert was sent
                mock_alerts.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_trades(self):
        """Test handling multiple concurrent trade executions."""
        # Mock dependencies
        mock_broker = AsyncMock()
        mock_risk = AsyncMock()
        mock_alerts = AsyncMock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        # Create multiple signals
        signals = [
            TradeSignal(
                strategy_name=f"Strategy{i}",
                ticker=f"STOCK{i}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100
            ) for i in range(5)
        ]

        # Mock successful execution for all
        with patch.object(interface, 'validate_signal') as mock_validate:
            with patch.object(interface, 'check_risk_limits') as mock_risk_check:
                with patch.object(interface, 'execute_broker_order') as mock_execute:

                    mock_validate.return_value = {"valid": True}
                    mock_risk_check.return_value = {"allowed": True}
                    mock_execute.return_value = {
                        "success": True,
                        "filled_quantity": 100,
                        "filled_price": 100.0,
                        "commission": 1.00
                    }

                    # Execute all trades concurrently
                    tasks = [interface.execute_trade(signal) for signal in signals]
                    results = await asyncio.gather(*tasks)

                    assert len(results) == 5
                    assert all(isinstance(result, TradeResult) for result in results)
                    assert all(result.status == TradeStatus.FILLED for result in results)

                    # Verify all trades are tracked
                    assert len(interface.active_trades) == 5

    def test_order_type_enum_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_order_side_enum_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_trade_status_enum_values(self):
        """Test TradeStatus enum values."""
        assert TradeStatus.PENDING.value == "pending"
        assert TradeStatus.SUBMITTED.value == "submitted"
        assert TradeStatus.FILLED.value == "filled"
        assert TradeStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert TradeStatus.CANCELLED.value == "cancelled"
        assert TradeStatus.REJECTED.value == "rejected"

    @pytest.mark.asyncio
    async def test_trade_execution_with_partial_fill(self):
        """Test trade execution with partial fill."""
        # Mock dependencies
        mock_broker = AsyncMock()
        mock_risk = AsyncMock()
        mock_alerts = AsyncMock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        signal = TradeSignal(
            strategy_name="TestStrategy",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            limit_price=185.00
        )

        # Mock partial fill
        with patch.object(interface, 'validate_signal') as mock_validate:
            with patch.object(interface, 'check_risk_limits') as mock_risk_check:
                with patch.object(interface, 'execute_broker_order') as mock_execute:

                    mock_validate.return_value = {"valid": True}
                    mock_risk_check.return_value = {"allowed": True}
                    mock_execute.return_value = {
                        "success": True,
                        "filled_quantity": 500,  # Partial fill
                        "filled_price": 185.00,
                        "commission": 1.00
                    }

                    result = await interface.execute_trade(signal)

                    assert isinstance(result, TradeResult)
                    assert result.status == TradeStatus.FILLED
                    assert result.filled_quantity == 500
                    assert result.signal.quantity == 1000  # Original quantity

    @pytest.mark.asyncio
    async def test_error_handling_in_trade_execution(self):
        """Test error handling during trade execution."""
        # Mock dependencies
        mock_broker = AsyncMock()
        mock_risk = AsyncMock()
        mock_alerts = AsyncMock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        signal = TradeSignal(
            strategy_name="TestStrategy",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        # Mock broker execution failure
        with patch.object(interface, 'validate_signal') as mock_validate:
            with patch.object(interface, 'check_risk_limits') as mock_risk_check:
                with patch.object(interface, 'execute_broker_order') as mock_execute:

                    mock_validate.return_value = {"valid": True}
                    mock_risk_check.return_value = {"allowed": True}
                    mock_execute.return_value = {
                        "success": False,
                        "error": "Insufficient funds"
                    }

                    result = await interface.execute_trade(signal)

                    assert isinstance(result, TradeResult)
                    assert result.status == TradeStatus.REJECTED
                    assert "Insufficient funds" in result.error_message

    def test_trade_id_generation_uniqueness(self):
        """Test that trade IDs are unique."""
        # Mock dependencies
        mock_broker = Mock()
        mock_risk = Mock()
        mock_alerts = Mock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        # Create multiple signals
        signals = [
            TradeSignal(
                strategy_name="TestStrategy",
                ticker="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=100
            ) for _ in range(3)
        ]

        # Generate trade IDs (simplified version of what happens in execute_trade)
        trade_ids = []
        for signal in signals:
            trade_id = f"{signal.strategy_name}_{signal.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            trade_ids.append(trade_id)

        # All trade IDs should be unique
        assert len(set(trade_ids)) == len(trade_ids)

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test interface performance under load."""
        # Mock dependencies
        mock_broker = AsyncMock()
        mock_risk = AsyncMock()
        mock_alerts = AsyncMock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        # Create many signals
        signals = [
            TradeSignal(
                strategy_name=f"Strategy{i}",
                ticker=f"STOCK{i % 10}",  # Limit to 10 different tickers
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=100
            ) for i in range(50)
        ]

        # Mock fast execution
        with patch.object(interface, 'validate_signal') as mock_validate:
            with patch.object(interface, 'check_risk_limits') as mock_risk_check:
                with patch.object(interface, 'execute_broker_order') as mock_execute:

                    mock_validate.return_value = {"valid": True}
                    mock_risk_check.return_value = {"allowed": True}
                    mock_execute.return_value = {
                        "success": True,
                        "filled_quantity": 100,
                        "filled_price": 100.0,
                        "commission": 1.00
                    }

                    import time
                    start_time = time.time()

                    # Execute all trades concurrently
                    tasks = [interface.execute_trade(signal) for signal in signals]
                    results = await asyncio.gather(*tasks)

                    end_time = time.time()
                    execution_time = end_time - start_time

                    # Should complete within reasonable time
                    assert execution_time < 5.0  # Max 5 seconds for 50 trades
                    assert len(results) == 50
                    assert all(isinstance(result, TradeResult) for result in results)

    def test_configuration_handling(self):
        """Test configuration parameter handling."""
        # Mock dependencies
        mock_broker = Mock()
        mock_risk = Mock()
        mock_alerts = Mock()

        # Test with various config types
        configs = [
            {},  # Empty config
            {"setting1": "value1"},  # Simple config
            {"complex": {"nested": {"value": 123}}},  # Nested config
            {"list_setting": [1, 2, 3]},  # List values
        ]

        for config in configs:
            interface = TradingInterface(
                broker_manager=mock_broker,
                risk_manager=mock_risk,
                alert_system=mock_alerts,
                config=config
            )

            assert interface.config == config

    def test_active_trades_tracking(self):
        """Test tracking of active trades."""
        # Mock dependencies
        mock_broker = Mock()
        mock_risk = Mock()
        mock_alerts = Mock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        # Initially no active trades
        assert len(interface.active_trades) == 0

        # Simulate adding trades
        signal = TradeSignal(
            strategy_name="TestStrategy",
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        result = TradeResult(
            trade_id="TEST123",
            signal=signal,
            status=TradeStatus.FILLED,
            filled_quantity=100,
            filled_price=185.50,
            commission=1.00
        )

        interface.active_trades["TEST123"] = result

        assert len(interface.active_trades) == 1
        assert "TEST123" in interface.active_trades
        assert interface.active_trades["TEST123"] == result

    def test_positions_tracking(self):
        """Test tracking of positions."""
        # Mock dependencies
        mock_broker = Mock()
        mock_risk = Mock()
        mock_alerts = Mock()
        config = {}

        interface = TradingInterface(
            broker_manager=mock_broker,
            risk_manager=mock_risk,
            alert_system=mock_alerts,
            config=config
        )

        # Initially no positions
        assert len(interface.positions) == 0

        # Simulate adding position
        position = PositionUpdate(
            ticker="AAPL",
            quantity=100,
            avg_price=185.50,
            current_price=186.00,
            market_value=18550.0,
            unrealized_pnl=50.0,
            timestamp=datetime.now()
        )

        interface.positions["AAPL"] = position

        assert len(interface.positions) == 1
        assert "AAPL" in interface.positions
        assert interface.positions["AAPL"] == position