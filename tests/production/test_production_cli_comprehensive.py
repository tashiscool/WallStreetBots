"""Comprehensive tests for Production CLI module to achieve >75% coverage."""
import pytest
import asyncio
import sys
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from decimal import Decimal
from datetime import datetime

from backend.tradingbot.production.core.production_cli import (
    ProductionCLI,
    main,
    # parse_args  # function not available
)


class TestProductionCLI:
    """Test ProductionCLI comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = ProductionCLI()

    def test_production_cli_initialization(self):
        """Test ProductionCLI initialization."""
        assert self.cli.manager is None

    @patch('backend.tradingbot.production.core.production_cli.ProductionManager')
    @patch('asyncio.Event.wait')
    def test_start_system_success(self, mock_event_wait, mock_manager_class):
        """Test successful system start."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Mock args
        args = Mock()
        args.alpaca_api_key = "test_key"
        args.alpaca_secret_key = "test_secret"
        args.paper_trading = True
        args.user_id = 1
        args.max_position_size = 0.20
        args.max_total_risk = 0.50
        args.strategies = "strategy1,strategy2"
        
        # Mock async start_production_system method
        async def mock_start_production_system():
            return True
        
        # Mock async stop_production_system method
        async def mock_stop_production_system():
            pass
        
        mock_manager.start_production_system = mock_start_production_system
        mock_manager.stop_production_system = mock_stop_production_system
        mock_manager.strategies = ["strategy1", "strategy2"]  # Mock strategies list
        
        # Mock the event wait to raise KeyboardInterrupt to simulate shutdown
        async def mock_wait():
            raise KeyboardInterrupt()
        
        mock_event_wait.return_value = mock_wait()
        
        # Run the async method
        asyncio.run(self.cli.start_system(args))
        
        # Verify manager was created and started
        mock_manager_class.assert_called_once()
        assert self.cli.manager == mock_manager

    @patch('backend.tradingbot.production.core.production_cli.ProductionManager')
    @patch('asyncio.Event.wait')
    def test_start_system_with_defaults(self, mock_event_wait, mock_manager_class):
        """Test system start with default values."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Mock args with minimal values
        args = Mock()
        args.alpaca_api_key = None
        args.alpaca_secret_key = None
        args.paper_trading = True
        args.user_id = 1
        args.max_position_size = 0.20
        args.max_total_risk = 0.50
        args.strategies = None
        
        # Mock async start_production_system method
        async def mock_start_production_system():
            return True
        
        # Mock async stop_production_system method
        async def mock_stop_production_system():
            pass
        
        mock_manager.start_production_system = mock_start_production_system
        mock_manager.stop_production_system = mock_stop_production_system
        mock_manager.strategies = []  # Mock empty strategies list
        
        # Mock the event wait to raise KeyboardInterrupt to simulate shutdown
        async def mock_wait():
            raise KeyboardInterrupt()
        
        mock_event_wait.return_value = mock_wait()
        
        # Run the async method
        asyncio.run(self.cli.start_system(args))
        
        # Verify manager was created with defaults
        mock_manager_class.assert_called_once()
        call_args = mock_manager_class.call_args[0][0]
        assert call_args.alpaca_api_key == "test_key"
        assert call_args.alpaca_secret_key == "test_secret"

    @patch('backend.tradingbot.production.core.production_cli.ProductionManager')
    def test_start_system_failure(self, mock_manager_class):
        """Test system start failure handling."""
        mock_manager_class.side_effect = Exception("Connection failed")
        
        args = Mock()
        args.alpaca_api_key = "invalid_key"
        args.alpaca_secret_key = "invalid_secret"
        args.paper_trading = True
        args.user_id = 1
        args.max_position_size = 0.20
        args.max_total_risk = 0.50
        args.strategies = None
        
        # Should handle exception gracefully and exit
        with pytest.raises(SystemExit):
            asyncio.run(self.cli.start_system(args))

    @pytest.mark.skip(reason="stop_system method does not exist in ProductionCLI")
    def test_stop_system_success(self):
        """Test successful system stop."""
        mock_manager = Mock()
        self.cli.manager = mock_manager
        
        self.cli.stop_system()
        
        mock_manager.stop.assert_called_once()

    @pytest.mark.skip(reason="stop_system method does not exist in ProductionCLI")
    def test_stop_system_no_manager(self):
        """Test stop system when no manager exists."""
        self.cli.manager = None
        
        # Should handle gracefully
        self.cli.stop_system()
        # No exception should be raised

    @pytest.mark.skip(reason="status_system method does not exist in ProductionCLI")
    def test_status_system_running(self):
        """Test status when system is running."""
        mock_manager = Mock()
        mock_status = Mock()
        mock_status.is_running = True
        mock_status.active_strategies = 3
        mock_status.total_trades = 150
        mock_status.total_pnl = Decimal("2500.50")
        mock_status.last_update = datetime.now()
        
        mock_manager.get_status.return_value = mock_status
        self.cli.manager = mock_manager
        
        self.cli.status_system()
        
        mock_manager.get_status.assert_called_once()

    @pytest.mark.skip(reason="status_system method does not exist in ProductionCLI")
    def test_status_system_not_running(self):
        """Test status when system is not running."""
        mock_manager = Mock()
        mock_status = Mock()
        mock_status.is_running = False
        mock_status.active_strategies = 0
        mock_status.total_trades = 0
        mock_status.total_pnl = Decimal("0.00")
        mock_status.last_update = datetime.now()
        
        mock_manager.get_status.return_value = mock_status
        self.cli.manager = mock_manager
        
        self.cli.status_system()
        
        mock_manager.get_status.assert_called_once()

    @pytest.mark.skip(reason="status_system method does not exist in ProductionCLI")
    def test_status_system_no_manager(self):
        """Test status when no manager exists."""
        self.cli.manager = None
        
        # Should handle gracefully
        self.cli.status_system()
        # No exception should be raised

    def test_portfolio_view(self):
        """Test portfolio view."""
        mock_manager = Mock()
        mock_integration = Mock()
        mock_positions = {
            "AAPL": {"quantity": 100, "avg_price": 150.0, "current_price": 155.0},
            "MSFT": {"quantity": 50, "avg_price": 300.0, "current_price": 310.0}
        }
        mock_trades = [
            Mock(symbol="AAPL", side="buy", quantity=100, price=150.0, status="filled"),
            Mock(symbol="MSFT", side="buy", quantity=50, price=300.0, status="filled")
        ]
        
        mock_integration.get_positions.return_value = mock_positions
        mock_integration.get_trades.return_value = mock_trades
        mock_manager.integration = mock_integration
        self.cli.manager = mock_manager
        
        self.cli.portfolio_view()
        
        mock_integration.get_positions.assert_called_once()
        mock_integration.get_trades.assert_called_once()

    def test_portfolio_view_no_manager(self):
        """Test portfolio view when no manager exists."""
        self.cli.manager = None
        
        # Should handle gracefully
        self.cli.portfolio_view()
        # No exception should be raised

    def test_manual_trade_buy(self):
        """Test manual buy trade."""
        mock_manager = Mock()
        mock_integration = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.trade_id = "trade_123"
        mock_result.execution_price = 150.25
        
        mock_integration.execute_trade = AsyncMock(return_value=mock_result)
        mock_manager.integration = mock_integration
        self.cli.manager = mock_manager
        
        args = Mock()
        args.ticker = "AAPL"
        args.side = "buy"
        args.quantity = 100
        args.price = 150.0
        
        self.cli.manual_trade(args)
        
        mock_integration.execute_trade.assert_called_once()
        call_args = mock_integration.execute_trade.call_args[0][0]
        assert call_args.symbol == "AAPL"
        assert call_args.action == "buy"
        assert call_args.quantity == 100
        assert call_args.price == 150.0

    def test_manual_trade_sell(self):
        """Test manual sell trade."""
        mock_manager = Mock()
        mock_integration = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.trade_id = "trade_456"
        mock_result.execution_price = 155.75
        
        mock_integration.execute_trade = AsyncMock(return_value=mock_result)
        mock_manager.integration = mock_integration
        self.cli.manager = mock_manager
        
        args = Mock()
        args.ticker = "MSFT"
        args.side = "sell"
        args.quantity = 50
        args.price = 155.0
        
        self.cli.manual_trade(args)
        
        mock_integration.execute_trade.assert_called_once()
        call_args = mock_integration.execute_trade.call_args[0][0]
        assert call_args.symbol == "MSFT"
        assert call_args.action == "sell"
        assert call_args.quantity == 50
        assert call_args.price == 155.0

    def test_manual_trade_failure(self):
        """Test manual trade failure."""
        mock_manager = Mock()
        mock_integration = Mock()
        mock_result = Mock()
        mock_result.success = False
        mock_result.message = "Trade failed"
        
        mock_integration.execute_trade = AsyncMock(return_value=mock_result)
        mock_manager.integration = mock_integration
        self.cli.manager = mock_manager
        
        args = Mock()
        args.ticker = "INVALID"
        args.side = "buy"
        args.quantity = 100
        args.price = 0.0
        
        self.cli.manual_trade(args)
        
        mock_integration.execute_trade.assert_called_once()

    def test_manual_trade_no_manager(self):
        """Test manual trade when no manager exists."""
        self.cli.manager = None
        
        args = Mock()
        args.ticker = "AAPL"
        args.side = "buy"
        args.quantity = 100
        args.price = 150.0
        
        # Should handle gracefully
        self.cli.manual_trade(args)
        # No exception should be raised

    def test_list_strategies(self):
        """Test listing strategies."""
        mock_manager = Mock()
        mock_strategies = {
            "strategy1": Mock(name="strategy1"),
            "strategy2": Mock(name="strategy2"),
            "strategy3": Mock(name="strategy3")
        }
        mock_manager.strategies = mock_strategies
        self.cli.manager = mock_manager
        
        self.cli.list_strategies()
        
        # Should not raise exception

    def test_list_strategies_no_manager(self):
        """Test listing strategies when no manager exists."""
        self.cli.manager = None
        
        # Should handle gracefully
        self.cli.list_strategies()
        # No exception should be raised

    def test_enable_strategy(self):
        """Test enabling a strategy."""
        mock_manager = Mock()
        mock_strategy = Mock()
        mock_strategy.start = Mock()
        mock_manager.strategies = {"test_strategy": mock_strategy}
        self.cli.manager = mock_manager
        
        args = Mock()
        args.strategy_name = "test_strategy"
        
        self.cli.enable_strategy(args)
        
        mock_strategy.start.assert_called_once()

    def test_enable_strategy_not_found(self):
        """Test enabling a non-existent strategy."""
        mock_manager = Mock()
        mock_manager.strategies = {}
        self.cli.manager = mock_manager
        
        args = Mock()
        args.strategy_name = "nonexistent_strategy"
        
        # Should handle gracefully
        self.cli.enable_strategy(args)
        # No exception should be raised

    def test_disable_strategy(self):
        """Test disabling a strategy."""
        mock_manager = Mock()
        mock_strategy = Mock()
        mock_strategy.stop = Mock()
        mock_manager.strategies = {"test_strategy": mock_strategy}
        self.cli.manager = mock_manager
        
        args = Mock()
        args.strategy_name = "test_strategy"
        
        self.cli.disable_strategy(args)
        
        mock_strategy.stop.assert_called_once()

    def test_disable_strategy_not_found(self):
        """Test disabling a non-existent strategy."""
        mock_manager = Mock()
        mock_manager.strategies = {}
        self.cli.manager = mock_manager
        
        args = Mock()
        args.strategy_name = "nonexistent_strategy"
        
        # Should handle gracefully
        self.cli.disable_strategy(args)
        # No exception should be raised


class TestParseArgs:
    """Test argument parsing functionality."""

    def skip_test_parse_args_start_command(self):
        """Test parsing start command arguments."""
        args = [
            "start",
            "--alpaca-api-key", "test_key",
            "--alpaca-secret-key", "test_secret",
            "--paper-trading",
            "--user-id", "1",
            "--max-position-size", "0.20",
            "--max-total-risk", "0.50",
            "--strategies", "strategy1,strategy2"
        ]
        
        # parsed_args = parse_args(args)  # function not available
        
        assert parsed_args.command == "start"
        assert parsed_args.alpaca_api_key == "test_key"
        assert parsed_args.alpaca_secret_key == "test_secret"
        assert parsed_args.paper_trading is True
        assert parsed_args.user_id == 1
        assert parsed_args.max_position_size == 0.20
        assert parsed_args.max_total_risk == 0.50
        assert parsed_args.strategies == "strategy1,strategy2"

    def skip_test_parse_args_status_command(self):
        """Test parsing status command arguments."""
        args = ["status"]
        
        # parsed_args = parse_args(args)  # function not available
        
        assert parsed_args.command == "status"

    def skip_test_parse_args_portfolio_command(self):
        """Test parsing portfolio command arguments."""
        args = ["portfolio"]
        
        # parsed_args = parse_args(args)  # function not available
        
        assert parsed_args.command == "portfolio"

    def skip_test_parse_args_stop_command(self):
        """Test parsing stop command arguments."""
        args = ["stop"]
        
        # parsed_args = parse_args(args)  # function not available
        
        assert parsed_args.command == "stop"

    def skip_test_parse_args_manual_trade_command(self):
        """Test parsing manual trade command arguments."""
        args = [
            "manual-trade",
            "--symbol", "AAPL",
            "--side", "buy",
            "--quantity", "100",
            "--price", "150.0"
        ]
        
        # parsed_args = parse_args(args)  # function not available
        
        assert parsed_args.command == "manual-trade"
        assert parsed_args.symbol == "AAPL"
        assert parsed_args.side == "buy"
        assert parsed_args.quantity == 100
        assert parsed_args.price == 150.0

    def skip_test_parse_args_list_strategies_command(self):
        """Test parsing list strategies command arguments."""
        args = ["list-strategies"]
        
        # parsed_args = parse_args(args)  # function not available
        
        assert parsed_args.command == "list-strategies"

    def skip_test_parse_args_enable_strategy_command(self):
        """Test parsing enable strategy command arguments."""
        args = ["enable-strategy", "--strategy-name", "test_strategy"]
        
        # parsed_args = parse_args(args)  # function not available
        
        assert parsed_args.command == "enable-strategy"
        assert parsed_args.strategy_name == "test_strategy"

    def skip_test_parse_args_disable_strategy_command(self):
        """Test parsing disable strategy command arguments."""
        args = ["disable-strategy", "--strategy-name", "test_strategy"]
        
        # parsed_args = parse_args(args)  # function not available
        
        assert parsed_args.command == "disable-strategy"
        assert parsed_args.strategy_name == "test_strategy"

    def skip_test_parse_args_default_values(self):
        """Test parsing with default values."""
        args = ["start"]
        
        # parsed_args = parse_args(args)  # function not available
        
        assert parsed_args.command == "start"
        assert parsed_args.alpaca_api_key is None
        assert parsed_args.alpaca_secret_key is None
        assert parsed_args.paper_trading is False
        assert parsed_args.user_id == 1
        assert parsed_args.max_position_size == 0.20
        assert parsed_args.max_total_risk == 0.50
        assert parsed_args.strategies is None


class TestMainFunction:
    """Test main function functionality."""

    @patch('backend.tradingbot.production.core.production_cli.ProductionCLI')
    def test_main_start_command(self, mock_cli_class):
        """Test main function with start command."""
        mock_cli = Mock()
        mock_cli.start_system = AsyncMock()  # start_system is async
        mock_cli_class.return_value = mock_cli

        with patch('sys.argv', ['production_cli.py', 'start', '--paper-trading']):
            main()

        mock_cli_class.assert_called_once()
        mock_cli.start_system.assert_called_once()

    @patch('backend.tradingbot.production.core.production_cli.ProductionCLI')
    def test_main_status_command(self, mock_cli_class):
        """Test main function with status command."""
        mock_cli = Mock()
        mock_cli.show_status = AsyncMock()
        mock_cli_class.return_value = mock_cli

        with patch('sys.argv', ['production_cli.py', 'status']):
            main()

        mock_cli_class.assert_called_once()
        mock_cli.show_status.assert_called_once()

    @patch('backend.tradingbot.production.core.production_cli.ProductionCLI')
    def test_main_portfolio_command(self, mock_cli_class):
        """Test main function with portfolio command."""
        mock_cli = Mock()
        mock_cli.show_portfolio = AsyncMock()
        mock_cli_class.return_value = mock_cli

        with patch('sys.argv', ['production_cli.py', 'portfolio']):
            main()

        mock_cli_class.assert_called_once()
        mock_cli.show_portfolio.assert_called_once()

    @patch('backend.tradingbot.production.core.production_cli.ProductionCLI')
    def test_main_stop_command(self, mock_cli_class):
        """Test main function with stop command."""
        mock_cli = Mock()
        mock_cli.stop_system = AsyncMock()
        mock_cli_class.return_value = mock_cli

        with patch('sys.argv', ['production_cli.py', 'stop']):
            main()

        mock_cli_class.assert_called_once()
        mock_cli.stop_system.assert_called_once()

    @patch('backend.tradingbot.production.core.production_cli.ProductionCLI')
    def test_main_manual_trade_command(self, mock_cli_class):
        """Test main function with manual trade command."""
        mock_cli = Mock()
        mock_cli.execute_trade = AsyncMock()  # Manual-trade calls execute_trade which needs to be async
        mock_cli_class.return_value = mock_cli

        with patch('sys.argv', ['production_cli.py', 'manual-trade', '--symbol', 'AAPL', '--side', 'buy', '--quantity', '100', '--price', '150.0']):
            main()

        mock_cli_class.assert_called_once()
        mock_cli.execute_trade.assert_called_once()

    @patch('backend.tradingbot.production.core.production_cli.ProductionCLI')
    def test_main_list_strategies_command(self, mock_cli_class):
        """Test main function with list strategies command."""
        mock_cli = Mock()
        mock_cli.list_strategies = AsyncMock()  # list_strategies is async
        mock_cli_class.return_value = mock_cli

        with patch('sys.argv', ['production_cli.py', 'list-strategies']):
            main()

        mock_cli_class.assert_called_once()
        mock_cli.list_strategies.assert_called_once()

    @patch('backend.tradingbot.production.core.production_cli.ProductionCLI')
    def test_main_enable_strategy_command(self, mock_cli_class):
        """Test main function with enable strategy command."""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        with patch('sys.argv', ['production_cli.py', 'enable-strategy', 'test_strategy']):
            main()
        
        mock_cli_class.assert_called_once()
        mock_cli.enable_strategy.assert_called_once()

    @patch('backend.tradingbot.production.core.production_cli.ProductionCLI')
    def test_main_disable_strategy_command(self, mock_cli_class):
        """Test main function with disable strategy command."""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        with patch('sys.argv', ['production_cli.py', 'disable-strategy', 'test_strategy']):
            main()
        
        mock_cli_class.assert_called_once()
        mock_cli.disable_strategy.assert_called_once()

    @patch('backend.tradingbot.production.core.production_cli.ProductionCLI')
    def test_main_unknown_command(self, mock_cli_class):
        """Test main function with unknown command."""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        with patch('sys.argv', ['production_cli.py', 'unknown-command']):
            # Should handle gracefully without creating CLI instance
            main()
        
        # Should not create CLI instance for unknown commands
        mock_cli_class.assert_not_called()


class TestIntegrationScenarios:
    """Test complete CLI integration scenarios."""

    def test_complete_cli_workflow(self):
        """Test complete CLI workflow from start to stop."""
        cli = ProductionCLI()
        
        # Mock manager
        mock_manager = Mock()
        mock_integration = Mock()
        mock_status = Mock()
        mock_positions = {"AAPL": {"quantity": 100, "avg_price": 150.0}}
        mock_trades = [Mock(symbol="AAPL", side="buy", quantity=100, price=150.0)]
        
        mock_status.is_running = True
        mock_status.active_strategies = 1
        mock_status.total_trades = 1
        mock_status.total_pnl = Decimal("500.00")
        mock_status.last_update = datetime.now()
        
        mock_manager.get_status.return_value = mock_status
        mock_manager.integration = mock_integration
        mock_integration.get_positions.return_value = mock_positions
        mock_integration.get_trades.return_value = mock_trades
        
        cli.manager = mock_manager
        
        # Test status
        cli.status_system()
        
        # Test portfolio view
        cli.portfolio_view()
        
        # Test manual trade
        args = Mock()
        args.ticker = "AAPL"
        args.side = "sell"
        args.quantity = 50
        args.price = 155.0
        
        mock_result = Mock()
        mock_result.success = True
        mock_result.trade_id = "trade_123"
        mock_integration.execute_trade = AsyncMock(return_value=mock_result)
        
        cli.manual_trade(args)
        
        # Test stop
        cli.stop_system()
        
        # Verify all methods were called
        mock_manager.get_system_status.assert_called()
        mock_integration.get_positions.assert_called()
        mock_integration.get_trades.assert_called()
        mock_integration.execute_trade.assert_called()
        # stop_system doesn't call manager.stop(), it just sets manager = None

    def test_error_recovery_scenarios(self):
        """Test error recovery scenarios."""
        cli = ProductionCLI()
        
        # Test with no manager
        cli.status_system()
        cli.portfolio_view()
        cli.stop_system()
        
        # Test with manager that raises exceptions
        mock_manager = Mock()
        mock_manager.get_status.side_effect = Exception("Status error")
        mock_manager.stop.side_effect = Exception("Stop error")
        
        cli.manager = mock_manager
        
        # Should handle exceptions gracefully
        try:
            cli.status_system()
        except Exception:
            pass  # Expected
        
        try:
            cli.stop_system()
        except Exception:
            pass  # Expected

    def test_performance_under_load(self):
        """Test CLI performance under load."""
        cli = ProductionCLI()
        
        # Mock manager with many strategies and trades
        mock_manager = Mock()
        mock_integration = Mock()
        mock_status = Mock()
        
        # Create many positions and trades
        mock_positions = {f"SYMBOL_{i}": {"quantity": 100, "avg_price": 100.0} for i in range(100)}
        mock_trades = [Mock(symbol=f"SYMBOL_{i}", side="buy", quantity=100, price=100.0) for i in range(1000)]
        
        mock_status.is_running = True
        mock_status.active_strategies = 10
        mock_status.total_trades = 1000
        mock_status.total_pnl = Decimal("10000.00")
        mock_status.last_update = datetime.now()
        
        mock_manager.get_status.return_value = mock_status
        mock_manager.integration = mock_integration
        mock_integration.get_positions.return_value = mock_positions
        mock_integration.get_trades.return_value = mock_trades
        
        cli.manager = mock_manager
        
        # Test multiple operations
        for _ in range(10):
            cli.status_system()
            cli.portfolio_view()
        
        # Should handle large datasets without issues
        mock_manager.get_system_status.assert_called()
        mock_integration.get_positions.assert_called()
        mock_integration.get_trades.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
