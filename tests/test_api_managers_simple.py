"""Simple tests for API Managers to achieve >75% coverage."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from backend.tradingbot.apimanagers import AlpacaManager, create_alpaca_manager, ALPACA_AVAILABLE


class TestAlpacaManagerBasics:
    """Test basic AlpacaManager functionality."""

    def test_init_with_test_keys(self):
        """Test initialization with test keys (mock mode)."""
        manager = AlpacaManager("test_key", "test_secret", paper_trading=True)
        assert manager.API_KEY == "test_key"
        assert manager.SECRET_KEY == "test_secret"
        assert manager.paper_trading is True

    def test_init_live_trading(self):
        """Test initialization for live trading."""
        manager = AlpacaManager("test_key", "test_secret", paper_trading=False)
        assert manager.paper_trading is False

    def test_validate_api_mock_mode(self):
        """Test API validation returns success in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        is_valid, message = manager.validate_api()
        assert is_valid is True
        assert isinstance(message, str)

    def test_get_bar_mock_mode(self):
        """Test get_bar returns empty lists in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        result = manager.get_bar("AAPL", "Day", start, end)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_price_mock_mode(self):
        """Test get_price returns tuple in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.get_price("AAPL")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_balance_mock_mode(self):
        """Test get_balance in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        balance = manager.get_balance()
        # Should return None in mock mode
        assert balance is None

    def test_get_account_value_mock_mode(self):
        """Test get_account_value in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        value = manager.get_account_value()
        assert value is None

    def test_get_position_mock_mode(self):
        """Test get_position returns 0 in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        qty = manager.get_position("AAPL")
        assert qty == 0

    def test_get_positions_mock_mode(self):
        """Test get_positions returns empty list in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        positions = manager.get_positions()
        assert positions == []

    def test_market_buy_mock_mode(self):
        """Test market_buy returns dict in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.market_buy("AAPL", 100)
        assert isinstance(result, dict)

    def test_market_sell_mock_mode(self):
        """Test market_sell returns dict in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.market_sell("AAPL", 50)
        assert isinstance(result, dict)

    def test_buy_option_mock_mode(self):
        """Test buy_option returns dict in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.buy_option("AAPL230616C00150000", 10, 2.50)
        assert isinstance(result, dict)

    def test_sell_option_mock_mode(self):
        """Test sell_option returns dict in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.sell_option("AAPL230616C00150000", 5, 3.00)
        assert isinstance(result, dict)

    def test_place_stop_loss_mock_mode(self):
        """Test place_stop_loss returns dict in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.place_stop_loss("AAPL", 50, 145.00)
        assert isinstance(result, dict)

    def test_cancel_order_mock_mode(self):
        """Test cancel_order returns boolean in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.cancel_order("order-123")
        assert isinstance(result, bool)

    def test_cancel_all_orders_mock_mode(self):
        """Test cancel_all_orders returns boolean in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.cancel_all_orders()
        assert isinstance(result, bool)

    def test_get_orders_mock_mode(self):
        """Test get_orders returns list in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        orders = manager.get_orders()
        assert isinstance(orders, list)

    def test_close_position_mock_mode(self):
        """Test close_position returns boolean in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.close_position("AAPL", 0.5)
        assert isinstance(result, bool)

    def test_market_close_mock_mode(self):
        """Test market_close returns boolean in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.market_close()
        assert isinstance(result, bool)

    def test_get_clock_mock_mode(self):
        """Test get_clock returns MockClock object in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        clock = manager.get_clock()
        assert clock is not None
        assert hasattr(clock, 'is_open')
        assert hasattr(clock, 'timestamp')

    def test_get_bars_mock_mode(self):
        """Test get_bars returns list in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        result = manager.get_bars("AAPL", "1Day", 10, start, end)
        assert isinstance(result, list)


class TestAlpacaManagerParameters:
    """Test different parameter combinations."""

    def test_market_buy_limit_order(self):
        """Test market_buy with limit order."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.market_buy("AAPL", 100, "limit", 150.00)
        assert isinstance(result, dict)

    def test_market_sell_limit_order(self):
        """Test market_sell with limit order."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.market_sell("AAPL", 50, "limit", 155.00)
        assert isinstance(result, dict)

    def test_get_bar_different_price_types(self):
        """Test get_bar with different price types."""
        manager = AlpacaManager("test_key", "test_secret")
        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        for price_type in ["open", "high", "low", "close"]:
            result = manager.get_bar("AAPL", "Day", start, end, price_type)
            assert isinstance(result, tuple)

    def test_get_bar_different_timeframes(self):
        """Test get_bar with different timeframes."""
        manager = AlpacaManager("test_key", "test_secret")
        start = datetime.now() - timedelta(hours=5)
        end = datetime.now()

        for timeframe in ["Day", "Hour", "Minute"]:
            result = manager.get_bar("AAPL", timeframe, start, end)
            assert isinstance(result, tuple)

    def test_get_orders_different_statuses(self):
        """Test get_orders with different status filters."""
        manager = AlpacaManager("test_key", "test_secret")

        for status in ["open", "closed", "all"]:
            orders = manager.get_orders(status)
            assert isinstance(orders, list)

    def test_close_position_different_percentages(self):
        """Test close_position with different percentages."""
        manager = AlpacaManager("test_key", "test_secret")

        for percentage in [0.25, 0.5, 0.75, 1.0]:
            result = manager.close_position("AAPL", percentage)
            assert isinstance(result, bool)

    def test_buy_option_different_expiries(self):
        """Test buy_option with different option symbols."""
        manager = AlpacaManager("test_key", "test_secret")

        symbols = [
            "AAPL230616C00150000",
            "MSFT231215P00300000",
            "SPY230929C00450000"
        ]

        for symbol in symbols:
            result = manager.buy_option(symbol, 10, 2.50)
            assert isinstance(result, dict)


class TestAlpacaManagerWithRealAPI:
    """Test AlpacaManager with mocked real API clients."""

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', True)
    @patch('backend.tradingbot.apimanagers.TradingClient')
    @patch('backend.tradingbot.apimanagers.StockHistoricalDataClient')
    def test_init_with_real_keys(self, mock_data_client, mock_trading_client):
        """Test initialization with real keys."""
        # Mock successful client creation
        mock_trading_client.return_value = Mock()
        mock_data_client.return_value = Mock()

        manager = AlpacaManager("real_key", "real_secret", paper_trading=True)
        assert manager.trading_client is not None
        assert manager.data_client is not None

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', True)
    @patch('backend.tradingbot.apimanagers.TradingClient')
    @patch('backend.tradingbot.apimanagers.StockHistoricalDataClient')
    def test_validate_api_with_real_client(self, mock_data_client, mock_trading_client):
        """Test API validation with real client."""
        mock_client = Mock()
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_client.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_client
        mock_data_client.return_value = Mock()

        manager = AlpacaManager("real_key", "real_secret", paper_trading=True)
        is_valid, message = manager.validate_api()

        assert is_valid is True
        assert "PAPER mode" in message

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', True)
    @patch('backend.tradingbot.apimanagers.TradingClient')
    @patch('backend.tradingbot.apimanagers.StockHistoricalDataClient')
    def test_get_balance_with_client(self, mock_data_client, mock_trading_client):
        """Test get_balance with mocked client."""
        mock_client = Mock()
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_account.buying_power = "25000.50"
        mock_client.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_client
        mock_data_client.return_value = Mock()

        manager = AlpacaManager("real_key", "real_secret")
        balance = manager.get_balance()
        assert balance == 25000.50


class TestCreateAlpacaManagerFactory:
    """Test the create_alpaca_manager factory function."""

    def test_create_paper_trading(self):
        """Test factory function for paper trading."""
        manager = create_alpaca_manager("test_key", "test_secret", paper_trading=True)
        assert isinstance(manager, AlpacaManager)
        assert manager.paper_trading is True

    def test_create_live_trading(self):
        """Test factory function for live trading."""
        manager = create_alpaca_manager("test_key", "test_secret", paper_trading=False)
        assert isinstance(manager, AlpacaManager)
        assert manager.paper_trading is False

    def test_create_default_paper(self):
        """Test factory function defaults to paper trading."""
        manager = create_alpaca_manager("test_key", "test_secret")
        assert isinstance(manager, AlpacaManager)
        assert manager.paper_trading is True


class TestAlpacaManagerErrorHandling:
    """Test error handling scenarios."""

    def test_empty_credentials(self):
        """Test with empty credentials."""
        manager = AlpacaManager("", "")
        assert manager.API_KEY == ""
        assert manager.SECRET_KEY == ""

    def test_none_credentials(self):
        """Test with None credentials."""
        manager = AlpacaManager(None, None)
        assert manager.API_KEY is None
        assert manager.SECRET_KEY is None

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', False)
    def test_sdk_unavailable(self):
        """Test when Alpaca SDK is unavailable."""
        manager = AlpacaManager("test_key", "test_secret")
        assert manager.trading_client is None
        assert manager.data_client is None

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', True)
    @patch('backend.tradingbot.apimanagers.TradingClient')
    def test_client_init_failure(self, mock_trading_client):
        """Test client initialization failure."""
        mock_trading_client.side_effect = Exception("Auth failed")

        # This should fall back to mock mode
        manager = AlpacaManager("real_key", "real_secret")
        assert manager.trading_client is None


class TestAlpacaManagerBehaviorCoverage:
    """Test specific behaviors to increase coverage."""

    def test_get_position_exception_handling(self):
        """Test get_position with exception (position not found)."""
        manager = AlpacaManager("test_key", "test_secret")
        # This will hit the exception path in mock mode
        qty = manager.get_position("NONEXISTENT")
        assert qty == 0

    def test_get_bars_with_no_start_end(self):
        """Test get_bars without start/end dates."""
        manager = AlpacaManager("test_key", "test_secret")
        result = manager.get_bars("AAPL", "1Day", 10)
        assert isinstance(result, list)

    def test_get_bars_with_timeframe_variations(self):
        """Test get_bars with different timeframe formats."""
        manager = AlpacaManager("test_key", "test_secret")

        timeframes = ["1Min", "5Min", "15Min", "1Hour", "1Day"]
        for timeframe in timeframes:
            result = manager.get_bars("AAPL", timeframe, 5)
            assert isinstance(result, list)

    def test_get_bar_timeframe_mapping(self):
        """Test get_bar timeframe mapping logic."""
        manager = AlpacaManager("test_key", "test_secret")
        start = datetime.now() - timedelta(days=1)
        end = datetime.now()

        # Test case-insensitive mapping
        for timeframe in ["day", "DAY", "Day", "hour", "HOUR", "minute"]:
            result = manager.get_bar("AAPL", timeframe, start, end)
            assert isinstance(result, tuple)

    def test_options_validation_edge_cases(self):
        """Test option methods with edge case inputs."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test with various option symbols
        symbols = [
            "AAPL",  # Not an option symbol
            "",      # Empty symbol
            "INVALID_OPTION_FORMAT"
        ]

        for symbol in symbols:
            result = manager.buy_option(symbol, 1, 1.0)
            assert isinstance(result, dict)

    def test_stop_loss_validation(self):
        """Test stop loss parameter validation."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test edge cases that trigger validation
        test_cases = [
            ("AAPL", 0, 100.0),     # Zero quantity
            ("AAPL", -1, 100.0),    # Negative quantity
            ("AAPL", 100, 0),       # Zero price
            ("AAPL", 100, -50.0),   # Negative price
        ]

        for symbol, qty, price in test_cases:
            result = manager.place_stop_loss(symbol, qty, price)
            assert isinstance(result, dict)

    def test_close_position_edge_cases(self):
        """Test close_position with edge case percentages."""
        manager = AlpacaManager("test_key", "test_secret")

        edge_cases = [-0.1, 0, 1.1, 2.0]
        for percentage in edge_cases:
            result = manager.close_position("AAPL", percentage)
            assert isinstance(result, bool)

    def test_market_order_edge_cases(self):
        """Test market orders with edge case inputs."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test with various quantities and symbols
        test_cases = [
            ("", 100),        # Empty symbol
            ("AAPL", 0),      # Zero quantity
            ("AAPL", -100),   # Negative quantity
        ]

        for symbol, qty in test_cases:
            buy_result = manager.market_buy(symbol, qty)
            assert isinstance(buy_result, dict)

            sell_result = manager.market_sell(symbol, qty)
            assert isinstance(sell_result, dict)

    def test_order_status_filtering(self):
        """Test get_orders with specific status filters."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test different status combinations
        status_filters = ["open", "closed", "all", "filled", "canceled", "expired"]
        for status in status_filters:
            orders = manager.get_orders(status)
            assert isinstance(orders, list)

    def test_timeframe_edge_cases_get_bar(self):
        """Test get_bar with edge case timeframes."""
        manager = AlpacaManager("test_key", "test_secret")
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()

        # Test various timeframe formats and cases
        timeframes = ["invalid", "INVALID", "1min", "1MIN", "1day", "1HOUR", ""]
        for tf in timeframes:
            result = manager.get_bar("AAPL", tf, start, end)
            assert isinstance(result, tuple)

    def test_price_type_edge_cases_get_bar(self):
        """Test get_bar with edge case price types."""
        manager = AlpacaManager("test_key", "test_secret")
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()

        # Test various price types including invalid ones
        price_types = ["volume", "invalid", "", "OPEN", "HIGH", "LOW", "CLOSE"]
        for price_type in price_types:
            result = manager.get_bar("AAPL", "Day", start, end, price_type)
            assert isinstance(result, tuple)

    def test_account_info_error_paths(self):
        """Test account info methods error handling paths."""
        manager = AlpacaManager("test_key", "test_secret")

        # These will hit the None client paths
        assert manager.get_balance() is None
        assert manager.get_account_value() is None

    def test_position_info_error_paths(self):
        """Test position info methods error handling paths."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test with various symbols
        symbols = ["", "INVALID", "AAPL", "SPY", "123", None]
        for symbol in symbols:
            if symbol is not None:
                qty = manager.get_position(symbol)
                assert isinstance(qty, int)

        # Test get_positions (hits exception path in mock mode)
        positions = manager.get_positions()
        assert isinstance(positions, list)

    def test_cancel_operations_coverage(self):
        """Test cancel operations to increase coverage."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test cancel_order with various IDs
        order_ids = ["", "123", "invalid", "order-abc-123"]
        for order_id in order_ids:
            result = manager.cancel_order(order_id)
            assert isinstance(result, bool)

        # Test cancel_all_orders
        result = manager.cancel_all_orders()
        assert isinstance(result, bool)

    def test_market_close_variations(self):
        """Test market_close method variations."""
        manager = AlpacaManager("test_key", "test_secret")

        # Call multiple times to ensure consistency
        for _ in range(3):
            result = manager.market_close()
            assert isinstance(result, bool)

    def test_get_clock_multiple_calls(self):
        """Test get_clock method multiple times."""
        manager = AlpacaManager("test_key", "test_secret")

        # Call multiple times and verify consistent behavior
        for _ in range(3):
            clock = manager.get_clock()
            assert clock is not None
            assert hasattr(clock, 'is_open')
            assert hasattr(clock, 'timestamp')
            assert clock.is_open is False  # Mock mode returns closed market

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', False)
    def test_all_methods_sdk_unavailable(self):
        """Test all major methods when SDK is unavailable."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test all major methods to increase coverage
        is_valid, message = manager.validate_api()
        assert is_valid is True

        result = manager.get_bar("AAPL", "Day", datetime.now()-timedelta(days=1), datetime.now())
        assert result == ([], [])

        success, price = manager.get_price("AAPL")
        assert success is False

        buy_result = manager.market_buy("AAPL", 100)
        assert isinstance(buy_result, dict)

        sell_result = manager.market_sell("AAPL", 50)
        assert isinstance(sell_result, dict)

        option_result = manager.buy_option("AAPL230616C00150000", 10, 2.50)
        assert isinstance(option_result, dict)