"""Focused tests for API Managers to achieve >75% coverage."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from backend.tradingbot.apimanagers import AlpacaManager, create_alpaca_manager, ALPACA_AVAILABLE


class TestAlpacaManagerMockMode:
    """Test AlpacaManager in mock mode (with test keys)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AlpacaManager("test_key", "test_secret", paper_trading=True)

    def test_init_mock_mode(self):
        """Test initialization in mock mode."""
        assert self.manager.API_KEY == "test_key"
        assert self.manager.SECRET_KEY == "test_secret"
        assert self.manager.paper_trading is True
        assert self.manager.trading_client is None
        assert self.manager.data_client is None

    def test_validate_api_mock_mode(self):
        """Test API validation in mock mode."""
        is_valid, message = self.manager.validate_api()
        assert is_valid is True
        assert "mock mode" in message.lower()

    def test_get_bar_mock_mode(self):
        """Test get_bar in mock mode."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        result = self.manager.get_bar("AAPL", "Day", start_date, end_date)
        # get_bar returns a tuple of (prices, timestamps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # prices
        assert isinstance(result[1], list)  # timestamps

    def test_get_price_mock_mode(self):
        """Test get_price in mock mode returns (False, 0.0)."""
        success, price = self.manager.get_price("AAPL")
        assert success is False
        assert price == 0.0

    def test_get_balance_mock_mode(self):
        """Test get_balance in mock mode."""
        balance = self.manager.get_balance()
        assert balance is None

    def test_get_account_value_mock_mode(self):
        """Test get_account_value in mock mode."""
        value = self.manager.get_account_value()
        assert value is None

    def test_get_position_mock_mode(self):
        """Test get_position in mock mode."""
        qty = self.manager.get_position("AAPL")
        assert qty == 0

    def test_get_positions_mock_mode(self):
        """Test get_positions in mock mode."""
        positions = self.manager.get_positions()
        assert positions == []

    def test_market_buy_mock_mode(self):
        """Test market_buy in mock mode."""
        result = self.manager.market_buy("AAPL", 100)
        assert "error" in result
        assert "Failed to place buy order" in result["error"]

    def test_market_sell_mock_mode(self):
        """Test market_sell in mock mode."""
        result = self.manager.market_sell("AAPL", 50)
        assert isinstance(result, dict)
        assert "error" in result
        assert "Failed to place sell order" in result["error"]

    def test_buy_option_mock_mode(self):
        """Test buy_option in mock mode."""
        result = self.manager.buy_option("AAPL", 10, "call", 150.0, "2023-06-16", 2.50)
        assert isinstance(result, dict)
        assert "error" in result
        assert "Failed to buy option" in result["error"]

    def test_sell_option_mock_mode(self):
        """Test sell_option in mock mode."""
        result = self.manager.sell_option("AAPL", 5, "call", 150.0, "2023-06-16", 3.00)
        assert isinstance(result, dict)
        assert "error" in result
        assert "Failed to sell option" in result["error"]

    def test_place_stop_loss_mock_mode(self):
        """Test place_stop_loss in mock mode."""
        result = self.manager.place_stop_loss("AAPL", 50, 145.00)
        assert isinstance(result, dict)
        assert "error" in result
        assert "Failed to place stop loss" in result["error"]

    def test_cancel_order_mock_mode(self):
        """Test cancel_order in mock mode."""
        success = self.manager.cancel_order("order-123")
        assert success is False

    def test_cancel_all_orders_mock_mode(self):
        """Test cancel_all_orders in mock mode."""
        success = self.manager.cancel_all_orders()
        assert success is False

    def test_get_orders_mock_mode(self):
        """Test get_orders in mock mode."""
        orders = self.manager.get_orders()
        assert orders == []

    def test_close_position_mock_mode(self):
        """Test close_position in mock mode."""
        success = self.manager.close_position("AAPL", 0.5)
        assert success is False

    def test_market_close_mock_mode(self):
        """Test market_close in mock mode."""
        is_closed = self.manager.market_close()
        assert is_closed is True  # Mock mode returns True

    def test_get_clock_mock_mode(self):
        """Test get_clock in mock mode."""
        clock = self.manager.get_clock()
        # get_clock returns a MockClock object in mock mode
        assert clock is not None
        assert hasattr(clock, 'is_open')

    def test_get_bars_mock_mode(self):
        """Test get_bars in mock mode."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        bars = self.manager.get_bars("AAPL", "Day", start_date, end_date)
        # get_bars returns a list of dictionaries
        assert isinstance(bars, list)
        assert len(bars) == 0  # Empty list in mock mode


class TestAlpacaManagerEdgeCases:
    """Test edge cases and validation logic."""

    def test_empty_credentials(self):
        """Test initialization with empty credentials."""
        manager = AlpacaManager("", "", paper_trading=True)
        assert manager.API_KEY == ""
        assert manager.SECRET_KEY == ""
        assert manager.trading_client is None

    def test_none_credentials(self):
        """Test initialization with None credentials."""
        manager = AlpacaManager(None, None, paper_trading=True)
        assert manager.API_KEY is None
        assert manager.SECRET_KEY is None
        assert manager.trading_client is None

    def test_invalid_quantity_market_buy(self):
        """Test market_buy with invalid quantities."""
        manager = AlpacaManager("test_key", "test_secret")

        # Zero quantity
        result = manager.market_buy("AAPL", 0)
        assert isinstance(result, dict)
        assert "error" in result

        # Negative quantity
        result = manager.market_buy("AAPL", -100)
        assert isinstance(result, dict)
        assert "error" in result

    def test_invalid_quantity_market_sell(self):
        """Test market_sell with invalid quantities."""
        manager = AlpacaManager("test_key", "test_secret")

        # Zero quantity
        result = manager.market_sell("AAPL", 0)
        assert isinstance(result, dict)
        assert "error" in result

        # Negative quantity
        result = manager.market_sell("AAPL", -50)
        assert isinstance(result, dict)
        assert "error" in result

    def test_invalid_percentage_close_position(self):
        """Test close_position with invalid percentages."""
        manager = AlpacaManager("test_key", "test_secret")

        # Negative percentage
        success = manager.close_position("AAPL", -0.5)
        assert success is False

        # Percentage > 1.0
        success = manager.close_position("AAPL", 1.5)
        assert success is False

    def test_invalid_option_parameters(self):
        """Test option trading with invalid parameters."""
        manager = AlpacaManager("test_key", "test_secret")

        # Zero contracts
        result = manager.buy_option("AAPL230616C00150000", 0, 2.50)
        assert isinstance(result, dict)
        assert "error" in result

        # Negative price
        result = manager.buy_option("AAPL230616C00150000", 10, -1.0)
        assert isinstance(result, dict)
        assert "error" in result

        # Zero contracts for sell
        result = manager.sell_option("AAPL230616C00150000", 0, 3.00)
        assert isinstance(result, dict)
        assert "error" in result

    def test_invalid_stop_loss_parameters(self):
        """Test stop loss with invalid parameters."""
        manager = AlpacaManager("test_key", "test_secret")

        # Zero quantity
        result = manager.place_stop_loss("AAPL", 0, 145.00)
        assert isinstance(result, dict)
        assert "error" in result

        # Negative stop price
        result = manager.place_stop_loss("AAPL", 50, -10.0)
        assert isinstance(result, dict)
        assert "error" in result


class TestAlpacaManagerWithRealClients:
    """Test AlpacaManager with real API keys (mocked clients)."""

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', True)
    @patch('backend.tradingbot.apimanagers.TradingClient')
    @patch('backend.tradingbot.apimanagers.StockHistoricalDataClient')
    def test_real_keys_success_init(self, mock_data_client, mock_trading_client):
        """Test initialization with real keys and successful client creation."""
        mock_trading_client.return_value = Mock()
        mock_data_client.return_value = Mock()

        manager = AlpacaManager("real_key", "real_secret", paper_trading=True)

        assert manager.API_KEY == "real_key"
        assert manager.SECRET_KEY == "real_secret"
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
        assert "ACTIVE" in message

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', True)
    @patch('backend.tradingbot.apimanagers.TradingClient')
    @patch('backend.tradingbot.apimanagers.StockHistoricalDataClient')
    def test_get_balance_with_real_client(self, mock_data_client, mock_trading_client):
        """Test get_balance with real client."""
        mock_client = Mock()
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_account.buying_power = "25000.50"
        mock_client.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_client
        mock_data_client.return_value = Mock()

        manager = AlpacaManager("real_key", "real_secret", paper_trading=True)
        balance = manager.get_balance()

        assert balance == 25000.50

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', True)
    @patch('backend.tradingbot.apimanagers.TradingClient')
    @patch('backend.tradingbot.apimanagers.StockHistoricalDataClient')
    def test_get_account_value_with_real_client(self, mock_data_client, mock_trading_client):
        """Test get_account_value with real client."""
        mock_client = Mock()
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_account.portfolio_value = "50000.75"  # API manager expects portfolio_value, not equity
        mock_client.get_account.return_value = mock_account
        mock_trading_client.return_value = mock_client
        mock_data_client.return_value = Mock()

        manager = AlpacaManager("real_key", "real_secret", paper_trading=True)
        value = manager.get_account_value()

        assert value == 50000.75

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', True)
    @patch('backend.tradingbot.apimanagers.TradingClient')
    @patch('backend.tradingbot.apimanagers.StockHistoricalDataClient')
    def test_get_position_with_real_client(self, mock_data_client, mock_trading_client):
        """Test get_position with real client."""
        mock_client = Mock()
        mock_account = Mock()
        mock_account.status = "ACTIVE"
        mock_position = Mock()
        mock_position.qty = "100"
        mock_client.get_account.return_value = mock_account
        mock_client.get_open_position.return_value = mock_position
        mock_trading_client.return_value = mock_client
        mock_data_client.return_value = Mock()

        manager = AlpacaManager("real_key", "real_secret", paper_trading=True)
        qty = manager.get_position("AAPL")

        assert qty == 100

    def test_get_price_with_real_client(self):
        """Test get_price with invalid credentials returns (False, 0.0)."""
        manager = AlpacaManager("invalid_key", "invalid_secret", paper_trading=True)
        success, price = manager.get_price("AAPL")

        # Should fail with invalid credentials — returns float 0.0
        assert success is False
        assert price == 0.0


class TestCreateAlpacaManagerFactory:
    """Test the create_alpaca_manager factory function."""

    def test_create_alpaca_manager_paper(self):
        """Test creating AlpacaManager for paper trading."""
        manager = create_alpaca_manager("test_key", "test_secret", paper=True)

        assert isinstance(manager, AlpacaManager)
        assert manager.API_KEY == "test_key"
        assert manager.SECRET_KEY == "test_secret"
        assert manager.paper_trading is True

    def test_create_alpaca_manager_live(self):
        """Test creating AlpacaManager for live trading."""
        manager = create_alpaca_manager("test_key", "test_secret", paper=False)

        assert isinstance(manager, AlpacaManager)
        assert manager.API_KEY == "test_key"
        assert manager.SECRET_KEY == "test_secret"
        assert manager.paper_trading is False

    def test_create_alpaca_manager_default(self):
        """Test creating AlpacaManager with default paper trading."""
        manager = create_alpaca_manager("test_key", "test_secret")

        assert isinstance(manager, AlpacaManager)
        assert manager.paper_trading is True


class TestAlpacaManagerErrorHandling:
    """Test error handling scenarios."""

    def test_get_balance_exception_handling(self):
        """Test get_balance exception handling in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        # In mock mode, should return None
        balance = manager.get_balance()
        assert balance is None

    def test_get_position_not_found(self):
        """Test get_position when position not found in mock mode."""
        manager = AlpacaManager("test_key", "test_secret")
        qty = manager.get_position("NONEXISTENT")
        assert qty == 0

    def test_get_bars_invalid_timeframe(self):
        """Test get_bars with invalid timeframe."""
        manager = AlpacaManager("test_key", "test_secret")
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        # Invalid timeframe should return empty lists
        result = manager.get_bars("AAPL", "InvalidTimeFrame", start_date, end_date)
        assert result == []

    def test_market_buy_validation_errors(self):
        """Test market_buy input validation."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test with empty symbol
        result = manager.market_buy("", 100)
        assert isinstance(result, dict)
        assert "error" in result  # Error response indicates failure

        # Test with None symbol
        result = manager.market_buy(None, 100)
        assert isinstance(result, dict)
        assert "error" in result  # Error response indicates failure

    def test_market_sell_validation_errors(self):
        """Test market_sell input validation."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test with empty symbol
        result = manager.market_sell("", 50)
        assert isinstance(result, dict)
        assert "error" in result  # Error response indicates failure

    def test_close_position_validation_errors(self):
        """Test close_position input validation."""
        manager = AlpacaManager("test_key", "test_secret")

        # Test with empty symbol
        success = manager.close_position("", 0.5)
        assert success is False

        # Test with None symbol
        success = manager.close_position(None, 0.5)
        assert success is False


class TestAlpacaManagerPriceTypes:
    """Test different price type handling in get_bar."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AlpacaManager("test_key", "test_secret")
        self.start_date = datetime.now() - timedelta(days=7)
        self.end_date = datetime.now()

    def test_get_bar_open_price(self):
        """Test get_bar with open price type."""
        result = self.manager.get_bar("AAPL", "Day", self.start_date, self.end_date, "open")
        # get_bar returns a tuple of (prices, timestamps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # prices
        assert isinstance(result[1], list)  # timestamps

    def test_get_bar_high_price(self):
        """Test get_bar with high price type."""
        result = self.manager.get_bar("AAPL", "Day", self.start_date, self.end_date, "high")
        # get_bar returns a tuple of (prices, timestamps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # prices
        assert isinstance(result[1], list)  # timestamps

    def test_get_bar_low_price(self):
        """Test get_bar with low price type."""
        result = self.manager.get_bar("AAPL", "Day", self.start_date, self.end_date, "low")
        # get_bar returns a tuple of (prices, timestamps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # prices
        assert isinstance(result[1], list)  # timestamps

    def test_get_bar_close_price_default(self):
        """Test get_bar with default close price type."""
        result = self.manager.get_bar("AAPL", "Day", self.start_date, self.end_date)
        # get_bar returns a tuple of (prices, timestamps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # prices
        assert isinstance(result[1], list)  # timestamps

    def test_get_bar_invalid_price_type(self):
        """Test get_bar with invalid price type."""
        result = self.manager.get_bar("AAPL", "Day", self.start_date, self.end_date, "invalid")
        # get_bar returns a tuple of (prices, timestamps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # prices
        assert isinstance(result[1], list)  # timestamps


class TestAlpacaManagerTimeframes:
    """Test different timeframe handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AlpacaManager("test_key", "test_secret")
        self.start_date = datetime.now() - timedelta(days=7)
        self.end_date = datetime.now()

    def test_get_bar_day_timeframe(self):
        """Test get_bar with Day timeframe."""
        result = self.manager.get_bar("AAPL", "Day", self.start_date, self.end_date)
        # get_bar returns a tuple of (prices, timestamps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # prices
        assert isinstance(result[1], list)  # timestamps

    def test_get_bar_hour_timeframe(self):
        """Test get_bar with Hour timeframe."""
        result = self.manager.get_bar("AAPL", "Hour", self.start_date, self.end_date)
        # get_bar returns a tuple of (prices, timestamps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # prices
        assert isinstance(result[1], list)  # timestamps

    def test_get_bar_minute_timeframe(self):
        """Test get_bar with Minute timeframe."""
        result = self.manager.get_bar("AAPL", "Minute", self.start_date, self.end_date)
        # get_bar returns a tuple of (prices, timestamps)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)  # prices
        assert isinstance(result[1], list)  # timestamps

    def test_get_bars_day_timeframe(self):
        """Test get_bars with Day timeframe."""
        result = self.manager.get_bars("AAPL", "Day", self.start_date, self.end_date)
        assert result == []

    def test_get_bars_hour_timeframe(self):
        """Test get_bars with Hour timeframe."""
        result = self.manager.get_bars("AAPL", "Hour", self.start_date, self.end_date)
        assert result == []


class TestAlpacaAvailableFlag:
    """Test ALPACA_AVAILABLE flag behavior."""

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', False)
    def test_sdk_unavailable_init(self):
        """Test initialization when SDK is unavailable."""
        manager = AlpacaManager("test_key", "test_secret")
        assert manager.trading_client is None
        assert manager.data_client is None
        assert manager.alpaca_available is False

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', False)
    def test_sdk_unavailable_validate_api(self):
        """Test validate_api when SDK is unavailable."""
        manager = AlpacaManager("test_key", "test_secret")
        is_valid, message = manager.validate_api()
        assert is_valid is True
        assert "not available" in message

    @patch('backend.tradingbot.apimanagers.ALPACA_AVAILABLE', True)
    def test_sdk_available_flag(self):
        """Test when SDK is marked as available."""
        # Should work normally with test keys (mock mode)
        manager = AlpacaManager("test_key", "test_secret")
        assert manager.alpaca_available is True  # But clients are None due to test keys
        assert manager.trading_client is None
        assert manager.data_client is None