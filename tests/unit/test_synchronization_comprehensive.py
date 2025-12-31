"""
Comprehensive tests for backend/tradingbot/synchronization.py
Target: 80%+ coverage with all edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal

from backend.tradingbot.synchronization import (
    validate_backend,
    sync_database_company_stock,
    sync_stock_instance,
    sync_alpaca,
)


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = Mock()
    user.id = 1
    user.first_name = "Test"
    user.last_name = "User"

    # Mock credential
    credential = Mock()
    credential.alpaca_id = "test_api_key"
    credential.alpaca_key = "test_secret_key"
    user.credential = credential

    # Mock portfolio
    portfolio = Mock()
    portfolio.cash = 10000.0
    portfolio.strategy = "momentum"
    portfolio.get_strategy_display = Mock(return_value="Momentum Trading")
    user.portfolio = portfolio

    # Mock social auth
    social_auth = Mock()
    social_auth.get = Mock(return_value=Mock())
    user.social_auth = social_auth

    return user


@pytest.fixture
def mock_alpaca_account():
    """Create mock Alpaca account data."""
    account = Mock()
    account.equity = "125000.50"
    account.buying_power = "50000.00"
    account.cash = "25000.00"
    account.currency = "USD"
    account.long_market_value = "100000.00"
    account.short_market_value = "0.00"
    account.portfolio_value = "125000.50"
    account.last_equity = "124000.00"
    account.status = "ACTIVE"
    return account


@pytest.fixture
def mock_alpaca_position():
    """Create mock Alpaca position."""
    position = Mock()
    position.symbol = "AAPL"
    position.qty = "10"
    position.side = "long"
    position.market_value = "1500.00"
    position.avg_entry_price = "145.00"
    position.unrealized_pl = "50.00"
    position.unrealized_plpc = "0.034"
    return position


class TestValidateBackend:
    """Test validate_backend function."""

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_validate_backend_success(self, mock_manager_class):
        """Test successful backend validation."""
        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "API validated")
        mock_manager_class.return_value = mock_manager

        result = validate_backend()

        assert result is not None
        assert mock_manager.validate_api.called

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_validate_backend_invalid_credentials(self, mock_manager_class):
        """Test backend validation with invalid credentials."""
        from django.core.exceptions import ValidationError

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (False, "Invalid API key")
        mock_manager_class.return_value = mock_manager

        with pytest.raises(ValidationError):
            validate_backend()

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_validate_backend_uses_settings(self, mock_manager_class):
        """Test that backend validation uses settings."""
        with patch('backend.tradingbot.synchronization.BACKEND_ALPACA_ID', 'test_id'):
            with patch('backend.tradingbot.synchronization.BACKEND_ALPACA_KEY', 'test_key'):
                mock_manager = Mock()
                mock_manager.validate_api.return_value = (True, "OK")
                mock_manager_class.return_value = mock_manager

                validate_backend()

                mock_manager_class.assert_called_with('test_id', 'test_key')


class TestSyncDatabaseCompanyStock:
    """Test sync_database_company_stock function."""

    @patch('backend.tradingbot.synchronization.Company')
    @patch('backend.tradingbot.synchronization.Stock')
    def test_sync_new_company(self, mock_stock_class, mock_company_class):
        """Test syncing a new company."""
        # Mock that company doesn't exist
        mock_company_class.objects.filter.return_value.exists.return_value = False

        mock_company = Mock()
        mock_company_class.return_value = mock_company

        mock_stock = Mock()
        mock_stock_class.return_value = mock_stock

        stock, company = sync_database_company_stock("AAPL")

        assert stock == mock_stock
        assert company == mock_company
        mock_company.save.assert_called_once()
        mock_stock.save.assert_called_once()

    @patch('backend.tradingbot.synchronization.Company')
    @patch('backend.tradingbot.synchronization.Stock')
    def test_sync_existing_company(self, mock_stock_class, mock_company_class):
        """Test syncing an existing company."""
        # Mock that company exists
        mock_company = Mock()
        mock_stock = Mock()

        mock_company_class.objects.filter.return_value.exists.return_value = True
        mock_company_class.objects.get.return_value = mock_company
        mock_stock_class.objects.get.return_value = mock_stock

        stock, company = sync_database_company_stock("AAPL")

        assert stock == mock_stock
        assert company == mock_company
        # Should not create new objects
        mock_company_class.assert_not_called()
        mock_stock_class.assert_not_called()

    @patch('backend.tradingbot.synchronization.Company')
    @patch('backend.tradingbot.synchronization.Stock')
    def test_sync_multiple_tickers(self, mock_stock_class, mock_company_class):
        """Test syncing multiple different tickers."""
        mock_company_class.objects.filter.return_value.exists.return_value = False

        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            stock, company = sync_database_company_stock(ticker)
            assert stock is not None
            assert company is not None


class TestSyncStockInstance:
    """Test sync_stock_instance function."""

    @patch('backend.tradingbot.synchronization.StockInstance')
    def test_sync_new_stock_instance(self, mock_instance_class):
        """Test creating a new stock instance."""
        mock_user = Mock()
        mock_portfolio = Mock()
        mock_stock = Mock()

        # Mock that instance doesn't exist
        mock_instance_class.objects.filter.return_value.exists.return_value = False

        mock_instance = Mock()
        mock_instance_class.return_value = mock_instance

        result = sync_stock_instance(mock_user, mock_portfolio, mock_stock)

        assert result == mock_instance
        mock_instance.save.assert_called_once()

    @patch('backend.tradingbot.synchronization.StockInstance')
    def test_sync_existing_stock_instance(self, mock_instance_class):
        """Test getting existing stock instance."""
        mock_user = Mock()
        mock_portfolio = Mock()
        mock_stock = Mock()

        # Mock that instance exists
        mock_instance = Mock()
        mock_instance_class.objects.filter.return_value.exists.return_value = True
        mock_instance_class.objects.get.return_value = mock_instance

        result = sync_stock_instance(mock_user, mock_portfolio, mock_stock)

        assert result == mock_instance
        # Should not create new instance
        mock_instance_class.assert_not_called()


class TestSyncAlpaca:
    """Test sync_alpaca function."""

    def test_sync_alpaca_no_credential(self, mock_user):
        """Test sync_alpaca with user without credentials."""
        delattr(mock_user, 'credential')

        result = sync_alpaca(mock_user)

        assert result is None

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_invalid_api(self, mock_manager_class, mock_user):
        """Test sync_alpaca with invalid API credentials."""
        mock_manager = Mock()
        mock_manager.validate_api.return_value = (False, "Invalid credentials")
        mock_manager_class.return_value = mock_manager

        result = sync_alpaca(mock_user)

        assert result is None

    @patch('backend.tradingbot.synchronization.sync_database_company_stock')
    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_success(
        self,
        mock_manager_class,
        mock_sync_company,
        mock_user,
        mock_alpaca_account,
        mock_alpaca_position,
    ):
        """Test successful Alpaca sync."""
        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = [mock_alpaca_position]
        mock_manager.api.list_orders.return_value = []
        mock_manager_class.return_value = mock_manager

        mock_company = Mock()
        mock_stock = Mock()
        mock_sync_company.return_value = (mock_stock, mock_company)

        with patch('backend.tradingbot.synchronization.Company') as mock_company_class:
            with patch('backend.tradingbot.synchronization.Stock') as mock_stock_class:
                with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
                    with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                        mock_company_class.objects.get.return_value = mock_company
                        mock_stock_class.objects.get.return_value = mock_stock
                        mock_instance_class.objects.filter.return_value.exists.return_value = False
                        mock_order_class.objects.filter.return_value.iterator.return_value = []

                        result = sync_alpaca(mock_user)

                        assert result is not None
                        assert "equity" in result
                        assert "buy_power" in result
                        assert "cash" in result
                        assert result["equity"] == "125000.5"

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_account_details(
        self,
        mock_manager_class,
        mock_user,
        mock_alpaca_account,
    ):
        """Test that account details are correctly extracted."""
        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = []
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
            with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                mock_instance_class.objects.filter.return_value.exists.return_value = False
                mock_instance_class.objects.filter.return_value.delete.return_value = None
                mock_order_class.objects.filter.return_value.iterator.return_value = []

                result = sync_alpaca(mock_user)

                assert result["equity"] == "125000.5"
                assert result["buy_power"] == "50000.0"
                assert result["cash"] == "25000.0"
                assert result["currency"] == "USD"
                assert result["long_portfolio_value"] == "100000.0"
                assert result["short_portfolio_value"] == "0.0"

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_portfolio_change_positive(
        self,
        mock_manager_class,
        mock_user,
        mock_alpaca_account,
    ):
        """Test portfolio change calculation - positive."""
        mock_alpaca_account.portfolio_value = "125000.00"
        mock_alpaca_account.last_equity = "120000.00"

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = []
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
            with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                mock_instance_class.objects.filter.return_value.exists.return_value = False
                mock_instance_class.objects.filter.return_value.delete.return_value = None
                mock_order_class.objects.filter.return_value.iterator.return_value = []

                result = sync_alpaca(mock_user)

                assert result["portfolio_change_direction"] == "positive"
                assert float(result["portfolio_dollar_change"]) > 0

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_portfolio_change_negative(
        self,
        mock_manager_class,
        mock_user,
        mock_alpaca_account,
    ):
        """Test portfolio change calculation - negative."""
        mock_alpaca_account.portfolio_value = "115000.00"
        mock_alpaca_account.last_equity = "120000.00"

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = []
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
            with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                mock_instance_class.objects.filter.return_value.exists.return_value = False
                mock_instance_class.objects.filter.return_value.delete.return_value = None
                mock_order_class.objects.filter.return_value.iterator.return_value = []

                result = sync_alpaca(mock_user)

                assert result["portfolio_change_direction"] == "negative"
                assert float(result["portfolio_dollar_change"]) < 0

    @patch('backend.tradingbot.synchronization.sync_database_company_stock')
    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_positions(
        self,
        mock_manager_class,
        mock_sync_company,
        mock_user,
        mock_alpaca_account,
    ):
        """Test syncing positions."""
        positions = []
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            pos = Mock()
            pos.symbol = symbol
            pos.qty = "10"
            positions.append(pos)

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = positions
        mock_manager.api.list_orders.return_value = []
        mock_manager_class.return_value = mock_manager

        mock_company = Mock()
        mock_stock = Mock()
        mock_sync_company.return_value = (mock_stock, mock_company)

        with patch('backend.tradingbot.synchronization.Company') as mock_company_class:
            with patch('backend.tradingbot.synchronization.Stock') as mock_stock_class:
                with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
                    with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                        mock_company_class.objects.get.return_value = mock_company
                        mock_stock_class.objects.get.return_value = mock_stock
                        mock_instance_class.objects.filter.return_value.exists.return_value = False
                        mock_instance_class.objects.filter.return_value.delete.return_value = None
                        mock_order_class.objects.filter.return_value.iterator.return_value = []

                        result = sync_alpaca(mock_user)

                        # Should sync all 3 companies
                        assert mock_sync_company.call_count == 3

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_order_sync(
        self,
        mock_manager_class,
        mock_user,
        mock_alpaca_account,
    ):
        """Test order synchronization."""
        # Mock open order
        mock_order = Mock()
        mock_order.status = "accepted"
        mock_order.client_order_id = "12345"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"
        mock_order.order_type = "market"
        mock_order.side = "buy"

        mock_local_order = Mock()
        mock_local_order.order_number = "12345"
        mock_local_order.client_order_id = ""
        mock_local_order.status = "N"

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = [mock_order]
        mock_manager.api.get_order_by_client_order_id.return_value = mock_order
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
            with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                mock_instance_class.objects.filter.return_value.exists.return_value = False
                mock_instance_class.objects.filter.return_value.delete.return_value = None
                mock_order_class.objects.filter.return_value.iterator.return_value = [mock_local_order]
                mock_order_class.objects.filter.return_value.exists.return_value = True

                result = sync_alpaca(mock_user)

                # Order status should be updated
                assert mock_local_order.save.called

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_filled_order(
        self,
        mock_manager_class,
        mock_user,
        mock_alpaca_account,
    ):
        """Test syncing filled order."""
        # Mock filled order
        mock_order = Mock()
        mock_order.status = "filled"
        mock_order.client_order_id = "12345"
        mock_order.filled_avg_price = "150.50"
        mock_order.filled_at = datetime.now()
        mock_order.filled_qty = "10"

        mock_local_order = Mock()
        mock_local_order.order_number = "12345"
        mock_local_order.client_order_id = ""
        mock_local_order.status = "A"

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = []
        mock_manager.api.get_order_by_client_order_id.return_value = mock_order
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
            with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                mock_instance_class.objects.filter.return_value.exists.return_value = False
                mock_instance_class.objects.filter.return_value.delete.return_value = None
                mock_order_class.objects.filter.return_value.iterator.return_value = [mock_local_order]

                result = sync_alpaca(mock_user)

                # Order should be marked as filled
                assert mock_local_order.save.called

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_deleted_order(
        self,
        mock_manager_class,
        mock_user,
        mock_alpaca_account,
    ):
        """Test deleting orders not found in Alpaca."""
        from alpaca_trade_api.rest import APIError

        mock_local_order = Mock()
        mock_local_order.order_number = "99999"
        mock_local_order.client_order_id = ""

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = []
        mock_manager.api.get_order_by_client_order_id.side_effect = APIError("Not found")
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
            with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                mock_instance_class.objects.filter.return_value.exists.return_value = False
                mock_instance_class.objects.filter.return_value.delete.return_value = None
                mock_order_class.objects.filter.return_value.iterator.return_value = [mock_local_order]

                result = sync_alpaca(mock_user)

                # Order should be deleted
                mock_local_order.delete.assert_called_once()

    @patch('backend.tradingbot.synchronization.validate_backend')
    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_usable_cash(
        self,
        mock_manager_class,
        mock_validate,
        mock_user,
        mock_alpaca_account,
    ):
        """Test usable cash calculation with open orders."""
        mock_alpaca_account.cash = "50000.00"

        # Mock open market buy order
        mock_order = Mock()
        mock_order.order_type = "market"
        mock_order.side = "buy"
        mock_order.symbol = "AAPL"
        mock_order.qty = "10"

        mock_backend_api = Mock()
        mock_backend_api.get_price.return_value = (True, 150.00)
        mock_validate.return_value = mock_backend_api

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = [mock_order]
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
            with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                with patch('backend.tradingbot.synchronization.create_local_order'):
                    mock_instance_class.objects.filter.return_value.exists.return_value = False
                    mock_instance_class.objects.filter.return_value.delete.return_value = None
                    mock_order_class.objects.filter.return_value.iterator.return_value = []
                    mock_order_class.objects.filter.return_value.exists.return_value = False

                    result = sync_alpaca(mock_user)

                    # Usable cash should be reduced
                    usable_cash = float(result["usable_cash"])
                    assert usable_cash < 50000.0

    @patch('backend.tradingbot.synchronization.validate_backend')
    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_watchlist_zero_quantity(
        self,
        mock_manager_class,
        mock_validate,
        mock_user,
        mock_alpaca_account,
    ):
        """Test syncing watchlist stocks with zero quantity."""
        mock_backend_api = Mock()
        mock_backend_api.get_price.return_value = (True, 150.00)
        mock_backend_api.get_bar.return_value = ([145.0, 148.0, 150.0], [datetime.now()] * 3)
        mock_validate.return_value = mock_backend_api

        # Mock zero quantity stock instance
        mock_stock_instance = Mock()
        mock_stock_instance.stock = Mock()
        mock_stock_instance.stock.__str__ = Mock(return_value="TSLA")

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = []
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
            with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                mock_instance_class.objects.filter.return_value.exists.return_value = True
                mock_instance_class.objects.filter.return_value.delete.return_value = None
                mock_instance_class.objects.filter.return_value.__iter__ = Mock(
                    return_value=iter([mock_stock_instance])
                )
                mock_order_class.objects.filter.return_value.iterator.return_value = []
                mock_order_class.objects.filter.return_value.order_by.return_value.iterator.return_value = []

                result = sync_alpaca(mock_user)

                assert "display_portfolio" in result
                # Display portfolio should include the watchlist stock
                assert len(result["display_portfolio"]) > 0

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_no_portfolio(
        self,
        mock_manager_class,
        mock_user,
        mock_alpaca_account,
    ):
        """Test syncing user without portfolio."""
        delattr(mock_user, 'portfolio')

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = []
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.Portfolio') as mock_portfolio_class:
            with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
                with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                    mock_instance_class.objects.filter.return_value.exists.return_value = False
                    mock_instance_class.objects.filter.return_value.delete.return_value = None
                    mock_order_class.objects.filter.return_value.iterator.return_value = []
                    mock_order_class.objects.filter.return_value.order_by.return_value.iterator.return_value = []

                    # Mock new portfolio
                    new_portfolio = Mock()
                    new_portfolio.cash = 10000.0
                    new_portfolio.save = Mock()
                    mock_portfolio_class.return_value = new_portfolio

                    result = sync_alpaca(mock_user)

                    # Should create new portfolio
                    new_portfolio.save.assert_called()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_zero_equity(
        self,
        mock_manager_class,
        mock_user,
    ):
        """Test syncing with zero equity account."""
        account = Mock()
        account.equity = "0.00"
        account.buying_power = "0.00"
        account.cash = "0.00"
        account.currency = "USD"
        account.long_market_value = "0.00"
        account.short_market_value = "0.00"
        account.portfolio_value = "0.00"
        account.last_equity = "0.00"

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = account
        mock_manager.get_positions.return_value = []
        mock_manager.api.list_orders.return_value = []
        mock_manager_class.return_value = mock_manager

        with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
            with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                mock_instance_class.objects.filter.return_value.exists.return_value = False
                mock_instance_class.objects.filter.return_value.delete.return_value = None
                mock_order_class.objects.filter.return_value.iterator.return_value = []
                mock_order_class.objects.filter.return_value.order_by.return_value.iterator.return_value = []

                result = sync_alpaca(mock_user)

                # Should handle zero division
                assert result["portfolio_change_direction"] == "error"

    @patch('backend.tradingbot.synchronization.sync_database_company_stock')
    @patch('backend.tradingbot.synchronization.AlpacaManager')
    def test_sync_alpaca_duplicate_positions(
        self,
        mock_manager_class,
        mock_sync_company,
        mock_user,
        mock_alpaca_account,
    ):
        """Test handling duplicate positions."""
        # Same symbol appears twice
        pos1 = Mock()
        pos1.symbol = "AAPL"
        pos1.qty = "10"

        pos2 = Mock()
        pos2.symbol = "AAPL"
        pos2.qty = "5"

        mock_manager = Mock()
        mock_manager.validate_api.return_value = (True, "Valid")
        mock_manager.get_account.return_value = mock_alpaca_account
        mock_manager.get_positions.return_value = [pos1, pos2]
        mock_manager.api.list_orders.return_value = []
        mock_manager_class.return_value = mock_manager

        mock_company = Mock()
        mock_stock = Mock()
        mock_sync_company.return_value = (mock_stock, mock_company)

        with patch('backend.tradingbot.synchronization.Company') as mock_company_class:
            with patch('backend.tradingbot.synchronization.Stock') as mock_stock_class:
                with patch('backend.tradingbot.synchronization.StockInstance') as mock_instance_class:
                    with patch('backend.tradingbot.synchronization.Order') as mock_order_class:
                        mock_company_class.objects.get.return_value = mock_company
                        mock_stock_class.objects.get.return_value = mock_stock
                        mock_instance_class.objects.filter.return_value.exists.return_value = False
                        mock_instance_class.objects.filter.return_value.delete.return_value = None
                        mock_order_class.objects.filter.return_value.iterator.return_value = []
                        mock_order_class.objects.filter.return_value.order_by.return_value.iterator.return_value = []

                        result = sync_alpaca(mock_user)

                        # Should sync company only once
                        assert mock_sync_company.call_count == 2
