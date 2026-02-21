from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from backend.auth0login.models import Credential
from backend.tradingbot.models import Order, TradeTransaction


class TestStockTradeTransactionLedger(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="stock_trade_ledger_user",
            password="testpass123",
        )
        Credential.objects.create(
            user=self.user,
            alpaca_id="alpaca-id",
            alpaca_key="alpaca-secret",
        )
        self.client.login(username="stock_trade_ledger_user", password="testpass123")

    @patch("backend.tradingbot.views.AlpacaManager")
    def test_stock_trade_buy_creates_canonical_trade_transaction(self, mock_alpaca_manager):
        mock_manager = mock_alpaca_manager.return_value
        mock_manager.market_buy.return_value = {
            "id": "order-buy-1",
            "status": "filled",
            "symbol": "AAPL",
            "qty": 2,
            "filled_price": 189.34,
        }

        response = self.client.post(
            reverse("stock_trade"),
            data={
                "ticker": "AAPL",
                "quantity": "2",
                "transaction_side": "buy",
                "order_type": "market",
            },
        )

        assert response.status_code == 201

        order = Order.objects.get(alpaca_order_id="order-buy-1")
        transaction = TradeTransaction.objects.get(order=order)

        assert transaction.transaction_type == "BUY"
        assert transaction.symbol == "AAPL"
        assert transaction.status == "filled"
        assert transaction.price == Decimal("189.34")
        assert transaction.gross_amount == Decimal("378.68")
        assert transaction.legacy_reference == "order-buy-1"

    @patch("backend.tradingbot.views.AlpacaManager")
    def test_stock_trade_sell_limit_uses_limit_price_when_fill_missing(self, mock_alpaca_manager):
        mock_manager = mock_alpaca_manager.return_value
        mock_manager.market_sell.return_value = {
            "id": "order-sell-1",
            "status": "new",
            "symbol": "MSFT",
            "qty": 3,
            "filled_price": None,
            "limit_price": 420.5,
        }

        response = self.client.post(
            reverse("stock_trade"),
            data={
                "ticker": "MSFT",
                "quantity": "3",
                "transaction_side": "sell",
                "order_type": "limit",
                "limit_price": "420.5",
            },
        )

        assert response.status_code == 201

        order = Order.objects.get(alpaca_order_id="order-sell-1")
        transaction = TradeTransaction.objects.get(order=order)

        assert transaction.transaction_type == "SELL"
        assert transaction.symbol == "MSFT"
        assert transaction.status == "new"
        assert transaction.price == Decimal("420.5")
        assert transaction.gross_amount == Decimal("1261.5")
