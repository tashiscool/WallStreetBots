"""
Tests for Alpaca Crypto Client

Tests cover:
- CryptoAsset enum
- Crypto dataclasses (Quote, Trade, Bar, Position, Order)
- CryptoMarketHours
- AlpacaCryptoClient methods
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from backend.tradingbot.crypto.alpaca_crypto_client import (
    CryptoAsset,
    CryptoQuote,
    CryptoTrade,
    CryptoBar,
    CryptoPosition,
    CryptoOrder,
    CryptoMarketHours,
    AlpacaCryptoClient,
    create_crypto_client,
    ALPACA_CRYPTO_AVAILABLE,
)


class TestCryptoAssetEnum:
    """Tests for CryptoAsset enum."""

    def test_btc_usd(self):
        """Test BTC/USD asset."""
        assert CryptoAsset.BTC_USD.value == "BTC/USD"

    def test_eth_usd(self):
        """Test ETH/USD asset."""
        assert CryptoAsset.ETH_USD.value == "ETH/USD"

    def test_sol_usd(self):
        """Test SOL/USD asset."""
        assert CryptoAsset.SOL_USD.value == "SOL/USD"

    def test_all_assets_have_usd(self):
        """Test all assets end with USD."""
        for asset in CryptoAsset:
            assert asset.value.endswith("/USD")


class TestCryptoQuote:
    """Tests for CryptoQuote dataclass."""

    def test_quote_creation(self):
        """Test creating a crypto quote."""
        quote = CryptoQuote(
            symbol="BTC/USD",
            bid=Decimal("50000.00"),
            ask=Decimal("50100.00"),
            bid_size=Decimal("1.5"),
            ask_size=Decimal("2.0"),
            timestamp=datetime.now(),
        )

        assert quote.symbol == "BTC/USD"
        assert quote.bid == Decimal("50000.00")
        assert quote.ask == Decimal("50100.00")

    def test_mid_price(self):
        """Test mid price calculation."""
        quote = CryptoQuote(
            symbol="BTC/USD",
            bid=Decimal("50000.00"),
            ask=Decimal("50100.00"),
            bid_size=Decimal("1.5"),
            ask_size=Decimal("2.0"),
            timestamp=datetime.now(),
        )

        # Mid = (50000 + 50100) / 2 = 50050
        assert quote.mid_price == Decimal("50050.00")

    def test_spread(self):
        """Test spread calculation."""
        quote = CryptoQuote(
            symbol="BTC/USD",
            bid=Decimal("50000.00"),
            ask=Decimal("50100.00"),
            bid_size=Decimal("1.5"),
            ask_size=Decimal("2.0"),
            timestamp=datetime.now(),
        )

        assert quote.spread == Decimal("100.00")

    def test_spread_pct(self):
        """Test spread percentage calculation."""
        quote = CryptoQuote(
            symbol="BTC/USD",
            bid=Decimal("50000.00"),
            ask=Decimal("50100.00"),
            bid_size=Decimal("1.5"),
            ask_size=Decimal("2.0"),
            timestamp=datetime.now(),
        )

        # Spread % = 100 / 50050 * 100 ≈ 0.1998
        assert 0.19 < quote.spread_pct < 0.21


class TestCryptoTrade:
    """Tests for CryptoTrade dataclass."""

    def test_trade_creation(self):
        """Test creating a crypto trade."""
        trade = CryptoTrade(
            symbol="BTC/USD",
            price=Decimal("50000.00"),
            size=Decimal("0.5"),
            timestamp=datetime.now(),
            exchange="CBSE",
        )

        assert trade.symbol == "BTC/USD"
        assert trade.price == Decimal("50000.00")
        assert trade.size == Decimal("0.5")
        assert trade.exchange == "CBSE"


class TestCryptoBar:
    """Tests for CryptoBar dataclass."""

    def test_bar_creation(self):
        """Test creating a crypto bar."""
        bar = CryptoBar(
            symbol="BTC/USD",
            open=Decimal("49000.00"),
            high=Decimal("51000.00"),
            low=Decimal("48500.00"),
            close=Decimal("50000.00"),
            volume=Decimal("1000.5"),
            timestamp=datetime.now(),
            vwap=Decimal("49800.00"),
        )

        assert bar.symbol == "BTC/USD"
        assert bar.open == Decimal("49000.00")
        assert bar.high == Decimal("51000.00")
        assert bar.low == Decimal("48500.00")
        assert bar.close == Decimal("50000.00")
        assert bar.vwap == Decimal("49800.00")


class TestCryptoPosition:
    """Tests for CryptoPosition dataclass."""

    def test_position_creation(self):
        """Test creating a crypto position."""
        position = CryptoPosition(
            symbol="BTC/USD",
            qty=Decimal("0.5"),
            avg_entry_price=Decimal("48000.00"),
            market_value=Decimal("25000.00"),
            current_price=Decimal("50000.00"),
            unrealized_pl=Decimal("1000.00"),
            unrealized_plpc=4.17,
        )

        assert position.symbol == "BTC/USD"
        assert position.qty == Decimal("0.5")

    def test_cost_basis(self):
        """Test cost basis calculation."""
        position = CryptoPosition(
            symbol="BTC/USD",
            qty=Decimal("0.5"),
            avg_entry_price=Decimal("48000.00"),
            market_value=Decimal("25000.00"),
            current_price=Decimal("50000.00"),
            unrealized_pl=Decimal("1000.00"),
            unrealized_plpc=4.17,
        )

        # Cost basis = 0.5 * 48000 = 24000
        assert position.cost_basis == Decimal("24000.00")


class TestCryptoOrder:
    """Tests for CryptoOrder dataclass."""

    def test_order_creation(self):
        """Test creating a crypto order."""
        order = CryptoOrder(
            id="order123",
            symbol="BTC/USD",
            side="buy",
            qty=Decimal("0.5"),
            filled_qty=Decimal("0.5"),
            order_type="market",
            status="filled",
            filled_avg_price=Decimal("50000.00"),
        )

        assert order.id == "order123"
        assert order.side == "buy"
        assert order.status == "filled"


class TestCryptoMarketHours:
    """Tests for CryptoMarketHours dataclass."""

    def test_is_open_default(self):
        """Test crypto market is always open by default."""
        hours = CryptoMarketHours()
        assert hours.is_open is True

    def test_is_market_open_static(self):
        """Test static method returns True."""
        assert CryptoMarketHours.is_market_open() is True


class TestAlpacaCryptoClient:
    """Tests for AlpacaCryptoClient."""

    def test_client_creation_without_sdk(self):
        """Test client creation when SDK not available."""
        with patch('backend.tradingbot.crypto.alpaca_crypto_client.ALPACA_CRYPTO_AVAILABLE', False):
            client = AlpacaCryptoClient("key", "secret")
            assert client.trading_client is None or not client.is_available()

    def test_normalize_symbol_with_slash(self):
        """Test symbol normalization with slash."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        assert client._normalize_symbol("BTC/USD") == "BTC/USD"

    def test_normalize_symbol_without_slash(self):
        """Test symbol normalization without slash."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        assert client._normalize_symbol("BTCUSD") == "BTC/USD"

    def test_normalize_symbol_lowercase(self):
        """Test symbol normalization with lowercase."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        assert client._normalize_symbol("btc/usd") == "BTC/USD"

    def test_get_supported_assets(self):
        """Test getting list of supported assets."""
        assets = AlpacaCryptoClient.get_supported_assets()
        assert "BTC/USD" in assets
        assert "ETH/USD" in assets
        assert len(assets) > 10

    def test_is_available_false(self):
        """Test is_available returns False when clients not initialized."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.trading_client = None
        client.data_client = None

        assert client.is_available() is False

    @pytest.mark.asyncio
    async def test_get_quote_no_data_client(self):
        """Test get_quote returns None when no data client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.get_quote("BTC/USD")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest_trade_no_data_client(self):
        """Test get_latest_trade returns None when no data client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.get_latest_trade("BTC/USD")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_price_no_data(self):
        """Test get_price returns None when no data."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.get_price("BTC/USD")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_historical_bars_no_data_client(self):
        """Test get_historical_bars returns empty list when no data client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.get_historical_bars("BTC/USD")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_positions_no_trading_client(self):
        """Test get_positions returns empty list when no trading client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.get_positions()
        assert result == []

    @pytest.mark.asyncio
    async def test_buy_no_trading_client(self):
        """Test buy returns None when no trading client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.buy("BTC/USD", qty=Decimal("0.1"))
        assert result is None

    @pytest.mark.asyncio
    async def test_sell_no_trading_client(self):
        """Test sell returns None when no trading client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.sell("BTC/USD", qty=Decimal("0.1"))
        assert result is None

    @pytest.mark.asyncio
    async def test_close_position_no_trading_client(self):
        """Test close_position returns None when no trading client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.close_position("BTC/USD")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_orders_no_trading_client(self):
        """Test get_orders returns empty list when no trading client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.get_orders()
        assert result == []

    @pytest.mark.asyncio
    async def test_cancel_order_no_trading_client(self):
        """Test cancel_order returns False when no trading client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.cancel_order("order123")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_account_crypto_info_no_trading_client(self):
        """Test get_account_crypto_info returns empty dict when no trading client."""
        client = AlpacaCryptoClient.__new__(AlpacaCryptoClient)
        client.api_key = "test"
        client.secret_key = "test"
        client.trading_client = None
        client.data_client = None

        result = await client.get_account_crypto_info()
        assert result == {}


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_crypto_client(self):
        """Test factory function creates client."""
        with patch('backend.tradingbot.crypto.alpaca_crypto_client.ALPACA_CRYPTO_AVAILABLE', False):
            client = create_crypto_client("key", "secret")
            assert isinstance(client, AlpacaCryptoClient)

    def test_create_crypto_client_paper(self):
        """Test factory function with paper trading."""
        with patch('backend.tradingbot.crypto.alpaca_crypto_client.ALPACA_CRYPTO_AVAILABLE', False):
            client = create_crypto_client("key", "secret", paper_trading=True)
            assert client.paper_trading is True
