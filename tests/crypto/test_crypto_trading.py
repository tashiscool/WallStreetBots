"""
Tests for Alpaca Crypto Trading.

Tests cryptocurrency trading functionality:
- Crypto client initialization
- Price fetching
- Buy/sell operations
- Crypto dip bot scanner
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from decimal import Decimal
from datetime import datetime, timedelta

from backend.tradingbot.crypto.alpaca_crypto_client import (
    CryptoAsset,
    CryptoQuote,
    CryptoBar,
    CryptoPosition,
    CryptoOrder,
    CryptoMarketHours,
    AlpacaCryptoClient,
)
from backend.tradingbot.crypto.crypto_dip_bot import (
    DipSeverity,
    DipSignal,
    CryptoDipBotConfig,
    CryptoDipBot,
)


class TestDipSignal:
    """Tests for DipSignal dataclass."""

    def test_signal_initialization(self):
        """Test signal initializes correctly."""
        signal = DipSignal(
            symbol="BTC/USD",
            current_price=Decimal("45000.00"),
            reference_price=Decimal("50000.00"),
            dip_percentage=10.0,
            severity=DipSeverity.MAJOR,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h"
        )
        assert signal.symbol == "BTC/USD"
        assert signal.current_price == Decimal("45000.00")
        assert signal.dip_percentage == 10.0
        assert signal.severity == DipSeverity.MAJOR

    def test_signal_is_actionable(self):
        """Test actionable signal detection."""
        # Actionable signal (>=5% dip with moderate+ severity)
        actionable_signal = DipSignal(
            symbol="ETH/USD",
            current_price=Decimal("2850.00"),
            reference_price=Decimal("3000.00"),
            dip_percentage=5.0,
            severity=DipSeverity.MODERATE,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h"
        )
        assert actionable_signal.is_actionable is True

        # Non-actionable signal (minor severity)
        minor_signal = DipSignal(
            symbol="SOL/USD",
            current_price=Decimal("97.00"),
            reference_price=Decimal("100.00"),
            dip_percentage=3.0,
            severity=DipSeverity.MINOR,
            volume_confirmation=False,
            timestamp=datetime.now(),
            timeframe="24h"
        )
        assert minor_signal.is_actionable is False


class TestCryptoPosition:
    """Tests for CryptoPosition dataclass."""

    def test_position_initialization(self):
        """Test position initializes correctly."""
        position = CryptoPosition(
            symbol="BTC/USD",
            qty=Decimal("0.5"),
            avg_entry_price=Decimal("44000.00"),
            market_value=Decimal("22500.00"),
            current_price=Decimal("45000.00"),
            unrealized_pl=Decimal("500.00"),
            unrealized_plpc=2.27
        )
        assert position.symbol == "BTC/USD"
        assert position.qty == Decimal("0.5")

    def test_position_cost_basis(self):
        """Test cost basis calculation."""
        position = CryptoPosition(
            symbol="BTC/USD",
            qty=Decimal("1.0"),
            avg_entry_price=Decimal("40000.00"),
            market_value=Decimal("45000.00"),
            current_price=Decimal("45000.00"),
            unrealized_pl=Decimal("5000.00"),
            unrealized_plpc=12.5
        )
        # Cost basis = qty * avg_entry_price = 1.0 * 40000 = 40000
        assert position.cost_basis == Decimal("40000.00")

    def test_position_with_loss(self):
        """Test position with unrealized loss."""
        position = CryptoPosition(
            symbol="ETH/USD",
            qty=Decimal("2.0"),
            avg_entry_price=Decimal("3000.00"),
            market_value=Decimal("5600.00"),
            current_price=Decimal("2800.00"),
            unrealized_pl=Decimal("-400.00"),
            unrealized_plpc=-6.67
        )
        assert position.unrealized_pl == Decimal("-400.00")
        assert position.unrealized_plpc < 0


class TestCryptoDipBot:
    """Tests for CryptoDipBot scanner."""

    @pytest.fixture
    def mock_crypto_client(self):
        """Create a mock Alpaca crypto client."""
        client = Mock(spec=AlpacaCryptoClient)
        client.get_price = AsyncMock(return_value=Decimal("45000.00"))
        client.get_positions = AsyncMock(return_value=[])
        client.get_orders = AsyncMock(return_value=[])
        client.is_available = Mock(return_value=True)
        return client

    def test_dip_bot_initialization(self, mock_crypto_client):
        """Test DipBot initializes correctly."""
        config = CryptoDipBotConfig(
            watch_list=["BTC/USD", "ETH/USD"],
            min_dip_percentage=5.0,
        )
        bot = CryptoDipBot(mock_crypto_client, config)
        assert bot is not None
        assert bot.config.min_dip_percentage == 5.0

    def test_dip_bot_config_defaults(self):
        """Test default configuration."""
        config = CryptoDipBotConfig()
        assert "BTC/USD" in config.watch_list
        assert "ETH/USD" in config.watch_list
        assert config.min_dip_percentage == 5.0
        assert config.max_open_positions == 5
        assert config.take_profit_pct == 10.0
        assert config.stop_loss_pct == 8.0

    def test_dip_severity_levels(self):
        """Test dip severity enumeration."""
        assert DipSeverity.MINOR.value == "minor"
        assert DipSeverity.MODERATE.value == "moderate"
        assert DipSeverity.MAJOR.value == "major"
        assert DipSeverity.SEVERE.value == "severe"

    def test_dip_bot_get_status(self, mock_crypto_client):
        """Test getting bot status."""
        bot = CryptoDipBot(mock_crypto_client)
        status = bot.get_status()

        assert "is_running" in status
        assert "daily_trades" in status
        assert "active_signals" in status
        assert "watch_list" in status


class TestCryptoMarketHours:
    """Tests for crypto market hours (24/7)."""

    def test_crypto_always_open(self):
        """Test that crypto markets are always open."""
        assert CryptoMarketHours.is_market_open() is True

    def test_market_hours_dataclass(self):
        """Test CryptoMarketHours dataclass."""
        hours = CryptoMarketHours()
        assert hours.is_open is True
        assert hours.next_open is None
        assert hours.next_close is None


class TestCryptoQuote:
    """Tests for CryptoQuote dataclass."""

    def test_quote_initialization(self):
        """Test quote initializes correctly."""
        quote = CryptoQuote(
            symbol="BTC/USD",
            bid=Decimal("44990.00"),
            ask=Decimal("45010.00"),
            bid_size=Decimal("1.5"),
            ask_size=Decimal("2.0"),
            timestamp=datetime.now()
        )
        assert quote.symbol == "BTC/USD"
        assert quote.bid == Decimal("44990.00")
        assert quote.ask == Decimal("45010.00")

    def test_quote_mid_price(self):
        """Test mid price calculation."""
        quote = CryptoQuote(
            symbol="ETH/USD",
            bid=Decimal("2990.00"),
            ask=Decimal("3010.00"),
            bid_size=Decimal("10.0"),
            ask_size=Decimal("8.0"),
            timestamp=datetime.now()
        )
        assert quote.mid_price == Decimal("3000.00")

    def test_quote_spread(self):
        """Test spread calculation."""
        quote = CryptoQuote(
            symbol="BTC/USD",
            bid=Decimal("45000.00"),
            ask=Decimal("45020.00"),
            bid_size=Decimal("1.0"),
            ask_size=Decimal("1.0"),
            timestamp=datetime.now()
        )
        assert quote.spread == Decimal("20.00")

    def test_quote_spread_percentage(self):
        """Test spread percentage calculation."""
        quote = CryptoQuote(
            symbol="BTC/USD",
            bid=Decimal("44990.00"),
            ask=Decimal("45010.00"),
            bid_size=Decimal("1.0"),
            ask_size=Decimal("1.0"),
            timestamp=datetime.now()
        )
        # Spread = 20, mid = 45000, spread_pct ~ 0.044%
        assert 0.04 < quote.spread_pct < 0.05


class TestCryptoAsset:
    """Tests for CryptoAsset enumeration."""

    def test_supported_assets(self):
        """Test supported crypto assets."""
        assert CryptoAsset.BTC_USD.value == "BTC/USD"
        assert CryptoAsset.ETH_USD.value == "ETH/USD"
        assert CryptoAsset.SOL_USD.value == "SOL/USD"

    def test_all_assets_have_usd_pair(self):
        """Test all assets are USD pairs."""
        for asset in CryptoAsset:
            assert asset.value.endswith("/USD")
