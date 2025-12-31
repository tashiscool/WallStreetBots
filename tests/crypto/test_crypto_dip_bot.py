"""
Tests for Crypto Dip Bot

Tests cover:
- DipSeverity enum
- DipSignal dataclass
- CryptoDipBotConfig
- DipBotState
- CryptoDipBot strategy
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from backend.tradingbot.crypto.crypto_dip_bot import (
    DipSeverity,
    DipSignal,
    CryptoDipBotConfig,
    DipBotState,
    CryptoDipBot,
    create_dip_bot,
)
from backend.tradingbot.crypto.alpaca_crypto_client import (
    CryptoBar,
    CryptoPosition,
)


class TestDipSeverityEnum:
    """Tests for DipSeverity enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert DipSeverity.MINOR.value == "minor"
        assert DipSeverity.MODERATE.value == "moderate"
        assert DipSeverity.MAJOR.value == "major"
        assert DipSeverity.SEVERE.value == "severe"


class TestDipSignal:
    """Tests for DipSignal dataclass."""

    def test_signal_creation(self):
        """Test creating a dip signal."""
        signal = DipSignal(
            symbol="BTC/USD",
            current_price=Decimal("45000"),
            reference_price=Decimal("50000"),
            dip_percentage=10.0,
            severity=DipSeverity.MAJOR,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h",
        )

        assert signal.symbol == "BTC/USD"
        assert signal.dip_percentage == 10.0
        assert signal.severity == DipSeverity.MAJOR

    def test_is_actionable_true(self):
        """Test is_actionable returns True for moderate+ dip."""
        signal = DipSignal(
            symbol="BTC/USD",
            current_price=Decimal("45000"),
            reference_price=Decimal("50000"),
            dip_percentage=10.0,
            severity=DipSeverity.MAJOR,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h",
        )

        assert signal.is_actionable is True

    def test_is_actionable_false_minor(self):
        """Test is_actionable returns False for minor dip."""
        signal = DipSignal(
            symbol="BTC/USD",
            current_price=Decimal("48000"),
            reference_price=Decimal("50000"),
            dip_percentage=4.0,
            severity=DipSeverity.MINOR,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h",
        )

        assert signal.is_actionable is False

    def test_is_actionable_false_low_percentage(self):
        """Test is_actionable returns False for low percentage."""
        signal = DipSignal(
            symbol="BTC/USD",
            current_price=Decimal("49000"),
            reference_price=Decimal("50000"),
            dip_percentage=2.0,
            severity=DipSeverity.MODERATE,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h",
        )

        assert signal.is_actionable is False


class TestCryptoDipBotConfig:
    """Tests for CryptoDipBotConfig."""

    def test_config_defaults(self):
        """Test config has sensible defaults."""
        config = CryptoDipBotConfig()

        assert len(config.watch_list) > 0
        assert "BTC/USD" in config.watch_list
        assert config.min_dip_percentage == 5.0
        assert config.severe_dip_percentage == 15.0
        assert config.max_open_positions == 5

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        config = CryptoDipBotConfig(
            watch_list=["BTC/USD", "ETH/USD"],
            min_dip_percentage=3.0,
            max_order_size=Decimal("5000"),
        )

        assert len(config.watch_list) == 2
        assert config.min_dip_percentage == 3.0
        assert config.max_order_size == Decimal("5000")

    def test_config_risk_management(self):
        """Test risk management settings."""
        config = CryptoDipBotConfig()

        assert config.take_profit_pct == 10.0
        assert config.stop_loss_pct == 8.0
        assert config.max_daily_trades == 10


class TestDipBotState:
    """Tests for DipBotState."""

    def test_state_defaults(self):
        """Test state has correct defaults."""
        state = DipBotState()

        assert state.is_running is False
        assert state.daily_trades_count == 0
        assert state.error_count == 0
        assert len(state.active_signals) == 0
        assert len(state.pending_orders) == 0


class TestCryptoDipBot:
    """Tests for CryptoDipBot."""

    @pytest.fixture
    def mock_crypto_client(self):
        """Create a mock crypto client."""
        client = Mock()
        client.get_price = AsyncMock(return_value=Decimal("45000"))
        client.get_historical_bars = AsyncMock(return_value=[
            CryptoBar(
                symbol="BTC/USD",
                open=Decimal("48000"),
                high=Decimal("50000"),
                low=Decimal("44000"),
                close=Decimal("45000"),
                volume=Decimal("1000"),
                timestamp=datetime.now() - timedelta(hours=i),
            )
            for i in range(24)
        ])
        client.get_positions = AsyncMock(return_value=[])
        client.buy = AsyncMock(return_value=Mock(id="order123"))
        client.sell = AsyncMock(return_value=Mock(id="order456"))
        client.get_orders = AsyncMock(return_value=[])
        client.cancel_order = AsyncMock(return_value=True)
        client.close_position = AsyncMock(return_value=Mock(id="close123"))
        return client

    def test_bot_creation(self, mock_crypto_client):
        """Test bot creation."""
        bot = CryptoDipBot(mock_crypto_client)
        assert bot.client is mock_crypto_client
        assert bot.state.is_running is False

    def test_bot_creation_with_config(self, mock_crypto_client):
        """Test bot creation with custom config."""
        config = CryptoDipBotConfig(min_dip_percentage=3.0)
        bot = CryptoDipBot(mock_crypto_client, config)
        assert bot.config.min_dip_percentage == 3.0

    def test_can_trade_no_cooldown(self, mock_crypto_client):
        """Test can_trade returns True when no cooldown."""
        bot = CryptoDipBot(mock_crypto_client)
        assert bot._can_trade("BTC/USD") is True

    def test_can_trade_global_cooldown(self, mock_crypto_client):
        """Test can_trade respects global cooldown."""
        bot = CryptoDipBot(mock_crypto_client)
        bot.state.last_trade_time = datetime.now()

        assert bot._can_trade("BTC/USD") is False

    def test_can_trade_asset_cooldown(self, mock_crypto_client):
        """Test can_trade respects asset cooldown."""
        bot = CryptoDipBot(mock_crypto_client)
        bot.state.asset_cooldowns["BTC/USD"] = datetime.now()

        assert bot._can_trade("BTC/USD") is False

    def test_calculate_order_size_base(self, mock_crypto_client):
        """Test order size calculation base."""
        bot = CryptoDipBot(mock_crypto_client)

        signal = DipSignal(
            symbol="BTC/USD",
            current_price=Decimal("45000"),
            reference_price=Decimal("50000"),
            dip_percentage=5.0,
            severity=DipSeverity.MODERATE,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h",
        )

        size = bot._calculate_order_size(signal)
        # Moderate = 1.5x base
        assert size == Decimal("15.00")

    def test_calculate_order_size_major(self, mock_crypto_client):
        """Test order size calculation for major dip."""
        bot = CryptoDipBot(mock_crypto_client)

        signal = DipSignal(
            symbol="BTC/USD",
            current_price=Decimal("45000"),
            reference_price=Decimal("50000"),
            dip_percentage=12.0,
            severity=DipSeverity.MAJOR,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h",
        )

        size = bot._calculate_order_size(signal)
        # Major = 2x base = 20
        assert size == Decimal("20.00")

    def test_calculate_order_size_severe(self, mock_crypto_client):
        """Test order size calculation for severe dip."""
        bot = CryptoDipBot(mock_crypto_client)

        signal = DipSignal(
            symbol="BTC/USD",
            current_price=Decimal("42500"),
            reference_price=Decimal("50000"),
            dip_percentage=15.0,
            severity=DipSeverity.SEVERE,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h",
        )

        size = bot._calculate_order_size(signal)
        # Severe = 3x base = 30
        assert size == Decimal("30.00")

    def test_calculate_order_size_capped(self, mock_crypto_client):
        """Test order size is capped at max."""
        config = CryptoDipBotConfig(max_order_size=Decimal("25.00"))
        bot = CryptoDipBot(mock_crypto_client, config)

        signal = DipSignal(
            symbol="BTC/USD",
            current_price=Decimal("42500"),
            reference_price=Decimal("50000"),
            dip_percentage=15.0,
            severity=DipSeverity.SEVERE,
            volume_confirmation=True,
            timestamp=datetime.now(),
            timeframe="24h",
        )

        size = bot._calculate_order_size(signal)
        # Should be capped at max
        assert size == Decimal("25.00")

    def test_check_volume_confirmation_true(self, mock_crypto_client):
        """Test volume confirmation when volume is high."""
        bot = CryptoDipBot(mock_crypto_client)

        # Recent volume is 1.5x+ average
        bars = [
            CryptoBar(
                symbol="BTC/USD",
                open=Decimal("48000"),
                high=Decimal("50000"),
                low=Decimal("44000"),
                close=Decimal("45000"),
                volume=Decimal("100"),
                timestamp=datetime.now() - timedelta(hours=i),
            )
            for i in range(10)
        ]
        # Make recent bars have higher volume
        for bar in bars[-3:]:
            bar.volume = Decimal("200")

        result = bot._check_volume_confirmation(bars)
        assert result is True

    def test_check_volume_confirmation_false(self, mock_crypto_client):
        """Test volume confirmation when volume is normal."""
        bot = CryptoDipBot(mock_crypto_client)

        bars = [
            CryptoBar(
                symbol="BTC/USD",
                open=Decimal("48000"),
                high=Decimal("50000"),
                low=Decimal("44000"),
                close=Decimal("45000"),
                volume=Decimal("100"),
                timestamp=datetime.now() - timedelta(hours=i),
            )
            for i in range(10)
        ]

        result = bot._check_volume_confirmation(bars)
        assert result is False

    def test_check_volume_confirmation_insufficient_data(self, mock_crypto_client):
        """Test volume confirmation with insufficient data."""
        bot = CryptoDipBot(mock_crypto_client)

        bars = [
            CryptoBar(
                symbol="BTC/USD",
                open=Decimal("48000"),
                high=Decimal("50000"),
                low=Decimal("44000"),
                close=Decimal("45000"),
                volume=Decimal("100"),
                timestamp=datetime.now(),
            )
        ]

        result = bot._check_volume_confirmation(bars)
        assert result is False

    def test_is_new_day_true(self, mock_crypto_client):
        """Test is_new_day returns True when new day."""
        bot = CryptoDipBot(mock_crypto_client)
        bot.state.last_scan_time = datetime.now() - timedelta(days=1)

        assert bot._is_new_day() is True

    def test_is_new_day_false(self, mock_crypto_client):
        """Test is_new_day returns False on same day."""
        bot = CryptoDipBot(mock_crypto_client)
        bot.state.last_scan_time = datetime.now()

        assert bot._is_new_day() is False

    def test_is_new_day_no_previous(self, mock_crypto_client):
        """Test is_new_day returns True when no previous scan."""
        bot = CryptoDipBot(mock_crypto_client)
        assert bot._is_new_day() is True

    def test_get_status(self, mock_crypto_client):
        """Test getting bot status."""
        bot = CryptoDipBot(mock_crypto_client)

        status = bot.get_status()

        assert "is_running" in status
        assert "daily_trades" in status
        assert "error_count" in status
        assert status["is_running"] is False

    def test_update_state_after_trade(self, mock_crypto_client):
        """Test state update after trade."""
        bot = CryptoDipBot(mock_crypto_client)

        bot._update_state_after_trade("BTC/USD")

        assert bot.state.daily_trades_count == 1
        assert bot.state.last_trade_time is not None
        assert "BTC/USD" in bot.state.asset_cooldowns

    @pytest.mark.asyncio
    async def test_get_signals(self, mock_crypto_client):
        """Test getting active signals."""
        bot = CryptoDipBot(mock_crypto_client)

        bot.state.active_signals = [
            DipSignal(
                symbol="BTC/USD",
                current_price=Decimal("45000"),
                reference_price=Decimal("50000"),
                dip_percentage=10.0,
                severity=DipSeverity.MAJOR,
                volume_confirmation=True,
                timestamp=datetime.now(),
                timeframe="24h",
            )
        ]

        signals = await bot.get_signals()
        assert len(signals) == 1
        assert signals[0]["symbol"] == "BTC/USD"

    @pytest.mark.asyncio
    async def test_stop(self, mock_crypto_client):
        """Test stopping the bot."""
        bot = CryptoDipBot(mock_crypto_client)
        bot.state.is_running = True

        await bot.stop()

        assert bot.state.is_running is False


class TestFactoryFunction:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_create_dip_bot(self):
        """Test factory function creates bot."""
        with patch('backend.tradingbot.crypto.crypto_dip_bot.AlpacaCryptoClient') as MockClient:
            MockClient.return_value = Mock()
            bot = await create_dip_bot("key", "secret")
            assert isinstance(bot, CryptoDipBot)

    @pytest.mark.asyncio
    async def test_create_dip_bot_with_config(self):
        """Test factory function with custom config."""
        with patch('backend.tradingbot.crypto.crypto_dip_bot.AlpacaCryptoClient') as MockClient:
            MockClient.return_value = Mock()
            config = CryptoDipBotConfig(min_dip_percentage=3.0)
            bot = await create_dip_bot("key", "secret", config=config)
            assert bot.config.min_dip_percentage == 3.0
