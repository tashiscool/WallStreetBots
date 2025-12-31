"""
Tests for FindatapyDataProvider

Tests cover:
- Provider initialization
- Historical data fetching
- Multi-source fallback
- ML data preparation
- Caching behavior
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestDataSourceConfig:
    """Tests for DataSourceConfig."""

    def test_config_defaults(self):
        """Test config has sensible defaults."""
        from backend.tradingbot.data.providers.findatapy_provider import DataSourceConfig

        config = DataSourceConfig()
        assert config.primary == "yahoo"
        assert "quandl" in config.fallbacks
        assert "fred" in config.fallbacks
        assert config.cache_algo == "internet_load_return"

    def test_config_custom_values(self):
        """Test config accepts custom values."""
        from backend.tradingbot.data.providers.findatapy_provider import DataSourceConfig

        config = DataSourceConfig(
            primary="quandl",
            fallbacks=["yahoo", "fred"],
            quandl_api_key="test_key",
        )
        assert config.primary == "quandl"
        assert config.fallbacks == ["yahoo", "fred"]
        assert config.quandl_api_key == "test_key"


class TestFetchResult:
    """Tests for FetchResult dataclass."""

    def test_fetch_result_success(self):
        """Test successful fetch result."""
        from backend.tradingbot.data.providers.findatapy_provider import FetchResult

        df = pd.DataFrame({"close": [100, 101, 102]})
        result = FetchResult(
            data=df,
            source="yahoo",
            success=True,
            fetch_time=0.5,
        )

        assert result.success
        assert result.source == "yahoo"
        assert result.error is None
        assert len(result.data) == 3

    def test_fetch_result_failure(self):
        """Test failed fetch result."""
        from backend.tradingbot.data.providers.findatapy_provider import FetchResult

        result = FetchResult(
            data=None,
            source="quandl",
            success=False,
            error="API rate limit exceeded",
        )

        assert not result.success
        assert result.error == "API rate limit exceeded"


class TestFindatapyProviderAvailability:
    """Tests for findatapy availability checking."""

    def test_availability_check(self):
        """Test is_findatapy_available function."""
        from backend.tradingbot.data.providers.findatapy_provider import is_findatapy_available

        # Should return bool
        available = is_findatapy_available()
        assert isinstance(available, bool)


@pytest.fixture
def mock_findatapy():
    """Mock findatapy for testing without actual API calls."""
    with patch.dict('sys.modules', {
        'findatapy': MagicMock(),
        'findatapy.market': MagicMock(),
    }):
        yield


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)

    base_price = 100
    returns = np.random.randn(100) * 0.02
    close = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "open": close * (1 + np.random.randn(100) * 0.005),
        "high": close * (1 + abs(np.random.randn(100) * 0.01)),
        "low": close * (1 - abs(np.random.randn(100) * 0.01)),
        "close": close,
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)


class TestMLDataFetcherIndicators:
    """Tests for ML data fetcher technical indicators."""

    def test_add_indicators_adds_sma(self, sample_ohlcv_data):
        """Test SMA indicators are added."""
        from backend.tradingbot.data.providers.findatapy_provider import MLDataFetcher

        # Create mock provider
        mock_provider = Mock()
        mock_provider.get_historical_data.return_value = sample_ohlcv_data

        fetcher = MLDataFetcher(provider=mock_provider)
        df = fetcher._add_technical_indicators(sample_ohlcv_data.copy())

        assert "sma_10" in df.columns
        assert "sma_20" in df.columns
        assert "sma_50" in df.columns

    def test_add_indicators_adds_ema(self, sample_ohlcv_data):
        """Test EMA indicators are added."""
        from backend.tradingbot.data.providers.findatapy_provider import MLDataFetcher

        mock_provider = Mock()
        fetcher = MLDataFetcher(provider=mock_provider)
        df = fetcher._add_technical_indicators(sample_ohlcv_data.copy())

        assert "ema_10" in df.columns
        assert "ema_20" in df.columns

    def test_add_indicators_adds_rsi(self, sample_ohlcv_data):
        """Test RSI indicator is added."""
        from backend.tradingbot.data.providers.findatapy_provider import MLDataFetcher

        mock_provider = Mock()
        fetcher = MLDataFetcher(provider=mock_provider)
        df = fetcher._add_technical_indicators(sample_ohlcv_data.copy())

        assert "rsi_14" in df.columns
        # RSI should be between 0 and 100
        valid_rsi = df["rsi_14"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_add_indicators_adds_macd(self, sample_ohlcv_data):
        """Test MACD indicators are added."""
        from backend.tradingbot.data.providers.findatapy_provider import MLDataFetcher

        mock_provider = Mock()
        fetcher = MLDataFetcher(provider=mock_provider)
        df = fetcher._add_technical_indicators(sample_ohlcv_data.copy())

        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_add_indicators_adds_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands are added."""
        from backend.tradingbot.data.providers.findatapy_provider import MLDataFetcher

        mock_provider = Mock()
        fetcher = MLDataFetcher(provider=mock_provider)
        df = fetcher._add_technical_indicators(sample_ohlcv_data.copy())

        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        assert "bb_width" in df.columns

    def test_add_indicators_adds_returns_volatility(self, sample_ohlcv_data):
        """Test returns and volatility are added."""
        from backend.tradingbot.data.providers.findatapy_provider import MLDataFetcher

        mock_provider = Mock()
        fetcher = MLDataFetcher(provider=mock_provider)
        df = fetcher._add_technical_indicators(sample_ohlcv_data.copy())

        assert "returns" in df.columns
        assert "volatility_20" in df.columns

    def test_add_indicators_adds_atr(self, sample_ohlcv_data):
        """Test ATR is added when high/low available."""
        from backend.tradingbot.data.providers.findatapy_provider import MLDataFetcher

        mock_provider = Mock()
        fetcher = MLDataFetcher(provider=mock_provider)
        df = fetcher._add_technical_indicators(sample_ohlcv_data.copy())

        assert "atr_14" in df.columns


class TestMLDataFetcherSequences:
    """Tests for ML sequence preparation."""

    def test_prepare_sequences_shape(self, sample_ohlcv_data):
        """Test sequence preparation produces correct shapes."""
        from backend.tradingbot.data.providers.findatapy_provider import MLDataFetcher

        mock_provider = Mock()
        fetcher = MLDataFetcher(provider=mock_provider)

        # Add indicators to have more features
        df = fetcher._add_technical_indicators(sample_ohlcv_data.copy())

        seq_length = 20
        X, y = fetcher.prepare_sequences(df, seq_length=seq_length)

        # X should be (samples, seq_length, features)
        assert len(X.shape) == 3
        assert X.shape[1] == seq_length

        # y should be (samples,)
        assert len(y.shape) == 1
        assert len(y) == len(X)

    def test_prepare_sequences_insufficient_data(self, sample_ohlcv_data):
        """Test error raised with insufficient data."""
        from backend.tradingbot.data.providers.findatapy_provider import MLDataFetcher

        mock_provider = Mock()
        fetcher = MLDataFetcher(provider=mock_provider)

        # Use only 10 rows but request 60-length sequences
        df_small = sample_ohlcv_data.head(10)

        with pytest.raises(ValueError, match="Not enough data"):
            fetcher.prepare_sequences(df_small, seq_length=60)


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_findatapy_provider(self):
        """Test factory function creates provider."""
        try:
            from backend.tradingbot.data.providers.findatapy_provider import (
                create_findatapy_provider,
                is_findatapy_available,
            )

            if not is_findatapy_available():
                pytest.skip("findatapy not available")

            provider = create_findatapy_provider(
                primary_source="yahoo",
                fallbacks=["quandl"],
                use_cache=False,
            )

            assert provider is not None
            assert provider.config.primary == "yahoo"
            assert provider.config.fallbacks == ["quandl"]

        except ImportError:
            pytest.skip("findatapy not available")


class TestSourceHealthTracking:
    """Tests for source health tracking."""

    def test_health_tracking_on_success(self):
        """Test health is tracked on successful fetch."""
        try:
            from backend.tradingbot.data.providers.findatapy_provider import (
                FindatapyDataProvider,
                is_findatapy_available,
            )

            if not is_findatapy_available():
                pytest.skip("findatapy not available")

            provider = FindatapyDataProvider()
            provider._update_source_health("yahoo", success=True)

            health = provider.get_source_health()
            assert "yahoo" in health
            assert health["yahoo"]["success_count"] == 1
            assert health["yahoo"]["failure_count"] == 0

        except ImportError:
            pytest.skip("findatapy not available")

    def test_health_tracking_on_failure(self):
        """Test health is tracked on failed fetch."""
        try:
            from backend.tradingbot.data.providers.findatapy_provider import (
                FindatapyDataProvider,
                is_findatapy_available,
            )

            if not is_findatapy_available():
                pytest.skip("findatapy not available")

            provider = FindatapyDataProvider()
            provider._update_source_health("quandl", success=False, error="Rate limit")

            health = provider.get_source_health()
            assert "quandl" in health
            assert health["quandl"]["failure_count"] == 1
            assert health["quandl"]["last_error"] == "Rate limit"

        except ImportError:
            pytest.skip("findatapy not available")
