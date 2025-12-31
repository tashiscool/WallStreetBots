"""
Comprehensive Tests for Market Data Fetcher

Tests the data fetching and preprocessing pipeline including:
- DataConfig
- MarketDataFetcher with multiple sources
- Technical indicators
- ML data preparation
- Multi-asset fetching
- Edge cases and error handling
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.tradingbots.data.market_data_fetcher import (
    DataConfig,
    MarketDataFetcher,
    MultiAssetDataFetcher,
    RECOMMENDED_HYPERPARAMETERS
)


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig()

        assert config.source == "yfinance"
        assert config.alphavantage_key is None
        assert config.lookback_days == 365 * 2
        assert config.interval == "1d"
        assert config.include_volume is True
        assert config.include_technical_indicators is True
        assert config.normalize is True
        assert config.handle_missing == "ffill"

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataConfig(
            source="alphavantage",
            alphavantage_key="test_key",
            lookback_days=90,
            interval="1h",
            include_volume=False,
            normalize=False
        )

        assert config.source == "alphavantage"
        assert config.alphavantage_key == "test_key"
        assert config.lookback_days == 90
        assert config.interval == "1h"
        assert config.include_volume is False
        assert config.normalize is False

    def test_findatapy_fallbacks_default(self):
        """Test default findatapy fallbacks."""
        config = DataConfig()

        assert config.findatapy_fallbacks == ["quandl", "fred"]

    def test_findatapy_fallbacks_custom(self):
        """Test custom findatapy fallbacks."""
        config = DataConfig(findatapy_fallbacks=["quandl", "yahoo"])

        assert config.findatapy_fallbacks == ["quandl", "yahoo"]


class TestMarketDataFetcherInitialization:
    """Tests for MarketDataFetcher initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        fetcher = MarketDataFetcher()

        assert fetcher.config is not None
        assert fetcher.config.source == "yfinance"

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = DataConfig(source="yfinance", lookback_days=180)
        fetcher = MarketDataFetcher(config=config)

        assert fetcher.config.lookback_days == 180

    def test_alphavantage_validation(self):
        """Test that Alpha Vantage requires API key."""
        config = DataConfig(source="alphavantage")

        with pytest.raises(ValueError, match="Alpha Vantage API key required"):
            MarketDataFetcher(config=config)

    def test_alphavantage_with_key(self):
        """Test Alpha Vantage with API key."""
        config = DataConfig(source="alphavantage", alphavantage_key="test_key")
        fetcher = MarketDataFetcher(config=config)

        assert fetcher.config.alphavantage_key == "test_key"


class TestYFinanceDataFetching:
    """Tests for yfinance data source."""

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_fetch_yfinance_success(self, mock_yf):
        """Test successful data fetching from yfinance."""
        # Mock ticker and history
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_df = pd.DataFrame({
            'Open': np.random.randn(100) * 5 + 100,
            'High': np.random.randn(100) * 5 + 105,
            'Low': np.random.randn(100) * 5 + 95,
            'Close': np.random.randn(100) * 5 + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = MarketDataFetcher()
        df = fetcher.fetch_prices("AAPL")

        assert not df.empty
        assert 'close' in df.columns
        assert 'open' in df.columns
        assert 'volume' in df.columns

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_fetch_yfinance_with_dates(self, mock_yf):
        """Test fetching with specific date range."""
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        mock_df = pd.DataFrame({
            'Open': [100] * 50,
            'High': [105] * 50,
            'Low': [95] * 50,
            'Close': [100] * 50,
            'Volume': [1000000] * 50
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = MarketDataFetcher()
        start = datetime(2023, 1, 1)
        end = datetime(2023, 2, 19)

        df = fetcher.fetch_prices("AAPL", start_date=start, end_date=end)

        assert not df.empty
        mock_ticker.history.assert_called_once()

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_fetch_yfinance_empty_dataframe(self, mock_yf):
        """Test handling of empty DataFrame."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = MarketDataFetcher()

        with pytest.raises(ValueError, match="No data found"):
            fetcher.fetch_prices("INVALID")

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_fetch_yfinance_column_standardization(self, mock_yf):
        """Test column name standardization."""
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        mock_df = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [100] * 10,
            'Volume': [1000000] * 10
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = MarketDataFetcher()
        df = fetcher.fetch_prices("AAPL")

        # Check lowercase columns
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'volume' in df.columns


class TestAlphaVantageDataFetching:
    """Tests for Alpha Vantage data source."""

    @patch('ml.tradingbots.data.market_data_fetcher.requests')
    def test_fetch_alphavantage_success(self, mock_requests):
        """Test successful Alpha Vantage fetch."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2023-01-01": {
                    "1. open": "100.0",
                    "2. high": "105.0",
                    "3. low": "95.0",
                    "4. close": "100.0",
                    "5. adjusted close": "100.0",
                    "6. volume": "1000000"
                },
                "2023-01-02": {
                    "1. open": "101.0",
                    "2. high": "106.0",
                    "3. low": "96.0",
                    "4. close": "101.0",
                    "5. adjusted close": "101.0",
                    "6. volume": "1100000"
                }
            }
        }
        mock_requests.get.return_value = mock_response

        config = DataConfig(source="alphavantage", alphavantage_key="test_key")
        fetcher = MarketDataFetcher(config=config)

        df = fetcher.fetch_prices("AAPL")

        assert not df.empty
        assert 'close' in df.columns
        assert 'volume' in df.columns

    @patch('ml.tradingbots.data.market_data_fetcher.requests')
    def test_fetch_alphavantage_error(self, mock_requests):
        """Test Alpha Vantage error handling."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "Note": "API call frequency limit reached"
        }
        mock_requests.get.return_value = mock_response

        config = DataConfig(source="alphavantage", alphavantage_key="test_key")
        fetcher = MarketDataFetcher(config=config)

        with pytest.raises(ValueError, match="Alpha Vantage error"):
            fetcher.fetch_prices("AAPL")

    @patch('ml.tradingbots.data.market_data_fetcher.requests')
    def test_fetch_alphavantage_date_filtering(self, mock_requests):
        """Test date range filtering for Alpha Vantage."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                f"2023-01-{i:02d}": {
                    "1. open": "100.0",
                    "2. high": "105.0",
                    "3. low": "95.0",
                    "4. close": "100.0",
                    "5. adjusted close": "100.0",
                    "6. volume": "1000000"
                }
                for i in range(1, 32)
            }
        }
        mock_requests.get.return_value = mock_response

        config = DataConfig(source="alphavantage", alphavantage_key="test_key")
        fetcher = MarketDataFetcher(config=config)

        start = datetime(2023, 1, 10)
        end = datetime(2023, 1, 20)
        df = fetcher.fetch_prices("AAPL", start_date=start, end_date=end)

        # Should only have dates in range
        assert df.index[0] >= start
        assert df.index[-1] <= end


class TestTechnicalIndicators:
    """Tests for technical indicator calculation."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 105 + np.cumsum(np.random.randn(100) * 0.5),
            'low': 95 + np.cumsum(np.random.randn(100) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

        return df

    def test_add_technical_indicators(self, sample_df):
        """Test adding technical indicators."""
        fetcher = MarketDataFetcher()
        df_with_indicators = fetcher.add_technical_indicators(sample_df)

        # Check moving averages
        assert 'sma_5' in df_with_indicators.columns
        assert 'sma_10' in df_with_indicators.columns
        assert 'sma_20' in df_with_indicators.columns
        assert 'sma_50' in df_with_indicators.columns

        # Check EMA
        assert 'ema_12' in df_with_indicators.columns
        assert 'ema_26' in df_with_indicators.columns

        # Check MACD
        assert 'macd' in df_with_indicators.columns
        assert 'macd_signal' in df_with_indicators.columns
        assert 'macd_hist' in df_with_indicators.columns

        # Check RSI
        assert 'rsi' in df_with_indicators.columns

        # Check Bollinger Bands
        assert 'bb_middle' in df_with_indicators.columns
        assert 'bb_upper' in df_with_indicators.columns
        assert 'bb_lower' in df_with_indicators.columns
        assert 'bb_width' in df_with_indicators.columns

        # Check ATR
        assert 'atr' in df_with_indicators.columns

        # Check volume indicators
        assert 'volume_sma' in df_with_indicators.columns
        assert 'volume_ratio' in df_with_indicators.columns

        # Check returns
        assert 'returns_1d' in df_with_indicators.columns
        assert 'returns_5d' in df_with_indicators.columns
        assert 'returns_20d' in df_with_indicators.columns

        # Check volatility
        assert 'volatility_20d' in df_with_indicators.columns

    def test_moving_averages_calculation(self, sample_df):
        """Test that moving averages are calculated correctly."""
        fetcher = MarketDataFetcher()
        df_with_indicators = fetcher.add_technical_indicators(sample_df)

        # SMA_5 should be close to manually calculated
        manual_sma5 = sample_df['close'].rolling(window=5).mean()
        pd.testing.assert_series_equal(
            df_with_indicators['sma_5'],
            manual_sma5,
            check_names=False
        )

    def test_rsi_range(self, sample_df):
        """Test that RSI is in valid range [0, 100]."""
        fetcher = MarketDataFetcher()
        df_with_indicators = fetcher.add_technical_indicators(sample_df)

        rsi = df_with_indicators['rsi'].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_bollinger_bands_ordering(self, sample_df):
        """Test that Bollinger Bands are ordered correctly."""
        fetcher = MarketDataFetcher()
        df_with_indicators = fetcher.add_technical_indicators(sample_df)

        # Upper should be >= Middle >= Lower
        valid_rows = df_with_indicators.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])
        assert (valid_rows['bb_upper'] >= valid_rows['bb_middle']).all()
        assert (valid_rows['bb_middle'] >= valid_rows['bb_lower']).all()

    def test_macd_calculation(self, sample_df):
        """Test MACD calculation."""
        fetcher = MarketDataFetcher()
        df_with_indicators = fetcher.add_technical_indicators(sample_df)

        # MACD should be EMA12 - EMA26
        expected_macd = df_with_indicators['ema_12'] - df_with_indicators['ema_26']
        pd.testing.assert_series_equal(
            df_with_indicators['macd'],
            expected_macd,
            check_names=False,
            atol=1e-10
        )


class TestMLDataPreparation:
    """Tests for ML data preparation."""

    @pytest.fixture
    def mock_fetcher_with_data(self):
        """Create fetcher with mocked data."""
        with patch('ml.tradingbots.data.market_data_fetcher.yf') as mock_yf:
            mock_ticker = Mock()
            dates = pd.date_range('2023-01-01', periods=300, freq='D')
            np.random.seed(42)

            mock_df = pd.DataFrame({
                'Open': 100 + np.cumsum(np.random.randn(300) * 0.5),
                'High': 105 + np.cumsum(np.random.randn(300) * 0.5),
                'Low': 95 + np.cumsum(np.random.randn(300) * 0.5),
                'Close': 100 + np.cumsum(np.random.randn(300) * 0.5),
                'Volume': np.random.randint(1000000, 10000000, 300)
            }, index=dates)

            mock_ticker.history.return_value = mock_df
            mock_yf.Ticker.return_value = mock_ticker

            yield MarketDataFetcher()

    def test_prepare_ml_data_basic(self, mock_fetcher_with_data):
        """Test basic ML data preparation."""
        data = mock_fetcher_with_data.prepare_ml_data(
            "AAPL",
            seq_length=60,
            target_horizon=1,
            train_split=0.8
        )

        assert 'X_train' in data
        assert 'X_test' in data
        assert 'y_train' in data
        assert 'y_test' in data
        assert 'prices' in data
        assert 'dates' in data

    def test_prepare_ml_data_shapes(self, mock_fetcher_with_data):
        """Test ML data shapes."""
        seq_length = 60
        data = mock_fetcher_with_data.prepare_ml_data(
            "AAPL",
            seq_length=seq_length,
            target_horizon=1,
            train_split=0.8
        )

        # X should have shape (samples, seq_length, features)
        assert data['X_train'].shape[1] == seq_length
        assert data['X_test'].shape[1] == seq_length

        # y should be 1D
        assert len(data['y_train'].shape) == 1
        assert len(data['y_test'].shape) == 1

    def test_prepare_ml_data_train_test_split(self, mock_fetcher_with_data):
        """Test train/test split."""
        data = mock_fetcher_with_data.prepare_ml_data(
            "AAPL",
            seq_length=60,
            train_split=0.8
        )

        total_samples = len(data['X_train']) + len(data['X_test'])
        train_ratio = len(data['X_train']) / total_samples

        assert 0.75 < train_ratio < 0.85  # Should be close to 0.8

    def test_prepare_ml_data_target_labels(self, mock_fetcher_with_data):
        """Test that targets are binary labels."""
        data = mock_fetcher_with_data.prepare_ml_data(
            "AAPL",
            seq_length=60
        )

        # Targets should be 0 or 1
        assert set(np.unique(data['y_train'])).issubset({0, 1})
        assert set(np.unique(data['y_test'])).issubset({0, 1})

    def test_prepare_ml_data_with_technical_indicators(self, mock_fetcher_with_data):
        """Test ML data with technical indicators."""
        config = DataConfig(include_technical_indicators=True)
        mock_fetcher_with_data.config = config

        data = mock_fetcher_with_data.prepare_ml_data(
            "AAPL",
            seq_length=60
        )

        # Should have more features with technical indicators
        num_features = data['X_train'].shape[2]
        assert num_features > 2  # More than just close and volume

    def test_prepare_ml_data_without_volume(self):
        """Test ML data without volume."""
        with patch('ml.tradingbots.data.market_data_fetcher.yf') as mock_yf:
            mock_ticker = Mock()
            dates = pd.date_range('2023-01-01', periods=300, freq='D')

            mock_df = pd.DataFrame({
                'Open': [100] * 300,
                'High': [105] * 300,
                'Low': [95] * 300,
                'Close': [100] * 300,
                'Volume': [1000000] * 300
            }, index=dates)

            mock_ticker.history.return_value = mock_df
            mock_yf.Ticker.return_value = mock_ticker

            config = DataConfig(
                include_volume=False,
                include_technical_indicators=False
            )
            fetcher = MarketDataFetcher(config=config)

            data = fetcher.prepare_ml_data("AAPL", seq_length=60)

            # Should only have close price
            assert data['X_train'].shape[2] == 1

    def test_prepare_ml_data_normalization(self, mock_fetcher_with_data):
        """Test data normalization."""
        config = DataConfig(normalize=True)
        mock_fetcher_with_data.config = config

        data = mock_fetcher_with_data.prepare_ml_data(
            "AAPL",
            seq_length=60
        )

        # Normalized data should be in [0, 1] range
        assert data['X_train'].min() >= 0
        assert data['X_train'].max() <= 1

    def test_prepare_rl_data(self, mock_fetcher_with_data):
        """Test RL data preparation."""
        prices = mock_fetcher_with_data.prepare_rl_data("AAPL")

        assert isinstance(prices, np.ndarray)
        assert len(prices) > 0


class TestDataHandling:
    """Tests for data handling and preprocessing."""

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_handle_missing_ffill(self, mock_yf):
        """Test forward fill for missing data."""
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=10, freq='D')

        # Create data with NaN
        close_prices = [100, 101, np.nan, 103, np.nan, 105, 106, 107, np.nan, 109]
        mock_df = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': close_prices,
            'Volume': [1000000] * 10
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        config = DataConfig(handle_missing="ffill", include_technical_indicators=False)
        fetcher = MarketDataFetcher(config=config)

        data = fetcher.prepare_ml_data("AAPL", seq_length=5)

        # Should not have NaN values
        assert not np.isnan(data['X_train']).any()
        assert not np.isnan(data['X_test']).any()

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_handle_missing_drop(self, mock_yf):
        """Test dropping missing data."""
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        close_prices = np.random.randn(100) + 100
        close_prices[10:15] = np.nan  # Add some NaN values

        mock_df = pd.DataFrame({
            'Open': [100] * 100,
            'High': [105] * 100,
            'Low': [95] * 100,
            'Close': close_prices,
            'Volume': [1000000] * 100
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        config = DataConfig(handle_missing="drop", include_technical_indicators=False)
        fetcher = MarketDataFetcher(config=config)

        data = fetcher.prepare_ml_data("AAPL", seq_length=5)

        # Should have fewer samples due to dropping
        assert len(data['X_train']) + len(data['X_test']) < 100 - 5


class TestMultiAssetDataFetcher:
    """Tests for multi-asset data fetching."""

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_fetch_multiple_symbols(self, mock_yf):
        """Test fetching multiple symbols."""
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        mock_df = pd.DataFrame({
            'Open': [100] * 100,
            'High': [105] * 100,
            'Low': [95] * 100,
            'Close': [100] * 100,
            'Volume': [1000000] * 100
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        multi_fetcher = MultiAssetDataFetcher()
        symbols = ["AAPL", "GOOGL", "MSFT"]

        data = multi_fetcher.fetch_multiple(symbols)

        assert len(data) == 3
        assert all(symbol in data for symbol in symbols)
        assert all(not df.empty for df in data.values())

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_fetch_multiple_with_error(self, mock_yf):
        """Test multi-fetch with one symbol failing."""
        def mock_ticker_side_effect(symbol):
            mock_ticker = Mock()
            if symbol == "INVALID":
                mock_ticker.history.return_value = pd.DataFrame()
            else:
                dates = pd.date_range('2023-01-01', periods=100, freq='D')
                mock_df = pd.DataFrame({
                    'Open': [100] * 100,
                    'High': [105] * 100,
                    'Low': [95] * 100,
                    'Close': [100] * 100,
                    'Volume': [1000000] * 100
                }, index=dates)
                mock_ticker.history.return_value = mock_df
            return mock_ticker

        mock_yf.Ticker.side_effect = mock_ticker_side_effect

        multi_fetcher = MultiAssetDataFetcher()
        symbols = ["AAPL", "INVALID", "GOOGL"]

        data = multi_fetcher.fetch_multiple(symbols)

        # Should have data for valid symbols
        assert "AAPL" in data
        assert "GOOGL" in data
        # INVALID might not be in data or might be empty
        assert len(data) >= 2

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_prepare_ensemble_data(self, mock_yf):
        """Test preparing ensemble data for multiple symbols."""
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        mock_df = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'High': 105 + np.cumsum(np.random.randn(300) * 0.5),
            'Low': 95 + np.cumsum(np.random.randn(300) * 0.5),
            'Close': 100 + np.cumsum(np.random.randn(300) * 0.5),
            'Volume': np.random.randint(1000000, 10000000, 300)
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        multi_fetcher = MultiAssetDataFetcher()
        symbols = ["AAPL", "GOOGL"]

        data = multi_fetcher.prepare_ensemble_data(symbols, seq_length=60)

        assert len(data) == 2
        assert all(symbol in data for symbol in symbols)
        assert all('X_train' in symbol_data for symbol_data in data.values())


class TestRecommendedHyperparameters:
    """Tests for recommended hyperparameters."""

    def test_lstm_hyperparameters(self):
        """Test LSTM hyperparameters are defined."""
        assert 'lstm' in RECOMMENDED_HYPERPARAMETERS
        lstm_params = RECOMMENDED_HYPERPARAMETERS['lstm']

        assert 'hidden_size' in lstm_params
        assert 'num_layers' in lstm_params
        assert 'seq_length' in lstm_params
        assert 'learning_rate' in lstm_params

    def test_transformer_hyperparameters(self):
        """Test Transformer hyperparameters are defined."""
        assert 'transformer' in RECOMMENDED_HYPERPARAMETERS
        transformer_params = RECOMMENDED_HYPERPARAMETERS['transformer']

        assert 'd_model' in transformer_params
        assert 'nhead' in transformer_params
        assert 'learning_rate' in transformer_params

    def test_ppo_hyperparameters(self):
        """Test PPO hyperparameters are defined."""
        assert 'ppo' in RECOMMENDED_HYPERPARAMETERS
        ppo_params = RECOMMENDED_HYPERPARAMETERS['ppo']

        assert 'hidden_dim' in ppo_params
        assert 'learning_rate' in ppo_params
        assert 'gamma' in ppo_params

    def test_dqn_hyperparameters(self):
        """Test DQN hyperparameters are defined."""
        assert 'dqn' in RECOMMENDED_HYPERPARAMETERS
        dqn_params = RECOMMENDED_HYPERPARAMETERS['dqn']

        assert 'hidden_dim' in dqn_params
        assert 'learning_rate' in dqn_params
        assert 'epsilon_start' in dqn_params


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_unknown_source(self):
        """Test unknown data source."""
        config = DataConfig(source="invalid_source")
        fetcher = MarketDataFetcher(config=config)

        with pytest.raises(ValueError, match="Unknown source"):
            fetcher.fetch_prices("AAPL")

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_very_short_data(self, mock_yf):
        """Test with very short data."""
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=10, freq='D')

        mock_df = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [100] * 10,
            'Volume': [1000000] * 10
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = MarketDataFetcher()

        # Should handle short data
        df = fetcher.fetch_prices("AAPL")
        assert len(df) == 10

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_constant_prices(self, mock_yf):
        """Test with constant prices."""
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        mock_df = pd.DataFrame({
            'Open': [100.0] * 100,
            'High': [100.0] * 100,
            'Low': [100.0] * 100,
            'Close': [100.0] * 100,
            'Volume': [1000000] * 100
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = MarketDataFetcher()
        df_with_indicators = fetcher.add_technical_indicators(
            fetcher.fetch_prices("AAPL")
        )

        # Should handle without crashing
        assert not df_with_indicators.empty

    @patch('ml.tradingbots.data.market_data_fetcher.yf')
    def test_extreme_volatility(self, mock_yf):
        """Test with extreme price volatility."""
        mock_ticker = Mock()
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        np.random.seed(42)
        prices = 100 + np.random.randn(100) * 50  # High volatility

        mock_df = pd.DataFrame({
            'Open': prices,
            'High': prices + 10,
            'Low': prices - 10,
            'Close': prices,
            'Volume': [1000000] * 100
        }, index=dates)

        mock_ticker.history.return_value = mock_df
        mock_yf.Ticker.return_value = mock_ticker

        fetcher = MarketDataFetcher()
        df = fetcher.fetch_prices("AAPL")
        df_with_indicators = fetcher.add_technical_indicators(df)

        # Should calculate indicators without errors
        assert 'rsi' in df_with_indicators.columns
        assert 'bb_width' in df_with_indicators.columns
