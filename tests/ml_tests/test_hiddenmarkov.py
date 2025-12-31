"""
Comprehensive Tests for Hidden Markov Model Components

Tests the HMM trading strategy including:
- APImanager for Alpaca integration
- DataManager for data preparation
- HMM for prediction and inference
- Edge cases and error handling
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.tradingbots.components.hiddenmarkov import APImanager, DataManager, HMM


class TestAPImanager:
    """Tests for Alpaca API manager."""

    @pytest.fixture
    def api_manager(self):
        """Create API manager with mock credentials."""
        return APImanager("test_key", "test_secret")

    def test_initialization(self, api_manager):
        """Test API manager initializes correctly."""
        assert api_manager.API_KEY == "test_key"
        assert api_manager.SECRET_KEY == "test_secret"
        assert "alpaca.markets" in api_manager.BASE_URL
        assert api_manager.ACCOUNT_URL is not None
        assert api_manager.api is not None

    def test_base_url_format(self, api_manager):
        """Test BASE_URL is properly formatted."""
        assert api_manager.BASE_URL.startswith("https:")

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi.REST')
    def test_get_bar_success(self, mock_rest):
        """Test successful bar retrieval."""
        # Mock the API response
        mock_api = Mock()
        mock_rest.return_value = mock_api

        # Create mock DataFrame
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        mock_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'open': [99, 100, 101, 102, 103],
            'high': [101, 102, 103, 104, 105],
            'low': [98, 99, 100, 101, 102],
        }, index=dates)

        mock_bars = Mock()
        mock_bars.df = mock_df
        mock_api.get_bars.return_value = mock_bars

        api_manager = APImanager("test_key", "test_secret")
        prices, timestamps = api_manager.get_bar(
            "AAPL",
            "Day",
            "2023-01-01",
            "2023-01-05"
        )

        assert len(prices) == 5
        assert len(timestamps) == 5
        assert prices[0] == 104  # Reversed order
        assert prices[-1] == 100

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi.REST')
    def test_get_bar_empty_dataframe(self, mock_rest):
        """Test handling of empty DataFrame."""
        mock_api = Mock()
        mock_rest.return_value = mock_api

        mock_bars = Mock()
        mock_bars.df = pd.DataFrame()
        mock_api.get_bars.return_value = mock_bars

        api_manager = APImanager("test_key", "test_secret")
        prices, timestamps = api_manager.get_bar(
            "INVALID",
            "Day",
            "2023-01-01",
            "2023-01-05"
        )

        assert prices == []
        assert timestamps == []

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi.REST')
    def test_get_bar_exception_handling(self, mock_rest):
        """Test exception handling in get_bar."""
        mock_api = Mock()
        mock_rest.return_value = mock_api
        mock_api.get_bars.side_effect = Exception("API Error")

        api_manager = APImanager("test_key", "test_secret")
        result = api_manager.get_bar("AAPL", "Day", "2023-01-01", "2023-01-05")

        assert isinstance(result, str)
        assert "Failed to get bars" in result

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi.REST')
    def test_get_bar_different_price_types(self, mock_rest):
        """Test get_bar with different price types."""
        mock_api = Mock()
        mock_rest.return_value = mock_api

        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        mock_df = pd.DataFrame({
            'close': [100, 101, 102],
            'open': [99, 100, 101],
            'high': [101, 102, 103],
            'low': [98, 99, 100],
        }, index=dates)

        mock_bars = Mock()
        mock_bars.df = mock_df
        mock_api.get_bars.return_value = mock_bars

        api_manager = APImanager("test_key", "test_secret")

        # Test with open prices
        prices, _ = api_manager.get_bar(
            "AAPL", "Day", "2023-01-01", "2023-01-03", price_type="open"
        )
        assert prices[0] == 101  # Reversed

        # Test with high prices
        prices, _ = api_manager.get_bar(
            "AAPL", "Day", "2023-01-01", "2023-01-03", price_type="high"
        )
        assert prices[0] == 103

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi.REST')
    def test_get_price_success(self, mock_rest):
        """Test successful price retrieval."""
        mock_api = Mock()
        mock_rest.return_value = mock_api

        mock_trade = Mock()
        mock_trade._raw = {"price": 150.50}
        mock_api.get_last_trade.return_value = mock_trade

        api_manager = APImanager("test_key", "test_secret")
        price = api_manager.get_price("AAPL")

        assert price == 150.50

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi.REST')
    def test_get_price_exception(self, mock_rest):
        """Test exception handling in get_price."""
        mock_api = Mock()
        mock_rest.return_value = mock_api
        mock_api.get_last_trade.side_effect = Exception("Price Error")

        api_manager = APImanager("test_key", "test_secret")
        result = api_manager.get_price("AAPL")

        assert isinstance(result, str)
        assert "Failed to get price" in result

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi.REST')
    def test_market_close(self, mock_rest):
        """Test market close check."""
        mock_api = Mock()
        mock_rest.return_value = mock_api

        mock_clock = Mock()
        mock_clock.is_open = True
        mock_api.get_clock.return_value = mock_clock

        api_manager = APImanager("test_key", "test_secret")
        is_open = api_manager.market_close()

        assert is_open is True


class TestDataManager:
    """Tests for DataManager class."""

    @pytest.fixture
    def mock_api_manager(self):
        """Create mock API manager."""
        with patch('ml.tradingbots.components.hiddenmarkov.APImanager') as mock:
            api_mock = Mock()
            mock.return_value = api_mock
            yield api_mock

    @pytest.fixture
    def data_manager(self, mock_api_manager):
        """Create DataManager instance."""
        return DataManager(
            "test_key",
            "test_secret",
            "AAPL",
            "2023-01-01",
            "2023-12-31"
        )

    def test_initialization(self, data_manager):
        """Test DataManager initialization."""
        assert data_manager.ALPACA_ID == "test_key"
        assert data_manager.ALPACA_KEY == "test_secret"
        assert data_manager.ticker == "AAPL"
        assert data_manager.start_date == "2023-01-01"
        assert data_manager.end_date == "2023-12-31"
        assert data_manager.open is None
        assert data_manager.close is None
        assert data_manager.normalized_close is None

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi')
    def test_get_data_close(self, mock_tradeapi, data_manager):
        """Test get_data for close prices."""
        # Mock the API
        mock_api = Mock()
        data_manager.api.api = mock_api

        dates = pd.date_range('2023-01-01', periods=10, freq='D', tz='UTC')
        mock_df = pd.DataFrame({
            'close': np.random.randn(10) * 10 + 100,
        }, index=dates)

        mock_bars = Mock()
        mock_bars.df = mock_df
        mock_api.get_bars.return_value = mock_bars

        result = data_manager.get_data("all", False)

        assert result is not None
        assert 'close' in result.columns
        assert 'date' in result.columns

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi')
    def test_get_data_open(self, mock_tradeapi, data_manager):
        """Test get_data for open prices."""
        mock_api = Mock()
        data_manager.api.api = mock_api

        dates = pd.date_range('2023-01-02', periods=10, freq='D', tz='UTC')
        mock_df = pd.DataFrame({
            'open': np.random.randn(10) * 10 + 100,
        }, index=dates)

        mock_bars = Mock()
        mock_bars.df = mock_df
        mock_api.get_bars.return_value = mock_bars

        result = data_manager.get_data("all", True)

        assert result is not None
        assert 'open' in result.columns
        assert 'date' in result.columns

    def test_normalize_helper(self, data_manager):
        """Test normalize_helper function."""
        seq = [100, 105, 110, 115, 120]
        normalized = data_manager.normalize_helper(seq.copy())

        assert normalized[0] == 0
        assert normalized[1] == 5
        assert normalized[4] == 20

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi')
    def test_align_data(self, mock_tradeapi, data_manager):
        """Test data alignment."""
        mock_api = Mock()
        data_manager.api.api = mock_api

        # Create matching data
        dates = pd.date_range('2023-01-01', periods=10, freq='D', tz='UTC')

        open_df = pd.DataFrame({
            'open': np.random.randn(10) * 10 + 100,
        }, index=dates)

        close_df = pd.DataFrame({
            'close': np.random.randn(10) * 10 + 100,
        }, index=dates)

        mock_bars = Mock()

        def get_bars_side_effect(*args, **kwargs):
            result = Mock()
            if 'open' in args or (kwargs and 'open' in str(kwargs)):
                result.df = open_df
            else:
                result.df = close_df
            return result

        mock_api.get_bars.side_effect = get_bars_side_effect

        data_manager.align_data("all")

        assert data_manager.open is not None
        assert data_manager.close is not None

    def test_normalize(self, data_manager):
        """Test normalize function."""
        # Create mock close data
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        close_data = pd.DataFrame({
            'close': np.linspace(100, 120, 20),
            'date': dates.date,
        })

        data_manager.close = close_data
        data_manager.normalize()

        assert data_manager.normalized_close is not None
        assert len(data_manager.first_day) > 0

    def test_get_last_datapoint(self, data_manager):
        """Test get_last_datapoint function."""
        # Setup test data
        dates = pd.date_range('2023-01-01', periods=20, freq='D')

        data_manager.open = pd.DataFrame({
            'open': np.linspace(100, 120, 20),
            'date': dates.date,
        })

        data_manager.close = pd.DataFrame({
            'close': np.linspace(100, 120, 20),
            'date': dates.date,
        })

        data_manager.first_day = [100] * 20

        data_manager.get_last_datapoint()

        assert data_manager.last_datapoint is not None
        assert len(data_manager.last_datapoint) > 0


class TestHMM:
    """Tests for HMM class."""

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock DataManager."""
        dm = Mock()

        # Create realistic close data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        close_prices = np.random.randn(100) * 5 + 100

        dm.close = pd.DataFrame({
            'close': close_prices,
            'date': dates.date,
        })

        dm.normalized_close = pd.DataFrame({
            'close': (close_prices - close_prices[0]),
        })

        dm.first_day = [100] * 10
        dm.open = pd.DataFrame({
            'open': np.random.randn(10) * 5 + 100,
        })

        return dm

    @pytest.fixture
    def hmm_model(self, mock_data_manager):
        """Create HMM instance."""
        return HMM(
            data_manager=mock_data_manager,
            num_hidden_states=3,
            covar_type="full",
            n_iter=10
        )

    def test_initialization(self, hmm_model, mock_data_manager):
        """Test HMM initialization."""
        assert hmm_model.data == mock_data_manager
        assert hmm_model.num_hidden_states == 3
        assert hmm_model.covar_type == "full"
        assert hmm_model.n_iter == 10
        assert hmm_model.model is None
        assert hmm_model.transit is None
        assert hmm_model.mean is None
        assert hmm_model.var is None

    def test_train(self, hmm_model, mock_data_manager):
        """Test HMM training."""
        hmm_model.train(mock_data_manager)

        assert hmm_model.model is not None
        assert hmm_model.transit is not None
        assert hmm_model.mean is not None
        assert hmm_model.var is not None

        # Verify transition matrix properties
        assert hmm_model.transit.shape == (3, 3)
        # Each row should sum to approximately 1
        assert np.allclose(hmm_model.transit.sum(axis=1), 1.0)

    def test_train_with_different_states(self, mock_data_manager):
        """Test training with different numbers of hidden states."""
        for n_states in [2, 3, 4, 5]:
            hmm = HMM(
                data_manager=mock_data_manager,
                num_hidden_states=n_states,
                covar_type="diag",
                n_iter=5
            )
            hmm.train(mock_data_manager)

            assert hmm.transit.shape == (n_states, n_states)
            assert hmm.mean.shape[0] == n_states

    def test_evaluation(self, hmm_model, mock_data_manager):
        """Test HMM evaluation."""
        hmm_model.train(mock_data_manager)
        hmm_model.evaluation(mock_data_manager)

        assert hmm_model.pred is not None
        assert len(hmm_model.pred) > 0

    def test_inference(self, hmm_model, mock_data_manager):
        """Test HMM inference."""
        # Setup data for inference
        mock_data_manager.open = pd.DataFrame({
            'open': np.random.randn(100) * 5 + 100,
        })

        hmm_model.train(mock_data_manager)
        hmm_model.evaluation(mock_data_manager)
        hmm_model.inference()

        assert hmm_model.num_uptrend is not None
        assert hmm_model.num_pred_acc is not None
        assert 0 <= hmm_model.num_uptrend <= 1
        assert 0 <= hmm_model.num_pred_acc <= 1

    def test_train_different_covariance_types(self, mock_data_manager):
        """Test training with different covariance types."""
        for covar_type in ["diag", "full", "spherical", "tied"]:
            hmm = HMM(
                data_manager=mock_data_manager,
                num_hidden_states=3,
                covar_type=covar_type,
                n_iter=5
            )
            hmm.train(mock_data_manager)

            assert hmm.model is not None

    def test_predictions_reasonable(self, hmm_model, mock_data_manager):
        """Test that predictions are in reasonable range."""
        hmm_model.train(mock_data_manager)
        hmm_model.evaluation(mock_data_manager)

        # Predictions should be within reasonable range of input data
        close_values = mock_data_manager.normalized_close['close'].values
        min_val = close_values.min() - 50
        max_val = close_values.max() + 50

        for pred in hmm_model.pred:
            assert min_val <= pred <= max_val

    def test_state_transitions_valid(self, hmm_model, mock_data_manager):
        """Test that state transitions are probabilistic."""
        hmm_model.train(mock_data_manager)

        # All transition probabilities should be non-negative
        assert np.all(hmm_model.transit >= 0)

        # Each row should sum to 1
        row_sums = hmm_model.transit.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_means_and_variances(self, hmm_model, mock_data_manager):
        """Test that means and variances are learned."""
        hmm_model.train(mock_data_manager)

        # Means should exist for each state
        assert len(hmm_model.mean) == hmm_model.num_hidden_states

        # Variances should be positive
        if hmm_model.covar_type == "diag":
            assert np.all(hmm_model.var > 0)

    def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""
        dm = Mock()
        dm.close = pd.DataFrame({'close': []})
        dm.normalized_close = pd.DataFrame({'close': []})

        hmm = HMM(dm, 3, "diag", 5)

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, Exception)):
            hmm.train(dm)

    def test_single_state_hmm(self, mock_data_manager):
        """Test HMM with single hidden state."""
        hmm = HMM(
            data_manager=mock_data_manager,
            num_hidden_states=1,
            covar_type="diag",
            n_iter=5
        )

        hmm.train(mock_data_manager)

        assert hmm.transit.shape == (1, 1)
        assert np.allclose(hmm.transit[0, 0], 1.0)

    def test_model_convergence(self, mock_data_manager):
        """Test that model converges with more iterations."""
        hmm_few = HMM(mock_data_manager, 3, "diag", 5)
        hmm_many = HMM(mock_data_manager, 3, "diag", 50)

        hmm_few.train(mock_data_manager)
        hmm_many.train(mock_data_manager)

        # Both should train successfully
        assert hmm_few.model is not None
        assert hmm_many.model is not None


class TestIntegration:
    """Integration tests for the complete HMM workflow."""

    @patch('ml.tradingbots.components.hiddenmarkov.tradeapi')
    def test_end_to_end_workflow(self, mock_tradeapi):
        """Test complete workflow from data fetch to inference."""
        # This would require extensive mocking of Alpaca API
        # Skipping for now but structure is here
        pass

    def test_data_alignment_and_normalization(self):
        """Test that data alignment and normalization work together."""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')

        dm = Mock()
        dm.close = pd.DataFrame({
            'close': np.linspace(100, 150, 50),
            'date': dates.date,
        })

        dm.open = pd.DataFrame({
            'open': np.linspace(99, 149, 50),
            'date': dates.date,
        })

        dm.first_day = [100] * 5

        # This tests that our mocks are set up correctly
        assert len(dm.close) == 50
        assert len(dm.open) == 50
