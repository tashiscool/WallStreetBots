"""
Comprehensive Tests for LSTM Pipeline

Tests the LSTM-based trading pipeline including:
- LSTMPortfolioManager
- LSTMPipeline
- LSTMSignalGenerator
- Portfolio rebalancing
- Trading signals
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

from ml.tradingbots.pipelines.lstm_pipeline import (
    LSTMPortfolioManager,
    LSTMPipeline,
    LSTMSignalGenerator
)
from ml.tradingbots.components.lstm_predictor import LSTMConfig


class TestLSTMPortfolioManagerInitialization:
    """Tests for LSTMPortfolioManager initialization."""

    @pytest.fixture
    def mock_fetcher(self):
        """Create mock data fetcher."""
        fetcher = Mock()
        fetcher.get_cur_price = Mock(return_value=150.0)
        fetcher.get_historical_prices = Mock(return_value=np.random.randn(100) * 10 + 150)
        return fetcher

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        return {
            "cash": 10000.0,
            "stocks": {
                "AAPL": 10,
                "GOOGL": 5,
                "MSFT": 8
            }
        }

    def test_initialization(self, sample_portfolio, mock_fetcher):
        """Test portfolio manager initialization."""
        manager = LSTMPortfolioManager(
            portfolio=sample_portfolio,
            data_fetcher=mock_fetcher
        )

        assert manager.portfolio == sample_portfolio
        assert manager.portfolio_cash == 10000.0
        assert len(manager.portfolio_stocks) == 3
        assert manager.buffer == 0.05
        assert manager.use_ensemble is False
        assert len(manager.models) == 0

    def test_custom_initialization(self, sample_portfolio, mock_fetcher):
        """Test initialization with custom parameters."""
        config = LSTMConfig(hidden_size=256, seq_length=30)
        manager = LSTMPortfolioManager(
            portfolio=sample_portfolio,
            data_fetcher=mock_fetcher,
            config=config,
            buffer=0.1,
            use_ensemble=True,
            ensemble_size=5,
            min_training_days=120,
            prediction_threshold=0.6
        )

        assert manager.config.hidden_size == 256
        assert manager.buffer == 0.1
        assert manager.use_ensemble is True
        assert manager.ensemble_size == 5
        assert manager.min_training_days == 120
        assert manager.prediction_threshold == 0.6

    def test_portfolio_value_calculation(self, sample_portfolio, mock_fetcher):
        """Test portfolio value calculation."""
        manager = LSTMPortfolioManager(
            portfolio=sample_portfolio,
            data_fetcher=mock_fetcher
        )

        # 10000 cash + (10 + 5 + 8) * 150 = 10000 + 3450 = 13450
        expected_value = 10000 + (10 + 5 + 8) * 150
        assert manager.total_portfolio_value == expected_value

    def test_price_dict_populated(self, sample_portfolio, mock_fetcher):
        """Test that price dictionary is populated."""
        manager = LSTMPortfolioManager(
            portfolio=sample_portfolio,
            data_fetcher=mock_fetcher
        )

        assert "AAPL" in manager.price_dict
        assert "GOOGL" in manager.price_dict
        assert "MSFT" in manager.price_dict
        assert all(price == 150.0 for price in manager.price_dict.values())

    def test_empty_portfolio(self, mock_fetcher):
        """Test with empty portfolio."""
        portfolio = {"cash": 5000.0, "stocks": {}}

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher
        )

        assert manager.total_portfolio_value == 5000.0
        assert len(manager.portfolio_stocks) == 0


class TestLSTMPortfolioManagerTraining:
    """Tests for model training in portfolio manager."""

    @pytest.fixture
    def mock_fetcher(self):
        """Create mock data fetcher."""
        fetcher = Mock()
        fetcher.get_cur_price = Mock(return_value=150.0)

        def get_historical_prices(ticker, start, end):
            np.random.seed(42)
            return np.cumsum(np.random.randn(100) * 2) + 150

        fetcher.get_historical_prices = Mock(side_effect=get_historical_prices)
        return fetcher

    @pytest.fixture
    def portfolio_manager(self, mock_fetcher):
        """Create portfolio manager with quick config."""
        portfolio = {
            "cash": 10000.0,
            "stocks": {"AAPL": 10, "GOOGL": 5}
        }

        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        return LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher,
            config=config,
            min_training_days=30
        )

    def test_train_models_success(self, portfolio_manager):
        """Test successful model training."""
        results = portfolio_manager.train_models(verbose=False)

        assert isinstance(results, dict)
        assert "AAPL" in results
        assert "GOOGL" in results
        assert len(portfolio_manager.models) == 2

    def test_train_models_with_ensemble(self, mock_fetcher):
        """Test training with ensemble."""
        portfolio = {
            "cash": 10000.0,
            "stocks": {"AAPL": 10}
        }

        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher,
            config=config,
            use_ensemble=True,
            ensemble_size=2,
            min_training_days=30
        )

        results = manager.train_models(verbose=False)

        assert "AAPL" in results
        assert "AAPL" in manager.ensembles
        assert "AAPL" not in manager.models

    def test_train_models_insufficient_data(self, mock_fetcher):
        """Test training with insufficient data."""
        # Mock returns very little data
        mock_fetcher.get_historical_prices = Mock(return_value=np.array([100, 101, 102]))

        portfolio = {"cash": 10000.0, "stocks": {"AAPL": 10}}

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher,
            min_training_days=100
        )

        results = manager.train_models(verbose=False)

        assert results["AAPL"]["error"] == "insufficient_data"

    def test_train_models_exception_handling(self, mock_fetcher):
        """Test exception handling during training."""
        # Mock raises exception
        mock_fetcher.get_historical_prices = Mock(side_effect=Exception("API Error"))

        portfolio = {"cash": 10000.0, "stocks": {"AAPL": 10}}

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher
        )

        results = manager.train_models(verbose=False)

        assert "error" in results["AAPL"]


class TestLSTMPortfolioManagerPredictions:
    """Tests for trend predictions."""

    @pytest.fixture
    def trained_manager(self, mock_fetcher):
        """Create trained portfolio manager."""
        portfolio = {"cash": 10000.0, "stocks": {"AAPL": 10}}

        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher,
            config=config,
            min_training_days=30
        )

        manager.train_models(verbose=False)
        return manager

    @pytest.fixture
    def mock_fetcher(self):
        """Create mock data fetcher."""
        fetcher = Mock()
        fetcher.get_cur_price = Mock(return_value=150.0)

        def get_historical_prices(ticker, start, end):
            np.random.seed(42)
            return np.cumsum(np.random.randn(100) * 2) + 150

        fetcher.get_historical_prices = Mock(side_effect=get_historical_prices)
        return fetcher

    def test_predict_trends_success(self, trained_manager):
        """Test successful trend prediction."""
        predictions = trained_manager.predict_trends()

        assert isinstance(predictions, dict)
        assert "AAPL" in predictions
        assert "trend" in predictions["AAPL"]
        assert "confidence" in predictions["AAPL"]
        assert "predicted_price" in predictions["AAPL"]

        assert predictions["AAPL"]["trend"] in ["up", "down", "sideways", "unknown"]

    def test_predict_trends_with_ensemble(self, mock_fetcher):
        """Test trend prediction with ensemble."""
        portfolio = {"cash": 10000.0, "stocks": {"AAPL": 10}}

        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher,
            config=config,
            use_ensemble=True,
            ensemble_size=2,
            min_training_days=30
        )

        manager.train_models(verbose=False)
        predictions = manager.predict_trends()

        assert "AAPL" in predictions
        assert "uncertainty" in predictions["AAPL"]

    def test_predict_trends_insufficient_data(self, trained_manager, mock_fetcher):
        """Test prediction with insufficient data."""
        # Mock returns insufficient data
        mock_fetcher.get_historical_prices = Mock(return_value=np.array([100, 101]))

        predictions = trained_manager.predict_trends()

        assert "AAPL" in predictions
        assert "error" in predictions["AAPL"]

    def test_predict_trends_exception(self, trained_manager, mock_fetcher):
        """Test exception handling in predictions."""
        mock_fetcher.get_historical_prices = Mock(side_effect=Exception("Error"))

        predictions = trained_manager.predict_trends()

        assert "AAPL" in predictions
        assert "error" in predictions["AAPL"]


class TestLSTMPortfolioManagerRebalancing:
    """Tests for portfolio rebalancing."""

    @pytest.fixture
    def mock_fetcher(self):
        """Create mock data fetcher."""
        fetcher = Mock()
        fetcher.get_cur_price = Mock(return_value=150.0)

        def get_historical_prices(ticker, start, end):
            np.random.seed(hash(ticker) % 100)
            return np.cumsum(np.random.randn(100) * 2) + 150

        fetcher.get_historical_prices = Mock(side_effect=get_historical_prices)
        return fetcher

    def test_rebalance_with_buy_signals(self, mock_fetcher):
        """Test rebalancing with buy signals."""
        portfolio = {
            "cash": 10000.0,
            "stocks": {"AAPL": 10, "GOOGL": 5, "MSFT": 8}
        }

        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher,
            config=config,
            prediction_threshold=0.3,  # Lower threshold for testing
            min_training_days=30
        )

        manager.train_models(verbose=False)
        new_portfolio = manager.rebalance()

        assert isinstance(new_portfolio, dict)
        # Should have allocations or be empty
        assert all(isinstance(v, (int, float)) for v in new_portfolio.values())

    def test_rebalance_no_buy_signals(self, mock_fetcher):
        """Test rebalancing with no buy signals."""
        portfolio = {"cash": 10000.0, "stocks": {"AAPL": 10}}

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher,
            prediction_threshold=0.99,  # Very high threshold
            min_training_days=30
        )

        manager.train_models(verbose=False)
        new_portfolio = manager.rebalance()

        # Should return empty portfolio (no buys)
        assert isinstance(new_portfolio, dict)

    def test_rebalance_respects_buffer(self, mock_fetcher):
        """Test that rebalancing respects cash buffer."""
        portfolio = {"cash": 10000.0, "stocks": {"AAPL": 10}}

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher,
            buffer=0.2,  # 20% buffer
            prediction_threshold=0.0,
            min_training_days=30
        )

        manager.train_models(verbose=False)

        # Mock a strong buy signal
        manager.predict_trends = Mock(return_value={
            "AAPL": {"trend": "up", "confidence": 0.9}
        })

        new_portfolio = manager.rebalance()

        if new_portfolio:
            # Total value should not exceed (total_portfolio_value * (1 - buffer))
            total_value = sum(
                qty * manager.price_dict.get(ticker, 0)
                for ticker, qty in new_portfolio.items()
            )
            max_value = manager.total_portfolio_value * (1 - manager.buffer)
            assert total_value <= max_value * 1.01  # Allow small rounding


class TestLSTMPipeline:
    """Tests for LSTMPipeline."""

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        return {
            "cash": 10000.0,
            "stocks": {"AAPL": 10, "GOOGL": 5}
        }

    @patch('ml.tradingbots.pipelines.lstm_pipeline.AlpacaFetcher')
    def test_initialization(self, mock_fetcher_class, sample_portfolio):
        """Test pipeline initialization."""
        pipeline = LSTMPipeline(
            name="TestPipeline",
            portfolio=sample_portfolio
        )

        assert pipeline.name == "TestPipeline"
        assert pipeline.portfolio == sample_portfolio
        assert pipeline.config is not None
        assert pipeline.use_ensemble is False

    @patch('ml.tradingbots.pipelines.lstm_pipeline.AlpacaFetcher')
    def test_custom_initialization(self, mock_fetcher_class, sample_portfolio):
        """Test pipeline with custom config."""
        config = LSTMConfig(hidden_size=256)

        pipeline = LSTMPipeline(
            name="TestPipeline",
            portfolio=sample_portfolio,
            config=config,
            use_ensemble=True,
            buffer=0.1,
            prediction_threshold=0.6
        )

        assert pipeline.config.hidden_size == 256
        assert pipeline.use_ensemble is True
        assert pipeline.buffer == 0.1
        assert pipeline.prediction_threshold == 0.6

    @patch('ml.tradingbots.pipelines.lstm_pipeline.AlpacaFetcher')
    def test_pipeline_execution(self, mock_fetcher_class, sample_portfolio):
        """Test pipeline execution."""
        # Mock fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_cur_price = Mock(return_value=150.0)
        mock_fetcher.get_historical_prices = Mock(
            return_value=np.random.randn(100) * 10 + 150
        )
        mock_fetcher_class.return_value = mock_fetcher

        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        pipeline = LSTMPipeline(
            name="TestPipeline",
            portfolio=sample_portfolio,
            config=config
        )

        # Execute pipeline
        result = pipeline.pipeline()

        assert isinstance(result, dict)
        assert pipeline.portfolio_manager is not None

    @patch('ml.tradingbots.pipelines.lstm_pipeline.AlpacaFetcher')
    def test_get_predictions(self, mock_fetcher_class, sample_portfolio):
        """Test getting predictions from pipeline."""
        mock_fetcher = Mock()
        mock_fetcher.get_cur_price = Mock(return_value=150.0)
        mock_fetcher.get_historical_prices = Mock(
            return_value=np.random.randn(100) * 10 + 150
        )
        mock_fetcher_class.return_value = mock_fetcher

        pipeline = LSTMPipeline(
            name="TestPipeline",
            portfolio=sample_portfolio
        )

        # Before execution
        predictions = pipeline.get_predictions()
        assert predictions == {}

        # Mock portfolio manager
        mock_manager = Mock()
        mock_manager.predict_trends = Mock(return_value={"AAPL": {"trend": "up"}})
        pipeline.portfolio_manager = mock_manager

        # After mock
        predictions = pipeline.get_predictions()
        assert "AAPL" in predictions


class TestLSTMSignalGenerator:
    """Tests for LSTMSignalGenerator."""

    @pytest.fixture
    def mock_fetcher(self):
        """Create mock data fetcher."""
        fetcher = Mock()
        return fetcher

    def test_initialization(self):
        """Test signal generator initialization."""
        generator = LSTMSignalGenerator()

        assert generator.config is not None
        assert generator.data_fetcher is not None
        assert len(generator.models) == 0

    def test_custom_initialization(self, mock_fetcher):
        """Test initialization with custom config."""
        config = LSTMConfig(hidden_size=256)

        generator = LSTMSignalGenerator(
            config=config,
            data_fetcher=mock_fetcher
        )

        assert generator.config.hidden_size == 256
        assert generator.data_fetcher == mock_fetcher

    def test_train_model(self):
        """Test training a model for specific symbol."""
        generator = LSTMSignalGenerator()

        np.random.seed(42)
        prices = np.cumsum(np.random.randn(250) * 2) + 100

        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )
        generator.config = config

        metrics = generator.train("AAPL", prices.tolist())

        assert isinstance(metrics, dict)
        assert "AAPL" in generator.models

    def test_generate_signal_success(self):
        """Test successful signal generation."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        generator = LSTMSignalGenerator(config=config)

        np.random.seed(42)
        prices = np.cumsum(np.random.randn(250) * 2) + 100

        # Train model
        generator.train("AAPL", prices.tolist())

        # Generate signal
        signal = generator.generate_signal("AAPL", prices.tolist())

        assert isinstance(signal, dict)
        assert signal["symbol"] == "AAPL"
        assert "signal" in signal
        assert "trend" in signal
        assert "confidence" in signal
        assert "predicted_price" in signal
        assert "current_price" in signal
        assert "expected_change_pct" in signal

    def test_generate_signal_classifications(self):
        """Test different signal classifications."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        generator = LSTMSignalGenerator(config=config)

        np.random.seed(42)
        prices = np.cumsum(np.random.randn(250) * 2) + 100

        generator.train("AAPL", prices.tolist())
        signal = generator.generate_signal("AAPL", prices.tolist())

        valid_signals = ["strong_buy", "buy", "hold", "sell", "strong_sell", "none"]
        assert signal["signal"] in valid_signals

    def test_generate_signal_not_trained(self):
        """Test signal generation without training."""
        generator = LSTMSignalGenerator()

        prices = [100, 101, 102, 103, 104]

        signal = generator.generate_signal("AAPL", prices)

        assert signal["signal"] == "none"
        assert signal["confidence"] == 0.0
        assert "error" in signal
        assert signal["error"] == "model_not_trained"

    def test_generate_signal_exception_handling(self):
        """Test exception handling in signal generation."""
        generator = LSTMSignalGenerator()

        # Create mock model that raises exception
        mock_model = Mock()
        mock_model.predict_trend.side_effect = Exception("Prediction error")
        generator.models["AAPL"] = mock_model

        prices = [100] * 100

        signal = generator.generate_signal("AAPL", prices)

        assert signal["signal"] == "none"
        assert "error" in signal


class TestEdgeCases:
    """Tests for edge cases."""

    @patch('ml.tradingbots.pipelines.lstm_pipeline.AlpacaFetcher')
    def test_empty_portfolio(self, mock_fetcher_class):
        """Test with empty portfolio."""
        portfolio = {"cash": 10000.0, "stocks": {}}

        pipeline = LSTMPipeline(
            name="TestPipeline",
            portfolio=portfolio
        )

        # Should handle empty stocks gracefully
        assert pipeline.portfolio == portfolio

    def test_zero_cash_portfolio(self):
        """Test portfolio with zero cash."""
        mock_fetcher = Mock()
        mock_fetcher.get_cur_price = Mock(return_value=150.0)

        portfolio = {"cash": 0.0, "stocks": {"AAPL": 10}}

        manager = LSTMPortfolioManager(
            portfolio=portfolio,
            data_fetcher=mock_fetcher
        )

        assert manager.portfolio_cash == 0.0
        assert manager.total_portfolio_value == 1500.0

    def test_negative_predictions(self):
        """Test handling of negative price predictions."""
        generator = LSTMSignalGenerator()

        # Mock model with negative prediction
        mock_model = Mock()
        mock_model.predict = Mock(return_value=-10.0)
        mock_model.predict_trend = Mock(return_value="down")
        mock_model.get_trend_confidence = Mock(return_value=0.8)

        generator.models["TEST"] = mock_model

        signal = generator.generate_signal("TEST", [100] * 100)

        # Should handle negative prediction
        assert isinstance(signal, dict)
        assert "predicted_price" in signal

    def test_very_high_confidence(self):
        """Test with very high confidence predictions."""
        generator = LSTMSignalGenerator()

        mock_model = Mock()
        mock_model.predict = Mock(return_value=120.0)
        mock_model.predict_trend = Mock(return_value="up")
        mock_model.get_trend_confidence = Mock(return_value=0.95)

        generator.models["TEST"] = mock_model

        signal = generator.generate_signal("TEST", [100] * 100)

        # High confidence up trend should give strong buy
        assert signal["signal"] in ["strong_buy", "buy"]

    def test_very_low_confidence(self):
        """Test with very low confidence predictions."""
        generator = LSTMSignalGenerator()

        mock_model = Mock()
        mock_model.predict = Mock(return_value=101.0)
        mock_model.predict_trend = Mock(return_value="up")
        mock_model.get_trend_confidence = Mock(return_value=0.1)

        generator.models["TEST"] = mock_model

        signal = generator.generate_signal("TEST", [100] * 100)

        # Low confidence should give hold
        assert signal["signal"] == "hold"
