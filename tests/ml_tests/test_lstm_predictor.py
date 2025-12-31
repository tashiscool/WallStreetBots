"""
Tests for LSTM Price Predictor.

Tests the LSTM model for time series price prediction including:
- Model initialization and configuration
- Data preparation and scaling
- Training functionality
- Prediction capabilities
- Model persistence (save/load)
- Ensemble predictions
"""

import numpy as np
import pytest
import tempfile
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.tradingbots.components.lstm_predictor import (
    LSTMConfig,
    LSTMModel,
    LSTMDataManager,
    LSTMPricePredictor,
    LSTMEnsemble,
)


class TestLSTMConfig:
    """Tests for LSTM configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LSTMConfig()
        assert config.input_size == 1
        assert config.hidden_size == 128
        assert config.num_layers == 2
        assert config.output_size == 1
        assert config.seq_length == 60
        assert config.dropout == 0.2
        assert config.learning_rate == 0.001
        assert config.epochs == 100
        assert config.batch_size == 32

    def test_custom_config(self):
        """Test custom configuration."""
        config = LSTMConfig(
            hidden_size=256,
            num_layers=3,
            seq_length=30,
            epochs=50
        )
        assert config.hidden_size == 256
        assert config.num_layers == 3
        assert config.seq_length == 30
        assert config.epochs == 50


class TestLSTMModel:
    """Tests for the PyTorch LSTM model."""

    def test_model_initialization(self):
        """Test model initializes correctly."""
        config = LSTMConfig()
        model = LSTMModel(config)
        assert model is not None
        assert model.hidden_size == 128
        assert model.num_layers == 2

    def test_model_forward_pass(self):
        """Test forward pass produces correct output shape."""
        import torch

        config = LSTMConfig(seq_length=60, batch_size=16)
        model = LSTMModel(config)

        # Create dummy input: (batch, seq_len, input_size)
        x = torch.randn(16, 60, 1)
        output = model(x)

        assert output.shape == (16, 1)

    def test_model_single_sample(self):
        """Test model with single sample."""
        import torch

        config = LSTMConfig()
        model = LSTMModel(config)

        x = torch.randn(1, 60, 1)
        output = model(x)

        assert output.shape == (1, 1)


class TestLSTMDataManager:
    """Tests for data preparation and scaling."""

    def test_data_manager_initialization(self):
        """Test data manager initializes correctly."""
        manager = LSTMDataManager(seq_length=60)
        assert manager.seq_length == 60
        assert not manager.is_fitted

    def test_prepare_data_shapes(self):
        """Test data preparation produces correct shapes."""
        manager = LSTMDataManager(seq_length=30)

        # Create sample price data
        prices = np.random.randn(100) * 10 + 100  # 100 prices around 100

        X, y = manager.prepare_data(prices, fit_scaler=True)

        # Should have (100 - 30) = 70 samples
        assert X.shape == (70, 30, 1)
        assert y.shape == (70, 1)
        assert manager.is_fitted

    def test_prepare_data_scaling(self):
        """Test that data is scaled to [0, 1] range."""
        manager = LSTMDataManager(seq_length=10)

        prices = np.array([100, 110, 105, 115, 120, 118, 125, 130, 128, 135, 140, 145])

        X, y = manager.prepare_data(prices, fit_scaler=True)

        # Scaled data should be in [0, 1] range
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_prepare_sequence(self):
        """Test single sequence preparation for prediction."""
        manager = LSTMDataManager(seq_length=10)

        prices = np.linspace(100, 200, 100)
        manager.prepare_data(prices, fit_scaler=True)

        # Prepare a sequence
        recent_prices = prices[-20:]
        sequence = manager.prepare_sequence(recent_prices)

        assert sequence.shape == (1, 10, 1)

    def test_inverse_transform(self):
        """Test inverse scaling returns original scale."""
        manager = LSTMDataManager(seq_length=10)

        prices = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
        X, y = manager.prepare_data(prices, fit_scaler=True)

        # Inverse transform should recover original values
        original = manager.inverse_transform(y)
        np.testing.assert_array_almost_equal(
            original,
            prices[10:],  # Targets start at index seq_length
            decimal=1
        )

    def test_scaler_not_fitted_error(self):
        """Test error when scaler not fitted."""
        manager = LSTMDataManager(seq_length=10)

        prices = np.random.randn(20) * 10 + 100

        with pytest.raises(ValueError, match="Scaler not fitted"):
            manager.prepare_data(prices, fit_scaler=False)


class TestLSTMPricePredictor:
    """Tests for the main LSTM predictor class."""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        # Generate a simple trending price series
        trend = np.linspace(100, 150, 200)
        noise = np.random.randn(200) * 2
        return trend + noise

    @pytest.fixture
    def quick_config(self):
        """Configuration for fast training in tests."""
        return LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=5,
            batch_size=16,
            early_stopping_patience=3
        )

    def test_predictor_initialization(self):
        """Test predictor initializes correctly."""
        predictor = LSTMPricePredictor()
        assert not predictor.is_trained
        assert len(predictor.training_loss_history) == 0

    def test_predictor_with_custom_config(self, quick_config):
        """Test predictor with custom configuration."""
        predictor = LSTMPricePredictor(quick_config)
        assert predictor.config.seq_length == 20
        assert predictor.config.hidden_size == 32

    def test_train_basic(self, sample_prices, quick_config):
        """Test basic training functionality."""
        predictor = LSTMPricePredictor(quick_config)

        metrics = predictor.train(sample_prices, verbose=False)

        assert predictor.is_trained
        assert "final_train_loss" in metrics
        assert "best_val_loss" in metrics
        assert "epochs_trained" in metrics
        assert len(predictor.training_loss_history) > 0

    def test_predict_after_training(self, sample_prices, quick_config):
        """Test prediction after training."""
        predictor = LSTMPricePredictor(quick_config)
        predictor.train(sample_prices, verbose=False)

        # Make prediction
        prediction = predictor.predict(sample_prices[-50:])

        assert isinstance(prediction, float)
        # Prediction should be in reasonable range
        assert 50 < prediction < 250

    def test_predict_without_training_raises(self, sample_prices):
        """Test that predicting without training raises error."""
        predictor = LSTMPricePredictor()

        with pytest.raises(RuntimeError, match="Model not trained"):
            predictor.predict(sample_prices)

    def test_predict_next_n(self, sample_prices, quick_config):
        """Test multi-step prediction."""
        predictor = LSTMPricePredictor(quick_config)
        predictor.train(sample_prices, verbose=False)

        predictions = predictor.predict_next_n(sample_prices, n=5)

        assert len(predictions) == 5
        assert all(isinstance(p, float) for p in predictions)

    def test_predict_trend(self, sample_prices, quick_config):
        """Test trend prediction."""
        predictor = LSTMPricePredictor(quick_config)
        predictor.train(sample_prices, verbose=False)

        trend = predictor.predict_trend(sample_prices)

        assert trend in ["up", "down", "sideways"]

    def test_get_trend_confidence(self, sample_prices, quick_config):
        """Test trend confidence calculation."""
        predictor = LSTMPricePredictor(quick_config)
        predictor.train(sample_prices, verbose=False)

        confidence = predictor.get_trend_confidence(sample_prices)

        assert 0 <= confidence <= 1

    def test_save_and_load_model(self, sample_prices, quick_config):
        """Test model save and load functionality."""
        predictor = LSTMPricePredictor(quick_config)
        predictor.train(sample_prices, verbose=False)

        # Use enough prices for the sequence length
        original_prediction = predictor.predict(sample_prices[-30:])

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pt")
            predictor.save_model(model_path)

            # Load into new predictor
            loaded_predictor = LSTMPricePredictor()
            loaded_predictor.load_model(model_path)

            loaded_prediction = loaded_predictor.predict(sample_prices[-30:])

        # Predictions should be identical
        np.testing.assert_almost_equal(original_prediction, loaded_prediction, decimal=5)


class TestLSTMEnsemble:
    """Tests for LSTM ensemble predictions."""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data."""
        np.random.seed(42)
        trend = np.linspace(100, 150, 200)
        noise = np.random.randn(200) * 2
        return trend + noise

    @pytest.fixture
    def quick_config(self):
        """Fast training configuration."""
        return LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            batch_size=16
        )

    def test_ensemble_initialization(self):
        """Test ensemble initializes correctly."""
        ensemble = LSTMEnsemble(n_models=3)
        assert ensemble.n_models == 3
        assert len(ensemble.models) == 0

    def test_ensemble_training(self, sample_prices, quick_config):
        """Test ensemble training."""
        ensemble = LSTMEnsemble(n_models=2, config=quick_config)

        metrics = ensemble.train(sample_prices, verbose=False)

        assert len(ensemble.models) == 2
        assert "models_trained" in metrics
        assert metrics["models_trained"] == 2

    def test_ensemble_prediction(self, sample_prices, quick_config):
        """Test ensemble prediction with uncertainty."""
        ensemble = LSTMEnsemble(n_models=2, config=quick_config)
        ensemble.train(sample_prices, verbose=False)

        mean_pred, std_pred = ensemble.predict(sample_prices[-50:])

        assert isinstance(mean_pred, float)
        assert isinstance(std_pred, float)
        assert std_pred >= 0  # Standard deviation should be non-negative

    def test_ensemble_trend_prediction(self, sample_prices, quick_config):
        """Test ensemble trend prediction."""
        ensemble = LSTMEnsemble(n_models=2, config=quick_config)
        ensemble.train(sample_prices, verbose=False)

        trend, confidence = ensemble.predict_trend(sample_prices[-50:])

        assert trend in ["up", "down", "sideways"]
        assert 0 <= confidence <= 1


class TestLSTMSignalCalculator:
    """Tests for LSTM signal calculator integration."""

    def test_signal_calculator_module_exists(self):
        """Test that signal calculator module can be imported."""
        try:
            from ml.tradingbots.components import lstm_signal_calculator
            assert lstm_signal_calculator is not None
        except ImportError:
            pytest.skip("lstm_signal_calculator module not available")

    def test_signal_calculator_classes_exist(self):
        """Test signal calculator classes exist."""
        try:
            from ml.tradingbots.components.lstm_signal_calculator import LSTMSignalCalculator
            assert LSTMSignalCalculator is not None
        except ImportError:
            pytest.skip("LSTMSignalCalculator not available")
