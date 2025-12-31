"""
Tests for ML Price Predictors (LSTM, Transformer, CNN)

Tests cover:
- Model initialization
- Data preparation
- Training loop
- Prediction functionality
- Model save/load
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pytest
import torch

# Test fixtures
@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    # Simulate price series with trend and noise
    n_samples = 500
    trend = np.linspace(100, 150, n_samples)
    noise = np.random.randn(n_samples) * 2
    prices = trend + noise
    return prices


@pytest.fixture
def short_prices():
    """Short price series for quick tests."""
    np.random.seed(42)
    return np.random.randn(100) * 10 + 100


class TestLSTMPredictor:
    """Tests for LSTM Price Predictor."""

    def test_lstm_config_defaults(self):
        """Test LSTM config has reasonable defaults."""
        from ml.tradingbots.components.lstm_predictor import LSTMConfig

        config = LSTMConfig()
        assert config.hidden_size == 128
        assert config.num_layers == 2
        assert config.seq_length == 60
        assert config.dropout == 0.2
        assert config.learning_rate == 0.001

    def test_lstm_model_creation(self):
        """Test LSTM model can be created."""
        from ml.tradingbots.components.lstm_predictor import LSTMConfig, LSTMModel

        config = LSTMConfig(hidden_size=64, num_layers=1)
        model = LSTMModel(config)

        assert model is not None
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'fc')

    def test_lstm_model_forward_pass(self):
        """Test LSTM model forward pass."""
        from ml.tradingbots.components.lstm_predictor import LSTMConfig, LSTMModel

        config = LSTMConfig(hidden_size=32, num_layers=1, seq_length=30)
        model = LSTMModel(config)

        # Create dummy input (batch_size=4, seq_length=30, input_size=1)
        x = torch.randn(4, 30, 1)
        output = model(x)

        assert output.shape == (4, 1)

    def test_lstm_data_manager_prepare_data(self, sample_prices):
        """Test data preparation for LSTM."""
        from ml.tradingbots.components.lstm_predictor import LSTMDataManager

        manager = LSTMDataManager(seq_length=30)
        X, y = manager.prepare_data(sample_prices)

        assert len(X) == len(sample_prices) - 30
        assert len(y) == len(sample_prices) - 30
        assert X.shape[1] == 30  # Sequence length
        assert X.shape[2] == 1   # Input features

    def test_lstm_predictor_train(self, short_prices):
        """Test LSTM predictor training (short run)."""
        from ml.tradingbots.components.lstm_predictor import (
            LSTMConfig, LSTMPricePredictor
        )

        config = LSTMConfig(
            hidden_size=16,
            num_layers=1,
            seq_length=20,
            epochs=2,  # Quick test
            batch_size=8,
        )
        predictor = LSTMPricePredictor(config)

        result = predictor.train(short_prices, verbose=False)

        assert predictor.is_trained
        assert 'final_train_loss' in result
        assert 'best_val_loss' in result
        assert result['epochs_trained'] >= 1

    def test_lstm_predictor_predict(self, short_prices):
        """Test LSTM prediction."""
        from ml.tradingbots.components.lstm_predictor import (
            LSTMConfig, LSTMPricePredictor
        )

        config = LSTMConfig(
            hidden_size=16,
            num_layers=1,
            seq_length=20,
            epochs=2,
        )
        predictor = LSTMPricePredictor(config)
        predictor.train(short_prices, verbose=False)

        prediction = predictor.predict(short_prices)

        assert isinstance(prediction, float)
        # Prediction should be in reasonable range
        assert 50 < prediction < 150

    def test_lstm_predict_trend(self, short_prices):
        """Test trend prediction."""
        from ml.tradingbots.components.lstm_predictor import (
            LSTMConfig, LSTMPricePredictor
        )

        config = LSTMConfig(
            hidden_size=16,
            num_layers=1,
            seq_length=20,
            epochs=2,
        )
        predictor = LSTMPricePredictor(config)
        predictor.train(short_prices, verbose=False)

        trend = predictor.predict_trend(short_prices)

        assert trend in ["up", "down", "sideways"]


class TestTransformerPredictor:
    """Tests for Transformer Price Predictor."""

    def test_transformer_config_defaults(self):
        """Test Transformer config defaults."""
        from ml.tradingbots.components.transformer_predictor import TransformerConfig

        config = TransformerConfig()
        assert config.d_model == 64
        assert config.nhead == 4
        assert config.num_encoder_layers == 3
        assert config.seq_length == 60

    def test_transformer_model_creation(self):
        """Test Transformer model can be created."""
        from ml.tradingbots.components.transformer_predictor import (
            TransformerConfig, TransformerModel
        )

        config = TransformerConfig(d_model=32, nhead=2, num_encoder_layers=1)
        model = TransformerModel(config)

        assert model is not None
        assert hasattr(model, 'transformer_encoder')
        assert hasattr(model, 'pos_encoder')

    def test_transformer_forward_pass(self):
        """Test Transformer forward pass."""
        from ml.tradingbots.components.transformer_predictor import (
            TransformerConfig, TransformerModel
        )

        config = TransformerConfig(
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            seq_length=20,
        )
        model = TransformerModel(config)

        x = torch.randn(4, 20, 1)  # batch=4, seq=20, features=1
        output = model(x)

        assert output.shape == (4, 1)

    def test_transformer_predictor_train(self, short_prices):
        """Test Transformer training."""
        from ml.tradingbots.components.transformer_predictor import (
            TransformerConfig, TransformerPricePredictor
        )

        config = TransformerConfig(
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            seq_length=20,
            epochs=2,
            batch_size=8,
        )
        predictor = TransformerPricePredictor(config)

        result = predictor.train(short_prices, verbose=False)

        assert predictor.is_trained
        assert 'final_train_loss' in result

    def test_transformer_predict(self, short_prices):
        """Test Transformer prediction."""
        from ml.tradingbots.components.transformer_predictor import (
            TransformerConfig, TransformerPricePredictor
        )

        config = TransformerConfig(
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            seq_length=20,
            epochs=2,
        )
        predictor = TransformerPricePredictor(config)
        predictor.train(short_prices, verbose=False)

        prediction = predictor.predict(short_prices)

        assert isinstance(prediction, float)


class TestCNNPredictor:
    """Tests for CNN Price Predictor."""

    def test_cnn_config_defaults(self):
        """Test CNN config defaults."""
        from ml.tradingbots.components.cnn_predictor import CNNConfig

        config = CNNConfig()
        assert config.seq_length == 60
        assert config.fc_hidden_size == 128
        assert config.dropout == 0.2

    def test_cnn_config_post_init(self):
        """Test CNN config post_init sets default lists."""
        from ml.tradingbots.components.cnn_predictor import CNNConfig

        config = CNNConfig()
        assert config.num_filters == [32, 64, 128]
        assert config.kernel_sizes == [3, 3, 3]
        assert config.pool_sizes == [2, 2, 2]

    def test_cnn_model_creation(self):
        """Test CNN model creation."""
        from ml.tradingbots.components.cnn_predictor import CNNConfig, CNNModel

        config = CNNConfig(
            num_filters=[16, 32],
            kernel_sizes=[3, 3],
            pool_sizes=[2, 2],
            seq_length=30,
        )
        model = CNNModel(config)

        assert model is not None
        assert hasattr(model, 'conv')
        assert hasattr(model, 'fc')

    def test_cnn_forward_pass(self):
        """Test CNN forward pass."""
        from ml.tradingbots.components.cnn_predictor import CNNConfig, CNNModel

        config = CNNConfig(
            num_filters=[16, 32],
            kernel_sizes=[3, 3],
            pool_sizes=[2, 2],
            seq_length=30,
        )
        model = CNNModel(config)

        x = torch.randn(4, 30, 1)  # batch=4, seq=30, features=1
        output = model(x)

        assert output.shape == (4, 1)

    def test_cnn_predictor_train(self, short_prices):
        """Test CNN training."""
        from ml.tradingbots.components.cnn_predictor import (
            CNNConfig, CNNPricePredictor
        )

        config = CNNConfig(
            num_filters=[8, 16],
            kernel_sizes=[3, 3],
            pool_sizes=[2, 2],
            seq_length=20,
            epochs=2,
            batch_size=8,
        )
        predictor = CNNPricePredictor(config)

        result = predictor.train(short_prices, verbose=False)

        assert predictor.is_trained
        assert 'final_train_loss' in result

    def test_cnn_predict(self, short_prices):
        """Test CNN prediction."""
        from ml.tradingbots.components.cnn_predictor import (
            CNNConfig, CNNPricePredictor
        )

        config = CNNConfig(
            num_filters=[8, 16],
            kernel_sizes=[3, 3],
            pool_sizes=[2, 2],
            seq_length=20,
            epochs=2,
        )
        predictor = CNNPricePredictor(config)
        predictor.train(short_prices, verbose=False)

        prediction = predictor.predict(short_prices)

        assert isinstance(prediction, float)

    def test_cnn_extract_features(self, short_prices):
        """Test CNN feature extraction."""
        from ml.tradingbots.components.cnn_predictor import (
            CNNConfig, CNNPricePredictor
        )

        config = CNNConfig(
            num_filters=[8, 16],
            kernel_sizes=[3, 3],
            pool_sizes=[2, 2],
            seq_length=20,
            epochs=2,
        )
        predictor = CNNPricePredictor(config)
        predictor.train(short_prices, verbose=False)

        features = predictor.extract_features(short_prices)

        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 2
        assert features.shape[0] == 1  # Single sample


class TestPredictorErrors:
    """Test error handling for predictors."""

    def test_lstm_predict_before_train_raises(self, short_prices):
        """Test prediction without training raises error."""
        from ml.tradingbots.components.lstm_predictor import LSTMPricePredictor

        predictor = LSTMPricePredictor()

        with pytest.raises(RuntimeError, match="not trained"):
            predictor.predict(short_prices)

    def test_transformer_predict_before_train_raises(self, short_prices):
        """Test Transformer prediction without training raises error."""
        from ml.tradingbots.components.transformer_predictor import TransformerPricePredictor

        predictor = TransformerPricePredictor()

        with pytest.raises(RuntimeError, match="not trained"):
            predictor.predict(short_prices)

    def test_cnn_predict_before_train_raises(self, short_prices):
        """Test CNN prediction without training raises error."""
        from ml.tradingbots.components.cnn_predictor import CNNPricePredictor

        predictor = CNNPricePredictor()

        with pytest.raises(RuntimeError, match="not trained"):
            predictor.predict(short_prices)

    def test_insufficient_data_raises(self):
        """Test insufficient data raises error."""
        from ml.tradingbots.components.lstm_predictor import LSTMDataManager

        manager = LSTMDataManager(seq_length=60)
        short_data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="at least"):
            manager.prepare_sequence(short_data)
