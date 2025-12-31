"""
Tests for Ensemble Price Predictor

Tests cover:
- EnsembleConfig configuration
- EnsembleMethod enum
- EnsemblePrediction results
- EnsemblePricePredictor training and prediction
- EnsembleHyperparameterTuner
"""

import numpy as np
import pytest


# Test fixtures
@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n_samples = 200
    trend = np.linspace(100, 150, n_samples)
    noise = np.random.randn(n_samples) * 2
    return trend + noise


@pytest.fixture
def short_prices():
    """Short price series for quick tests."""
    np.random.seed(42)
    return np.random.randn(100) * 10 + 100


class TestEnsembleConfig:
    """Tests for EnsembleConfig."""

    def test_ensemble_config_defaults(self):
        """Test config has reasonable defaults."""
        from ml.tradingbots.components.ensemble_predictor import EnsembleConfig

        config = EnsembleConfig()
        assert config.use_lstm is True
        assert config.use_transformer is True
        assert config.use_cnn is True
        assert config.validation_split == 0.2

    def test_ensemble_method_enum(self):
        """Test EnsembleMethod enum values."""
        from ml.tradingbots.components.ensemble_predictor import EnsembleMethod

        assert EnsembleMethod.SIMPLE_AVERAGE.value == "simple_average"
        assert EnsembleMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert EnsembleMethod.STACKING.value == "stacking"
        assert EnsembleMethod.VOTING.value == "voting"


class TestEnsemblePrediction:
    """Tests for EnsemblePrediction."""

    def test_ensemble_prediction_creation(self):
        """Test EnsemblePrediction can be created."""
        from ml.tradingbots.components.ensemble_predictor import EnsemblePrediction

        pred = EnsemblePrediction(
            predicted_price=155.0,
            current_price=150.0,
            expected_change_pct=3.33,
            trend="up",
            confidence=0.85,
            uncertainty=2.5,
            lstm_prediction=154.0,
            transformer_prediction=156.0,
            cnn_prediction=155.0,
        )

        assert pred.predicted_price == 155.0
        assert pred.trend == "up"
        assert pred.confidence == 0.85
        assert pred.lstm_prediction == 154.0

    def test_ensemble_prediction_to_dict(self):
        """Test EnsemblePrediction to_dict method."""
        from ml.tradingbots.components.ensemble_predictor import EnsemblePrediction

        pred = EnsemblePrediction(
            predicted_price=155.0,
            current_price=150.0,
            expected_change_pct=3.33,
            trend="up",
            confidence=0.85,
            uncertainty=2.5,
        )

        result = pred.to_dict()
        assert "predicted_price" in result
        assert "trend" in result
        assert result["predicted_price"] == 155.0


class TestEnsemblePredictor:
    """Tests for EnsemblePricePredictor."""

    def test_ensemble_creation_default(self):
        """Test ensemble predictor can be created with defaults."""
        from ml.tradingbots.components.ensemble_predictor import EnsemblePricePredictor

        predictor = EnsemblePricePredictor()
        assert 'lstm' in predictor.models
        assert 'transformer' in predictor.models
        assert 'cnn' in predictor.models

    def test_ensemble_creation_single_model(self):
        """Test ensemble with only one model."""
        from ml.tradingbots.components.ensemble_predictor import (
            EnsembleConfig,
            EnsemblePricePredictor,
        )

        config = EnsembleConfig(
            use_lstm=True,
            use_transformer=False,
            use_cnn=False,
        )
        predictor = EnsemblePricePredictor(config)

        assert 'lstm' in predictor.models
        assert 'transformer' not in predictor.models
        assert 'cnn' not in predictor.models

    def test_ensemble_train_single_model(self, short_prices):
        """Test training ensemble with single model."""
        from ml.tradingbots.components.ensemble_predictor import (
            EnsembleConfig,
            EnsemblePricePredictor,
        )
        from ml.tradingbots.components.lstm_predictor import LSTMConfig

        # Use small model for fast testing (epochs in config)
        lstm_config = LSTMConfig(
            hidden_size=16,
            num_layers=1,
            seq_length=10,
            epochs=2,
        )

        config = EnsembleConfig(
            lstm_config=lstm_config,
            use_lstm=True,
            use_transformer=False,
            use_cnn=False,
        )
        predictor = EnsemblePricePredictor(config)
        predictor.train(short_prices)

        assert predictor.is_trained

    def test_ensemble_predict(self, short_prices):
        """Test ensemble prediction."""
        from ml.tradingbots.components.ensemble_predictor import (
            EnsembleConfig,
            EnsemblePricePredictor,
            EnsemblePrediction,
        )
        from ml.tradingbots.components.lstm_predictor import LSTMConfig

        lstm_config = LSTMConfig(
            hidden_size=16,
            num_layers=1,
            seq_length=10,
            epochs=2,
        )

        config = EnsembleConfig(
            lstm_config=lstm_config,
            use_lstm=True,
            use_transformer=False,
            use_cnn=False,
        )
        predictor = EnsemblePricePredictor(config)
        predictor.train(short_prices)

        prediction = predictor.predict(short_prices)

        assert isinstance(prediction, EnsemblePrediction)
        assert prediction.predicted_price > 0
        assert prediction.trend in ["up", "down", "sideways"]


class TestHyperparameterTuner:
    """Tests for EnsembleHyperparameterTuner."""

    def test_tuner_creation(self):
        """Test tuner creation."""
        from ml.tradingbots.components.ensemble_predictor import EnsembleHyperparameterTuner

        tuner = EnsembleHyperparameterTuner(n_iter=2, cv_folds=2)
        assert tuner.n_iter == 2
        assert tuner.cv_folds == 2

    def test_tuner_default_param_grid(self):
        """Test tuner has default parameter grid."""
        from ml.tradingbots.components.ensemble_predictor import EnsembleHyperparameterTuner

        tuner = EnsembleHyperparameterTuner()
        assert 'lstm_hidden_size' in tuner.param_grid
        assert 'dropout' in tuner.param_grid
        assert 'learning_rate' in tuner.param_grid

    def test_tuner_custom_param_grid(self):
        """Test tuner with custom parameter grid."""
        from ml.tradingbots.components.ensemble_predictor import EnsembleHyperparameterTuner

        custom_grid = {
            'lstm_hidden_size': [32, 64],
            'dropout': [0.1, 0.2],
        }
        tuner = EnsembleHyperparameterTuner(param_grid=custom_grid)
        assert tuner.param_grid == custom_grid


class TestEnsembleErrors:
    """Tests for ensemble error handling."""

    def test_ensemble_predict_before_train(self, short_prices):
        """Test prediction before training raises error."""
        from ml.tradingbots.components.ensemble_predictor import EnsemblePricePredictor

        predictor = EnsemblePricePredictor()
        with pytest.raises(RuntimeError, match="trained"):
            predictor.predict(short_prices)

    def test_ensemble_no_models_config(self):
        """Test config with no models still creates predictor."""
        from ml.tradingbots.components.ensemble_predictor import (
            EnsembleConfig,
            EnsemblePricePredictor,
        )

        config = EnsembleConfig(
            use_lstm=False,
            use_transformer=False,
            use_cnn=False,
        )
        predictor = EnsemblePricePredictor(config)
        # Predictor is created but has no models
        assert len(predictor.models) == 0
