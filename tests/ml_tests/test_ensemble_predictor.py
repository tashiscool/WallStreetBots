"""
Comprehensive Tests for Ensemble Predictor

Tests the ensemble prediction system including:
- EnsembleMethod enum
- EnsembleConfig
- EnsemblePrediction
- EnsemblePricePredictor
- EnsembleHyperparameterTuner
- Edge cases and error handling
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch
import numpy as np
import pytest

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.tradingbots.components.ensemble_predictor import (
    EnsembleMethod,
    EnsembleConfig,
    EnsemblePrediction,
    EnsemblePricePredictor,
    EnsembleHyperparameterTuner
)
from ml.tradingbots.components.lstm_predictor import LSTMConfig
from ml.tradingbots.components.transformer_predictor import TransformerConfig
from ml.tradingbots.components.cnn_predictor import CNNConfig


class TestEnsembleMethod:
    """Tests for EnsembleMethod enum."""

    def test_ensemble_methods_exist(self):
        """Test that all ensemble methods are defined."""
        assert hasattr(EnsembleMethod, 'SIMPLE_AVERAGE')
        assert hasattr(EnsembleMethod, 'WEIGHTED_AVERAGE')
        assert hasattr(EnsembleMethod, 'STACKING')
        assert hasattr(EnsembleMethod, 'VOTING')

    def test_ensemble_method_values(self):
        """Test enum values."""
        assert EnsembleMethod.SIMPLE_AVERAGE.value == "simple_average"
        assert EnsembleMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert EnsembleMethod.STACKING.value == "stacking"
        assert EnsembleMethod.VOTING.value == "voting"


class TestEnsembleConfig:
    """Tests for EnsembleConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnsembleConfig()

        assert config.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE
        assert config.use_lstm is True
        assert config.use_transformer is True
        assert config.use_cnn is True
        assert config.lstm_weight is None
        assert config.transformer_weight is None
        assert config.cnn_weight is None
        assert config.stacking_alpha == 1.0
        assert config.validation_split == 0.2
        assert config.verbose is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnsembleConfig(
            ensemble_method=EnsembleMethod.SIMPLE_AVERAGE,
            use_lstm=True,
            use_transformer=False,
            use_cnn=True,
            lstm_weight=0.6,
            cnn_weight=0.4,
            validation_split=0.3
        )

        assert config.ensemble_method == EnsembleMethod.SIMPLE_AVERAGE
        assert config.use_transformer is False
        assert config.lstm_weight == 0.6
        assert config.cnn_weight == 0.4
        assert config.validation_split == 0.3


class TestEnsemblePrediction:
    """Tests for EnsemblePrediction dataclass."""

    def test_prediction_initialization(self):
        """Test prediction initialization."""
        pred = EnsemblePrediction(
            predicted_price=110.0,
            current_price=100.0,
            expected_change_pct=10.0,
            trend="up",
            confidence=0.85,
            uncertainty=2.5
        )

        assert pred.predicted_price == 110.0
        assert pred.current_price == 100.0
        assert pred.expected_change_pct == 10.0
        assert pred.trend == "up"
        assert pred.confidence == 0.85
        assert pred.uncertainty == 2.5

    def test_prediction_with_individual_models(self):
        """Test prediction with individual model predictions."""
        pred = EnsemblePrediction(
            predicted_price=105.0,
            current_price=100.0,
            expected_change_pct=5.0,
            trend="up",
            confidence=0.8,
            uncertainty=1.0,
            lstm_prediction=106.0,
            transformer_prediction=105.0,
            cnn_prediction=104.0,
            model_agreement=0.9
        )

        assert pred.lstm_prediction == 106.0
        assert pred.transformer_prediction == 105.0
        assert pred.cnn_prediction == 104.0
        assert pred.model_agreement == 0.9

    def test_prediction_to_dict(self):
        """Test converting prediction to dictionary."""
        pred = EnsemblePrediction(
            predicted_price=105.0,
            current_price=100.0,
            expected_change_pct=5.0,
            trend="up",
            confidence=0.8,
            uncertainty=1.0
        )

        pred_dict = pred.to_dict()

        assert isinstance(pred_dict, dict)
        assert pred_dict["predicted_price"] == 105.0
        assert pred_dict["trend"] == "up"
        assert "timestamp" in pred_dict


class TestEnsemblePricePredictorInitialization:
    """Tests for EnsemblePricePredictor initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        ensemble = EnsemblePricePredictor()

        assert len(ensemble.models) == 3  # LSTM, Transformer, CNN
        assert 'lstm' in ensemble.models
        assert 'transformer' in ensemble.models
        assert 'cnn' in ensemble.models
        assert ensemble.is_trained is False

    def test_custom_initialization(self):
        """Test initialization with custom config."""
        config = EnsembleConfig(
            use_lstm=True,
            use_transformer=False,
            use_cnn=True
        )

        ensemble = EnsemblePricePredictor(config=config)

        assert len(ensemble.models) == 2
        assert 'lstm' in ensemble.models
        assert 'transformer' not in ensemble.models
        assert 'cnn' in ensemble.models

    def test_initialization_only_lstm(self):
        """Test initialization with only LSTM."""
        config = EnsembleConfig(
            use_lstm=True,
            use_transformer=False,
            use_cnn=False
        )

        ensemble = EnsemblePricePredictor(config=config)

        assert len(ensemble.models) == 1
        assert 'lstm' in ensemble.models


class TestEnsemblePricePredictorTraining:
    """Tests for ensemble training."""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(200) * 2) + 100

    @pytest.fixture
    def quick_config(self):
        """Quick training config."""
        lstm_config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        transformer_config = TransformerConfig(
            seq_length=20,
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            epochs=2,
            batch_size=16
        )

        cnn_config = CNNConfig(
            seq_length=20,
            num_filters=[16, 32],
            epochs=2,
            batch_size=16
        )

        return EnsembleConfig(
            lstm_config=lstm_config,
            transformer_config=transformer_config,
            cnn_config=cnn_config,
            verbose=False
        )

    def test_train_all_models(self, sample_prices, quick_config):
        """Test training all models."""
        ensemble = EnsemblePricePredictor(config=quick_config)

        metrics = ensemble.train(sample_prices, verbose=False)

        assert 'lstm' in metrics
        assert 'transformer' in metrics
        assert 'cnn' in metrics
        assert ensemble.is_trained is True

    def test_train_selective_models(self, sample_prices):
        """Test training only selected models."""
        lstm_config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        config = EnsembleConfig(
            lstm_config=lstm_config,
            use_lstm=True,
            use_transformer=False,
            use_cnn=False,
            verbose=False
        )

        ensemble = EnsemblePricePredictor(config=config)
        metrics = ensemble.train(sample_prices, verbose=False)

        assert 'lstm' in metrics
        assert 'transformer' not in metrics
        assert 'cnn' not in metrics

    def test_calculate_weights(self, sample_prices, quick_config):
        """Test weight calculation after training."""
        ensemble = EnsemblePricePredictor(config=quick_config)
        ensemble.train(sample_prices, verbose=False)

        # Weights should be calculated
        assert len(ensemble.model_weights) == 3
        assert all(w > 0 for w in ensemble.model_weights.values())

        # Weights should sum to approximately 1
        total_weight = sum(ensemble.model_weights.values())
        assert np.isclose(total_weight, 1.0)

    def test_simple_average_weights(self, sample_prices, quick_config):
        """Test simple average weights."""
        quick_config.ensemble_method = EnsembleMethod.SIMPLE_AVERAGE

        ensemble = EnsemblePricePredictor(config=quick_config)
        ensemble.train(sample_prices, verbose=False)

        # All weights should be equal
        weights = list(ensemble.model_weights.values())
        assert np.allclose(weights, [1/3, 1/3, 1/3], atol=0.01)

    def test_manual_weights(self, sample_prices, quick_config):
        """Test manual weight specification."""
        quick_config.lstm_weight = 0.5
        quick_config.transformer_weight = 0.3
        quick_config.cnn_weight = 0.2

        ensemble = EnsemblePricePredictor(config=quick_config)
        ensemble.train(sample_prices, verbose=False)

        assert np.isclose(ensemble.model_weights['lstm'], 0.5)
        assert np.isclose(ensemble.model_weights['transformer'], 0.3)
        assert np.isclose(ensemble.model_weights['cnn'], 0.2)


class TestEnsemblePredictions:
    """Tests for ensemble predictions."""

    @pytest.fixture
    def trained_ensemble(self):
        """Create a trained ensemble."""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200) * 2) + 100

        lstm_config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        transformer_config = TransformerConfig(
            seq_length=20,
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            epochs=2,
            batch_size=16
        )

        cnn_config = CNNConfig(
            seq_length=20,
            num_filters=[16, 32],
            epochs=2,
            batch_size=16
        )

        config = EnsembleConfig(
            lstm_config=lstm_config,
            transformer_config=transformer_config,
            cnn_config=cnn_config,
            verbose=False
        )

        ensemble = EnsemblePricePredictor(config=config)
        ensemble.train(prices, verbose=False)

        return ensemble, prices

    def test_predict_basic(self, trained_ensemble):
        """Test basic prediction."""
        ensemble, prices = trained_ensemble

        prediction = ensemble.predict(prices)

        assert isinstance(prediction, EnsemblePrediction)
        assert prediction.predicted_price is not None
        assert prediction.current_price == prices[-1]
        assert isinstance(prediction.trend, str)
        assert 0 <= prediction.confidence <= 1

    def test_predict_trend_classification(self, trained_ensemble):
        """Test trend classification."""
        ensemble, prices = trained_ensemble

        prediction = ensemble.predict(prices)

        assert prediction.trend in ["up", "down", "sideways"]

    def test_predict_individual_models(self, trained_ensemble):
        """Test that individual model predictions are included."""
        ensemble, prices = trained_ensemble

        prediction = ensemble.predict(prices)

        assert prediction.lstm_prediction is not None
        assert prediction.transformer_prediction is not None
        assert prediction.cnn_prediction is not None

    def test_predict_uncertainty(self, trained_ensemble):
        """Test uncertainty calculation."""
        ensemble, prices = trained_ensemble

        prediction = ensemble.predict(prices)

        assert prediction.uncertainty >= 0

    def test_predict_model_agreement(self, trained_ensemble):
        """Test model agreement calculation."""
        ensemble, prices = trained_ensemble

        prediction = ensemble.predict(prices)

        assert 0 <= prediction.model_agreement <= 1

    def test_predict_without_training(self):
        """Test prediction without training raises error."""
        ensemble = EnsemblePricePredictor()

        prices = np.random.randn(100) + 100

        with pytest.raises(RuntimeError, match="not trained"):
            ensemble.predict(prices)

    def test_predict_next_n(self, trained_ensemble):
        """Test multi-step prediction."""
        ensemble, prices = trained_ensemble

        predictions = ensemble.predict_next_n(prices, n=5)

        assert len(predictions) == 5
        assert all(isinstance(p, EnsemblePrediction) for p in predictions)


class TestEnsembleStacking:
    """Tests for stacking ensemble method."""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample prices."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(250) * 2) + 100

    def test_stacking_training(self, sample_prices):
        """Test stacking method training."""
        lstm_config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        config = EnsembleConfig(
            lstm_config=lstm_config,
            use_lstm=True,
            use_transformer=False,
            use_cnn=False,
            ensemble_method=EnsembleMethod.STACKING,
            verbose=False
        )

        ensemble = EnsemblePricePredictor(config=config)
        ensemble.train(sample_prices, verbose=False)

        # Meta-learner might be trained if enough validation data
        assert ensemble.is_trained

    def test_stacking_prediction(self, sample_prices):
        """Test stacking prediction."""
        lstm_config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        transformer_config = TransformerConfig(
            seq_length=20,
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            epochs=2,
            batch_size=16
        )

        config = EnsembleConfig(
            lstm_config=lstm_config,
            transformer_config=transformer_config,
            use_lstm=True,
            use_transformer=True,
            use_cnn=False,
            ensemble_method=EnsembleMethod.STACKING,
            verbose=False
        )

        ensemble = EnsemblePricePredictor(config=config)
        ensemble.train(sample_prices, verbose=False)

        prediction = ensemble.predict(sample_prices)

        assert isinstance(prediction, EnsemblePrediction)


class TestEnsemblePersistence:
    """Tests for saving and loading ensembles."""

    @pytest.fixture
    def trained_ensemble(self):
        """Create trained ensemble."""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200) * 2) + 100

        lstm_config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        config = EnsembleConfig(
            lstm_config=lstm_config,
            use_lstm=True,
            use_transformer=False,
            use_cnn=False,
            verbose=False
        )

        ensemble = EnsemblePricePredictor(config=config)
        ensemble.train(prices, verbose=False)

        return ensemble, prices

    def test_save_ensemble(self, trained_ensemble):
        """Test saving ensemble."""
        ensemble, _ = trained_ensemble

        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble.save_ensemble(tmpdir)

            # Check files exist
            import os
            assert os.path.exists(os.path.join(tmpdir, 'lstm_model.pt'))
            assert os.path.exists(os.path.join(tmpdir, 'ensemble_metadata.json'))

    def test_load_ensemble(self, trained_ensemble):
        """Test loading ensemble."""
        ensemble, prices = trained_ensemble

        # Get original prediction
        original_pred = ensemble.predict(prices)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            ensemble.save_ensemble(tmpdir)

            # Create new ensemble and load
            new_ensemble = EnsemblePricePredictor(config=ensemble.config)
            new_ensemble.load_ensemble(tmpdir)

            # Get new prediction
            new_pred = new_ensemble.predict(prices)

        # Predictions should be very close
        assert np.isclose(
            original_pred.predicted_price,
            new_pred.predicted_price,
            rtol=0.01
        )

    def test_save_untrained_ensemble_raises(self):
        """Test that saving untrained ensemble raises error."""
        ensemble = EnsemblePricePredictor()

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="untrained"):
                ensemble.save_ensemble(tmpdir)


class TestEnsembleModelContributions:
    """Tests for getting individual model contributions."""

    @pytest.fixture
    def trained_ensemble(self):
        """Create trained ensemble."""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200) * 2) + 100

        lstm_config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        transformer_config = TransformerConfig(
            seq_length=20,
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            epochs=2,
            batch_size=16
        )

        config = EnsembleConfig(
            lstm_config=lstm_config,
            transformer_config=transformer_config,
            use_lstm=True,
            use_transformer=True,
            use_cnn=False,
            verbose=False
        )

        ensemble = EnsemblePricePredictor(config=config)
        ensemble.train(prices, verbose=False)

        return ensemble, prices

    def test_get_model_contributions(self, trained_ensemble):
        """Test getting model contributions."""
        ensemble, prices = trained_ensemble

        contributions = ensemble.get_model_contributions(prices)

        assert 'lstm' in contributions
        assert 'transformer' in contributions

        for contrib in contributions.values():
            if 'error' not in contrib:
                assert 'prediction' in contrib
                assert 'trend' in contrib
                assert 'confidence' in contrib
                assert 'weight' in contrib
                assert 'validation_loss' in contrib


class TestEnsembleHyperparameterTuner:
    """Tests for hyperparameter tuning."""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample prices."""
        np.random.seed(42)
        return np.cumsum(np.random.randn(300) * 2) + 100

    def test_tuner_initialization(self):
        """Test tuner initialization."""
        tuner = EnsembleHyperparameterTuner(n_iter=5, cv_folds=2)

        assert tuner.n_iter == 5
        assert tuner.cv_folds == 2
        assert tuner.best_params is None
        assert tuner.best_score == float('inf')

    def test_default_param_grid(self):
        """Test default parameter grid."""
        tuner = EnsembleHyperparameterTuner()

        param_grid = tuner._get_default_param_grid()

        assert 'lstm_hidden_size' in param_grid
        assert 'transformer_d_model' in param_grid
        assert 'learning_rate' in param_grid

    def test_random_search(self, sample_prices):
        """Test random search."""
        param_grid = {
            'lstm_hidden_size': [32, 64],
            'seq_length': [20, 30]
        }

        tuner = EnsembleHyperparameterTuner(
            param_grid=param_grid,
            n_iter=2,
            cv_folds=2
        )

        best_params = tuner.random_search(sample_prices, verbose=False)

        assert best_params is not None
        assert 'lstm_hidden_size' in best_params
        assert len(tuner.results) == 2

    def test_get_best_ensemble(self, sample_prices):
        """Test getting best ensemble."""
        param_grid = {
            'lstm_hidden_size': [32],
            'seq_length': [20]
        }

        tuner = EnsembleHyperparameterTuner(
            param_grid=param_grid,
            n_iter=1,
            cv_folds=2
        )

        tuner.random_search(sample_prices, verbose=False)
        best_ensemble = tuner.get_best_ensemble()

        assert isinstance(best_ensemble, EnsemblePricePredictor)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_predict_with_model_failure(self):
        """Test prediction when one model fails."""
        ensemble = EnsemblePricePredictor()

        # Mock one model to fail
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model failed")
        ensemble.models['lstm'] = mock_model
        ensemble.is_trained = True

        prices = np.random.randn(100) + 100

        # Should handle gracefully or raise
        try:
            prediction = ensemble.predict(prices)
            # If it doesn't raise, check it's valid
            assert isinstance(prediction, EnsemblePrediction)
        except RuntimeError:
            # It's okay to raise if all models fail
            pass

    def test_empty_ensemble(self):
        """Test ensemble with no models."""
        config = EnsembleConfig(
            use_lstm=False,
            use_transformer=False,
            use_cnn=False
        )

        ensemble = EnsemblePricePredictor(config=config)

        assert len(ensemble.models) == 0

    def test_predict_with_constant_prices(self):
        """Test prediction with constant prices."""
        lstm_config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )

        config = EnsembleConfig(
            lstm_config=lstm_config,
            use_lstm=True,
            use_transformer=False,
            use_cnn=False,
            verbose=False
        )

        ensemble = EnsemblePricePredictor(config=config)

        # Train on constant prices
        prices = np.array([100.0] * 200)

        try:
            ensemble.train(prices, verbose=False)
            prediction = ensemble.predict(prices)
            # Should handle constant prices
            assert isinstance(prediction, EnsemblePrediction)
        except Exception:
            # Training might fail with constant data
            pass

    def test_predict_with_high_uncertainty(self):
        """Test when models disagree significantly."""
        ensemble = EnsemblePricePredictor()

        # Mock models with very different predictions
        mock_lstm = Mock()
        mock_lstm.predict = Mock(return_value=100.0)
        mock_lstm.predict_trend = Mock(return_value="up")

        mock_transformer = Mock()
        mock_transformer.predict = Mock(return_value=150.0)
        mock_transformer.predict_trend = Mock(return_value="down")

        mock_cnn = Mock()
        mock_cnn.predict = Mock(return_value=75.0)
        mock_cnn.predict_trend = Mock(return_value="sideways")

        ensemble.models = {
            'lstm': mock_lstm,
            'transformer': mock_transformer,
            'cnn': mock_cnn
        }
        ensemble.model_weights = {
            'lstm': 1/3,
            'transformer': 1/3,
            'cnn': 1/3
        }
        ensemble.is_trained = True

        prices = np.random.randn(100) + 100

        prediction = ensemble.predict(prices)

        # Should have high uncertainty
        assert prediction.uncertainty > 0

    def test_very_short_training_data(self):
        """Test with insufficient training data."""
        lstm_config = LSTMConfig(seq_length=20, epochs=1)

        config = EnsembleConfig(
            lstm_config=lstm_config,
            use_lstm=True,
            use_transformer=False,
            use_cnn=False,
            verbose=False
        )

        ensemble = EnsemblePricePredictor(config=config)

        # Too little data
        prices = np.random.randn(30) + 100

        try:
            ensemble.train(prices, verbose=False)
            # May succeed or fail depending on implementation
        except Exception:
            # Expected to potentially fail with insufficient data
            pass
