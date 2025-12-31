"""
Comprehensive Tests for LSTM Signal Calculator

Tests the LSTM signal strength calculator including:
- Signal strength calculation
- Confidence calculation
- Model training and caching
- Prediction details
- Model persistence
- Edge cases and error handling
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.tradingbots.components.lstm_signal_calculator import LSTMSignalCalculator
from ml.tradingbots.components.lstm_predictor import LSTMConfig, LSTMPricePredictor, LSTMEnsemble
from backend.tradingbot.validation.signal_strength_validator import SignalType


class TestLSTMSignalCalculatorInitialization:
    """Tests for LSTMSignalCalculator initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        calculator = LSTMSignalCalculator()

        assert calculator.config is not None
        assert calculator.use_ensemble is False
        assert calculator.ensemble_size == 3
        assert calculator.min_training_samples == 200
        assert len(calculator._models) == 0
        assert len(calculator._ensembles) == 0
        assert len(calculator._trained_symbols) == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        config = LSTMConfig(hidden_size=256, seq_length=30)
        calculator = LSTMSignalCalculator(
            config=config,
            use_ensemble=True,
            ensemble_size=5,
            min_training_samples=300
        )

        assert calculator.config.hidden_size == 256
        assert calculator.config.seq_length == 30
        assert calculator.use_ensemble is True
        assert calculator.ensemble_size == 5
        assert calculator.min_training_samples == 300

    def test_params_stored_correctly(self):
        """Test that parameters are stored in params dict."""
        calculator = LSTMSignalCalculator(min_training_samples=250)

        assert 'config' in calculator.params
        assert 'use_ensemble' in calculator.params
        assert 'ensemble_size' in calculator.params
        assert 'min_training_samples' in calculator.params
        assert calculator.params['min_training_samples'] == 250


class TestSignalType:
    """Tests for signal type identification."""

    def test_get_signal_type(self):
        """Test that signal type is correctly identified."""
        calculator = LSTMSignalCalculator()
        signal_type = calculator.get_signal_type()

        assert signal_type == SignalType.TREND


class TestModelTraining:
    """Tests for model training functionality."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample market data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        prices = np.cumsum(np.random.randn(250) * 2) + 100

        df = pd.DataFrame({
            'Close': prices,
            'Open': prices - np.random.rand(250) * 2,
            'High': prices + np.random.rand(250) * 2,
            'Low': prices - np.random.rand(250) * 2,
            'Volume': np.random.randint(1000000, 10000000, 250)
        }, index=dates)

        return df

    @pytest.fixture
    def quick_config(self):
        """Quick training config for tests."""
        return LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            batch_size=16,
            early_stopping_patience=2
        )

    def test_ensure_model_trained_success(self, sample_data, quick_config):
        """Test successful model training."""
        calculator = LSTMSignalCalculator(config=quick_config)
        symbol = "AAPL"

        success = calculator._ensure_model_trained(symbol, sample_data)

        assert success is True
        assert symbol in calculator._trained_symbols
        assert symbol in calculator._models

    def test_ensure_model_trained_insufficient_data(self, quick_config):
        """Test handling of insufficient data."""
        calculator = LSTMSignalCalculator(
            config=quick_config,
            min_training_samples=100
        )

        # Create small dataset
        df = pd.DataFrame({
            'Close': np.random.randn(50) + 100
        })

        success = calculator._ensure_model_trained("AAPL", df)

        assert success is False
        assert "AAPL" not in calculator._trained_symbols

    def test_ensure_model_trained_already_trained(self, sample_data, quick_config):
        """Test that already trained models are not retrained."""
        calculator = LSTMSignalCalculator(config=quick_config)
        symbol = "AAPL"

        # Train once
        calculator._ensure_model_trained(symbol, sample_data)
        initial_model = calculator._models[symbol]

        # Try to train again
        calculator._ensure_model_trained(symbol, sample_data)
        second_model = calculator._models[symbol]

        # Should be the same model instance
        assert initial_model is second_model

    def test_ensure_model_trained_with_ensemble(self, sample_data, quick_config):
        """Test model training with ensemble."""
        calculator = LSTMSignalCalculator(
            config=quick_config,
            use_ensemble=True,
            ensemble_size=2
        )
        symbol = "AAPL"

        success = calculator._ensure_model_trained(symbol, sample_data)

        assert success is True
        assert symbol in calculator._trained_symbols
        assert symbol in calculator._ensembles
        assert symbol not in calculator._models

    def test_ensure_model_trained_exception_handling(self, quick_config):
        """Test exception handling during training."""
        calculator = LSTMSignalCalculator(config=quick_config)

        # Create invalid data
        df = pd.DataFrame({'Close': [np.nan] * 100})

        success = calculator._ensure_model_trained("AAPL", df)

        assert success is False
        assert "AAPL" not in calculator._trained_symbols


class TestCalculateRawStrength:
    """Tests for raw signal strength calculation."""

    @pytest.fixture
    def trained_calculator(self):
        """Create a trained calculator."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            batch_size=16
        )
        calculator = LSTMSignalCalculator(config=config)

        # Create training data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        prices = np.cumsum(np.random.randn(250) * 2) + 100

        df = pd.DataFrame({
            'Close': prices,
        }, index=dates)

        calculator._ensure_model_trained("AAPL", df)
        return calculator, df

    def test_calculate_raw_strength_success(self, trained_calculator):
        """Test successful strength calculation."""
        calculator, df = trained_calculator

        strength = calculator.calculate_raw_strength(df, symbol="AAPL")

        assert isinstance(strength, float)
        assert 0 <= strength <= 100

    def test_calculate_raw_strength_insufficient_data(self, trained_calculator):
        """Test with insufficient data."""
        calculator, _ = trained_calculator

        # Too few rows
        small_df = pd.DataFrame({'Close': [100, 101, 102]})

        strength = calculator.calculate_raw_strength(small_df, symbol="AAPL")

        assert strength == 0.0

    def test_calculate_raw_strength_none_data(self):
        """Test with None data."""
        calculator = LSTMSignalCalculator()

        strength = calculator.calculate_raw_strength(None, symbol="AAPL")

        assert strength == 0.0

    def test_calculate_raw_strength_fallback(self):
        """Test fallback strength calculation."""
        calculator = LSTMSignalCalculator()

        # Create data without training
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Close': np.linspace(100, 110, 50)
        }, index=dates)

        strength = calculator.calculate_raw_strength(df, symbol="UNKNOWN")

        assert isinstance(strength, float)
        assert 0 <= strength <= 100

    def test_calculate_raw_strength_with_ensemble(self):
        """Test strength calculation with ensemble."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            batch_size=16
        )
        calculator = LSTMSignalCalculator(
            config=config,
            use_ensemble=True,
            ensemble_size=2
        )

        # Create data
        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        prices = np.cumsum(np.random.randn(250) * 2) + 100
        df = pd.DataFrame({'Close': prices}, index=dates)

        calculator._ensure_model_trained("AAPL", df)
        strength = calculator.calculate_raw_strength(df, symbol="AAPL")

        assert isinstance(strength, float)
        assert 0 <= strength <= 100

    def test_calculate_raw_strength_components(self, trained_calculator):
        """Test that strength considers all components."""
        calculator, df = trained_calculator

        strength = calculator.calculate_raw_strength(df, symbol="AAPL")

        # Should include magnitude, confidence, and direction scores
        # These combine to make a score between 0-100
        assert strength >= 0
        assert strength <= 100

    def test_fallback_strength_calculation(self):
        """Test fallback strength with momentum."""
        calculator = LSTMSignalCalculator()

        # Create trending data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Close': np.linspace(100, 120, 30)
        }, index=dates)

        strength = calculator._fallback_strength(df)

        assert strength > 0  # Should detect upward trend

    def test_fallback_strength_short_data(self):
        """Test fallback with short data."""
        calculator = LSTMSignalCalculator()

        df = pd.DataFrame({'Close': [100, 101, 102]})

        strength = calculator._fallback_strength(df)

        assert strength == 0.0

    def test_fallback_strength_exception(self):
        """Test fallback exception handling."""
        calculator = LSTMSignalCalculator()

        # Invalid data
        df = pd.DataFrame({'Close': [np.nan] * 30})

        strength = calculator._fallback_strength(df)

        assert strength == 0.0


class TestCalculateConfidence:
    """Tests for confidence calculation."""

    @pytest.fixture
    def trained_calculator(self):
        """Create a trained calculator."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            batch_size=16
        )
        calculator = LSTMSignalCalculator(config=config)

        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        prices = np.cumsum(np.random.randn(250) * 2) + 100
        df = pd.DataFrame({'Close': prices}, index=dates)

        calculator._ensure_model_trained("AAPL", df)
        return calculator, df

    def test_calculate_confidence_success(self, trained_calculator):
        """Test successful confidence calculation."""
        calculator, df = trained_calculator

        confidence = calculator.calculate_confidence(df, symbol="AAPL")

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_calculate_confidence_insufficient_data(self, trained_calculator):
        """Test confidence with insufficient data."""
        calculator, _ = trained_calculator

        small_df = pd.DataFrame({'Close': [100, 101]})

        confidence = calculator.calculate_confidence(small_df, symbol="AAPL")

        assert confidence == 0.0

    def test_calculate_confidence_with_ensemble(self):
        """Test confidence with ensemble."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            batch_size=16
        )
        calculator = LSTMSignalCalculator(
            config=config,
            use_ensemble=True,
            ensemble_size=2
        )

        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        df = pd.DataFrame({
            'Close': np.cumsum(np.random.randn(250) * 2) + 100
        }, index=dates)

        calculator._ensure_model_trained("AAPL", df)
        confidence = calculator.calculate_confidence(df, symbol="AAPL")

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_calculate_confidence_fallback(self):
        """Test fallback confidence calculation."""
        calculator = LSTMSignalCalculator()

        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Close': np.linspace(100, 110, 30)
        }, index=dates)

        confidence = calculator.calculate_confidence(df, symbol="UNKNOWN")

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_fallback_confidence_consistent_trend(self):
        """Test fallback confidence with consistent trend."""
        calculator = LSTMSignalCalculator()

        # All up moves
        df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        })

        confidence = calculator._fallback_confidence(df)

        assert confidence > 0.8  # High confidence for consistent trend

    def test_fallback_confidence_random(self):
        """Test fallback confidence with random moves."""
        calculator = LSTMSignalCalculator()

        np.random.seed(42)
        df = pd.DataFrame({
            'Close': 100 + np.random.randn(20)
        })

        confidence = calculator._fallback_confidence(df)

        assert 0 <= confidence <= 1

    def test_fallback_confidence_short_data(self):
        """Test fallback confidence with short data."""
        calculator = LSTMSignalCalculator()

        df = pd.DataFrame({'Close': [100, 101]})

        confidence = calculator._fallback_confidence(df)

        assert confidence == 0.0


class TestPredictionDetails:
    """Tests for get_prediction_details."""

    @pytest.fixture
    def trained_calculator(self):
        """Create a trained calculator."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            batch_size=16
        )
        calculator = LSTMSignalCalculator(config=config)

        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        df = pd.DataFrame({
            'Close': np.cumsum(np.random.randn(250) * 2) + 100
        }, index=dates)

        calculator._ensure_model_trained("AAPL", df)
        return calculator, df

    def test_get_prediction_details_success(self, trained_calculator):
        """Test getting prediction details."""
        calculator, df = trained_calculator

        details = calculator.get_prediction_details(df, symbol="AAPL")

        assert isinstance(details, dict)
        assert "symbol" in details
        assert "current_price" in details
        assert "predicted_price" in details
        assert "expected_change_pct" in details
        assert "trend" in details
        assert "confidence" in details
        assert "model_type" in details

        assert details["symbol"] == "AAPL"
        assert details["model_type"] == "single"

    def test_get_prediction_details_with_ensemble(self):
        """Test prediction details with ensemble."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            batch_size=16
        )
        calculator = LSTMSignalCalculator(
            config=config,
            use_ensemble=True,
            ensemble_size=2
        )

        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        df = pd.DataFrame({
            'Close': np.cumsum(np.random.randn(250) * 2) + 100
        }, index=dates)

        calculator._ensure_model_trained("AAPL", df)
        details = calculator.get_prediction_details(df, symbol="AAPL")

        assert details["model_type"] == "ensemble"
        assert "uncertainty" in details
        assert "ensemble_size" in details

    def test_get_prediction_details_insufficient_data(self, trained_calculator):
        """Test prediction details with insufficient data."""
        calculator, _ = trained_calculator

        small_df = pd.DataFrame({'Close': [100, 101]})

        details = calculator.get_prediction_details(small_df, symbol="AAPL")

        assert "error" in details
        assert details["error"] == "insufficient_data"

    def test_get_prediction_details_not_trained(self):
        """Test prediction details without training."""
        calculator = LSTMSignalCalculator()

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({'Close': np.random.randn(100) + 100}, index=dates)

        details = calculator.get_prediction_details(df, symbol="UNKNOWN")

        assert "error" in details
        assert details["error"] == "model_not_trained"

    def test_get_prediction_details_includes_multistep(self, trained_calculator):
        """Test that single model includes multi-step predictions."""
        calculator, df = trained_calculator

        details = calculator.get_prediction_details(df, symbol="AAPL")

        if details.get("model_type") == "single":
            assert "predictions_5_steps" in details
            assert len(details["predictions_5_steps"]) == 5


class TestModelPersistence:
    """Tests for model saving and loading."""

    @pytest.fixture
    def trained_calculator(self):
        """Create a trained calculator."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=3,
            batch_size=16
        )
        calculator = LSTMSignalCalculator(config=config)

        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        df = pd.DataFrame({
            'Close': np.cumsum(np.random.randn(250) * 2) + 100
        }, index=dates)

        calculator._ensure_model_trained("AAPL", df)
        calculator._ensure_model_trained("GOOGL", df)

        return calculator, df

    def test_save_models(self, trained_calculator):
        """Test saving models."""
        calculator, _ = trained_calculator

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = calculator.save_models(tmpdir)

            assert isinstance(saved_paths, dict)
            assert "AAPL" in saved_paths
            assert "GOOGL" in saved_paths

            # Check files exist
            for path in saved_paths.values():
                assert os.path.exists(path)

    def test_load_model(self, trained_calculator):
        """Test loading a model."""
        calculator, df = trained_calculator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            saved_paths = calculator.save_models(tmpdir)
            aapl_path = saved_paths["AAPL"]

            # Create new calculator and load
            new_calculator = LSTMSignalCalculator()
            success = new_calculator.load_model("AAPL", aapl_path)

            assert success is True
            assert "AAPL" in new_calculator._trained_symbols
            assert "AAPL" in new_calculator._models

            # Test prediction
            prediction = new_calculator.calculate_raw_strength(df, symbol="AAPL")
            assert isinstance(prediction, float)

    def test_load_model_invalid_path(self):
        """Test loading from invalid path."""
        calculator = LSTMSignalCalculator()

        success = calculator.load_model("AAPL", "/invalid/path/model.pt")

        assert success is False
        assert "AAPL" not in calculator._trained_symbols


class TestModelManagement:
    """Tests for model cache management."""

    def test_clear_models(self):
        """Test clearing all models."""
        calculator = LSTMSignalCalculator()

        # Add some mock data
        calculator._trained_symbols.add("AAPL")
        calculator._trained_symbols.add("GOOGL")
        calculator._models["AAPL"] = Mock()
        calculator._ensembles["GOOGL"] = Mock()

        calculator.clear_models()

        assert len(calculator._models) == 0
        assert len(calculator._ensembles) == 0
        assert len(calculator._trained_symbols) == 0

    def test_multiple_symbols(self):
        """Test handling multiple symbols."""
        config = LSTMConfig(
            seq_length=20,
            hidden_size=32,
            num_layers=1,
            epochs=2,
            batch_size=16
        )
        calculator = LSTMSignalCalculator(config=config)

        dates = pd.date_range('2023-01-01', periods=250, freq='D')

        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            df = pd.DataFrame({
                'Close': np.cumsum(np.random.randn(250) * 2) + 100
            }, index=dates)
            calculator._ensure_model_trained(symbol, df)

        assert len(calculator._trained_symbols) == 3
        assert len(calculator._models) == 3


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        calculator = LSTMSignalCalculator()

        df = pd.DataFrame()

        strength = calculator.calculate_raw_strength(df)
        assert strength == 0.0

        confidence = calculator.calculate_confidence(df)
        assert confidence == 0.0

    def test_nan_values(self):
        """Test with NaN values."""
        calculator = LSTMSignalCalculator()

        df = pd.DataFrame({
            'Close': [100, np.nan, 102, 103, np.nan]
        })

        # Should handle gracefully
        strength = calculator.calculate_raw_strength(df, symbol="TEST")
        assert isinstance(strength, float)

    def test_constant_prices(self):
        """Test with constant prices."""
        calculator = LSTMSignalCalculator()

        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Close': [100.0] * 30
        }, index=dates)

        strength = calculator.calculate_raw_strength(df, symbol="TEST")
        confidence = calculator.calculate_confidence(df, symbol="TEST")

        assert isinstance(strength, float)
        assert isinstance(confidence, float)

    def test_extreme_volatility(self):
        """Test with extreme price volatility."""
        calculator = LSTMSignalCalculator()

        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Close': 100 + np.random.randn(30) * 50
        }, index=dates)

        strength = calculator.calculate_raw_strength(df, symbol="TEST")

        assert isinstance(strength, float)
        assert 0 <= strength <= 100

    def test_negative_prices(self):
        """Test handling of negative prices."""
        calculator = LSTMSignalCalculator()

        df = pd.DataFrame({
            'Close': [-10, -5, 0, 5, 10]
        })

        # Should handle without crashing
        strength = calculator.calculate_raw_strength(df, symbol="TEST")
        assert isinstance(strength, float)

    def test_very_large_prices(self):
        """Test with very large prices."""
        calculator = LSTMSignalCalculator()

        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Close': np.linspace(1000000, 1100000, 30)
        }, index=dates)

        strength = calculator.calculate_raw_strength(df, symbol="TEST")

        assert isinstance(strength, float)
        assert 0 <= strength <= 100

    def test_prediction_exception_handling(self):
        """Test exception handling in predictions."""
        calculator = LSTMSignalCalculator()

        # Mock a model that raises an exception
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        calculator._models["TEST"] = mock_model
        calculator._trained_symbols.add("TEST")

        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({'Close': np.random.randn(100) + 100}, index=dates)

        # Should fall back gracefully
        strength = calculator.calculate_raw_strength(df, symbol="TEST")

        assert isinstance(strength, float)
