"""
LSTM Signal Strength Calculator

Integrates LSTM predictions with the signal validation framework
for standardized signal strength scoring.
"""

import logging
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from backend.tradingbot.validation.signal_strength_validator import (
    SignalStrengthCalculator,
    SignalType,
)
from .lstm_predictor import LSTMPricePredictor, LSTMConfig, LSTMEnsemble


class LSTMSignalCalculator(SignalStrengthCalculator):
    """
    Signal strength calculator using LSTM deep learning predictions.

    Implements the SignalStrengthCalculator interface for integration
    with the signal validation framework.
    """

    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
        use_ensemble: bool = False,
        ensemble_size: int = 3,
        min_training_samples: int = 200,
    ):
        """
        Initialize the LSTM signal calculator.

        Args:
            config: LSTM configuration
            use_ensemble: Whether to use ensemble predictions
            ensemble_size: Number of models in ensemble
            min_training_samples: Minimum samples needed for training
        """
        self.config = config or LSTMConfig()
        self.use_ensemble = use_ensemble
        self.ensemble_size = ensemble_size
        self.min_training_samples = min_training_samples

        # Models cache (symbol -> trained model)
        self._models: Dict[str, LSTMPricePredictor] = {}
        self._ensembles: Dict[str, LSTMEnsemble] = {}

        # Training status
        self._trained_symbols: set = set()

        self.logger = logging.getLogger(__name__)
        self.params = {
            'config': self.config,
            'use_ensemble': use_ensemble,
            'ensemble_size': ensemble_size,
            'min_training_samples': min_training_samples,
        }

    def get_signal_type(self) -> SignalType:
        """Return the signal type this calculator handles."""
        return SignalType.TREND

    def _ensure_model_trained(
        self,
        symbol: str,
        market_data: pd.DataFrame,
    ) -> bool:
        """
        Ensure a model is trained for the given symbol.

        Args:
            symbol: Stock symbol
            market_data: Historical market data

        Returns:
            True if model is trained and ready
        """
        if symbol in self._trained_symbols:
            return True

        if len(market_data) < self.min_training_samples:
            self.logger.warning(
                f"Insufficient data for {symbol}: "
                f"{len(market_data)} < {self.min_training_samples}"
            )
            return False

        try:
            prices = market_data['Close'].values

            if self.use_ensemble:
                ensemble = LSTMEnsemble(self.ensemble_size, self.config)
                ensemble.train(prices, verbose=False)
                self._ensembles[symbol] = ensemble
            else:
                model = LSTMPricePredictor(self.config)
                model.train(prices, verbose=False)
                self._models[symbol] = model

            self._trained_symbols.add(symbol)
            return True

        except Exception as e:
            self.logger.error(f"Failed to train model for {symbol}: {e}")
            return False

    def calculate_raw_strength(
        self,
        market_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        **kwargs
    ) -> float:
        """
        Calculate raw signal strength based on LSTM predictions.

        Args:
            market_data: DataFrame with 'Close' column
            symbol: Stock symbol for model lookup
            **kwargs: Additional parameters

        Returns:
            Signal strength score (0-100)
        """
        if market_data is None or len(market_data) < self.config.seq_length:
            return 0.0

        # Ensure model is trained
        if not self._ensure_model_trained(symbol, market_data):
            # Fall back to simple momentum if model can't be trained
            return self._fallback_strength(market_data)

        try:
            prices = market_data['Close'].values

            if self.use_ensemble and symbol in self._ensembles:
                ensemble = self._ensembles[symbol]
                predicted_price, uncertainty = ensemble.predict(prices)
                trend, confidence = ensemble.predict_trend(prices)
            elif symbol in self._models:
                model = self._models[symbol]
                predicted_price = model.predict(prices)
                trend = model.predict_trend(prices)
                confidence = model.get_trend_confidence(prices)
                uncertainty = 0.0
            else:
                return self._fallback_strength(market_data)

            current_price = prices[-1]
            pct_change = abs((predicted_price - current_price) / current_price * 100)

            # Calculate strength based on:
            # 1. Magnitude of predicted change
            # 2. Confidence in prediction
            # 3. Trend direction clarity

            # Magnitude score (0-50): Higher change = stronger signal
            magnitude_score = min(50, pct_change * 10)

            # Confidence score (0-30): Higher confidence = stronger signal
            confidence_score = confidence * 30

            # Direction score (0-20): Clear up/down = stronger signal
            if trend in ("up", "down"):
                direction_score = 20
            else:
                direction_score = 5

            # Reduce strength for high uncertainty (ensemble only)
            if self.use_ensemble and uncertainty > 0:
                # High uncertainty reduces strength
                uncertainty_penalty = min(20, uncertainty / current_price * 100)
                total_strength = (
                    magnitude_score + confidence_score + direction_score
                    - uncertainty_penalty
                )
            else:
                total_strength = magnitude_score + confidence_score + direction_score

            return min(100.0, max(0.0, total_strength))

        except Exception as e:
            self.logger.error(f"Error calculating LSTM strength: {e}")
            return self._fallback_strength(market_data)

    def _fallback_strength(self, market_data: pd.DataFrame) -> float:
        """
        Calculate fallback strength using simple momentum.

        Used when LSTM model isn't available.
        """
        try:
            prices = market_data['Close'].values
            if len(prices) < 20:
                return 0.0

            # Simple momentum calculation
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-20:])
            momentum = (short_ma - long_ma) / long_ma * 100

            # Convert to strength score
            strength = abs(momentum) * 10
            return min(100.0, max(0.0, strength))

        except Exception:
            return 0.0

    def calculate_confidence(
        self,
        market_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        **kwargs
    ) -> float:
        """
        Calculate signal confidence based on LSTM predictions.

        Args:
            market_data: DataFrame with 'Close' column
            symbol: Stock symbol for model lookup
            **kwargs: Additional parameters

        Returns:
            Confidence score (0-1)
        """
        if market_data is None or len(market_data) < self.config.seq_length:
            return 0.0

        try:
            prices = market_data['Close'].values

            if self.use_ensemble and symbol in self._ensembles:
                ensemble = self._ensembles[symbol]
                _, confidence = ensemble.predict_trend(prices)
                return confidence
            elif symbol in self._models:
                model = self._models[symbol]
                return model.get_trend_confidence(prices)
            else:
                # Fallback confidence based on trend consistency
                return self._fallback_confidence(market_data)

        except Exception as e:
            self.logger.error(f"Error calculating LSTM confidence: {e}")
            return self._fallback_confidence(market_data)

    def _fallback_confidence(self, market_data: pd.DataFrame) -> float:
        """
        Calculate fallback confidence using price consistency.
        """
        try:
            prices = market_data['Close'].values
            if len(prices) < 10:
                return 0.0

            # Count consecutive up/down moves
            changes = np.diff(prices[-10:])
            positive = sum(1 for c in changes if c > 0)
            negative = sum(1 for c in changes if c < 0)

            # Higher consistency = higher confidence
            consistency = max(positive, negative) / len(changes)
            return min(1.0, max(0.0, consistency))

        except Exception:
            return 0.0

    def get_prediction_details(
        self,
        market_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """
        Get detailed prediction information.

        Args:
            market_data: DataFrame with price data
            symbol: Stock symbol

        Returns:
            Dictionary with prediction details
        """
        if market_data is None or len(market_data) < self.config.seq_length:
            return {"error": "insufficient_data"}

        if symbol not in self._trained_symbols:
            if not self._ensure_model_trained(symbol, market_data):
                return {"error": "model_not_trained"}

        try:
            prices = market_data['Close'].values
            current_price = prices[-1]

            if self.use_ensemble and symbol in self._ensembles:
                ensemble = self._ensembles[symbol]
                predicted_price, uncertainty = ensemble.predict(prices)
                trend, confidence = ensemble.predict_trend(prices)

                return {
                    "symbol": symbol,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "expected_change_pct": (predicted_price - current_price) / current_price * 100,
                    "trend": trend,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "model_type": "ensemble",
                    "ensemble_size": self.ensemble_size,
                }
            elif symbol in self._models:
                model = self._models[symbol]
                predicted_price = model.predict(prices)
                trend = model.predict_trend(prices)
                confidence = model.get_trend_confidence(prices)

                # Get multi-step predictions
                predictions_5 = model.predict_next_n(prices, n=5)

                return {
                    "symbol": symbol,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "expected_change_pct": (predicted_price - current_price) / current_price * 100,
                    "trend": trend,
                    "confidence": confidence,
                    "predictions_5_steps": predictions_5,
                    "model_type": "single",
                }
            else:
                return {"error": "model_not_available"}

        except Exception as e:
            return {"error": str(e)}

    def clear_models(self) -> None:
        """Clear all cached models."""
        self._models.clear()
        self._ensembles.clear()
        self._trained_symbols.clear()

    def save_models(self, directory: str) -> Dict[str, str]:
        """
        Save all trained models to disk.

        Args:
            directory: Directory to save models

        Returns:
            Dictionary mapping symbols to saved file paths
        """
        import os
        os.makedirs(directory, exist_ok=True)

        saved_paths = {}
        for symbol, model in self._models.items():
            path = os.path.join(directory, f"lstm_{symbol}.pt")
            model.save_model(path)
            saved_paths[symbol] = path

        return saved_paths

    def load_model(self, symbol: str, path: str) -> bool:
        """
        Load a model from disk.

        Args:
            symbol: Stock symbol
            path: Path to saved model

        Returns:
            True if loaded successfully
        """
        try:
            model = LSTMPricePredictor(self.config)
            model.load_model(path)
            self._models[symbol] = model
            self._trained_symbols.add(symbol)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model for {symbol}: {e}")
            return False
