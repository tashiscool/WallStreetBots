"""
Ensemble Price Predictor

Combines LSTM, Transformer, and CNN models for robust price prediction
with weighted averaging, stacking, and uncertainty estimation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

from .lstm_predictor import LSTMPricePredictor, LSTMConfig
from .transformer_predictor import TransformerPricePredictor, TransformerConfig
from .cnn_predictor import CNNPricePredictor, CNNConfig


class EnsembleMethod(Enum):
    """Methods for combining model predictions."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking"
    VOTING = "voting"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""
    # Model configs
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    cnn_config: CNNConfig = field(default_factory=CNNConfig)

    # Ensemble settings
    ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE
    use_lstm: bool = True
    use_transformer: bool = True
    use_cnn: bool = True

    # Weights for weighted average (auto-calculated if None)
    lstm_weight: Optional[float] = None
    transformer_weight: Optional[float] = None
    cnn_weight: Optional[float] = None

    # Stacking settings
    stacking_alpha: float = 1.0  # Ridge regression alpha

    # Training settings
    validation_split: float = 0.2
    verbose: bool = True


@dataclass
class EnsemblePrediction:
    """Result of ensemble prediction."""
    predicted_price: float
    current_price: float
    expected_change_pct: float
    trend: str
    confidence: float
    uncertainty: float  # Standard deviation of model predictions

    # Individual model predictions
    lstm_prediction: Optional[float] = None
    transformer_prediction: Optional[float] = None
    cnn_prediction: Optional[float] = None

    # Model agreement
    model_agreement: float = 0.0  # 0-1, how much models agree
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_price": self.predicted_price,
            "current_price": self.current_price,
            "expected_change_pct": self.expected_change_pct,
            "trend": self.trend,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "lstm_prediction": self.lstm_prediction,
            "transformer_prediction": self.transformer_prediction,
            "cnn_prediction": self.cnn_prediction,
            "model_agreement": self.model_agreement,
            "timestamp": self.timestamp.isoformat(),
        }


class EnsemblePricePredictor:
    """
    Full Ensemble predictor combining LSTM, Transformer, and CNN.

    Features:
    - Multiple model architectures for diverse perspectives
    - Weighted averaging based on validation performance
    - Stacking with meta-learner
    - Uncertainty estimation
    - Model agreement tracking
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()

        # Initialize models
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.model_val_losses: Dict[str, float] = {}

        if self.config.use_lstm:
            self.models['lstm'] = LSTMPricePredictor(self.config.lstm_config)

        if self.config.use_transformer:
            self.models['transformer'] = TransformerPricePredictor(self.config.transformer_config)

        if self.config.use_cnn:
            self.models['cnn'] = CNNPricePredictor(self.config.cnn_config)

        # Meta-learner for stacking
        self.meta_learner: Optional[Ridge] = None
        self.meta_scaler = MinMaxScaler()

        self.is_trained = False
        self.training_metrics: Dict[str, Any] = {}

    def train(
        self,
        prices: np.ndarray,
        validation_split: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Train all ensemble models.

        Args:
            prices: Historical price data
            validation_split: Fraction for validation
            verbose: Print training progress

        Returns:
            Training metrics for all models
        """
        validation_split = validation_split or self.config.validation_split
        verbose = verbose if verbose is not None else self.config.verbose

        all_metrics = {}

        # Train each model
        for name, model in self.models.items():
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training {name.upper()} model...")
                print('='*50)

            # Set random seed for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)

            metrics = model.train(
                prices,
                validation_split=validation_split,
                verbose=verbose
            )
            all_metrics[name] = metrics
            self.model_val_losses[name] = metrics['best_val_loss']

        # Calculate weights based on validation performance
        self._calculate_weights()

        # Train meta-learner for stacking
        if self.config.ensemble_method == EnsembleMethod.STACKING:
            self._train_meta_learner(prices, validation_split)

        self.is_trained = True
        self.training_metrics = all_metrics

        if verbose:
            print(f"\n{'='*50}")
            print("ENSEMBLE TRAINING COMPLETE")
            print('='*50)
            print(f"Model weights: {self.model_weights}")

        return all_metrics

    def _calculate_weights(self) -> None:
        """Calculate model weights based on validation loss."""
        if self.config.ensemble_method == EnsembleMethod.SIMPLE_AVERAGE:
            n_models = len(self.models)
            self.model_weights = {name: 1.0 / n_models for name in self.models}
            return

        # Check if weights are manually specified
        if all([
            self.config.lstm_weight is not None,
            self.config.transformer_weight is not None,
            self.config.cnn_weight is not None,
        ]):
            total = (
                (self.config.lstm_weight if self.config.use_lstm else 0) +
                (self.config.transformer_weight if self.config.use_transformer else 0) +
                (self.config.cnn_weight if self.config.use_cnn else 0)
            )
            if self.config.use_lstm:
                self.model_weights['lstm'] = self.config.lstm_weight / total
            if self.config.use_transformer:
                self.model_weights['transformer'] = self.config.transformer_weight / total
            if self.config.use_cnn:
                self.model_weights['cnn'] = self.config.cnn_weight / total
            return

        # Calculate weights inversely proportional to validation loss
        # Lower loss = higher weight
        inv_losses = {name: 1.0 / (loss + 1e-8) for name, loss in self.model_val_losses.items()}
        total_inv = sum(inv_losses.values())
        self.model_weights = {name: inv_loss / total_inv for name, inv_loss in inv_losses.items()}

    def _train_meta_learner(self, prices: np.ndarray, validation_split: float) -> None:
        """Train meta-learner for stacking."""
        # Generate predictions on validation set
        split_idx = int(len(prices) * (1 - validation_split))
        val_prices = prices[split_idx:]

        if len(val_prices) < self.config.lstm_config.seq_length + 10:
            return

        # Collect predictions
        X_meta = []
        y_meta = []

        seq_length = self.config.lstm_config.seq_length

        for i in range(seq_length, len(val_prices) - 1):
            input_seq = val_prices[i - seq_length:i]
            actual = val_prices[i]

            preds = []
            for name, model in self.models.items():
                try:
                    pred = model.predict(input_seq)
                    preds.append(pred)
                except Exception:
                    continue

            if len(preds) == len(self.models):
                X_meta.append(preds)
                y_meta.append(actual)

        if len(X_meta) < 10:
            return

        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)

        # Scale features
        X_meta_scaled = self.meta_scaler.fit_transform(X_meta)

        # Train Ridge regression
        self.meta_learner = Ridge(alpha=self.config.stacking_alpha)
        self.meta_learner.fit(X_meta_scaled, y_meta)

    def predict(self, prices: np.ndarray) -> EnsemblePrediction:
        """
        Generate ensemble prediction.

        Args:
            prices: Recent price history

        Returns:
            EnsemblePrediction with all details
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call train() first.")

        current_price = prices[-1]
        predictions = {}
        trends = {}

        # Get predictions from each model
        for name, model in self.models.items():
            try:
                pred = model.predict(prices)
                predictions[name] = pred
                trends[name] = model.predict_trend(prices)
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
                continue

        if not predictions:
            raise RuntimeError("All model predictions failed")

        # Combine predictions
        if self.config.ensemble_method == EnsembleMethod.STACKING and self.meta_learner:
            pred_values = [predictions[name] for name in self.models.keys() if name in predictions]
            X_meta = np.array([pred_values])
            X_meta_scaled = self.meta_scaler.transform(X_meta)
            final_prediction = float(self.meta_learner.predict(X_meta_scaled)[0])
        else:
            # Weighted average
            total_weight = sum(self.model_weights.get(name, 0) for name in predictions)
            final_prediction = sum(
                pred * self.model_weights.get(name, 0) / total_weight
                for name, pred in predictions.items()
            )

        # Calculate uncertainty (std dev of predictions)
        pred_values = list(predictions.values())
        uncertainty = float(np.std(pred_values)) if len(pred_values) > 1 else 0.0

        # Calculate model agreement
        trend_counts = {}
        for trend in trends.values():
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        max_agreement = max(trend_counts.values()) if trend_counts else 0
        model_agreement = max_agreement / len(trends) if trends else 0.0

        # Determine final trend
        pct_change = (final_prediction - current_price) / current_price * 100
        if pct_change > 0.5:
            final_trend = "up"
        elif pct_change < -0.5:
            final_trend = "down"
        else:
            final_trend = "sideways"

        # Calculate confidence based on agreement and prediction strength
        strength_confidence = min(abs(pct_change) / 2.0, 1.0)
        agreement_confidence = model_agreement
        confidence = (strength_confidence * 0.4 + agreement_confidence * 0.6)

        return EnsemblePrediction(
            predicted_price=final_prediction,
            current_price=current_price,
            expected_change_pct=pct_change,
            trend=final_trend,
            confidence=confidence,
            uncertainty=uncertainty,
            lstm_prediction=predictions.get('lstm'),
            transformer_prediction=predictions.get('transformer'),
            cnn_prediction=predictions.get('cnn'),
            model_agreement=model_agreement,
        )

    def predict_next_n(self, prices: np.ndarray, n: int = 5) -> List[EnsemblePrediction]:
        """Predict the next n prices using autoregressive prediction."""
        predictions = []
        current_sequence = list(prices[-self.config.lstm_config.seq_length:])

        for _ in range(n):
            pred = self.predict(np.array(current_sequence))
            predictions.append(pred)
            current_sequence = current_sequence[1:] + [pred.predicted_price]

        return predictions

    def get_model_contributions(self, prices: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Get detailed contribution from each model."""
        contributions = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(prices)
                trend = model.predict_trend(prices)
                confidence = model.get_trend_confidence(prices)

                contributions[name] = {
                    "prediction": pred,
                    "trend": trend,
                    "confidence": confidence,
                    "weight": self.model_weights.get(name, 0),
                    "validation_loss": self.model_val_losses.get(name, 0),
                }
            except Exception as e:
                contributions[name] = {"error": str(e)}

        return contributions

    def save_ensemble(self, directory: str) -> None:
        """Save all models in the ensemble."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained ensemble")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model_path = path / f"{name}_model.pt"
            model.save_model(str(model_path))

        # Save ensemble metadata
        import json
        metadata = {
            "model_weights": self.model_weights,
            "model_val_losses": self.model_val_losses,
            "ensemble_method": self.config.ensemble_method.value,
            "models_used": list(self.models.keys()),
        }
        with open(path / "ensemble_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save meta-learner if stacking
        if self.meta_learner:
            import pickle
            with open(path / "meta_learner.pkl", "wb") as f:
                pickle.dump({
                    'meta_learner': self.meta_learner,
                    'meta_scaler': self.meta_scaler,
                }, f)

    def load_ensemble(self, directory: str) -> None:
        """Load all models in the ensemble."""
        path = Path(directory)

        # Load metadata
        import json
        with open(path / "ensemble_metadata.json", "r") as f:
            metadata = json.load(f)

        self.model_weights = metadata["model_weights"]
        self.model_val_losses = metadata["model_val_losses"]

        # Load models
        for name in metadata["models_used"]:
            if name in self.models:
                model_path = path / f"{name}_model.pt"
                self.models[name].load_model(str(model_path))

        # Load meta-learner if exists
        meta_path = path / "meta_learner.pkl"
        if meta_path.exists():
            import pickle
            with open(meta_path, "rb") as f:
                meta_data = pickle.load(f)
                self.meta_learner = meta_data['meta_learner']
                self.meta_scaler = meta_data['meta_scaler']

        self.is_trained = True


class EnsembleHyperparameterTuner:
    """
    Hyperparameter tuning for ensemble models.

    Supports grid search and random search for optimal configurations.
    """

    def __init__(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        n_iter: int = 10,
        cv_folds: int = 3,
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            param_grid: Parameter search space
            n_iter: Number of iterations for random search
            cv_folds: Number of cross-validation folds
        """
        self.param_grid = param_grid or self._get_default_param_grid()
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.best_params: Optional[Dict] = None
        self.best_score: float = float('inf')
        self.results: List[Dict] = []

    def _get_default_param_grid(self) -> Dict[str, List[Any]]:
        """Get default parameter search space."""
        return {
            'lstm_hidden_size': [64, 128, 256],
            'lstm_num_layers': [1, 2, 3],
            'transformer_d_model': [32, 64, 128],
            'transformer_nhead': [2, 4, 8],
            'cnn_num_filters': [[32, 64], [32, 64, 128], [64, 128, 256]],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.001, 0.01],
            'seq_length': [30, 60, 90],
        }

    def random_search(
        self,
        prices: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform random search hyperparameter tuning.

        Args:
            prices: Training data
            verbose: Print progress

        Returns:
            Best parameters found
        """
        import random

        for i in range(self.n_iter):
            # Sample random parameters
            params = {
                key: random.choice(values)
                for key, values in self.param_grid.items()
            }

            if verbose:
                print(f"\nIteration {i+1}/{self.n_iter}")
                print(f"Testing params: {params}")

            # Create config with sampled params
            config = self._create_config_from_params(params)

            # Cross-validation score
            score = self._cross_validate(prices, config)

            self.results.append({
                'params': params,
                'score': score,
            })

            if score < self.best_score:
                self.best_score = score
                self.best_params = params
                if verbose:
                    print(f"New best score: {score:.6f}")

        if verbose:
            print(f"\nBest parameters: {self.best_params}")
            print(f"Best score: {self.best_score:.6f}")

        return self.best_params

    def _create_config_from_params(self, params: Dict) -> EnsembleConfig:
        """Create EnsembleConfig from parameter dict."""
        lstm_config = LSTMConfig(
            hidden_size=params.get('lstm_hidden_size', 128),
            num_layers=params.get('lstm_num_layers', 2),
            dropout=params.get('dropout', 0.2),
            learning_rate=params.get('learning_rate', 0.001),
            seq_length=params.get('seq_length', 60),
        )

        transformer_config = TransformerConfig(
            d_model=params.get('transformer_d_model', 64),
            nhead=params.get('transformer_nhead', 4),
            dropout=params.get('dropout', 0.2),
            learning_rate=params.get('learning_rate', 0.001),
            seq_length=params.get('seq_length', 60),
        )

        cnn_config = CNNConfig(
            num_filters=params.get('cnn_num_filters', [32, 64, 128]),
            dropout=params.get('dropout', 0.2),
            learning_rate=params.get('learning_rate', 0.001),
            seq_length=params.get('seq_length', 60),
        )

        return EnsembleConfig(
            lstm_config=lstm_config,
            transformer_config=transformer_config,
            cnn_config=cnn_config,
            verbose=False,
        )

    def _cross_validate(self, prices: np.ndarray, config: EnsembleConfig) -> float:
        """Perform time-series cross-validation."""
        n = len(prices)
        fold_size = n // (self.cv_folds + 1)
        scores = []

        for fold in range(self.cv_folds):
            # Time-series split (no shuffling)
            train_end = fold_size * (fold + 1)
            val_start = train_end
            val_end = min(val_start + fold_size, n)

            train_prices = prices[:train_end]
            val_prices = prices[val_start:val_end]

            if len(train_prices) < config.lstm_config.seq_length + 50:
                continue

            try:
                # Train ensemble
                ensemble = EnsemblePricePredictor(config)
                ensemble.train(train_prices, verbose=False)

                # Evaluate on validation
                mse = self._evaluate(ensemble, train_prices, val_prices)
                scores.append(mse)
            except Exception as e:
                print(f"Fold {fold} failed: {e}")
                continue

        return np.mean(scores) if scores else float('inf')

    def _evaluate(
        self,
        ensemble: EnsemblePricePredictor,
        train_prices: np.ndarray,
        val_prices: np.ndarray,
    ) -> float:
        """Evaluate ensemble on validation set."""
        seq_length = ensemble.config.lstm_config.seq_length
        errors = []

        # Use end of training data as initial context
        context = list(train_prices[-seq_length:])

        for i, actual in enumerate(val_prices[:20]):  # Limit for speed
            try:
                pred = ensemble.predict(np.array(context))
                error = (pred.predicted_price - actual) ** 2
                errors.append(error)
                context = context[1:] + [actual]
            except Exception:
                continue

        return np.mean(errors) if errors else float('inf')

    def get_best_ensemble(self) -> EnsemblePricePredictor:
        """Get ensemble with best parameters."""
        if self.best_params is None:
            raise RuntimeError("No tuning performed yet")

        config = self._create_config_from_params(self.best_params)
        return EnsemblePricePredictor(config)
