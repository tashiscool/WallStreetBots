"""
LSTM Price Predictor Component

A PyTorch-based LSTM model for price/trend prediction that integrates
with the existing signal validation framework.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    input_size: int = 1  # Number of features (price only by default)
    hidden_size: int = 128  # LSTM hidden layer size
    num_layers: int = 2  # Number of LSTM layers
    output_size: int = 1  # Prediction output size
    seq_length: int = 60  # Lookback window (60 time steps)
    dropout: float = 0.2  # Dropout rate for regularization
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10


class LSTMModel(nn.Module):
    """PyTorch LSTM model for time series prediction."""

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM."""
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        last_out = lstm_out[:, -1, :]
        predictions = self.fc(last_out)
        return predictions


class LSTMDataManager:
    """Manages data preparation for LSTM model."""

    def __init__(self, seq_length: int = 60):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def prepare_data(
        self,
        prices: np.ndarray,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.

        Args:
            prices: 1D array of prices
            fit_scaler: Whether to fit the scaler (True for training)

        Returns:
            X: Sequences of shape (samples, seq_length, 1)
            y: Targets of shape (samples, 1)
        """
        prices = np.array(prices).reshape(-1, 1)

        if fit_scaler:
            scaled_prices = self.scaler.fit_transform(prices)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            scaled_prices = self.scaler.transform(prices)

        X, y = [], []
        for i in range(len(scaled_prices) - self.seq_length):
            X.append(scaled_prices[i:i + self.seq_length])
            y.append(scaled_prices[i + self.seq_length])

        return np.array(X), np.array(y)

    def prepare_sequence(self, prices: np.ndarray) -> np.ndarray:
        """Prepare a single sequence for prediction."""
        if len(prices) < self.seq_length:
            raise ValueError(f"Need at least {self.seq_length} prices")

        prices = np.array(prices[-self.seq_length:]).reshape(-1, 1)
        scaled = self.scaler.transform(prices)
        return scaled.reshape(1, self.seq_length, 1)

    def inverse_transform(self, scaled_values: np.ndarray) -> np.ndarray:
        """Convert scaled values back to original scale."""
        return self.scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()


class LSTMPricePredictor:
    """
    LSTM-based price predictor for trading signal generation.

    Follows the existing component patterns (HMM, Monte Carlo) for
    integration with the signal validation framework.
    """

    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.model = LSTMModel(self.config)
        self.data_manager = LSTMDataManager(self.config.seq_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.is_trained = False
        self.training_loss_history: List[float] = []
        self.validation_loss_history: List[float] = []

    def train(
        self,
        prices: np.ndarray,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> dict:
        """
        Train the LSTM model on historical price data.

        Args:
            prices: Array of historical prices
            validation_split: Fraction of data for validation
            verbose: Print training progress

        Returns:
            Training metrics dictionary
        """
        # Prepare data
        X, y = self.data_manager.prepare_data(prices, fit_scaler=True)

        # Split into train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            self.training_loss_history.append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            self.validation_loss_history.append(val_loss)
            self.model.train()

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} - "
                      f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(best_state)
        self.is_trained = True

        return {
            "final_train_loss": self.training_loss_history[-1],
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.training_loss_history),
            "early_stopped": patience_counter >= self.config.early_stopping_patience
        }

    def predict(self, prices: np.ndarray) -> float:
        """
        Predict the next price.

        Args:
            prices: Recent price history (at least seq_length prices)

        Returns:
            Predicted next price
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        sequence = self.data_manager.prepare_sequence(prices)
        X = torch.FloatTensor(sequence).to(self.device)

        self.model.eval()
        with torch.no_grad():
            scaled_pred = self.model(X).cpu().numpy()

        return float(self.data_manager.inverse_transform(scaled_pred)[0])

    def predict_next_n(self, prices: np.ndarray, n: int = 5) -> List[float]:
        """
        Predict the next n prices using autoregressive prediction.

        Args:
            prices: Recent price history
            n: Number of steps to predict

        Returns:
            List of n predicted prices
        """
        predictions = []
        current_sequence = list(prices[-self.config.seq_length:])

        for _ in range(n):
            next_price = self.predict(np.array(current_sequence))
            predictions.append(next_price)
            current_sequence = current_sequence[1:] + [next_price]

        return predictions

    def predict_trend(self, prices: np.ndarray) -> str:
        """
        Predict the price trend direction.

        Args:
            prices: Recent price history

        Returns:
            "up", "down", or "sideways"
        """
        current_price = prices[-1]
        predicted_price = self.predict(prices)

        pct_change = (predicted_price - current_price) / current_price * 100

        if pct_change > 0.5:  # More than 0.5% predicted increase
            return "up"
        elif pct_change < -0.5:  # More than 0.5% predicted decrease
            return "down"
        else:
            return "sideways"

    def get_trend_confidence(self, prices: np.ndarray) -> float:
        """
        Calculate confidence in the trend prediction.

        Returns a value between 0 and 1 based on the strength
        of the predicted movement.
        """
        current_price = prices[-1]
        predicted_price = self.predict(prices)

        pct_change = abs((predicted_price - current_price) / current_price * 100)

        # Confidence scales with the magnitude of predicted change
        # 2% change = high confidence, <0.5% = low confidence
        confidence = min(pct_change / 2.0, 1.0)
        return confidence

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler_data_min': self.data_manager.scaler.data_min_,
            'scaler_data_max': self.data_manager.scaler.data_max_,
            'scaler_data_range': self.data_manager.scaler.data_range_,
            'scaler_scale': self.data_manager.scaler.scale_,
            'scaler_min': self.data_manager.scaler.min_,
            'scaler_n_features': self.data_manager.scaler.n_features_in_,
            'scaler_n_samples': self.data_manager.scaler.n_samples_seen_,
            'training_loss': self.training_loss_history,
            'validation_loss': self.validation_loss_history,
        }, save_path)

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        self.model = LSTMModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Update data_manager with loaded config's seq_length
        self.data_manager = LSTMDataManager(seq_length=self.config.seq_length)

        # Restore scaler state
        self.data_manager.scaler.data_min_ = checkpoint['scaler_data_min']
        self.data_manager.scaler.data_max_ = checkpoint['scaler_data_max']
        self.data_manager.scaler.data_range_ = checkpoint['scaler_data_range']
        self.data_manager.scaler.scale_ = checkpoint['scaler_scale']
        self.data_manager.scaler.min_ = checkpoint['scaler_min']
        self.data_manager.scaler.n_features_in_ = checkpoint['scaler_n_features']
        self.data_manager.scaler.n_samples_seen_ = checkpoint['scaler_n_samples']
        self.data_manager.is_fitted = True

        self.training_loss_history = checkpoint['training_loss']
        self.validation_loss_history = checkpoint['validation_loss']
        self.is_trained = True
        self.model.eval()


class LSTMEnsemble:
    """
    Ensemble of LSTM models for more robust predictions.

    Trains multiple models with different random seeds and
    averages their predictions.
    """

    def __init__(self, n_models: int = 3, config: Optional[LSTMConfig] = None):
        self.n_models = n_models
        self.config = config or LSTMConfig()
        self.models: List[LSTMPricePredictor] = []

    def train(self, prices: np.ndarray, verbose: bool = True) -> dict:
        """Train the ensemble of models."""
        metrics_list = []

        for i in range(self.n_models):
            if verbose:
                print(f"\n--- Training model {i+1}/{self.n_models} ---")

            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            model = LSTMPricePredictor(self.config)
            metrics = model.train(prices, verbose=verbose)
            self.models.append(model)
            metrics_list.append(metrics)

        return {
            "models_trained": len(self.models),
            "avg_val_loss": np.mean([m["best_val_loss"] for m in metrics_list])
        }

    def predict(self, prices: np.ndarray) -> Tuple[float, float]:
        """
        Get ensemble prediction with uncertainty estimate.

        Returns:
            Tuple of (mean_prediction, std_deviation)
        """
        predictions = [model.predict(prices) for model in self.models]
        return float(np.mean(predictions)), float(np.std(predictions))

    def predict_trend(self, prices: np.ndarray) -> Tuple[str, float]:
        """
        Get ensemble trend prediction with confidence.

        Returns:
            Tuple of (trend, confidence)
        """
        trends = [model.predict_trend(prices) for model in self.models]

        # Majority vote
        trend_counts = {}
        for t in trends:
            trend_counts[t] = trend_counts.get(t, 0) + 1

        majority_trend = max(trend_counts, key=trend_counts.get)
        confidence = trend_counts[majority_trend] / len(trends)

        return majority_trend, confidence
