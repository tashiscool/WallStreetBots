"""
CNN Price Predictor Component

A PyTorch-based 1D CNN model for price/trend prediction that captures
local patterns and features in time series data.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


@dataclass
class CNNConfig:
    """Configuration for CNN model."""
    input_size: int = 1  # Number of features
    seq_length: int = 60  # Input sequence length
    num_filters: List[int] = None  # Filters per conv layer
    kernel_sizes: List[int] = None  # Kernel size per conv layer
    pool_sizes: List[int] = None  # Pool size per layer
    fc_hidden_size: int = 128  # Fully connected hidden size
    dropout: float = 0.2
    output_size: int = 1
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10

    def __post_init__(self):
        if self.num_filters is None:
            self.num_filters = [32, 64, 128]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3]
        if self.pool_sizes is None:
            self.pool_sizes = [2, 2, 2]


class CNNModel(nn.Module):
    """PyTorch 1D CNN model for time series prediction."""

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config

        # Build convolutional layers
        conv_layers = []
        in_channels = config.input_size

        for i, (filters, kernel, pool) in enumerate(zip(
            config.num_filters,
            config.kernel_sizes,
            config.pool_sizes
        )):
            conv_layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool),
                nn.Dropout(config.dropout),
            ])
            in_channels = filters

        self.conv = nn.Sequential(*conv_layers)

        # Calculate output size after convolutions
        test_input = torch.zeros(1, config.input_size, config.seq_length)
        with torch.no_grad():
            test_output = self.conv(test_input)
            conv_output_size = test_output.view(1, -1).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, config.fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden_size, config.fc_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden_size // 2, config.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Predictions of shape (batch, output_size)
        """
        # x shape: (batch, seq_len, input_size)
        # Conv1d expects: (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Convolutional layers
        x = self.conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        output = self.fc(x)
        return output


class CNNDataManager:
    """Manages data preparation for CNN model."""

    def __init__(self, seq_length: int = 60):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def prepare_data(
        self,
        prices: np.ndarray,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for CNN training."""
        prices = np.array(prices).reshape(-1, 1)

        if fit_scaler:
            scaled_prices = self.scaler.fit_transform(prices)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted.")
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


class CNNPricePredictor:
    """
    CNN-based price predictor for trading signal generation.

    Uses 1D convolutions to capture local patterns in price series.
    """

    def __init__(self, config: Optional[CNNConfig] = None):
        self.config = config or CNNConfig()
        self.model = CNNModel(self.config)
        self.data_manager = CNNDataManager(self.config.seq_length)
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
        """Train the CNN model on historical price data."""
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
        best_state = None

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
        if best_state:
            self.model.load_state_dict(best_state)
        self.is_trained = True

        return {
            "final_train_loss": self.training_loss_history[-1],
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.training_loss_history),
            "early_stopped": patience_counter >= self.config.early_stopping_patience
        }

    def predict(self, prices: np.ndarray) -> float:
        """Predict the next price."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        sequence = self.data_manager.prepare_sequence(prices)
        X = torch.FloatTensor(sequence).to(self.device)

        self.model.eval()
        with torch.no_grad():
            scaled_pred = self.model(X).cpu().numpy()

        return float(self.data_manager.inverse_transform(scaled_pred)[0])

    def predict_trend(self, prices: np.ndarray) -> str:
        """Predict the price trend direction."""
        current_price = prices[-1]
        predicted_price = self.predict(prices)
        pct_change = (predicted_price - current_price) / current_price * 100

        if pct_change > 0.5:
            return "up"
        elif pct_change < -0.5:
            return "down"
        else:
            return "sideways"

    def get_trend_confidence(self, prices: np.ndarray) -> float:
        """Calculate confidence in the trend prediction."""
        current_price = prices[-1]
        predicted_price = self.predict(prices)
        pct_change = abs((predicted_price - current_price) / current_price * 100)
        confidence = min(pct_change / 2.0, 1.0)
        return confidence

    def extract_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract learned features from the CNN."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        sequence = self.data_manager.prepare_sequence(prices)
        X = torch.FloatTensor(sequence).to(self.device)
        X = X.transpose(1, 2)  # Conv1d format

        self.model.eval()
        with torch.no_grad():
            # Get output of conv layers before FC
            features = self.model.conv(X)
            features = features.view(features.size(0), -1)

        return features.cpu().numpy()

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler_min': self.data_manager.scaler.data_min_,
            'scaler_max': self.data_manager.scaler.data_max_,
            'scaler_scale': self.data_manager.scaler.scale_,
            'training_loss': self.training_loss_history,
            'validation_loss': self.validation_loss_history,
        }, path)

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        self.config = checkpoint['config']
        self.model = CNNModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        self.data_manager.scaler.data_min_ = checkpoint['scaler_min']
        self.data_manager.scaler.data_max_ = checkpoint['scaler_max']
        self.data_manager.scaler.scale_ = checkpoint['scaler_scale']
        self.data_manager.is_fitted = True

        self.training_loss_history = checkpoint['training_loss']
        self.validation_loss_history = checkpoint['validation_loss']
        self.is_trained = True
        self.model.eval()
