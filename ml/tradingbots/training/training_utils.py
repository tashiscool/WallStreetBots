"""
Training Utilities with Best Practices

Provides:
- Walk-forward validation for time series
- Progress tracking with rich console output
- Model checkpointing and early stopping
- Comprehensive metrics evaluation
- Training summaries with visualizations
"""

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Try importing optional dependencies for rich UX
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training with best practices."""
    # Validation strategy
    validation_strategy: str = "walk_forward"  # "walk_forward", "holdout", "kfold"
    train_ratio: float = 0.8
    n_splits: int = 5  # For walk-forward or k-fold

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    learning_rate_decay: float = 0.5
    lr_patience: int = 5

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-6

    # Regularization
    dropout: float = 0.2
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # Display
    verbose: bool = True
    progress_bar: bool = True
    log_interval: int = 10  # Log every N epochs

    # Reproducibility
    seed: Optional[int] = 42


@dataclass
class TrainingMetrics:
    """Stores and computes training metrics."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)

    # Trading-specific metrics
    sharpe_ratios: List[float] = field(default_factory=list)
    max_drawdowns: List[float] = field(default_factory=list)
    win_rates: List[float] = field(default_factory=list)
    profit_factors: List[float] = field(default_factory=list)

    # Timing
    epoch_times: List[float] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def add_epoch(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        epoch_time: Optional[float] = None,
    ) -> None:
        """Add metrics for one epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)

    def add_trading_metrics(
        self,
        sharpe: float,
        max_dd: float,
        win_rate: float,
        profit_factor: float,
    ) -> None:
        """Add trading-specific metrics."""
        self.sharpe_ratios.append(sharpe)
        self.max_drawdowns.append(max_dd)
        self.win_rates.append(win_rate)
        self.profit_factors.append(profit_factor)

    @property
    def best_val_loss(self) -> float:
        return min(self.val_losses) if self.val_losses else float('inf')

    @property
    def best_epoch(self) -> int:
        if not self.val_losses:
            return 0
        return int(np.argmin(self.val_losses))

    @property
    def total_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return sum(self.epoch_times)

    @property
    def avg_epoch_time(self) -> float:
        return np.mean(self.epoch_times) if self.epoch_times else 0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        return {
            "epochs_trained": len(self.train_losses),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "best_sharpe": max(self.sharpe_ratios) if self.sharpe_ratios else None,
            "best_win_rate": max(self.win_rates) if self.win_rates else None,
            "total_time_sec": self.total_time,
            "avg_epoch_time_sec": self.avg_epoch_time,
        }


class ProgressTracker:
    """
    Rich progress tracking for training.

    Provides:
    - Progress bar with ETA
    - Live metrics display
    - Epoch summaries
    """

    def __init__(
        self,
        total_epochs: int,
        desc: str = "Training",
        use_tqdm: bool = True,
    ):
        self.total_epochs = total_epochs
        self.desc = desc
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.pbar = None
        self.current_epoch = 0

    def __enter__(self):
        if self.use_tqdm:
            self.pbar = tqdm(
                total=self.total_epochs,
                desc=self.desc,
                unit="epoch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        else:
            self._print_header()
        return self

    def __exit__(self, *args):
        if self.pbar:
            self.pbar.close()
        else:
            self._print_footer()

    def _print_header(self) -> None:
        """Print training header."""
        print("\n" + "=" * 70)
        print(f"  {self.desc}")
        print("=" * 70)
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Time':>8} | Status")
        print("-" * 70)

    def _print_footer(self) -> None:
        """Print training footer."""
        print("-" * 70)
        print("Training complete!")
        print("=" * 70 + "\n")

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        epoch_time: float,
        extra_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> None:
        """Update progress with epoch results."""
        self.current_epoch = epoch

        status = "* BEST *" if is_best else ""

        if self.use_tqdm:
            self.pbar.update(1)
            self.pbar.set_postfix({
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "best": "Yes" if is_best else "No",
            })
        else:
            print(
                f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                f"{epoch_time:>7.2f}s | {status}"
            )

        # Print extra metrics if provided
        if extra_metrics and not self.use_tqdm:
            extras = " | ".join([f"{k}: {v:.4f}" for k, v in extra_metrics.items()])
            print(f"       | {extras}")

    def log_message(self, message: str) -> None:
        """Log a message during training."""
        if self.use_tqdm:
            tqdm.write(message)
        else:
            print(f"       | {message}")


class EarlyStoppingCallback:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training when no improvement
    is seen for `patience` epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def __call__(
        self,
        epoch: int,
        val_loss: float,
        model_state: Optional[Dict] = None,
    ) -> bool:
        """
        Check if training should stop.

        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
            model_state: Current model state dict (for restoration)

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if model_state is not None:
                self.best_state = {k: v.clone() if hasattr(v, 'clone') else v
                                   for k, v in model_state.items()}
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False

    def get_best_state(self) -> Optional[Dict]:
        """Get the best model state."""
        return self.best_state


class ModelCheckpointer:
    """
    Saves model checkpoints during training.

    Features:
    - Save best model only or all checkpoints
    - Automatic checkpoint naming
    - Load best checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_best_only: bool = True,
        model_name: str = "model",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.model_name = model_name
        self.best_loss = float('inf')

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model_state: Dict,
        epoch: int,
        val_loss: float,
        metrics: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Save a checkpoint.

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        is_best = val_loss < self.best_loss

        if self.save_best_only and not is_best:
            return None

        if is_best:
            self.best_loss = val_loss

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_epoch{epoch}_{timestamp}.pt"
        if is_best:
            filename = f"{self.model_name}_best.pt"

        path = os.path.join(self.checkpoint_dir, filename)

        try:
            import torch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "val_loss": val_loss,
                "metrics": metrics or {},
            }
            torch.save(checkpoint, path)
            return path
        except ImportError:
            # Save with numpy if torch not available
            np.savez(path.replace('.pt', '.npz'), **model_state)
            return path

    def load_best(self) -> Optional[Dict]:
        """Load the best checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
        if not os.path.exists(path):
            return None

        try:
            import torch
            return torch.load(path)
        except ImportError:
            return None


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    This is the gold standard for validating trading models because:
    1. Respects temporal ordering (no future data leakage)
    2. Simulates real trading conditions
    3. Tests model adaptability over time
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.8,
        min_train_size: Optional[int] = None,
        gap: int = 0,  # Gap between train and validation to prevent leakage
    ):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.min_train_size = min_train_size
        self.gap = gap

    def split(
        self,
        data_length: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation splits.

        Args:
            data_length: Total number of samples

        Returns:
            List of (train_indices, val_indices) tuples
        """
        splits = []
        fold_size = data_length // self.n_splits

        for i in range(self.n_splits):
            # Calculate split points
            val_start = (i + 1) * fold_size
            val_end = min((i + 2) * fold_size, data_length)

            if val_start >= data_length:
                break

            # Training includes all data before validation (with gap)
            train_end = val_start - self.gap
            train_start = 0

            if self.min_train_size and train_end < self.min_train_size:
                continue

            train_indices = np.arange(train_start, train_end)
            val_indices = np.arange(val_start, val_end)

            splits.append((train_indices, val_indices))

        return splits

    def get_expanding_splits(
        self,
        data_length: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate expanding window splits.

        Each subsequent fold uses more training data (expanding window).
        This is preferred for small datasets.
        """
        splits = []
        fold_size = data_length // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            val_start = train_end + self.gap
            val_end = min(val_start + fold_size, data_length)

            if val_start >= data_length:
                break

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)

            splits.append((train_indices, val_indices))

        return splits


def calculate_trading_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Calculate comprehensive trading performance metrics.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Dictionary of trading metrics
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return": 0.0,
            "volatility": 0.0,
        }

    # Basic stats
    avg_return = np.mean(returns)
    volatility = np.std(returns)
    annualized_return = avg_return * periods_per_year
    annualized_vol = volatility * np.sqrt(periods_per_year)

    # Sharpe Ratio
    rf_per_period = risk_free_rate / periods_per_year
    sharpe = (avg_return - rf_per_period) / (volatility + 1e-8) * np.sqrt(periods_per_year)

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
    sortino = (avg_return - rf_per_period) / (downside_std + 1e-8) * np.sqrt(periods_per_year)

    # Maximum Drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # Calmar Ratio
    calmar = annualized_return / (abs(max_drawdown) + 1e-8) if max_drawdown != 0 else 0

    # Win Rate
    winning_trades = np.sum(returns > 0)
    total_trades = len(returns)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / (gross_loss + 1e-8) if gross_loss > 0 else float('inf')

    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_return": float(avg_return),
        "volatility": float(annualized_vol),
        "total_return": float(cumulative[-1] - 1) if len(cumulative) > 0 else 0,
    }


def display_training_summary(
    metrics: TrainingMetrics,
    model_name: str = "Model",
    show_plot: bool = False,
) -> None:
    """
    Display a comprehensive training summary.

    Args:
        metrics: TrainingMetrics object with training history
        model_name: Name of the model for display
        show_plot: Whether to show matplotlib plots
    """
    summary = metrics.get_summary()

    print("\n" + "=" * 70)
    print(f"  TRAINING SUMMARY: {model_name}")
    print("=" * 70)

    # Training progress
    print("\n  Training Progress:")
    print(f"    Epochs trained:     {summary['epochs_trained']}")
    print(f"    Best epoch:         {summary['best_epoch']}")
    print(f"    Total time:         {summary['total_time_sec']:.2f}s")
    print(f"    Avg epoch time:     {summary['avg_epoch_time_sec']:.2f}s")

    # Loss metrics
    print("\n  Loss Metrics:")
    print(f"    Final train loss:   {summary['final_train_loss']:.6f}")
    print(f"    Final val loss:     {summary['final_val_loss']:.6f}")
    print(f"    Best val loss:      {summary['best_val_loss']:.6f}")

    # Trading metrics (if available)
    if summary.get('best_sharpe') is not None:
        print("\n  Trading Metrics:")
        print(f"    Best Sharpe Ratio:  {summary['best_sharpe']:.4f}")
        if summary.get('best_win_rate'):
            print(f"    Best Win Rate:      {summary['best_win_rate']:.2%}")

    print("\n" + "=" * 70)

    # Show plot if requested
    if show_plot and metrics.train_losses:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Loss plot
            axes[0].plot(metrics.train_losses, label='Train Loss', alpha=0.8)
            axes[0].plot(metrics.val_losses, label='Val Loss', alpha=0.8)
            axes[0].axvline(x=metrics.best_epoch, color='r', linestyle='--',
                           label=f'Best Epoch ({metrics.best_epoch})')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training & Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Trading metrics plot
            if metrics.sharpe_ratios:
                axes[1].plot(metrics.sharpe_ratios, label='Sharpe Ratio', alpha=0.8)
                axes[1].set_xlabel('Evaluation')
                axes[1].set_ylabel('Sharpe Ratio')
                axes[1].set_title('Trading Performance')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  (Install matplotlib for training plots)")


def train_with_best_practices(
    model: Any,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    config: TrainingConfig,
    train_step_fn: Callable,
    eval_fn: Callable,
) -> Tuple[Any, TrainingMetrics]:
    """
    Train a model with all best practices applied.

    This is a generic training loop that works with any model that has:
    - A train_step function
    - An eval function

    Args:
        model: The model to train
        train_data: Tuple of (X_train, y_train)
        val_data: Tuple of (X_val, y_val)
        config: TrainingConfig with all settings
        train_step_fn: Function(model, batch_X, batch_y) -> loss
        eval_fn: Function(model, X, y) -> (loss, metrics)

    Returns:
        Tuple of (trained_model, training_metrics)
    """
    # Set seed for reproducibility
    if config.seed is not None:
        np.random.seed(config.seed)
        try:
            import torch
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
        except ImportError:
            pass

    # Initialize components
    metrics = TrainingMetrics()
    metrics.start_time = time.time()

    early_stopping = None
    if config.early_stopping:
        early_stopping = EarlyStoppingCallback(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

    checkpointer = None
    if config.checkpoint_dir:
        checkpointer = ModelCheckpointer(
            checkpoint_dir=config.checkpoint_dir,
            save_best_only=config.save_best_only,
        )

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Create batches
    n_samples = len(X_train)
    n_batches = max(1, n_samples // config.batch_size)

    # Training loop with progress tracking
    with ProgressTracker(
        config.epochs,
        desc="Training",
        use_tqdm=config.progress_bar,
    ) as progress:

        for epoch in range(config.epochs):
            epoch_start = time.time()

            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Train epoch
            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, n_samples)

                batch_X = X_train_shuffled[start_idx:end_idx]
                batch_y = y_train_shuffled[start_idx:end_idx]

                batch_loss = train_step_fn(model, batch_X, batch_y)
                epoch_loss += batch_loss

            avg_train_loss = epoch_loss / n_batches

            # Evaluate
            val_loss, val_metrics = eval_fn(model, X_val, y_val)

            epoch_time = time.time() - epoch_start

            # Track metrics
            metrics.add_epoch(
                train_loss=avg_train_loss,
                val_loss=val_loss,
                epoch_time=epoch_time,
            )

            # Check if best
            is_best = val_loss <= metrics.best_val_loss

            # Update progress
            progress.update(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                epoch_time=epoch_time,
                extra_metrics=val_metrics,
                is_best=is_best,
            )

            # Checkpointing
            if checkpointer and hasattr(model, 'state_dict'):
                checkpointer.save(
                    model.state_dict(),
                    epoch=epoch,
                    val_loss=val_loss,
                    metrics=val_metrics,
                )

            # Early stopping
            if early_stopping:
                model_state = model.state_dict() if hasattr(model, 'state_dict') else None
                if early_stopping(epoch, val_loss, model_state):
                    progress.log_message(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {config.early_stopping_patience} epochs)"
                    )
                    break

    metrics.end_time = time.time()

    # Restore best model if early stopping was used
    if early_stopping and early_stopping.restore_best and early_stopping.best_state:
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(early_stopping.best_state)

    return model, metrics


def evaluate_trading_model(
    model: Any,
    test_prices: np.ndarray,
    predict_fn: Callable,
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """
    Evaluate a trading model on test data.

    Args:
        model: Trained model
        test_prices: Array of test prices
        predict_fn: Function(model, prices) -> signal (-1, 0, 1)
        initial_capital: Starting capital

    Returns:
        Dictionary with evaluation results
    """
    capital = initial_capital
    position = 0.0
    trades = []
    portfolio_values = [capital]
    returns = []

    for i in range(60, len(test_prices)):
        # Get prediction
        price_window = test_prices[i-60:i]
        signal = predict_fn(model, price_window)

        current_price = test_prices[i]
        prev_price = test_prices[i-1]

        # Execute trade
        if signal != position:
            # Calculate PnL from position change
            if position != 0:
                price_change = (current_price - prev_price) / prev_price
                pnl = position * price_change * capital
                capital += pnl
                returns.append(pnl / portfolio_values[-1])

            position = signal
            trades.append({
                "step": i,
                "signal": signal,
                "price": current_price,
                "capital": capital,
            })
        elif position != 0:
            # Mark-to-market
            price_change = (current_price - prev_price) / prev_price
            pnl = position * price_change * capital
            capital += pnl
            returns.append(pnl / portfolio_values[-1])

        portfolio_values.append(capital)

    # Calculate metrics
    trading_metrics = calculate_trading_metrics(np.array(returns))

    return {
        "final_capital": capital,
        "total_return": (capital - initial_capital) / initial_capital,
        "num_trades": len(trades),
        "trades": trades,
        "portfolio_values": portfolio_values,
        **trading_metrics,
    }
