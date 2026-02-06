"""
Modular Loss Functions for Hyperparameter Optimization

Inspired by Freqtrade's HyperOptLoss classes.
Each loss function converts backtest metrics into a single objective value.
"""

from abc import ABC, abstractmethod
from typing import Dict


class BaseLossFunction(ABC):
    """Base class for optimization loss functions."""

    @abstractmethod
    def calculate(self, metrics: Dict[str, float]) -> float:
        """Calculate loss (lower is better for minimization).

        Args:
            metrics: Dictionary of backtest metrics.

        Returns:
            Loss value to minimize (negate for maximization).
        """
        pass


class SharpeLoss(BaseLossFunction):
    """Maximize Sharpe ratio."""

    def calculate(self, metrics: Dict[str, float]) -> float:
        return -metrics.get("sharpe_ratio", 0.0)


class SortinoLoss(BaseLossFunction):
    """Maximize Sortino ratio."""

    def calculate(self, metrics: Dict[str, float]) -> float:
        return -metrics.get("sortino_ratio", 0.0)


class CalmarLoss(BaseLossFunction):
    """Maximize Calmar ratio (return / max drawdown)."""

    def calculate(self, metrics: Dict[str, float]) -> float:
        return -metrics.get("calmar_ratio", 0.0)


class ProfitLoss(BaseLossFunction):
    """Maximize total return."""

    def calculate(self, metrics: Dict[str, float]) -> float:
        return -metrics.get("total_return_pct", 0.0)


class MaxDrawdownLoss(BaseLossFunction):
    """Minimize maximum drawdown."""

    def calculate(self, metrics: Dict[str, float]) -> float:
        return metrics.get("max_drawdown_pct", 100.0)


class CustomWeightedLoss(BaseLossFunction):
    """Weighted combination of multiple metrics.

    Example:
        loss = CustomWeightedLoss({
            'sharpe_ratio': 0.4,
            'total_return_pct': 0.3,
            'max_drawdown_pct': -0.3,  # negative = penalize high values
        })
    """

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def calculate(self, metrics: Dict[str, float]) -> float:
        score = 0.0
        for metric, weight in self.weights.items():
            score += weight * metrics.get(metric, 0.0)
        return -score  # Negate so higher weighted score = lower loss


LOSS_FUNCTIONS = {
    "sharpe": SharpeLoss,
    "sortino": SortinoLoss,
    "calmar": CalmarLoss,
    "profit": ProfitLoss,
    "max_drawdown": MaxDrawdownLoss,
}


def get_loss_function(name: str) -> BaseLossFunction:
    """Get loss function by name."""
    cls = LOSS_FUNCTIONS.get(name)
    if cls is None:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(LOSS_FUNCTIONS)}")
    return cls()
