"""
Alpha Model Implementations

Pre-built alpha models for signal generation.
Use get_alpha_model(name) to instantiate by name.
"""

from typing import Any, Dict, Optional

from .rsi_alpha import RSIAlphaModel
from .macd_alpha import MACDAlphaModel
from .momentum_alpha import MomentumAlphaModel
from .mean_reversion_alpha import MeanReversionAlphaModel
from .breakout_alpha import BreakoutAlphaModel
from .ensemble_alpha import EnsembleAlphaModel
from .pairs_trading_alpha import PairsTradingAlphaModel
from .bollinger_alpha import BollingerAlphaModel
from .volatility_alpha import VolatilityAlphaModel
from .stochastic_alpha import StochasticAlphaModel
from .atr_reversion_alpha import ATRReversionAlphaModel
from .sentiment_alpha import SentimentAlphaModel

# Registry: name -> class
ALPHA_MODELS: Dict[str, type] = {
    'rsi': RSIAlphaModel,
    'macd': MACDAlphaModel,
    'momentum': MomentumAlphaModel,
    'mean_reversion': MeanReversionAlphaModel,
    'breakout': BreakoutAlphaModel,
    'ensemble': EnsembleAlphaModel,
    'pairs_trading': PairsTradingAlphaModel,
    'bollinger': BollingerAlphaModel,
    'volatility': VolatilityAlphaModel,
    'stochastic': StochasticAlphaModel,
    'atr_reversion': ATRReversionAlphaModel,
    'sentiment': SentimentAlphaModel,
}


def get_alpha_model(name: str, **kwargs) -> Any:
    """Instantiate an alpha model by name.

    Args:
        name: Model name (e.g. 'rsi', 'bollinger', 'pairs_trading')
        **kwargs: Passed to model constructor

    Returns:
        Alpha model instance

    Raises:
        ValueError: If name is not registered
    """
    cls = ALPHA_MODELS.get(name)
    if cls is None:
        available = ", ".join(sorted(ALPHA_MODELS.keys()))
        raise ValueError(f"Unknown alpha model '{name}'. Available: {available}")
    return cls(**kwargs)


def list_alpha_models() -> Dict[str, str]:
    """List available alpha models with their class names."""
    return {name: cls.__name__ for name, cls in ALPHA_MODELS.items()}


__all__ = [
    'RSIAlphaModel',
    'MACDAlphaModel',
    'MomentumAlphaModel',
    'MeanReversionAlphaModel',
    'BreakoutAlphaModel',
    'EnsembleAlphaModel',
    'PairsTradingAlphaModel',
    'BollingerAlphaModel',
    'VolatilityAlphaModel',
    'StochasticAlphaModel',
    'ATRReversionAlphaModel',
    'SentimentAlphaModel',
    'ALPHA_MODELS',
    'get_alpha_model',
    'list_alpha_models',
]
