"""
Hyperparameter Optimization Framework

Inspired by freqtrade's hyperopt system, this module provides ML-based
parameter optimization for trading strategies.

Usage:
    from backend.tradingbot.optimize import HyperoptEngine, IntParameter, RealParameter

    class MyStrategy(BaseStrategy):
        buy_rsi = IntParameter(low=10, high=40, default=30, space='buy')
        sell_rsi = IntParameter(low=60, high=90, default=70, space='sell')
        stop_loss = RealParameter(low=0.02, high=0.10, default=0.05, space='stoploss')

    engine = HyperoptEngine(
        strategy_class=MyStrategy,
        data_provider=data_client,
        loss_function='sharpe'
    )
    best_params = engine.optimize(epochs=100)
"""

from .parameters import (
    IntParameter,
    RealParameter,
    DecimalParameter,
    CategoricalParameter,
    BooleanParameter,
    HyperoptSpace,
)
from .hyperopt import HyperoptEngine, HyperoptConfig
from .hyperopt_loss import (
    IHyperoptLoss,
    SharpeHyperoptLoss,
    SortinoHyperoptLoss,
    MaxDrawdownHyperoptLoss,
    CalmarHyperoptLoss,
    ProfitHyperoptLoss,
    WinRateHyperoptLoss,
    MultiMetricHyperoptLoss,
)

__all__ = [
    # Parameters
    'IntParameter',
    'RealParameter',
    'DecimalParameter',
    'CategoricalParameter',
    'BooleanParameter',
    'HyperoptSpace',
    # Engine
    'HyperoptEngine',
    'HyperoptConfig',
    # Loss Functions
    'IHyperoptLoss',
    'SharpeHyperoptLoss',
    'SortinoHyperoptLoss',
    'MaxDrawdownHyperoptLoss',
    'CalmarHyperoptLoss',
    'ProfitHyperoptLoss',
    'WinRateHyperoptLoss',
    'MultiMetricHyperoptLoss',
]
