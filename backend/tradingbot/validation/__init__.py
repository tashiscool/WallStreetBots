"""
Trading Strategy Signal Validation Framework
===========================================

Comprehensive signal strength validation and integration layer for trading strategies.
"""

from .signal_strength_validator import (
    SignalStrengthValidator,
    SignalType,
    SignalQuality,
    SignalMetrics,
    SignalValidationResult,
    BreakoutSignalCalculator,
    MomentumSignalCalculator
)

from .strategy_signal_integration import (
    StrategySignalMixin,
    StrategySignalConfig,
    SignalValidationIntegrator,
    SwingTradingSignalCalculator,
    MomentumWeekliesSignalCalculator,
    LEAPSSignalCalculator,
    signal_integrator
)

__all__ = [
    'BreakoutSignalCalculator',
    'LEAPSSignalCalculator',
    'MomentumSignalCalculator',
    'MomentumWeekliesSignalCalculator',
    'SignalMetrics',
    'SignalQuality',
    'SignalStrengthValidator',
    'SignalType',
    'SignalValidationIntegrator',
    'SignalValidationResult',
    'StrategySignalConfig',
    'StrategySignalMixin',
    'SwingTradingSignalCalculator',
    'signal_integrator'
]

__version__ = '1.0.0'