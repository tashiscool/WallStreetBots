"""
Strategy Signal Integration Layer
================================

Integrates the comprehensive signal strength validation framework
with existing trading strategies to provide standardized signal validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from .signal_strength_validator import (
    SignalStrengthValidator,
    SignalType,
    SignalValidationResult,
    SignalStrengthCalculator
)


@dataclass
class StrategySignalConfig:
    """Configuration for strategy-specific signal validation."""
    strategy_name: str
    default_signal_type: SignalType
    minimum_strength_threshold: float = 65.0
    minimum_confidence_threshold: float = 0.6
    consistency_threshold: float = 0.7
    volume_threshold: float = 1.5
    risk_reward_minimum: float = 1.8
    max_position_size_multiplier: float = 1.0
    enable_regime_filtering: bool = True
    custom_validation_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_validation_params is None:
            self.custom_validation_params = {}


class StrategySignalMixin:
    """
    Mixin class to add standardized signal validation to existing strategies.

    Add this mixin to strategy classes to enable comprehensive signal validation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signal_validator: Optional[SignalStrengthValidator] = None
        self._signal_config: Optional[StrategySignalConfig] = None
        self._signal_history: List[SignalValidationResult] = []
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)

    def initialize_signal_validation(self, config: StrategySignalConfig):
        """Initialize signal validation for this strategy."""
        self._signal_config = config

        # Ensure logger exists
        if not hasattr(self, 'logger') or self.logger is None:
            self.logger = logging.getLogger(__name__)

        # Create validator with strategy-specific configuration
        validator_config = {
            'minimum_strength_threshold': config.minimum_strength_threshold,
            'minimum_confidence_threshold': config.minimum_confidence_threshold,
            'risk_reward_minimum': config.risk_reward_minimum,
            'regime_filter_enabled': config.enable_regime_filtering,
            **config.custom_validation_params
        }

        self._signal_validator = SignalStrengthValidator(validator_config)

        self.logger.info(f"Signal validation initialized for {config.strategy_name}")

    def validate_signal(self,
                       symbol: str,
                       market_data: pd.DataFrame,
                       signal_type: Optional[SignalType] = None,
                       signal_params: Optional[Dict[str, Any]] = None) -> SignalValidationResult:
        """
        Validate a trading signal using the comprehensive validation framework.

        Args:
            symbol: Symbol to validate signal for
            market_data: Market data for validation
            signal_type: Type of signal (uses default if None)
            signal_params: Additional signal parameters

        Returns:
            SignalValidationResult with comprehensive validation metrics
        """
        if not self._signal_validator or not self._signal_config:
            raise ValueError("Signal validation not initialized. Call initialize_signal_validation first.")

        signal_type = signal_type or self._signal_config.default_signal_type
        signal_params = signal_params or {}

        # Add strategy-specific context
        signal_params.update({
            'strategy_name': self._signal_config.strategy_name,
            'max_position_size_multiplier': self._signal_config.max_position_size_multiplier
        })

        result = self._signal_validator.validate_signal(
            signal_type=signal_type,
            symbol=symbol,
            market_data=market_data,
            signal_params=signal_params
        )

        # Store result for strategy-specific analysis
        self._signal_history.append(result)

        return result

    def get_signal_strength_score(self,
                                 symbol: str,
                                 market_data: pd.DataFrame,
                                 signal_type: Optional[SignalType] = None) -> float:
        """
        Get just the signal strength score (0-100) for quick filtering.

        Returns:
            Signal strength score (0-100)
        """
        result = self.validate_signal(symbol, market_data, signal_type)
        return result.normalized_score

    def filter_signals_by_strength(self,
                                  signals: List[Dict[str, Any]],
                                  market_data_getter: Callable[[str], pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Filter a list of signals by strength validation.

        Args:
            signals: List of signal dictionaries (must contain 'symbol')
            market_data_getter: Function that returns market data for a symbol

        Returns:
            Filtered list of signals that pass validation
        """
        validated_signals = []

        for signal in signals:
            try:
                symbol = signal['symbol']
                market_data = market_data_getter(symbol)

                validation_result = self.validate_signal(symbol, market_data)

                if validation_result.recommended_action == "trade":
                    # Add validation metadata to signal
                    signal_with_validation = signal.copy()
                    signal_with_validation.update({
                        'validation_score': validation_result.normalized_score,
                        'validation_confidence': validation_result.confidence_level,
                        'validation_quality': validation_result.quality_grade.value,
                        'suggested_position_size': validation_result.suggested_position_size
                    })
                    validated_signals.append(signal_with_validation)

            except Exception as e:
                self.logger.error(f"Error validating signal for {signal.get('symbol', 'unknown')}: {e}")

        return validated_signals

    def get_strategy_signal_summary(self) -> Dict[str, Any]:
        """Get summary of signal validation performance for this strategy."""
        if not self._signal_history:
            return {"message": "No signal validation history"}

        recent_signals = self._signal_history[-50:]  # Last 50 signals

        return {
            'strategy_name': self._signal_config.strategy_name if self._signal_config else 'Unknown',
            'total_signals_validated': len(recent_signals),
            'average_strength_score': np.mean([s.normalized_score for s in recent_signals]),
            'signals_recommended_for_trading': sum(1 for s in recent_signals if s.recommended_action == "trade"),
            'average_confidence': np.mean([s.confidence_level for s in recent_signals]),
            'quality_distribution': {
                'excellent': sum(1 for s in recent_signals if s.quality_grade.value == 'excellent'),
                'good': sum(1 for s in recent_signals if s.quality_grade.value == 'good'),
                'fair': sum(1 for s in recent_signals if s.quality_grade.value == 'fair'),
                'poor': sum(1 for s in recent_signals if s.quality_grade.value == 'poor'),
                'very_poor': sum(1 for s in recent_signals if s.quality_grade.value == 'very_poor')
            }
        }


class SwingTradingSignalCalculator(SignalStrengthCalculator):
    """Specialized signal calculator for swing trading strategies."""

    def get_signal_type(self) -> SignalType:
        return SignalType.BREAKOUT

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate swing trading specific signal strength."""
        if len(market_data) < 20:
            return 0.0

        try:
            prices = market_data['Close'].values
            volumes = market_data['Volume'].values if 'Volume' in market_data else None

            # Swing trading specific metrics
            current_price = prices[-1]

            # 1. Breakout strength from consolidation
            lookback = min(20, len(prices) - 1)
            consolidation_high = np.max(prices[-lookback:-1])
            consolidation_low = np.min(prices[-lookback:-1])

            range_size = consolidation_high - consolidation_low
            breakout_distance = max(0, current_price - consolidation_high)

            breakout_strength = (breakout_distance / range_size * 50) if range_size > 0 else 0

            # 2. Volume confirmation (swing trading needs volume)
            volume_strength = 0
            if volumes is not None:
                recent_volume = np.mean(volumes[-3:])
                avg_volume = np.mean(volumes[-lookback:])
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                volume_strength = min(30, (volume_ratio - 1) * 20)  # Cap at 30 points

            # 3. Momentum component
            momentum_periods = min(5, len(prices) - 1)
            recent_momentum = (prices[-1] - prices[-momentum_periods]) / prices[-momentum_periods] * 100 if prices[-momentum_periods] > 0 else 0
            momentum_strength = min(20, max(0, recent_momentum * 5))  # Cap at 20 points

            total_strength = breakout_strength + volume_strength + momentum_strength

            return min(100.0, max(0.0, total_strength))

        except Exception as e:
            logging.error(f"Error in swing trading signal calculation: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate confidence specific to swing trading."""
        if len(market_data) < 10:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Confidence based on trend clarity and volatility
            lookback = min(10, len(prices) - 1)
            price_changes = np.diff(prices[-lookback:])

            # Trend consistency
            positive_moves = sum(1 for change in price_changes if change > 0)
            trend_consistency = max(positive_moves, len(price_changes) - positive_moves) / len(price_changes)

            # Volatility component (lower volatility = higher confidence for swing trades)
            volatility = np.std(prices[-lookback:]) / np.mean(prices[-lookback:]) if np.mean(prices[-lookback:]) > 0 else 0
            volatility_confidence = max(0, 1 - volatility * 10)  # Penalize high volatility

            return (trend_consistency + volatility_confidence) / 2.0

        except Exception:
            return 0.5


class MomentumWeekliesSignalCalculator(SignalStrengthCalculator):
    """Specialized signal calculator for momentum weeklies strategies."""

    def get_signal_type(self) -> SignalType:
        return SignalType.MOMENTUM

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate momentum-specific signal strength for weekly options."""
        if len(market_data) < 15:
            return 0.0

        try:
            prices = market_data['Close'].values

            # 1. Short-term momentum (1-3 days)
            short_momentum = (prices[-1] - prices[-4]) / prices[-4] * 100 if len(prices) >= 4 and prices[-4] > 0 else 0

            # 2. Medium-term momentum (1 week)
            medium_momentum = (prices[-1] - prices[-8]) / prices[-8] * 100 if len(prices) >= 8 and prices[-8] > 0 else 0

            # 3. Acceleration (momentum of momentum)
            if len(prices) >= 8:
                early_momentum = (prices[-4] - prices[-8]) / prices[-8] * 100 if prices[-8] > 0 else 0
                acceleration = short_momentum - early_momentum
            else:
                acceleration = 0

            # 4. Volume trend (if available)
            volume_trend = 0
            if 'Volume' in market_data.columns:
                volumes = market_data['Volume'].values
                recent_volume = np.mean(volumes[-3:])
                earlier_volume = np.mean(volumes[-6:-3]) if len(volumes) >= 6 else recent_volume
                volume_trend = (recent_volume - earlier_volume) / earlier_volume * 100 if earlier_volume > 0 else 0

            # Combine components (weekly options need strong momentum)
            momentum_score = (
                abs(short_momentum) * 0.4 +  # Current momentum
                abs(medium_momentum) * 0.3 +  # Sustained momentum
                max(0, acceleration) * 0.2 +  # Accelerating momentum
                max(0, volume_trend) * 0.1    # Volume confirmation
            )

            return min(100.0, max(0.0, momentum_score * 2))  # Scale up for weekly options

        except Exception as e:
            logging.error(f"Error in momentum weeklies signal calculation: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate confidence for momentum weeklies."""
        if len(market_data) < 10:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Momentum consistency over multiple timeframes
            timeframes = [3, 5, 8] if len(prices) >= 8 else [3, 5] if len(prices) >= 5 else [3]
            momentum_directions = []

            for tf in timeframes:
                if len(prices) > tf:
                    momentum = (prices[-1] - prices[-tf]) / prices[-tf] if prices[-tf] > 0 else 0
                    momentum_directions.append(1 if momentum > 0 else 0)

            # Confidence based on consistency across timeframes
            if momentum_directions:
                consistency = abs(sum(momentum_directions) / len(momentum_directions) - 0.5) * 2  # Convert to 0-1
            else:
                consistency = 0.5

            return min(1.0, max(0.0, consistency))

        except Exception:
            return 0.5


class LEAPSSignalCalculator(SignalStrengthCalculator):
    """Specialized signal calculator for LEAPS strategies."""

    def get_signal_type(self) -> SignalType:
        return SignalType.TREND

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate LEAPS-specific signal strength (long-term trend)."""
        if len(market_data) < 30:
            return 0.0

        try:
            prices = market_data['Close'].values

            # 1. Long-term trend strength
            long_term_return = (prices[-1] - prices[-30]) / prices[-30] * 100 if prices[-30] > 0 else 0

            # 2. Trend consistency (how often price is above moving average)
            ma_20 = np.mean(prices[-20:])
            above_ma_count = sum(1 for p in prices[-20:] if p > ma_20)
            trend_consistency = above_ma_count / 20 * 100

            # 3. Fundamental momentum proxy (steady growth)
            # Look for steady, sustainable growth rather than explosive moves
            growth_periods = []
            for i in range(5, 25, 5):  # Check 5, 10, 15, 20 day returns
                if len(prices) > i:
                    period_return = (prices[-1] - prices[-i]) / prices[-i] * 100 if prices[-i] > 0 else 0
                    growth_periods.append(period_return)

            # Prefer consistent positive growth across periods
            positive_periods = sum(1 for r in growth_periods if r > 0)
            growth_consistency = positive_periods / len(growth_periods) * 100 if growth_periods else 0

            # 4. Volatility penalty (LEAPS prefer stable moves)
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) if np.mean(prices[-20:]) > 0 else 0
            volatility_penalty = max(0, volatility * 100 - 10)  # Penalize volatility > 10%

            # Combine for LEAPS scoring
            leaps_score = (
                min(40, abs(long_term_return)) +  # Long-term performance (capped)
                trend_consistency * 0.3 +         # Trend consistency
                growth_consistency * 0.2 -        # Growth consistency
                volatility_penalty                # Volatility penalty
            )

            return min(100.0, max(0.0, leaps_score))

        except Exception as e:
            logging.error(f"Error in LEAPS signal calculation: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate confidence for LEAPS signals."""
        if len(market_data) < 20:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Confidence based on trend stability and duration
            # Check if trend has been consistent over extended period
            ma_periods = [5, 10, 15, 20]
            trend_alignment = 0

            for period in ma_periods:
                if len(prices) >= period:
                    ma = np.mean(prices[-period:])
                    if prices[-1] > ma:  # Uptrend alignment
                        trend_alignment += 1

            trend_confidence = trend_alignment / len(ma_periods)

            # Factor in volatility (lower is better for LEAPS)
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) if np.mean(prices[-20:]) > 0 else 0
            volatility_confidence = max(0, 1 - volatility * 5)  # Penalize high volatility

            return (trend_confidence + volatility_confidence) / 2.0

        except Exception:
            return 0.5


class SignalValidationIntegrator:
    """
    Central integrator for adding signal validation to existing strategies.
    """

    def __init__(self):
        self.strategy_configs = {}
        self.custom_calculators = {}
        self.logger = logging.getLogger(__name__)

    def register_strategy_config(self, strategy_name: str, config: StrategySignalConfig):
        """Register configuration for a strategy."""
        self.strategy_configs[strategy_name] = config
        self.logger.info(f"Registered signal validation config for {strategy_name}")

    def register_custom_calculator(self, strategy_name: str, calculator: SignalStrengthCalculator):
        """Register custom signal calculator for a strategy."""
        self.custom_calculators[strategy_name] = calculator
        self.logger.info(f"Registered custom calculator for {strategy_name}")

    def enhance_strategy_with_validation(self, strategy_instance, strategy_name: str):
        """
        Enhance an existing strategy instance with signal validation capabilities.

        Args:
            strategy_instance: Instance of strategy to enhance
            strategy_name: Name of the strategy for configuration lookup
        """
        if not hasattr(strategy_instance, 'initialize_signal_validation'):
            # Add mixin methods to existing instance
            for method_name in ['initialize_signal_validation', 'validate_signal',
                              'get_signal_strength_score', 'filter_signals_by_strength',
                              'get_strategy_signal_summary']:
                if hasattr(StrategySignalMixin, method_name):
                    setattr(strategy_instance, method_name,
                           getattr(StrategySignalMixin, method_name).__get__(strategy_instance))

            # Initialize mixin attributes
            if not hasattr(strategy_instance, '_signal_validator'):
                strategy_instance._signal_validator = None
            if not hasattr(strategy_instance, '_signal_config'):
                strategy_instance._signal_config = None
            if not hasattr(strategy_instance, '_signal_history'):
                strategy_instance._signal_history = []
            if not hasattr(strategy_instance, 'logger'):
                strategy_instance.logger = logging.getLogger(__name__)

        # Get or create config
        config = self.strategy_configs.get(strategy_name)
        if not config:
            config = self._create_default_config(strategy_name)

        # Initialize validation
        strategy_instance.initialize_signal_validation(config)

        # Register custom calculator if available
        if strategy_name in self.custom_calculators:
            calculator = self.custom_calculators[strategy_name]
            strategy_instance._signal_validator.register_calculator(
                calculator.get_signal_type(), calculator
            )

        self.logger.info(f"Enhanced {strategy_name} strategy with signal validation")

    def _create_default_config(self, strategy_name: str) -> StrategySignalConfig:
        """Create default configuration based on strategy name."""
        # Strategy-specific defaults
        if 'swing' in strategy_name.lower():
            return StrategySignalConfig(
                strategy_name=strategy_name,
                default_signal_type=SignalType.BREAKOUT,
                minimum_strength_threshold=70.0,
                risk_reward_minimum=2.0
            )
        elif 'momentum' in strategy_name.lower() or 'weeklies' in strategy_name.lower():
            return StrategySignalConfig(
                strategy_name=strategy_name,
                default_signal_type=SignalType.MOMENTUM,
                minimum_strength_threshold=65.0,
                risk_reward_minimum=1.8
            )
        elif 'leaps' in strategy_name.lower():
            return StrategySignalConfig(
                strategy_name=strategy_name,
                default_signal_type=SignalType.TREND,
                minimum_strength_threshold=60.0,
                risk_reward_minimum=1.5
            )
        else:
            return StrategySignalConfig(
                strategy_name=strategy_name,
                default_signal_type=SignalType.TECHNICAL,
                minimum_strength_threshold=65.0
            )


# Pre-configured integrator instance
signal_integrator = SignalValidationIntegrator()

# Register specialized calculators
signal_integrator.register_custom_calculator("swing_trading", SwingTradingSignalCalculator())
signal_integrator.register_custom_calculator("momentum_weeklies", MomentumWeekliesSignalCalculator())
signal_integrator.register_custom_calculator("leaps_tracker", LEAPSSignalCalculator())

# Register strategy configurations
signal_integrator.register_strategy_config(
    "swing_trading",
    StrategySignalConfig(
        strategy_name="swing_trading",
        default_signal_type=SignalType.BREAKOUT,
        minimum_strength_threshold=70.0,
        minimum_confidence_threshold=0.7,
        risk_reward_minimum=2.0,
        custom_validation_params={'volume_confirmation_weight': 0.4}
    )
)

signal_integrator.register_strategy_config(
    "momentum_weeklies",
    StrategySignalConfig(
        strategy_name="momentum_weeklies",
        default_signal_type=SignalType.MOMENTUM,
        minimum_strength_threshold=75.0,
        minimum_confidence_threshold=0.6,
        risk_reward_minimum=1.8,
        custom_validation_params={'max_time_decay_hours': 72}
    )
)

signal_integrator.register_strategy_config(
    "leaps_tracker",
    StrategySignalConfig(
        strategy_name="leaps_tracker",
        default_signal_type=SignalType.TREND,
        minimum_strength_threshold=60.0,
        minimum_confidence_threshold=0.65,
        risk_reward_minimum=1.5,
        custom_validation_params={'regime_filter_enabled': True}
    )
)