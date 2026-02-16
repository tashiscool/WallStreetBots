"""
Comprehensive Signal Strength Validation Framework
=================================================

Standardized signal strength validation across all trading strategies.
Provides consistent scoring, validation, and reporting for signal quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union, Protocol, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json
import warnings

try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, using simplified statistical methods")


class SignalType(Enum):
    """Types of trading signals."""
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


class SignalQuality(Enum):
    """Signal quality levels (see _grade_signal_quality for thresholds)."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"          # 80-89
    FAIR = "fair"          # 70-79
    POOR = "poor"          # 50-69
    VERY_POOR = "very_poor" # 0-49


@dataclass
class SignalMetrics:
    """Core signal strength metrics."""
    strength_score: float  # 0-100 normalized score
    confidence: float      # 0-1 confidence level
    consistency: float     # 0-1 historical consistency
    volume_confirmation: float  # 0-1 volume support
    technical_confluence: float  # 0-1 multiple indicator agreement
    risk_reward_ratio: float    # Expected return/risk ratio
    time_decay_factor: float    # Time sensitivity (lower = more urgent)
    market_regime_fit: float    # 0-1 fit with current regime


@dataclass
class SignalValidationResult:
    """Result of signal strength validation."""
    signal_id: str
    signal_type: SignalType
    timestamp: datetime
    symbol: str

    # Core metrics
    raw_metrics: SignalMetrics
    normalized_score: float  # 0-100 final score
    quality_grade: SignalQuality

    # Validation results
    passes_minimum_threshold: bool
    passes_consistency_check: bool
    passes_regime_filter: bool
    passes_risk_check: bool

    # Recommendations
    recommended_action: str  # "trade", "monitor", "reject"
    confidence_level: float  # 0-1 overall confidence
    suggested_position_size: float  # 0-1 position sizing multiplier

    # Metadata
    validation_notes: List[str] = field(default_factory=list)
    historical_performance_percentile: Optional[float] = None


class SignalStrengthCalculator(ABC):
    """Abstract base for strategy-specific signal strength calculators."""

    @abstractmethod
    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate raw signal strength (0-100)."""
        pass

    @abstractmethod
    def get_signal_type(self) -> SignalType:
        """Return the signal type this calculator handles."""
        pass

    @abstractmethod
    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate signal confidence (0-1)."""
        pass


class BreakoutSignalCalculator(SignalStrengthCalculator):
    """Calculator for breakout signal strength."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, lookback_periods: int = 20, volume_weight: float = 0.3):
        # Handle both old-style and new-style parameters
        if params is not None:
            self.lookback_periods = params.get('lookback_period', params.get('lookback_periods', lookback_periods))
            self.volume_weight = params.get('volume_weight', volume_weight)
            self.params = params
        else:
            self.lookback_periods = lookback_periods
            self.volume_weight = volume_weight
            self.params = {
                'lookback_periods': lookback_periods,
                'volume_weight': volume_weight
            }

    def get_signal_type(self) -> SignalType:
        return SignalType.BREAKOUT

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate breakout strength based on price action and volume."""
        if market_data is None or len(market_data) < 10:  # Need at least 10 periods for meaningful breakout calculation
            return 0.0

        try:
            prices = market_data['Close'].values
            volumes = market_data['Volume'].values if 'Volume' in market_data else None

            # Current price vs resistance
            current_price = prices[-1]
            
            # Calculate resistance level from available data
            # Look at the first half of the data to find resistance, excluding recent periods
            resistance_periods = min(self.lookback_periods, len(prices) // 2)
            resistance_level = np.max(prices[:resistance_periods]) if len(prices) > resistance_periods else np.max(prices[:-5])

            # Breakout strength (% above resistance)
            breakout_pct = max(0, (current_price - resistance_level) / resistance_level * 100)
            
            # Scale up breakout percentage for better scoring
            breakout_strength = min(100, breakout_pct * 20)  # Scale up by 20x

            # Volume confirmation
            volume_strength = 0.0
            if volumes is not None and len(volumes) >= 3:
                recent_volume = np.mean(volumes[-3:])
                # Use available data for average volume calculation
                lookback_for_volume = min(self.lookback_periods, len(volumes) - 3)
                if lookback_for_volume > 0:
                    avg_volume = np.mean(volumes[-lookback_for_volume-3:-3])
                    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                    volume_strength = min(100, (volume_ratio - 1) * 50)  # Normalize to 0-100

            # Combine price and volume signals
            price_weight = 1.0 - self.volume_weight
            strength = (breakout_strength * price_weight + volume_strength * self.volume_weight)

            return min(100.0, max(0.0, strength))

        except Exception as e:
            logging.error(f"Error calculating breakout strength: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate confidence based on breakout clarity and volume."""
        if market_data is None or len(market_data) < 10:  # Need at least 10 periods for meaningful confidence calculation
            return 0.0

        try:
            prices = market_data['Close'].values
            volumes = market_data['Volume'].values if 'Volume' in market_data else None

            # Price clarity (how clear the breakout is)
            current_price = prices[-1]
            # Use same resistance calculation as strength calculation
            resistance_periods = min(self.lookback_periods, len(prices) // 2)
            resistance_level = np.max(prices[:resistance_periods]) if len(prices) > resistance_periods else np.max(prices[:-5])
            support_level = np.min(prices[:resistance_periods]) if len(prices) > resistance_periods else np.min(prices[:-5])

            range_size = resistance_level - support_level
            breakout_distance = current_price - resistance_level

            # If we have a clear breakout, give high confidence
            if breakout_distance > 0:
                # For clear breakouts, give high confidence
                breakout_pct = breakout_distance / resistance_level * 100
                if breakout_pct > 2:  # More than 2% breakout
                    price_confidence = 0.9
                elif breakout_pct > 1:  # More than 1% breakout
                    price_confidence = 0.8
                else:
                    price_confidence = min(1.0, max(0.0, breakout_distance / range_size)) if range_size > 0 else 0.7
            else:
                price_confidence = 0.0

            # Volume confidence
            volume_confidence = 0.5  # Default if no volume data
            if volumes is not None:
                recent_volume = np.mean(volumes[-3:])
                avg_volume = np.mean(volumes[-self.lookback_periods:])
                volume_confidence = min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5

            return (price_confidence + volume_confidence) / 2.0

        except Exception as e:
            logging.error(f"Error calculating breakout confidence: {e}")
            return 0.0


class MomentumSignalCalculator(SignalStrengthCalculator):
    """Calculator for momentum signal strength."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, short_window: int = 5, long_window: int = 20):
        # Handle both old-style and new-style parameters
        if params is not None:
            self.short_window = params.get('momentum_period', params.get('short_window', short_window))
            self.long_window = params.get('long_window', long_window)
            self.params = params
        else:
            self.short_window = short_window
            self.long_window = long_window
            self.params = {
                'short_window': short_window,
                'long_window': long_window
            }

    def get_signal_type(self) -> SignalType:
        return SignalType.MOMENTUM

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate momentum strength based on moving averages and price acceleration."""
        if len(market_data) < self.long_window:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Moving averages
            short_ma = np.mean(prices[-self.short_window:])
            long_ma = np.mean(prices[-self.long_window:])

            # Momentum strength (% difference between MAs)
            ma_divergence = (short_ma - long_ma) / long_ma * 100 if long_ma > 0 else 0

            # Price acceleration (rate of change of rate of change)
            if len(prices) >= 10:
                roc_5 = (prices[-1] - prices[-6]) / prices[-6] * 100 if prices[-6] > 0 else 0
                roc_10 = (prices[-6] - prices[-11]) / prices[-11] * 100 if len(prices) >= 11 and prices[-11] > 0 else 0
                acceleration = roc_5 - roc_10
            else:
                acceleration = 0

            # Combine signals
            strength = abs(ma_divergence) * 5 + max(0, acceleration) * 2

            return min(100.0, max(0.0, strength))

        except Exception as e:
            logging.error(f"Error calculating momentum strength: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate confidence based on trend consistency."""
        if len(market_data) < self.long_window:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Trend consistency (how many periods in same direction)
            short_ma = np.mean(prices[-self.short_window:])
            long_ma = np.mean(prices[-self.long_window:])

            if short_ma > long_ma:  # Uptrend
                consistent_periods = sum(1 for i in range(self.short_window)
                                       if prices[-(i+1)] > prices[-(i+2)] if len(prices) > i+1)
            else:  # Downtrend
                consistent_periods = sum(1 for i in range(self.short_window)
                                       if prices[-(i+1)] < prices[-(i+2)] if len(prices) > i+1)

            consistency = consistent_periods / (self.short_window - 1) if self.short_window > 1 else 0

            return min(1.0, max(0.0, consistency))

        except Exception as e:
            logging.error(f"Error calculating momentum confidence: {e}")
            return 0.0


class SignalStrengthValidator:
    """Main validator for signal strength across all strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # Register signal calculators
        self.calculators = {
            SignalType.BREAKOUT: BreakoutSignalCalculator(),
            SignalType.MOMENTUM: MomentumSignalCalculator(),
        }

        # Historical performance tracking
        self.signal_history: List[SignalValidationResult] = []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'minimum_strength_threshold': 60.0,
            'minimum_confidence_threshold': 0.6,
            'minimum_consistency_threshold': 0.7,
            'consistency_threshold': 0.7,  # Alias for backward compatibility
            'volume_confirmation_weight': 0.3,
            'technical_confluence_weight': 0.2,
            'risk_reward_minimum': 1.5,
            'max_time_decay_hours': 24,
            'regime_filter_enabled': True,
            'consistency_lookback_days': 30,
        }

    def register_calculator(self, signal_type: SignalType, calculator: SignalStrengthCalculator):
        """Register a custom signal strength calculator."""
        self.calculators[signal_type] = calculator

    def validate_signal(self,
                       signal_type: SignalType,
                       symbol: str,
                       market_data: pd.DataFrame,
                       signal_params: Optional[Dict[str, Any]] = None) -> SignalValidationResult:
        """Validate a trading signal and return comprehensive results."""
        # Handle None signal_type
        if signal_type is None:
            signal_type = SignalType.BREAKOUT  # Default fallback
        
        signal_id = f"{symbol}_{signal_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        signal_params = signal_params or {}

        try:
            # Get appropriate calculator
            calculator = self.calculators.get(signal_type)
            if not calculator:
                return self._create_failed_result(signal_id, signal_type, symbol,
                                                "No calculator available for signal type")

            # Calculate raw metrics
            raw_strength = calculator.calculate_raw_strength(market_data, **signal_params)
            confidence = calculator.calculate_confidence(market_data, **signal_params)

            # Calculate additional metrics
            metrics = self._calculate_comprehensive_metrics(
                market_data, raw_strength, confidence, signal_params
            )

            # Normalize final score
            normalized_score = self._calculate_normalized_score(metrics)

            # Grade quality
            quality_grade = self._grade_signal_quality(normalized_score)

            # Run validation checks
            validation_checks = self._run_validation_checks(metrics, normalized_score)

            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, normalized_score, validation_checks)

            # Create result
            result = SignalValidationResult(
                signal_id=signal_id,
                signal_type=signal_type,
                timestamp=datetime.now(),
                symbol=symbol,
                raw_metrics=metrics,
                normalized_score=normalized_score,
                quality_grade=quality_grade,
                passes_minimum_threshold=validation_checks['minimum_threshold'],
                passes_consistency_check=validation_checks['consistency_check'],
                passes_regime_filter=validation_checks['regime_filter'],
                passes_risk_check=validation_checks['risk_check'],
                recommended_action=recommendations['action'],
                confidence_level=recommendations['confidence'],
                suggested_position_size=recommendations['position_size'],
                validation_notes=recommendations['notes'],
                historical_performance_percentile=self._calculate_historical_percentile(normalized_score)
            )

            # Store for historical analysis
            self.signal_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return self._create_failed_result(signal_id, signal_type, symbol, str(e))

    def _calculate_comprehensive_metrics(self,
                                       market_data: pd.DataFrame,
                                       raw_strength: float,
                                       confidence: float,
                                       signal_params: Dict[str, Any]) -> SignalMetrics:
        """Calculate comprehensive signal metrics."""
        # If raw strength is 0 (insufficient data), return all zeros
        if raw_strength == 0.0:
            return SignalMetrics(
                strength_score=0.0,
                confidence=0.0,
                consistency=0.0,
                volume_confirmation=0.0,
                technical_confluence=0.0,
                risk_reward_ratio=0.0,
                time_decay_factor=0.0,
                market_regime_fit=0.0
            )

        # Volume confirmation
        volume_confirmation = self._calculate_volume_confirmation(market_data)

        # Technical confluence
        technical_confluence = self._calculate_technical_confluence(market_data)

        # Risk-reward ratio
        risk_reward_ratio = signal_params.get('risk_reward_ratio', 2.0)

        # Time decay factor
        time_decay_factor = self._calculate_time_decay_factor(signal_params)

        # Market regime fit
        market_regime_fit = self._calculate_market_regime_fit(market_data)

        # Consistency (simplified for now)
        consistency = confidence  # Can be enhanced with historical analysis

        return SignalMetrics(
            strength_score=raw_strength,
            confidence=confidence,
            consistency=consistency,
            volume_confirmation=volume_confirmation,
            technical_confluence=technical_confluence,
            risk_reward_ratio=risk_reward_ratio,
            time_decay_factor=time_decay_factor,
            market_regime_fit=market_regime_fit
        )

    def _calculate_volume_confirmation(self, market_data: pd.DataFrame) -> float:
        """Calculate volume confirmation score (0-1)."""
        if 'Volume' not in market_data.columns or len(market_data) < 10:
            return 0.0  # No confirmation if no volume data

        try:
            volumes = market_data['Volume'].values
            recent_volume = np.mean(volumes[-3:])
            avg_volume = np.mean(volumes[-20:])

            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            return min(1.0, max(0.0, (volume_ratio - 0.5) * 2))

        except Exception:
            return 0.5

    def _calculate_technical_confluence(self, market_data: pd.DataFrame, signal_type: SignalType = None) -> float:
        """Calculate technical indicator confluence (0-1)."""
        if len(market_data) < 20:
            return 0.5

        try:
            prices = market_data['Close'].values

            # Simple technical signals
            signals = []

            # Moving average signal
            if len(prices) >= 20:
                ma_20 = np.mean(prices[-20:])
                signals.append(1.0 if prices[-1] > ma_20 else 0.0)

            # Price momentum signal
            if len(prices) >= 10:
                momentum = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
                signals.append(1.0 if momentum > 0 else 0.0)

            # Volatility signal (prefer lower volatility for signal quality)
            if len(prices) >= 10:
                volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if np.mean(prices[-10:]) > 0 else 0
                signals.append(1.0 if volatility < 0.05 else 0.0)  # Low volatility = good

            return np.mean(signals) if signals else 0.5

        except Exception:
            return 0.5

    def _calculate_time_decay_factor(self, signal_params: Dict[str, Any]) -> float:
        """Calculate time decay factor (0-1, higher = more urgent)."""
        # Handle both dict and datetime inputs for backward compatibility
        if isinstance(signal_params, datetime):
            # If signal_params is a datetime, calculate time since then
            signal_time = signal_params
            hours_ago = (datetime.now() - signal_time).total_seconds() / 3600
            max_decay_hours = self.config.get('max_time_decay_hours', 24)
            urgency = 1.0 - min(1.0, hours_ago / max_decay_hours)
            return max(0.0, min(1.0, urgency))
        elif isinstance(signal_params, dict) and 'signal_timestamp' in signal_params:
            # Handle signal_timestamp in dict
            signal_time = signal_params['signal_timestamp']
            hours_ago = (datetime.now() - signal_time).total_seconds() / 3600
            max_decay_hours = self.config.get('max_time_decay_hours', 24)
            urgency = 1.0 - min(1.0, hours_ago / max_decay_hours)
            return max(0.0, min(1.0, urgency))
        else:
            # Original logic for dict input
            max_hold_hours = signal_params.get('max_hold_hours', 24)
            max_decay_hours = self.config.get('max_time_decay_hours', 24)

            # Higher urgency for shorter-term signals
            urgency = 1.0 - min(1.0, max_hold_hours / max_decay_hours)
            return max(0.0, min(1.0, urgency))

    def _calculate_market_regime_fit(self, market_data: pd.DataFrame, signal_type: SignalType = None) -> float:
        """Calculate how well signal fits current market regime (0-1)."""
        if len(market_data) < 20:
            return 0.5

        try:
            prices = market_data['Close'].values

            # Simple regime detection based on volatility and trend
            recent_volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if np.mean(prices[-10:]) > 0 else 0
            recent_trend = (prices[-1] - prices[-20]) / prices[-20] if prices[-20] > 0 else 0

            # Prefer signals in trending, low-volatility environments
            trend_score = min(1.0, abs(recent_trend) * 10)  # Strong trend = good
            volatility_score = max(0.0, 1.0 - recent_volatility * 20)  # Low volatility = good

            return (trend_score + volatility_score) / 2.0

        except Exception:
            return 0.5

    def _calculate_normalized_score(self, metrics: SignalMetrics) -> float:
        """Calculate final normalized score (0-100)."""
        # Weighted combination of all metrics
        weights = {
            'strength': 0.30,
            'confidence': 0.20,
            'consistency': 0.15,
            'volume_confirmation': 0.10,
            'technical_confluence': 0.10,
            'risk_reward': 0.10,
            'market_regime_fit': 0.05
        }

        # Normalize risk-reward ratio to 0-1 scale
        risk_reward_normalized = min(1.0, metrics.risk_reward_ratio / 3.0)

        score = (
            metrics.strength_score * weights['strength'] +
            metrics.confidence * 100 * weights['confidence'] +
            metrics.consistency * 100 * weights['consistency'] +
            metrics.volume_confirmation * 100 * weights['volume_confirmation'] +
            metrics.technical_confluence * 100 * weights['technical_confluence'] +
            risk_reward_normalized * 100 * weights['risk_reward'] +
            metrics.market_regime_fit * 100 * weights['market_regime_fit']
        )

        return min(100.0, max(0.0, score))

    def _grade_signal_quality(self, score: float) -> SignalQuality:
        """Grade signal quality based on score."""
        if score >= 90:
            return SignalQuality.EXCELLENT
        elif score >= 80:
            return SignalQuality.GOOD
        elif score >= 70:
            return SignalQuality.FAIR
        elif score >= 50:
            return SignalQuality.POOR
        else:
            return SignalQuality.VERY_POOR

    def _run_validation_checks(self, metrics: SignalMetrics, score: float) -> Dict[str, bool]:
        """Run comprehensive validation checks."""
        return {
            'minimum_threshold': score >= self.config.get('minimum_strength_threshold', 60.0),
            'consistency_check': metrics.consistency >= self.config.get('minimum_consistency_threshold', 0.6),
            'regime_filter': metrics.market_regime_fit >= 0.3 if self.config.get('regime_filter_enabled', True) else True,
            'risk_check': metrics.risk_reward_ratio >= self.config.get('risk_reward_minimum', 1.5)
        }

    def _generate_recommendations(self,
                                metrics: SignalMetrics,
                                score: float,
                                validation_checks: Dict[str, bool]) -> Dict[str, Any]:
        """Generate trading recommendations based on validation results."""
        all_checks_pass = all(validation_checks.values())

        if all_checks_pass and score >= 80:
            action = "trade"
            confidence = 0.9
            position_size = min(1.0, score / 100.0)
            notes = ["High quality signal - recommended for trading"]
        elif all_checks_pass and score >= 60:
            action = "trade"
            confidence = 0.7
            position_size = min(0.7, score / 100.0)
            notes = ["Good quality signal - trade with reduced size"]
        elif score >= 50:
            action = "monitor"
            confidence = 0.5
            position_size = 0.0
            notes = ["Marginal signal - monitor for improvement"]
        else:
            action = "reject"
            confidence = 0.2
            position_size = 0.0
            notes = ["Poor quality signal - reject"]

        # Add specific notes for failed checks
        for check, passed in validation_checks.items():
            if not passed:
                notes.append(f"Failed {check} validation")

        return {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'notes': notes
        }

    def _calculate_historical_percentile(self, score: float) -> Optional[float]:
        """Calculate percentile rank against historical signals."""
        if len(self.signal_history) < 10:
            return None

        historical_scores = [s.normalized_score for s in self.signal_history[-100:]]  # Last 100 signals
        if not historical_scores:
            return None

        percentile = sum(1 for s in historical_scores if s <= score) / len(historical_scores) * 100
        return percentile

    def _create_failed_result(self, signal_id: str, signal_type: SignalType,
                            symbol: str, error_msg: str, reason: Optional[str] = None) -> SignalValidationResult:
        """Create a failed validation result."""
        notes = [f"Validation failed: {error_msg}"]
        if reason:
            notes.append(f"Reason: {reason}")
        
        return SignalValidationResult(
            signal_id=signal_id,
            signal_type=signal_type,
            timestamp=datetime.now(),
            symbol=symbol,
            raw_metrics=SignalMetrics(0, 0, 0, 0, 0, 0, 0, 0),
            normalized_score=0.0,
            quality_grade=SignalQuality.VERY_POOR,
            passes_minimum_threshold=False,
            passes_consistency_check=False,
            passes_regime_filter=False,
            passes_risk_check=False,
            recommended_action="reject",
            confidence_level=0.0,
            suggested_position_size=0.0,
            validation_notes=notes
        )

    def _determine_quality_grade(self, score: float) -> SignalQuality:
        """Determine signal quality grade based on score."""
        return self._grade_signal_quality(score)

    def _calculate_position_size(self, metrics: SignalMetrics, score: float) -> float:
        """Calculate suggested position size based on signal quality."""
        if score < 50:
            return 0.0  # Reject low quality signals
        
        # Base position size on score and risk metrics
        base_size = score / 100.0
        
        # Adjust based on risk-reward ratio
        risk_adjustment = min(1.0, metrics.risk_reward_ratio / 2.0)
        
        # Adjust based on confidence
        confidence_adjustment = metrics.confidence
        
        final_size = base_size * risk_adjustment * confidence_adjustment
        return min(1.0, max(0.0, final_size))

    def _generate_validation_notes(self, result: SignalValidationResult) -> List[str]:
        """Generate validation notes based on result."""
        notes = []
        
        # Quality-based notes
        if result.quality_grade == SignalQuality.EXCELLENT:
            notes.append("Excellent signal quality - high confidence trade")
        elif result.quality_grade == SignalQuality.GOOD:
            notes.append("Good quality signal - recommended for trading")
        elif result.quality_grade == SignalQuality.FAIR:
            notes.append("Fair quality signal - trade with caution")
        elif result.quality_grade == SignalQuality.POOR:
            notes.append("Poor quality signal - monitor only")
        else:
            notes.append("Very poor signal quality - reject")
        
        # Add specific validation notes
        notes.extend(result.validation_notes)
        
        return notes

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of recent validation results."""
        if not self.signal_history:
            return {
                "message": "No validation history available",
                "total_signals_validated": 0,
                "total_signals": 0,
                "average_score": 0.0,
                "average_strength_score": 0.0,
                "quality_distribution": {},
                "pass_rates": {},
                "recommendation_distribution": {}
            }

        recent_signals = self.signal_history[-50:]  # Last 50 signals

        return {
            'total_signals_validated': len(recent_signals),
            'total_signals': len(recent_signals),
            'average_score': np.mean([s.normalized_score for s in recent_signals]),
            'average_strength_score': np.mean([s.normalized_score for s in recent_signals]),  # Alias
            'quality_distribution': {
                quality.value: sum(1 for s in recent_signals if s.quality_grade == quality)
                for quality in SignalQuality
            },
            'signals_by_quality': {  # Alias for quality_distribution
                quality.value: sum(1 for s in recent_signals if s.quality_grade == quality)
                for quality in SignalQuality
            },
            'pass_rates': {
                'minimum_threshold': sum(1 for s in recent_signals if s.passes_minimum_threshold) / len(recent_signals),
                'consistency_check': sum(1 for s in recent_signals if s.passes_consistency_check) / len(recent_signals),
                'regime_filter': sum(1 for s in recent_signals if s.passes_regime_filter) / len(recent_signals),
                'risk_check': sum(1 for s in recent_signals if s.passes_risk_check) / len(recent_signals),
            },
            'recommendation_distribution': {
                action: sum(1 for s in recent_signals if s.recommended_action == action)
                for action in ['trade', 'monitor', 'reject']
            }
        }

    def export_validation_history(self, filepath: Optional[str] = None, format: str = 'json'):
        """Export validation history to file."""
        history_data = []
        for result in self.signal_history:
            history_data.append({
                'signal_id': result.signal_id,
                'signal_type': result.signal_type.value,
                'timestamp': result.timestamp.isoformat(),
                'symbol': result.symbol,
                'normalized_score': result.normalized_score,
                'quality_grade': result.quality_grade.value,
                'recommended_action': result.recommended_action,
                'confidence_level': result.confidence_level,
                'validation_notes': result.validation_notes
            })

        if filepath is None:
            # Return data directly if no filepath provided
            if format.lower() == 'json':
                return json.dumps(history_data, indent=2)
            elif format.lower() == 'csv':
                import pandas as pd
                df = pd.DataFrame(history_data)
                return df.to_csv(index=False)
            elif format.lower() == 'dataframe':
                import pandas as pd
                return pd.DataFrame(history_data)
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            # Write to file
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(history_data, f, indent=2)
            elif format.lower() == 'csv':
                import pandas as pd
                df = pd.DataFrame(history_data)
                df.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")


class TrendSignalCalculator(SignalStrengthCalculator):
    """Calculator for trend signal strength."""

    def __init__(self, trend_periods: int = 20):
        self.trend_periods = trend_periods
        self.params = {
            'trend_periods': trend_periods
        }

    def get_signal_type(self) -> SignalType:
        return SignalType.TREND

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate trend strength based on moving averages and price action."""
        if len(market_data) < self.trend_periods:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Calculate trend strength
            ma_short = np.mean(prices[-10:])
            ma_long = np.mean(prices[-self.trend_periods:])

            # Trend direction and strength
            trend_direction = 1 if ma_short > ma_long else -1
            trend_strength = abs(ma_short - ma_long) / ma_long * 100 if ma_long > 0 else 0

            # Price momentum
            momentum = (prices[-1] - prices[-5]) / prices[-5] * 100 if len(prices) >= 5 and prices[-5] > 0 else 0

            # Combine trend and momentum
            strength = trend_strength * 0.7 + abs(momentum) * 0.3

            return min(100.0, max(0.0, strength))

        except Exception as e:
            logging.error(f"Error calculating trend strength: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate confidence based on trend consistency."""
        if len(market_data) < self.trend_periods:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Trend consistency over time
            ma_periods = [5, 10, 15, 20]
            trend_alignment = 0

            for period in ma_periods:
                if len(prices) >= period:
                    ma = np.mean(prices[-period:])
                    if prices[-1] > ma:
                        trend_alignment += 1

            consistency = trend_alignment / len(ma_periods)
            return min(1.0, max(0.0, consistency))

        except Exception as e:
            logging.error(f"Error calculating trend confidence: {e}")
            return 0.0


class CustomSignalCalculator(SignalStrengthCalculator):
    """Custom signal calculator for specialized strategies."""

    def __init__(self, signal_type: SignalType = SignalType.TECHNICAL, custom_params: Optional[Dict[str, Any]] = None,
                 strength_function: Optional[Callable] = None, confidence_function: Optional[Callable] = None):
        self.signal_type = signal_type
        self.custom_params = custom_params or {}
        self.strength_function = strength_function
        self.confidence_function = confidence_function
        self.params = {
            'signal_type': signal_type.value,
            'custom_params': self.custom_params,
            'strength_function': strength_function is not None,
            'confidence_function': confidence_function is not None
        }

    def get_signal_type(self) -> SignalType:
        return self.signal_type

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate custom signal strength."""
        if market_data is None or len(market_data) < 10:
            return 0.0

        # Use custom function if provided
        if self.strength_function:
            try:
                return self.strength_function(market_data)
            except Exception:
                return 0.0

        try:
            prices = market_data['Close'].values

            # Simple custom calculation based on price action
            recent_return = (prices[-1] - prices[-10]) / prices[-10] * 100 if prices[-10] > 0 else 0
            
            # Apply custom multiplier if provided
            multiplier = self.custom_params.get('strength_multiplier', 1.0)
            strength = abs(recent_return) * multiplier

            return min(100.0, max(0.0, strength))

        except Exception as e:
            logging.error(f"Error calculating custom signal strength: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate custom signal confidence."""
        if market_data is None or len(market_data) < 5:
            return 0.0

        # Use custom function if provided
        if self.confidence_function:
            try:
                return self.confidence_function(market_data)
            except Exception:
                return 0.0

        try:
            prices = market_data['Close'].values

            # Simple confidence based on recent price stability
            recent_volatility = np.std(prices[-5:]) / np.mean(prices[-5:]) if np.mean(prices[-5:]) > 0 else 0
            confidence = max(0.0, 1.0 - recent_volatility * 10)

            return min(1.0, confidence)

        except Exception as e:
            logging.error(f"Error calculating custom signal confidence: {e}")
            return 0.0


class SwingTradingSignalCalculator(SignalStrengthCalculator):
    """Signal calculator for swing trading strategies."""

    def __init__(self, lookback_periods: int = 20, volume_weight: float = 0.3):
        self.lookback_periods = lookback_periods
        self.volume_weight = volume_weight
        self.params = {
            'lookback_periods': lookback_periods,
            'volume_weight': volume_weight
        }

    def get_signal_type(self) -> SignalType:
        return SignalType.BREAKOUT

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate swing trading signal strength."""
        if market_data is None or len(market_data) < 10:
            return 0.0

        try:
            prices = market_data['Close'].values
            volumes = market_data['Volume'].values if 'Volume' in market_data else None

            # Swing trading focuses on price momentum and volume
            recent_return = (prices[-1] - prices[-5]) / prices[-5] * 100 if prices[-5] > 0 else 0
            momentum_strength = min(100, abs(recent_return) * 10)

            # Volume confirmation
            volume_strength = 0.0
            if volumes is not None and len(volumes) >= 5:
                recent_volume = np.mean(volumes[-3:])
                avg_volume = np.mean(volumes[-self.lookback_periods:])
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                volume_strength = min(100, (volume_ratio - 1) * 50)

            # Combine momentum and volume
            price_weight = 1.0 - self.volume_weight
            strength = (momentum_strength * price_weight + volume_strength * self.volume_weight)

            return min(100.0, max(0.0, strength))

        except Exception as e:
            logging.error(f"Error calculating swing trading strength: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate swing trading confidence."""
        if market_data is None or len(market_data) < 10:
            return 0.0

        try:
            prices = market_data['Close'].values
            volumes = market_data['Volume'].values if 'Volume' in market_data else None

            # Price trend consistency
            recent_trend = np.polyfit(range(5), prices[-5:], 1)[0]
            trend_consistency = min(1.0, abs(recent_trend) / np.std(prices[-5:]) if np.std(prices[-5:]) > 0 else 0)

            # Volume consistency
            volume_consistency = 0.5
            if volumes is not None:
                recent_volume = np.mean(volumes[-3:])
                avg_volume = np.mean(volumes[-self.lookback_periods:])
                volume_consistency = min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5

            return (trend_consistency + volume_consistency) / 2.0

        except Exception as e:
            logging.error(f"Error calculating swing trading confidence: {e}")
            return 0.0


class MomentumWeekliesSignalCalculator(SignalStrengthCalculator):
    """Signal calculator for momentum weekly strategies."""

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window
        self.params = {
            'short_window': short_window,
            'long_window': long_window
        }

    def get_signal_type(self) -> SignalType:
        return SignalType.MOMENTUM

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate momentum weekly signal strength."""
        if market_data is None or len(market_data) < self.long_window:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Calculate moving averages
            short_ma = np.mean(prices[-self.short_window:])
            long_ma = np.mean(prices[-self.long_window:])

            # Momentum strength based on MA crossover
            momentum_pct = (short_ma - long_ma) / long_ma * 100 if long_ma > 0 else 0
            strength = min(100, abs(momentum_pct) * 20)

            return min(100.0, max(0.0, strength))

        except Exception as e:
            logging.error(f"Error calculating momentum weekly strength: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate momentum weekly confidence."""
        if market_data is None or len(market_data) < self.long_window:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Trend consistency over time
            short_trend = np.polyfit(range(self.short_window), prices[-self.short_window:], 1)[0]
            long_trend = np.polyfit(range(self.long_window), prices[-self.long_window:], 1)[0]

            # Confidence based on trend alignment
            if short_trend * long_trend > 0:  # Same direction
                confidence = min(1.0, abs(short_trend) / abs(long_trend)) if long_trend != 0 else 0.5
            else:  # Opposite directions
                confidence = 0.3

            return confidence

        except Exception as e:
            logging.error(f"Error calculating momentum weekly confidence: {e}")
            return 0.0


class LEAPSTrackerSignalCalculator(SignalStrengthCalculator):
    """Signal calculator for LEAPS tracker strategies."""

    def __init__(self, trend_periods: int = 30):
        self.trend_periods = trend_periods
        self.params = {
            'trend_periods': trend_periods
        }

    def get_signal_type(self) -> SignalType:
        return SignalType.TREND

    def calculate_raw_strength(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate LEAPS tracker signal strength."""
        if market_data is None or len(market_data) < self.trend_periods:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Long-term trend strength
            trend_slope = np.polyfit(range(self.trend_periods), prices[-self.trend_periods:], 1)[0]
            trend_strength = min(100, abs(trend_slope) * 1000)

            # Volatility-adjusted strength
            volatility = np.std(prices[-self.trend_periods:])
            if volatility > 0:
                trend_strength = trend_strength / volatility * 10

            return min(100.0, max(0.0, trend_strength))

        except Exception as e:
            logging.error(f"Error calculating LEAPS tracker strength: {e}")
            return 0.0

    def calculate_confidence(self, market_data: pd.DataFrame, **kwargs) -> float:
        """Calculate LEAPS tracker confidence."""
        if market_data is None or len(market_data) < self.trend_periods:
            return 0.0

        try:
            prices = market_data['Close'].values

            # Trend consistency over different timeframes
            short_trend = np.polyfit(range(10), prices[-10:], 1)[0]
            long_trend = np.polyfit(range(self.trend_periods), prices[-self.trend_periods:], 1)[0]

            # Confidence based on trend alignment and strength
            if short_trend * long_trend > 0:  # Same direction
                trend_strength = abs(long_trend)
                confidence = min(1.0, trend_strength * 1000)
            else:  # Opposite directions
                confidence = 0.2

            return confidence

        except Exception as e:
            logging.error(f"Error calculating LEAPS tracker confidence: {e}")
            return 0.0