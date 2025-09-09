"""
Market Regime Filter and Signal Detection System
Implements the bull regime filters and pullback reversal signals from the successful playbook.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification"""
    BULL="bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNDEFINED = "undefined"


class SignalType(Enum):
    """Trading signal types"""
    BUY="buy"
    SELL = "sell"
    HOLD = "hold"
    NO_SIGNAL = "no_signal"


@dataclass
class TechnicalIndicators:
    """Container for technical analysis indicators"""
    price: float
    ema_20: float
    ema_50: float
    ema_200: float
    rsi_14: float
    atr_14: float
    volume: int
    high_24h: float
    low_24h: float

    # Derived indicators
    distance_from_20ema: float = 0.0
    distance_from_50ema: float = 0.0
    ema_20_slope: float = 0.0
    ema_50_slope: float = 0.0

    def __post_init__(self):
        """Calculate derived indicators"""
        self.distance_from_20ema=(self.price - self.ema_20) / self.price if self.price > 0 else 0
        self.distance_from_50ema=(self.price - self.ema_50) / self.price if self.price > 0 else 0


@dataclass
class MarketSignal:
    """Trading signal with context"""
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    regime: MarketRegime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: List[str] = None
    timestamp: datetime=None

    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning=[]
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TechnicalAnalysis:
    """Technical analysis calculations for regime detection"""

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return [np.nan] * len(prices)

        alpha=2.0 / (period + 1)
        ema=[prices[0]]  # Start with first price

        for price in prices[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])

        return ema

    @staticmethod
    def calculate_rsi(prices: List[float], period: int=14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return [np.nan] * len(prices)

        deltas=[prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains=[max(0, delta) for delta in deltas]
        losses=[-min(0, delta) for delta in deltas]

        # Calculate initial averages
        avg_gain=sum(gains[:period]) / period
        avg_loss=sum(losses[:period]) / period

        rsi_values=[np.nan] * (period)  # First 'period' values are NaN

        if avg_loss== 0:
            rsi_values.append(100.0)
        else:
            rs=avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

        # Calculate subsequent RSI values
        for i in range(period, len(deltas)):
            gain=gains[i]
            loss = losses[i]

            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss=(avg_loss * (period - 1) + loss) / period

            if avg_loss== 0:
                rsi_values.append(100.0)
            else:
                rs=avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))

        return rsi_values

    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int=14) -> List[float]:
        """Calculate Average True Range"""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return [np.nan] * len(closes)

        true_ranges=[]
        for i in range(1, len(closes)):
            tr=max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)

        # Calculate ATR using simple moving average initially
        atr_values=[np.nan]  # First value is NaN

        if len(true_ranges) >= period:
            # Initial ATR
            initial_atr=sum(true_ranges[:period]) / period
            atr_values.extend([np.nan] * (period - 1))
            atr_values.append(initial_atr)

            # Subsequent ATR values using Wilder's smoothing
            for i in range(period, len(true_ranges)):
                atr=(atr_values[-1] * (period - 1) + true_ranges[i]) / period
                atr_values.append(atr)

        return atr_values

    @staticmethod
    def calculate_slope(values: List[float], period: int=5) -> float:
        """Calculate slope of recent values using linear regression"""
        if len(values) < period:
            return 0.0

        recent_values=values[-period:]
        x = np.arange(len(recent_values))

        # Linear regression slope
        n=len(recent_values)
        sum_x=sum(x)
        sum_y=sum(recent_values)
        sum_xy=sum(x[i] * recent_values[i] for i in range(n))
        sum_x2=sum(xi ** 2 for xi in x)

        slope=(n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope


class MarketRegimeFilter:
    """
    Market regime detection based on successful playbook criteria:
    - Bull: Close > 50-DMA and 50-DMA > 200-DMA and 20-DMA slope positive
    - Avoid binary events (earnings ±7 days, major macro)
    """

    def __init__(self):
        self.ta=TechnicalAnalysis()

        # Regime parameters from playbook
        self.min_trend_slope=0.001  # Minimum positive slope for 20-EMA
        self.rsi_pullback_min = 35
        self.rsi_pullback_max = 50
        self.max_distance_from_20ema = 0.01  # 1% maximum distance for pullback

    def determine_regime(self, indicators: TechnicalIndicators) -> MarketRegime:
        """
        Determine market regime based on EMA alignment and trend

        Bull Regime Criteria:
        1. Close > 50-EMA
        2. 50-EMA > 200-EMA
        3. 20-EMA slope positive
        """
        # Bull regime checks
        above_50ema=indicators.price > indicators.ema_50
        ema_alignment = indicators.ema_50 > indicators.ema_200
        positive_trend = indicators.ema_20_slope > self.min_trend_slope

        if above_50ema and ema_alignment and positive_trend:
            return MarketRegime.BULL

        # Bear regime (inverse conditions)
        below_50ema=indicators.price < indicators.ema_50
        ema_bearish = indicators.ema_50 < indicators.ema_200
        negative_trend = indicators.ema_20_slope < -self.min_trend_slope

        if below_50ema and ema_bearish and negative_trend:
            return MarketRegime.BEAR

        # Sideways if EMAs are close and slope is flat
        if abs(indicators.ema_20_slope) <= self.min_trend_slope:
            return MarketRegime.SIDEWAYS

        return MarketRegime.UNDEFINED

    def detect_pullback_setup(self, indicators: TechnicalIndicators, prev_indicators: TechnicalIndicators) -> bool:
        """
        Detect pullback setup: red day into 20-EMA support with RSI 35-50

        Setup Criteria:
        1. Previous day was red (close < open - approximated by price decline)
        2. Current price near/touching 20-EMA
        3. RSI between 35-50
        4. Still above 50-EMA (regime intact)
        """
        # Previous day was red (approximated by significant decline)
        prev_day_red=indicators.price < prev_indicators.price * 0.998  # At least 0.2% decline

        # Near 20-EMA (within 1% or touching)
        near_20ema=(
            abs(indicators.distance_from_20ema) <= self.max_distance_from_20ema or
            indicators.low_24h <= indicators.ema_20
        )

        # RSI in pullback range
        rsi_in_range=self.rsi_pullback_min <= indicators.rsi_14 <= self.rsi_pullback_max

        # Still above 50-EMA (trend intact)
        above_50ema=indicators.price > indicators.ema_50

        return prev_day_red and near_20ema and rsi_in_range and above_50ema

    def detect_reversal_trigger(self, indicators: TechnicalIndicators, prev_indicators: TechnicalIndicators) -> bool:
        """
        Detect reversal trigger: intraday recovery above 20-EMA and prior high

        Trigger Criteria:
        1. Current close > 20-EMA
        2. Current close > previous high (momentum confirmation)
        3. Volume expansion (if available)
        """
        # Recovery above 20-EMA
        above_20ema=indicators.price > indicators.ema_20

        # Above previous high (momentum)
        above_prev_high=indicators.price > prev_indicators.high_24h

        # Volume confirmation (if volume significantly higher)
        volume_expansion=indicators.volume > prev_indicators.volume * 1.2

        return above_20ema and above_prev_high and volume_expansion


class SignalGenerator:
    """Generate trading signals based on regime and technical setup"""

    def __init__(self):
        self.regime_filter=MarketRegimeFilter()

    def generate_signal(
        self,
        current_indicators: TechnicalIndicators,
        previous_indicators: TechnicalIndicators,
        earnings_risk: bool=False,
        macro_risk: bool=False
    ) -> MarketSignal:
        """
        Generate trading signal based on regime and setup

        Args:
            current_indicators: Current market indicators
            previous_indicators: Previous period indicators
            earnings_risk: True if within ±7 days of earnings
            macro_risk: True if major macro event imminent

        Returns:
            MarketSignal with recommendation
        """
        regime=self.regime_filter.determine_regime(current_indicators)
        reasoning=[]

        # Risk filters first
        if earnings_risk:
            reasoning.append("Earnings risk detected - avoiding new positions")
            return MarketSignal(SignalType.HOLD, 0.0, regime, reasoning=reasoning)

        if macro_risk:
            reasoning.append("Major macro event risk - avoiding new positions")
            return MarketSignal(SignalType.HOLD, 0.0, regime, reasoning=reasoning)

        # Only trade in bull regime
        if regime != MarketRegime.BULL:
            reasoning.append(f"Market regime is {regime.value} - not bull market")
            return MarketSignal(SignalType.NO_SIGNAL, 0.0, regime, reasoning=reasoning)

        # Check for pullback setup
        has_pullback_setup=self.regime_filter.detect_pullback_setup(
            current_indicators, previous_indicators
        )

        if has_pullback_setup:
            reasoning.append("Pullback setup detected: red day into 20-EMA support with RSI 35-50")

            # Check for reversal trigger
            has_reversal_trigger=self.regime_filter.detect_reversal_trigger(
                current_indicators, previous_indicators
            )

            if has_reversal_trigger:
                reasoning.extend([
                    "Reversal trigger confirmed: price above 20-EMA and previous high",
                    "Consider ~5% OTM ~30 DTE calls"
                ])

                # Calculate confidence based on signal strength
                confidence=self._calculate_signal_confidence(current_indicators, previous_indicators)

                return MarketSignal(
                    SignalType.BUY,
                    confidence,
                    regime,
                    reasoning=reasoning
                )
            else:
                reasoning.append("Pullback setup present but awaiting reversal trigger")
                return MarketSignal(SignalType.HOLD, 0.3, regime, reasoning=reasoning)

        # Default to hold in bull market
        reasoning.append("Bull regime confirmed but no setup detected")
        return MarketSignal(SignalType.HOLD, 0.1, regime, reasoning=reasoning)

    def _calculate_signal_confidence(
        self,
        current: TechnicalIndicators,
        previous: TechnicalIndicators
    ) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence=0.0

        # Base confidence for bull regime
        confidence += 0.3

        # EMA alignment strength
        if current.ema_20 > current.ema_50 > current.ema_200:
            confidence += 0.2

        # Trend strength (20-EMA slope)
        if current.ema_20_slope > 0.002:  # Strong positive slope
            confidence += 0.2
        elif current.ema_20_slope > 0.001:  # Moderate positive slope
            confidence += 0.1

        # Volume confirmation
        volume_ratio=current.volume / previous.volume if previous.volume > 0 else 1.0
        if volume_ratio > 1.5:  # Strong volume
            confidence += 0.2
        elif volume_ratio > 1.2:  # Moderate volume
            confidence += 0.1

        # Price action strength
        price_momentum = (current.price - previous.price) / previous.price
        if price_momentum > 0.01:  # Strong upward momentum
            confidence += 0.1

        return min(confidence, 1.0)


def create_sample_indicators(
    price: float,
    ema_20: float,
    ema_50: float,
    ema_200: float,
    rsi: float,
    volume: int=1000000,
    high: float=None,
    low: float=None
) -> TechnicalIndicators:
    """Helper function to create sample indicators for testing"""
    return TechnicalIndicators(
        price=price,
        ema_20=ema_20,
        ema_50=ema_50,
        ema_200=ema_200,
        rsi_14=rsi,
        atr_14=price * 0.02,  # 2% ATR
        volume=volume,
        high_24h=high or price * 1.01,
        low_24h=low or price * 0.99
    )


if __name__== "__main__":# Test the regime filter and signal generator
    signal_gen = SignalGenerator()

    # Test case 1: Bull regime pullback setup
    print("=== TEST CASE 1: BULL REGIME PULLBACK===")

    # Previous day (red day)
    prev_indicators=create_sample_indicators(
        price=210.0,
        ema_20=208.0,
        ema_50=205.0,
        ema_200=200.0,
        rsi=45,
        volume=1500000
    )

    # Current day (pullback to 20-EMA with reversal)
    current_indicators=create_sample_indicators(
        price=208.5,
        ema_20=208.0,
        ema_50=205.0,
        ema_200=200.0,
        rsi=42,
        volume=2000000,
        high=211.0,
        low=207.0
    )

    signal=signal_gen.generate_signal(current_indicators, prev_indicators)
    print(f"Signal: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Regime: {signal.regime.value}")
    print("Reasoning:")
    for reason in signal.reasoning:
        print(f"  - {reason}")

    # Test case 2: Bear regime
    print("\n=== TEST CASE 2: BEAR REGIME ===")

    bear_indicators=create_sample_indicators(
        price=195.0,
        ema_20=198.0,
        ema_50=202.0,
        ema_200=205.0,
        rsi=35,
        volume=1000000
    )

    signal=signal_gen.generate_signal(bear_indicators, bear_indicators)
    print(f"Signal: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Regime: {signal.regime.value}")
    print("Reasoning:")
    for reason in signal.reasoning:
        print(f"  - {reason}")
