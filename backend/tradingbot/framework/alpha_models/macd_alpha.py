"""
MACD Alpha Model

Generates insights based on MACD crossover signals.
"""

from datetime import timedelta
from typing import Any, Dict, List, Tuple
import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class MACDAlphaModel(AlphaModel):
    """
    Alpha model using MACD for trend-following signals.

    Generates:
    - Long insight when MACD crosses above signal line
    - Short insight when MACD crosses below signal line
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        insight_period: timedelta = timedelta(days=10),
        name: str = "MACDAlpha",
    ):
        """
        Initialize MACD Alpha Model.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            insight_period: How long insights are valid
            name: Model name
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.insight_period = insight_period

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """Generate MACD-based insights."""
        insights = []

        for symbol in symbols:
            symbol_data = data.get(symbol)
            if symbol_data is None:
                continue

            closes = symbol_data.get('close', [])
            min_length = self.slow_period + self.signal_period + 1
            if len(closes) < min_length:
                continue

            # Calculate MACD
            macd, signal, histogram = self._calculate_macd(closes)
            if macd is None:
                continue

            # Calculate previous values for crossover detection
            prev_macd, prev_signal, prev_histogram = self._calculate_macd(closes[:-1])
            if prev_macd is None:
                continue

            # Detect crossovers
            current_diff = macd - signal
            prev_diff = prev_macd - prev_signal

            if current_diff > 0 and prev_diff <= 0:
                # Bullish crossover
                confidence = min(abs(histogram) / abs(macd) if macd != 0 else 0.5, 0.9)
                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=0.03,  # 3% expected move
                    confidence=0.5 + confidence * 0.3,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={
                        'macd': macd,
                        'signal': signal,
                        'histogram': histogram,
                        'crossover': 'bullish',
                    },
                ))

            elif current_diff < 0 and prev_diff >= 0:
                # Bearish crossover
                confidence = min(abs(histogram) / abs(macd) if macd != 0 else 0.5, 0.9)
                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=0.03,
                    confidence=0.5 + confidence * 0.3,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={
                        'macd': macd,
                        'signal': signal,
                        'histogram': histogram,
                        'crossover': 'bearish',
                    },
                ))

        return insights

    def _calculate_macd(
        self,
        closes: List[float],
    ) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram."""
        closes = np.array(closes)

        if len(closes) < self.slow_period + self.signal_period:
            return None, None, None

        # Calculate EMAs
        fast_ema = self._ema(closes, self.fast_period)
        slow_ema = self._ema(closes, self.slow_period)

        # MACD line
        macd_line = fast_ema - slow_ema

        # We need enough MACD values to calculate signal line
        if len(closes) < self.slow_period + self.signal_period:
            return None, None, None

        # Calculate MACD for signal line calculation
        macd_values = []
        for i in range(self.signal_period + 1):
            idx = len(closes) - self.signal_period - 1 + i
            if idx >= self.slow_period:
                fast = self._ema(closes[:idx + 1], self.fast_period)
                slow = self._ema(closes[:idx + 1], self.slow_period)
                macd_values.append(fast - slow)

        if len(macd_values) < self.signal_period:
            return None, None, None

        # Signal line
        signal_line = self._ema(np.array(macd_values), self.signal_period)

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return np.mean(data)

        multiplier = 2 / (period + 1)
        ema = data[:period].mean()

        for price in data[period:]:
            ema = (price - ema) * multiplier + ema

        return ema
