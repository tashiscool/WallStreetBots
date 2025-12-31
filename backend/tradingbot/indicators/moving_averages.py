"""
Moving Average Indicators - Ported from QuantConnect LEAN.

Includes: SMA, EMA, DEMA, TEMA, Wilder's, KAMA

Original: https://github.com/QuantConnect/Lean/blob/master/Indicators/
License: Apache 2.0
"""

from collections import deque
from datetime import datetime
from typing import Optional

from .base import Indicator, IndicatorDataPoint, WindowIndicator


class SimpleMovingAverage(WindowIndicator):
    """
    Simple Moving Average (SMA).

    Calculates the arithmetic mean of prices over a specified period.

    Formula: SMA = (P1 + P2 + ... + Pn) / n
    """

    def __init__(self, period: int, name: Optional[str] = None):
        name = name or f"SMA({period})"
        super().__init__(name, period)
        self._sum = 0.0

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        # Maintain running sum for efficiency
        if len(self._values) >= self._period:
            oldest = self._values[0]
            self._sum -= oldest
        self._values.append(input_point.value)
        self._sum += input_point.value

        if len(self._values) < self._period:
            return self._sum / len(self._values)
        return self._sum / self._period

    def _compute_window_value(self) -> float:
        """Compute SMA from window."""
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        super().reset()
        self._sum = 0.0


class ExponentialMovingAverage(Indicator):
    """
    Exponential Moving Average (EMA).

    Gives more weight to recent prices using an exponential decay.

    Formula: EMA = Price * k + EMA_prev * (1 - k)
    where k = 2 / (n + 1)
    """

    def __init__(self, period: int, smoothing_factor: Optional[float] = None,
                 name: Optional[str] = None):
        name = name or f"EMA({period})"
        super().__init__(name, period)
        self._k = smoothing_factor if smoothing_factor else 2.0 / (period + 1)
        self._initialized = False

    @staticmethod
    def smoothing_factor_default(period: int) -> float:
        """Default EMA smoothing factor: 2 / (n + 1)"""
        return 2.0 / (period + 1)

    @property
    def smoothing_factor(self) -> float:
        return self._k

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        if not self._initialized:
            if self._samples < self._period:
                # Use SMA for warmup
                self._window.append(input_point.value)
                return sum(self._window) / len(self._window)
            else:
                # Initialize with SMA
                self._initialized = True
                return sum(self._window) / self._period

        # EMA calculation
        return input_point.value * self._k + self._current.value * (1 - self._k)

    def reset(self) -> None:
        super().reset()
        self._initialized = False


class WilderMovingAverage(Indicator):
    """
    Wilder's Smoothed Moving Average.

    Similar to EMA but with smoothing factor = 1/n instead of 2/(n+1).
    Used in RSI, ATR, and ADX calculations.

    Formula: WSM = Price * (1/n) + WSM_prev * (1 - 1/n)
    """

    def __init__(self, period: int, name: Optional[str] = None):
        name = name or f"WILDERS({period})"
        super().__init__(name, period)
        self._k = 1.0 / period
        self._initialized = False

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        if not self._initialized:
            if self._samples < self._period:
                self._window.append(input_point.value)
                return sum(self._window) / len(self._window)
            else:
                self._initialized = True
                return sum(self._window) / self._period

        return input_point.value * self._k + self._current.value * (1 - self._k)

    def reset(self) -> None:
        super().reset()
        self._initialized = False


class DoubleExponentialMovingAverage(Indicator):
    """
    Double Exponential Moving Average (DEMA).

    Reduces lag compared to regular EMA by applying EMA twice.

    Formula: DEMA = 2 * EMA(price) - EMA(EMA(price))
    """

    def __init__(self, period: int, name: Optional[str] = None):
        name = name or f"DEMA({period})"
        super().__init__(name, period * 2 - 1)  # Warm-up period
        self._ema1 = ExponentialMovingAverage(period)
        self._ema2 = ExponentialMovingAverage(period)

    @property
    def is_ready(self) -> bool:
        return self._ema2.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._ema1.update(input_point.time, input_point.value)

        if not self._ema1.is_ready:
            return 0.0

        self._ema2.update(input_point.time, self._ema1.value)

        if not self._ema2.is_ready:
            return 0.0

        return 2 * self._ema1.value - self._ema2.value

    def reset(self) -> None:
        super().reset()
        self._ema1.reset()
        self._ema2.reset()


class TripleExponentialMovingAverage(Indicator):
    """
    Triple Exponential Moving Average (TEMA).

    Further reduces lag by applying EMA three times.

    Formula: TEMA = 3*EMA1 - 3*EMA2 + EMA3
    where EMA1 = EMA(price), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
    """

    def __init__(self, period: int, name: Optional[str] = None):
        name = name or f"TEMA({period})"
        super().__init__(name, period * 3 - 2)
        self._ema1 = ExponentialMovingAverage(period)
        self._ema2 = ExponentialMovingAverage(period)
        self._ema3 = ExponentialMovingAverage(period)

    @property
    def is_ready(self) -> bool:
        return self._ema3.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._ema1.update(input_point.time, input_point.value)
        if not self._ema1.is_ready:
            return 0.0

        self._ema2.update(input_point.time, self._ema1.value)
        if not self._ema2.is_ready:
            return 0.0

        self._ema3.update(input_point.time, self._ema2.value)
        if not self._ema3.is_ready:
            return 0.0

        return 3 * self._ema1.value - 3 * self._ema2.value + self._ema3.value

    def reset(self) -> None:
        super().reset()
        self._ema1.reset()
        self._ema2.reset()
        self._ema3.reset()


class KaufmanAdaptiveMovingAverage(Indicator):
    """
    Kaufman Adaptive Moving Average (KAMA).

    Adapts to market volatility by adjusting the smoothing constant
    based on the efficiency ratio.

    Efficiency Ratio = Direction / Volatility
    - High ER (trending): KAMA moves quickly
    - Low ER (choppy): KAMA moves slowly
    """

    def __init__(self, period: int, fast_period: int = 2, slow_period: int = 30,
                 name: Optional[str] = None):
        name = name or f"KAMA({period},{fast_period},{slow_period})"
        super().__init__(name, period)
        self._fast_sc = 2.0 / (fast_period + 1)
        self._slow_sc = 2.0 / (slow_period + 1)
        self._diff_window = deque(maxlen=period)
        self._initialized = False

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._diff_window.append(input_point.value)

        if len(self._diff_window) < self._period:
            return input_point.value

        if not self._initialized:
            self._initialized = True
            return input_point.value

        # Calculate efficiency ratio
        direction = abs(input_point.value - self._diff_window[0])
        volatility = sum(
            abs(self._diff_window[i] - self._diff_window[i - 1])
            for i in range(1, len(self._diff_window))
        )

        if volatility == 0:
            er = 0.0
        else:
            er = direction / volatility

        # Calculate smoothing constant
        sc = (er * (self._fast_sc - self._slow_sc) + self._slow_sc) ** 2

        # KAMA calculation
        return self._current.value + sc * (input_point.value - self._current.value)

    def reset(self) -> None:
        super().reset()
        self._diff_window.clear()
        self._initialized = False


class HullMovingAverage(Indicator):
    """
    Hull Moving Average (HMA).

    Reduces lag while maintaining smoothness using weighted moving averages.

    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """

    def __init__(self, period: int, name: Optional[str] = None):
        name = name or f"HMA({period})"
        sqrt_period = int(period ** 0.5)
        super().__init__(name, period + sqrt_period - 1)

        self._half_period = period // 2
        self._sqrt_period = sqrt_period

        self._wma_half = WeightedMovingAverage(self._half_period)
        self._wma_full = WeightedMovingAverage(period)
        self._wma_sqrt = WeightedMovingAverage(sqrt_period)

    @property
    def is_ready(self) -> bool:
        return self._wma_sqrt.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._wma_half.update(input_point.time, input_point.value)
        self._wma_full.update(input_point.time, input_point.value)

        if not (self._wma_half.is_ready and self._wma_full.is_ready):
            return 0.0

        raw_hma = 2 * self._wma_half.value - self._wma_full.value
        self._wma_sqrt.update(input_point.time, raw_hma)

        if not self._wma_sqrt.is_ready:
            return 0.0

        return self._wma_sqrt.value

    def reset(self) -> None:
        super().reset()
        self._wma_half.reset()
        self._wma_full.reset()
        self._wma_sqrt.reset()


class WeightedMovingAverage(WindowIndicator):
    """
    Weighted Moving Average (WMA).

    Gives linearly increasing weights to more recent data.

    Formula: WMA = (n*Pn + (n-1)*Pn-1 + ... + 1*P1) / (n + n-1 + ... + 1)
    """

    def __init__(self, period: int, name: Optional[str] = None):
        name = name or f"WMA({period})"
        super().__init__(name, period)
        # Pre-calculate denominator: 1 + 2 + ... + n = n*(n+1)/2
        self._weight_sum = period * (period + 1) / 2

    def _compute_window_value(self) -> float:
        if len(self._values) < self._period:
            return sum(self._values) / len(self._values)

        weighted_sum = sum(
            (i + 1) * v for i, v in enumerate(self._values)
        )
        return weighted_sum / self._weight_sum


class TriangularMovingAverage(WindowIndicator):
    """
    Triangular Moving Average (TMA).

    Double-smoothed moving average using SMA of SMA.

    Formula: TMA = SMA(SMA(price, n), n)
    """

    def __init__(self, period: int, name: Optional[str] = None):
        name = name or f"TMA({period})"
        super().__init__(name, period)
        self._sma1 = SimpleMovingAverage(period)
        self._sma2 = SimpleMovingAverage(period)

    @property
    def is_ready(self) -> bool:
        return self._sma2.is_ready

    def _compute_window_value(self) -> float:
        return self._sma2.value

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._sma1.update(input_point.time, input_point.value)
        if not self._sma1.is_ready:
            return 0.0

        self._sma2.update(input_point.time, self._sma1.value)
        return self._sma2.value if self._sma2.is_ready else 0.0

    def reset(self) -> None:
        super().reset()
        self._sma1.reset()
        self._sma2.reset()
