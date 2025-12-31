"""
Momentum Indicators - Ported from QuantConnect LEAN.

Includes: MACD, ROC, Momentum, Awesome Oscillator, PPO

Original: https://github.com/QuantConnect/Lean/blob/master/Indicators/
License: Apache 2.0
"""

from collections import deque
from datetime import datetime
from typing import Optional

from .base import Indicator, IndicatorDataPoint, TradeBar
from .moving_averages import ExponentialMovingAverage, SimpleMovingAverage


class MACD(Indicator):
    """
    Moving Average Convergence Divergence (MACD).

    Shows relationship between two moving averages of price.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line)
    Histogram = MACD Line - Signal Line

    Typical settings: 12, 26, 9
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, name: Optional[str] = None):
        name = name or f"MACD({fast_period},{slow_period},{signal_period})"
        super().__init__(name, slow_period + signal_period - 1)
        self._fast = ExponentialMovingAverage(fast_period, name=f"{name}Fast")
        self._slow = ExponentialMovingAverage(slow_period, name=f"{name}Slow")
        self._signal = ExponentialMovingAverage(signal_period, name=f"{name}Signal")

        self._macd_line = 0.0
        self._signal_line = 0.0
        self._histogram = 0.0

    @property
    def fast(self) -> ExponentialMovingAverage:
        return self._fast

    @property
    def slow(self) -> ExponentialMovingAverage:
        return self._slow

    @property
    def signal(self) -> ExponentialMovingAverage:
        return self._signal

    @property
    def macd_line(self) -> float:
        """MACD Line = Fast EMA - Slow EMA"""
        return self._macd_line

    @property
    def signal_line(self) -> float:
        """Signal Line = EMA of MACD Line"""
        return self._signal_line

    @property
    def histogram(self) -> float:
        """Histogram = MACD Line - Signal Line"""
        return self._histogram

    @property
    def is_ready(self) -> bool:
        return self._signal.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._fast.update(input_point.time, input_point.value)
        self._slow.update(input_point.time, input_point.value)

        if not self._slow.is_ready:
            return 0.0

        self._macd_line = self._fast.value - self._slow.value
        self._signal.update(input_point.time, self._macd_line)
        self._signal_line = self._signal.value
        self._histogram = self._macd_line - self._signal_line

        return self._macd_line

    def reset(self) -> None:
        super().reset()
        self._fast.reset()
        self._slow.reset()
        self._signal.reset()
        self._macd_line = 0.0
        self._signal_line = 0.0
        self._histogram = 0.0


class RateOfChange(Indicator):
    """
    Rate of Change (ROC).

    Measures percentage change in price over a specified period.

    ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    """

    def __init__(self, period: int = 10, name: Optional[str] = None):
        name = name or f"ROC({period})"
        super().__init__(name, period + 1)
        self._prices = deque(maxlen=period + 1)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._prices.append(input_point.value)

        if len(self._prices) <= self._period:
            return 0.0

        old_price = self._prices[0]
        if old_price == 0:
            return 0.0

        return ((input_point.value - old_price) / old_price) * 100

    def reset(self) -> None:
        super().reset()
        self._prices.clear()


class Momentum(Indicator):
    """
    Momentum Indicator.

    Measures absolute price change over a period.

    Momentum = Current Price - Price n periods ago
    """

    def __init__(self, period: int = 10, name: Optional[str] = None):
        name = name or f"MOM({period})"
        super().__init__(name, period + 1)
        self._prices = deque(maxlen=period + 1)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._prices.append(input_point.value)

        if len(self._prices) <= self._period:
            return 0.0

        return input_point.value - self._prices[0]

    def reset(self) -> None:
        super().reset()
        self._prices.clear()


class AwesomeOscillator(Indicator):
    """
    Awesome Oscillator (AO) - Developed by Bill Williams.

    Measures market momentum using difference of two SMAs
    of the median price.

    AO = SMA(5, Median Price) - SMA(34, Median Price)
    Median Price = (High + Low) / 2
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 34,
                 name: Optional[str] = None):
        name = name or f"AO({fast_period},{slow_period})"
        super().__init__(name, slow_period)
        self._fast_sma = SimpleMovingAverage(fast_period)
        self._slow_sma = SimpleMovingAverage(slow_period)

    @property
    def is_ready(self) -> bool:
        return self._slow_sma.is_ready

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        median_price = bar.median_price
        return self.update(bar.time, median_price)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._fast_sma.update(input_point.time, input_point.value)
        self._slow_sma.update(input_point.time, input_point.value)

        if not self._slow_sma.is_ready:
            return 0.0

        return self._fast_sma.value - self._slow_sma.value

    def reset(self) -> None:
        super().reset()
        self._fast_sma.reset()
        self._slow_sma.reset()


class PercentagePriceOscillator(Indicator):
    """
    Percentage Price Oscillator (PPO).

    Similar to MACD but expressed as a percentage.
    Allows comparison between securities with different prices.

    PPO = ((EMA_fast - EMA_slow) / EMA_slow) * 100
    Signal = EMA of PPO
    Histogram = PPO - Signal
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, name: Optional[str] = None):
        name = name or f"PPO({fast_period},{slow_period},{signal_period})"
        super().__init__(name, slow_period + signal_period - 1)
        self._fast = ExponentialMovingAverage(fast_period)
        self._slow = ExponentialMovingAverage(slow_period)
        self._signal = ExponentialMovingAverage(signal_period)

        self._ppo_line = 0.0
        self._signal_line = 0.0
        self._histogram = 0.0

    @property
    def ppo_line(self) -> float:
        return self._ppo_line

    @property
    def signal_line(self) -> float:
        return self._signal_line

    @property
    def histogram(self) -> float:
        return self._histogram

    @property
    def is_ready(self) -> bool:
        return self._signal.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._fast.update(input_point.time, input_point.value)
        self._slow.update(input_point.time, input_point.value)

        if not self._slow.is_ready or self._slow.value == 0:
            return 0.0

        self._ppo_line = ((self._fast.value - self._slow.value) / self._slow.value) * 100
        self._signal.update(input_point.time, self._ppo_line)
        self._signal_line = self._signal.value
        self._histogram = self._ppo_line - self._signal_line

        return self._ppo_line

    def reset(self) -> None:
        super().reset()
        self._fast.reset()
        self._slow.reset()
        self._signal.reset()
        self._ppo_line = 0.0
        self._signal_line = 0.0
        self._histogram = 0.0


class TrueStrengthIndex(Indicator):
    """
    True Strength Index (TSI).

    Double-smoothed momentum indicator.

    TSI = (EMA(EMA(PC, 25), 13) / EMA(EMA(|PC|, 25), 13)) * 100
    PC = Price Change = Close - Previous Close
    """

    def __init__(self, long_period: int = 25, short_period: int = 13,
                 signal_period: int = 7, name: Optional[str] = None):
        name = name or f"TSI({long_period},{short_period},{signal_period})"
        super().__init__(name, long_period + short_period)

        self._pc_ema1 = ExponentialMovingAverage(long_period)
        self._pc_ema2 = ExponentialMovingAverage(short_period)
        self._abs_pc_ema1 = ExponentialMovingAverage(long_period)
        self._abs_pc_ema2 = ExponentialMovingAverage(short_period)
        self._signal = ExponentialMovingAverage(signal_period)

        self._previous_price: Optional[float] = None
        self._tsi_value = 0.0
        self._signal_value = 0.0

    @property
    def tsi(self) -> float:
        return self._tsi_value

    @property
    def signal_line(self) -> float:
        return self._signal_value

    @property
    def is_ready(self) -> bool:
        return self._pc_ema2.is_ready and self._abs_pc_ema2.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        if self._previous_price is None:
            self._previous_price = input_point.value
            return 0.0

        price_change = input_point.value - self._previous_price
        abs_price_change = abs(price_change)
        self._previous_price = input_point.value

        # Double smooth price change
        self._pc_ema1.update(input_point.time, price_change)
        if self._pc_ema1.is_ready:
            self._pc_ema2.update(input_point.time, self._pc_ema1.value)

        # Double smooth absolute price change
        self._abs_pc_ema1.update(input_point.time, abs_price_change)
        if self._abs_pc_ema1.is_ready:
            self._abs_pc_ema2.update(input_point.time, self._abs_pc_ema1.value)

        if not self.is_ready or self._abs_pc_ema2.value == 0:
            return 0.0

        self._tsi_value = (self._pc_ema2.value / self._abs_pc_ema2.value) * 100
        self._signal.update(input_point.time, self._tsi_value)
        self._signal_value = self._signal.value

        return self._tsi_value

    def reset(self) -> None:
        super().reset()
        self._pc_ema1.reset()
        self._pc_ema2.reset()
        self._abs_pc_ema1.reset()
        self._abs_pc_ema2.reset()
        self._signal.reset()
        self._previous_price = None
        self._tsi_value = 0.0
        self._signal_value = 0.0
