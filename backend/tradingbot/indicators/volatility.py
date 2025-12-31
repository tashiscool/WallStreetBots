"""
Volatility Indicators - Ported from QuantConnect LEAN.

Includes: Bollinger Bands, ATR, Standard Deviation, Keltner, Donchian

Original: https://github.com/QuantConnect/Lean/blob/master/Indicators/
License: Apache 2.0
"""

from collections import deque
from datetime import datetime
from math import sqrt
from typing import Optional

from .base import Indicator, IndicatorDataPoint, TradeBar
from .moving_averages import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WilderMovingAverage,
)


class StandardDeviation(Indicator):
    """
    Standard Deviation Indicator.

    Measures the dispersion of prices from their mean.
    """

    def __init__(self, period: int = 20, name: Optional[str] = None):
        name = name or f"STD({period})"
        super().__init__(name, period)
        self._values = deque(maxlen=period)
        self._sma = SimpleMovingAverage(period)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._values.append(input_point.value)
        self._sma.update(input_point.time, input_point.value)

        if len(self._values) < self._period:
            return 0.0

        mean = self._sma.value
        variance = sum((x - mean) ** 2 for x in self._values) / self._period
        return sqrt(variance)

    def reset(self) -> None:
        super().reset()
        self._values.clear()
        self._sma.reset()


class BollingerBands(Indicator):
    """
    Bollinger Bands - Developed by John Bollinger.

    Consists of:
    - Middle Band: SMA of price
    - Upper Band: Middle Band + (k * Standard Deviation)
    - Lower Band: Middle Band - (k * Standard Deviation)

    Typical settings: 20 period, 2 standard deviations
    """

    def __init__(self, period: int = 20, k: float = 2.0,
                 name: Optional[str] = None):
        name = name or f"BB({period},{k})"
        super().__init__(name, period)
        self._k = k
        self._sma = SimpleMovingAverage(period)
        self._std = StandardDeviation(period)

        self._upper = 0.0
        self._middle = 0.0
        self._lower = 0.0
        self._bandwidth = 0.0
        self._percent_b = 0.0

    @property
    def upper_band(self) -> float:
        """Upper Bollinger Band."""
        return self._upper

    @property
    def middle_band(self) -> float:
        """Middle Bollinger Band (SMA)."""
        return self._middle

    @property
    def lower_band(self) -> float:
        """Lower Bollinger Band."""
        return self._lower

    @property
    def bandwidth(self) -> float:
        """Bandwidth = (Upper - Lower) / Middle"""
        return self._bandwidth

    @property
    def percent_b(self) -> float:
        """%B = (Price - Lower) / (Upper - Lower)"""
        return self._percent_b

    @property
    def is_ready(self) -> bool:
        return self._sma.is_ready and self._std.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._sma.update(input_point.time, input_point.value)
        self._std.update(input_point.time, input_point.value)

        if not self.is_ready:
            return 0.0

        self._middle = self._sma.value
        band_width = self._k * self._std.value
        self._upper = self._middle + band_width
        self._lower = self._middle - band_width

        if self._middle != 0:
            self._bandwidth = (self._upper - self._lower) / self._middle

        band_range = self._upper - self._lower
        if band_range != 0:
            self._percent_b = (input_point.value - self._lower) / band_range

        return self._middle

    def reset(self) -> None:
        super().reset()
        self._sma.reset()
        self._std.reset()
        self._upper = 0.0
        self._middle = 0.0
        self._lower = 0.0
        self._bandwidth = 0.0
        self._percent_b = 0.0


class AverageTrueRange(Indicator):
    """
    Average True Range (ATR) - Developed by J. Welles Wilder.

    Measures market volatility by calculating the average of true ranges.

    True Range = Max(High - Low, |High - Previous Close|, |Low - Previous Close|)
    """

    def __init__(self, period: int = 14, name: Optional[str] = None):
        name = name or f"ATR({period})"
        super().__init__(name, period + 1)
        self._atr = WilderMovingAverage(period, f"{name}Avg")
        self._previous_close: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return self._atr.is_ready

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1

        if self._previous_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._previous_close),
                abs(bar.low - self._previous_close)
            )

        self._previous_close = bar.close
        self._atr.update(bar.time, tr)
        self._current = IndicatorDataPoint(bar.time, self._atr.value)

        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._atr.reset()
        self._previous_close = None


class KeltnerChannels(Indicator):
    """
    Keltner Channels - Developed by Chester Keltner.

    Similar to Bollinger Bands but uses ATR instead of standard deviation.

    Middle = EMA of Close
    Upper = Middle + (multiplier * ATR)
    Lower = Middle - (multiplier * ATR)
    """

    def __init__(self, ema_period: int = 20, atr_period: int = 10,
                 multiplier: float = 2.0, name: Optional[str] = None):
        name = name or f"KC({ema_period},{atr_period},{multiplier})"
        super().__init__(name, max(ema_period, atr_period + 1))
        self._multiplier = multiplier
        self._ema = ExponentialMovingAverage(ema_period)
        self._atr = AverageTrueRange(atr_period)

        self._upper = 0.0
        self._middle = 0.0
        self._lower = 0.0

    @property
    def upper_channel(self) -> float:
        return self._upper

    @property
    def middle_channel(self) -> float:
        return self._middle

    @property
    def lower_channel(self) -> float:
        return self._lower

    @property
    def is_ready(self) -> bool:
        return self._ema.is_ready and self._atr.is_ready

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1
        self._ema.update(bar.time, bar.close)
        self._atr.update_bar(bar)

        if not self.is_ready:
            self._current = IndicatorDataPoint(bar.time, 0.0)
            return False

        self._middle = self._ema.value
        channel_width = self._multiplier * self._atr.value
        self._upper = self._middle + channel_width
        self._lower = self._middle - channel_width

        self._current = IndicatorDataPoint(bar.time, self._middle)
        return True

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._ema.reset()
        self._atr.reset()
        self._upper = 0.0
        self._middle = 0.0
        self._lower = 0.0


class DonchianChannel(Indicator):
    """
    Donchian Channel - Developed by Richard Donchian.

    Uses highest high and lowest low over a period.

    Upper = Highest High over n periods
    Lower = Lowest Low over n periods
    Middle = (Upper + Lower) / 2
    """

    def __init__(self, period: int = 20, name: Optional[str] = None):
        name = name or f"DC({period})"
        super().__init__(name, period)
        self._highs = deque(maxlen=period)
        self._lows = deque(maxlen=period)

        self._upper = 0.0
        self._middle = 0.0
        self._lower = 0.0

    @property
    def upper_channel(self) -> float:
        return self._upper

    @property
    def middle_channel(self) -> float:
        return self._middle

    @property
    def lower_channel(self) -> float:
        return self._lower

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        if len(self._highs) < self._period:
            self._current = IndicatorDataPoint(bar.time, 0.0)
            return False

        self._upper = max(self._highs)
        self._lower = min(self._lows)
        self._middle = (self._upper + self._lower) / 2

        self._current = IndicatorDataPoint(bar.time, self._middle)
        return True

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._highs.clear()
        self._lows.clear()
        self._upper = 0.0
        self._middle = 0.0
        self._lower = 0.0


class ChoppinessIndex(Indicator):
    """
    Choppiness Index.

    Determines whether the market is trending or ranging.

    Values 0-38.2: Trending market
    Values 61.8-100: Choppy/ranging market
    """

    def __init__(self, period: int = 14, name: Optional[str] = None):
        name = name or f"CHOP({period})"
        super().__init__(name, period + 1)
        self._atr_values = deque(maxlen=period)
        self._highs = deque(maxlen=period)
        self._lows = deque(maxlen=period)
        self._previous_close: Optional[float] = None
        from math import log10
        self._log10 = log10

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        # Calculate True Range
        if self._previous_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._previous_close),
                abs(bar.low - self._previous_close)
            )

        self._atr_values.append(tr)
        self._previous_close = bar.close

        if len(self._atr_values) < self._period:
            self._current = IndicatorDataPoint(bar.time, 0.0)
            return False

        atr_sum = sum(self._atr_values)
        highest_high = max(self._highs)
        lowest_low = min(self._lows)
        hl_range = highest_high - lowest_low

        if hl_range == 0:
            chop = 50.0
        else:
            chop = 100 * self._log10(atr_sum / hl_range) / self._log10(self._period)

        self._current = IndicatorDataPoint(bar.time, chop)
        return True

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._atr_values.clear()
        self._highs.clear()
        self._lows.clear()
        self._previous_close = None
