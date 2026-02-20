"""
Volume Indicators - Ported from QuantConnect LEAN.

Includes: OBV, A/D, CMF, VWAP, Force Index

Original: https://github.com/QuantConnect/Lean/blob/master/Indicators/
License: Apache 2.0
"""

from collections import deque
from typing import Optional, TYPE_CHECKING

from .base import Indicator, IndicatorDataPoint, TradeBar
from .moving_averages import ExponentialMovingAverage

if TYPE_CHECKING:
    from datetime import datetime


class OnBalanceVolume(Indicator):
    """
    On Balance Volume (OBV) - Developed by Joseph Granville.

    Cumulative indicator that adds volume on up days
    and subtracts volume on down days.

    If Close > Previous Close: OBV = Previous OBV + Volume
    If Close < Previous Close: OBV = Previous OBV - Volume
    If Close = Previous Close: OBV = Previous OBV
    """

    def __init__(self, name: Optional[str] = None):
        name = name or "OBV"
        super().__init__(name, 2)
        self._obv = 0.0
        self._previous_close: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return self._samples >= 2

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data with volume."""
        self._samples += 1

        if self._previous_close is not None:
            if bar.close > self._previous_close:
                self._obv += bar.volume
            elif bar.close < self._previous_close:
                self._obv -= bar.volume
            # If equal, OBV stays the same

        self._previous_close = bar.close
        self._current = IndicatorDataPoint(bar.time, self._obv)

        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._obv

    def reset(self) -> None:
        super().reset()
        self._obv = 0.0
        self._previous_close = None


class AccumulationDistribution(Indicator):
    """
    Accumulation/Distribution Line - Developed by Marc Chaikin.

    Measures the cumulative flow of money into and out of a security.

    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = MFM * Volume
    A/D = Previous A/D + Money Flow Volume
    """

    def __init__(self, name: Optional[str] = None):
        name = name or "AD"
        super().__init__(name, 1)
        self._ad = 0.0

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data with volume."""
        self._samples += 1

        hl_range = bar.high - bar.low
        if hl_range == 0:
            mfm = 0.0
        else:
            mfm = ((bar.close - bar.low) - (bar.high - bar.close)) / hl_range

        mfv = mfm * bar.volume
        self._ad += mfv

        self._current = IndicatorDataPoint(bar.time, self._ad)
        return True

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._ad

    def reset(self) -> None:
        super().reset()
        self._ad = 0.0


class ChaikinMoneyFlow(Indicator):
    """
    Chaikin Money Flow (CMF) - Developed by Marc Chaikin.

    Measures buying and selling pressure over a period.

    CMF = Sum(Money Flow Volume) / Sum(Volume) over n periods

    Values above 0 indicate buying pressure.
    Values below 0 indicate selling pressure.
    """

    def __init__(self, period: int = 20, name: Optional[str] = None):
        name = name or f"CMF({period})"
        super().__init__(name, period)
        self._mfv_window = deque(maxlen=period)
        self._volume_window = deque(maxlen=period)

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data with volume."""
        self._samples += 1

        hl_range = bar.high - bar.low
        if hl_range == 0:
            mfm = 0.0
        else:
            mfm = ((bar.close - bar.low) - (bar.high - bar.close)) / hl_range

        mfv = mfm * bar.volume

        self._mfv_window.append(mfv)
        self._volume_window.append(bar.volume)

        if len(self._mfv_window) < self._period:
            self._current = IndicatorDataPoint(bar.time, 0.0)
            return False

        volume_sum = sum(self._volume_window)
        if volume_sum == 0:
            cmf = 0.0
        else:
            cmf = sum(self._mfv_window) / volume_sum

        self._current = IndicatorDataPoint(bar.time, cmf)
        return True

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._mfv_window.clear()
        self._volume_window.clear()


class VolumeWeightedAveragePrice(Indicator):
    """
    Volume Weighted Average Price (VWAP).

    Average price weighted by volume, typically reset daily.

    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    """

    def __init__(self, name: Optional[str] = None):
        name = name or "VWAP"
        super().__init__(name, 1)
        self._cumulative_tp_volume = 0.0
        self._cumulative_volume = 0.0
        self._current_date: Optional[datetime] = None

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data with volume."""
        self._samples += 1

        # Reset on new day
        bar_date = bar.time.date() if hasattr(bar.time, 'date') else bar.time
        if self._current_date is not None and bar_date != self._current_date:
            self._cumulative_tp_volume = 0.0
            self._cumulative_volume = 0.0
        self._current_date = bar_date

        typical_price = bar.typical_price
        self._cumulative_tp_volume += typical_price * bar.volume
        self._cumulative_volume += bar.volume

        if self._cumulative_volume == 0:
            vwap = 0.0
        else:
            vwap = self._cumulative_tp_volume / self._cumulative_volume

        self._current = IndicatorDataPoint(bar.time, vwap)
        return True

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._cumulative_tp_volume = 0.0
        self._cumulative_volume = 0.0
        self._current_date = None


class ForceIndex(Indicator):
    """
    Force Index - Developed by Alexander Elder.

    Measures the power behind price movements using price and volume.

    Force Index = (Close - Previous Close) * Volume
    Smoothed Force Index = EMA(Force Index)
    """

    def __init__(self, period: int = 13, name: Optional[str] = None):
        name = name or f"FI({period})"
        super().__init__(name, period + 1)
        self._ema = ExponentialMovingAverage(period)
        self._previous_close: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return self._ema.is_ready

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data with volume."""
        self._samples += 1

        if self._previous_close is None:
            self._previous_close = bar.close
            return False

        force = (bar.close - self._previous_close) * bar.volume
        self._previous_close = bar.close

        self._ema.update(bar.time, force)
        self._current = IndicatorDataPoint(bar.time, self._ema.value)

        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._ema.reset()
        self._previous_close = None


class NegativeVolumeIndex(Indicator):
    """
    Negative Volume Index (NVI).

    Focuses on days when volume decreases from the previous day.
    Starts at 1000 and changes only on days with lower volume.
    """

    def __init__(self, name: Optional[str] = None):
        name = name or "NVI"
        super().__init__(name, 2)
        self._nvi = 1000.0
        self._previous_close: Optional[float] = None
        self._previous_volume: Optional[float] = None

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data with volume."""
        self._samples += 1

        if self._previous_close is not None and self._previous_volume is not None:
            if bar.volume < self._previous_volume:
                pct_change = (bar.close - self._previous_close) / self._previous_close
                self._nvi = self._nvi * (1 + pct_change)

        self._previous_close = bar.close
        self._previous_volume = bar.volume
        self._current = IndicatorDataPoint(bar.time, self._nvi)

        return self._samples >= 2

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._nvi

    def reset(self) -> None:
        super().reset()
        self._nvi = 1000.0
        self._previous_close = None
        self._previous_volume = None


class PositiveVolumeIndex(Indicator):
    """
    Positive Volume Index (PVI).

    Focuses on days when volume increases from the previous day.
    Starts at 1000 and changes only on days with higher volume.
    """

    def __init__(self, name: Optional[str] = None):
        name = name or "PVI"
        super().__init__(name, 2)
        self._pvi = 1000.0
        self._previous_close: Optional[float] = None
        self._previous_volume: Optional[float] = None

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data with volume."""
        self._samples += 1

        if self._previous_close is not None and self._previous_volume is not None:
            if bar.volume > self._previous_volume:
                pct_change = (bar.close - self._previous_close) / self._previous_close
                self._pvi = self._pvi * (1 + pct_change)

        self._previous_close = bar.close
        self._previous_volume = bar.volume
        self._current = IndicatorDataPoint(bar.time, self._pvi)

        return self._samples >= 2

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._pvi

    def reset(self) -> None:
        super().reset()
        self._pvi = 1000.0
        self._previous_close = None
        self._previous_volume = None
