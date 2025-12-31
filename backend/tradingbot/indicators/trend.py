"""
Trend Indicators - Ported from QuantConnect LEAN.

Includes: ADX, Parabolic SAR, Aroon, Ichimoku, SuperTrend

Original: https://github.com/QuantConnect/Lean/blob/master/Indicators/
License: Apache 2.0
"""

from collections import deque
from datetime import datetime
from typing import Optional

from .base import Indicator, IndicatorDataPoint, TradeBar
from .moving_averages import WilderMovingAverage, SimpleMovingAverage


class AverageDirectionalIndex(Indicator):
    """
    Average Directional Index (ADX) - Developed by J. Welles Wilder.

    Measures trend strength regardless of direction.

    +DI = (Smoothed +DM / ATR) * 100
    -DI = (Smoothed -DM / ATR) * 100
    DX = |(+DI - -DI)| / (+DI + -DI) * 100
    ADX = Smoothed DX

    Values above 25 indicate strong trend.
    Values below 20 indicate weak trend or ranging.
    """

    def __init__(self, period: int = 14, name: Optional[str] = None):
        name = name or f"ADX({period})"
        super().__init__(name, period * 2)
        self._period = period

        self._positive_dm = WilderMovingAverage(period)
        self._negative_dm = WilderMovingAverage(period)
        self._atr = WilderMovingAverage(period)
        self._adx = WilderMovingAverage(period)

        self._previous_bar: Optional[TradeBar] = None
        self._positive_di = 0.0
        self._negative_di = 0.0
        self._dx = 0.0

    @property
    def positive_di(self) -> float:
        """+DI: Positive Directional Indicator"""
        return self._positive_di

    @property
    def negative_di(self) -> float:
        """-DI: Negative Directional Indicator"""
        return self._negative_di

    @property
    def dx(self) -> float:
        """DX: Directional Movement Index"""
        return self._dx

    @property
    def is_ready(self) -> bool:
        return self._adx.is_ready

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1

        if self._previous_bar is None:
            self._previous_bar = bar
            return False

        # Calculate True Range
        tr = max(
            bar.high - bar.low,
            abs(bar.high - self._previous_bar.close),
            abs(bar.low - self._previous_bar.close)
        )

        # Calculate Directional Movement
        up_move = bar.high - self._previous_bar.high
        down_move = self._previous_bar.low - bar.low

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0

        self._previous_bar = bar

        # Update smoothed values
        self._positive_dm.update(bar.time, plus_dm)
        self._negative_dm.update(bar.time, minus_dm)
        self._atr.update(bar.time, tr)

        if not self._atr.is_ready or self._atr.value == 0:
            return False

        # Calculate DI values
        self._positive_di = (self._positive_dm.value / self._atr.value) * 100
        self._negative_di = (self._negative_dm.value / self._atr.value) * 100

        # Calculate DX
        di_sum = self._positive_di + self._negative_di
        if di_sum == 0:
            self._dx = 0.0
        else:
            self._dx = (abs(self._positive_di - self._negative_di) / di_sum) * 100

        # Update ADX
        self._adx.update(bar.time, self._dx)
        self._current = IndicatorDataPoint(bar.time, self._adx.value)

        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._positive_dm.reset()
        self._negative_dm.reset()
        self._atr.reset()
        self._adx.reset()
        self._previous_bar = None
        self._positive_di = 0.0
        self._negative_di = 0.0
        self._dx = 0.0


class ParabolicSAR(Indicator):
    """
    Parabolic SAR (Stop and Reverse) - Developed by J. Welles Wilder.

    Trailing stop indicator that follows price action.

    During uptrend: SAR = Previous SAR + AF * (EP - Previous SAR)
    During downtrend: SAR = Previous SAR - AF * (Previous SAR - EP)

    Where:
    - AF (Acceleration Factor) starts at 0.02, increases by 0.02 on new highs/lows
    - EP (Extreme Point) is highest high in uptrend, lowest low in downtrend
    """

    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02,
                 af_max: float = 0.2, name: Optional[str] = None):
        name = name or f"PSAR({af_start},{af_increment},{af_max})"
        super().__init__(name, 2)
        self._af_start = af_start
        self._af_increment = af_increment
        self._af_max = af_max

        self._af = af_start
        self._ep = 0.0
        self._sar = 0.0
        self._is_rising = True
        self._initialized = False
        self._previous_bar: Optional[TradeBar] = None

    @property
    def is_rising(self) -> bool:
        """True if SAR indicates uptrend."""
        return self._is_rising

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1

        if not self._initialized:
            if self._previous_bar is None:
                self._previous_bar = bar
                return False

            # Initialize SAR
            self._is_rising = bar.close > self._previous_bar.close
            if self._is_rising:
                self._sar = self._previous_bar.low
                self._ep = bar.high
            else:
                self._sar = self._previous_bar.high
                self._ep = bar.low

            self._initialized = True
            self._previous_bar = bar
            self._current = IndicatorDataPoint(bar.time, self._sar)
            return True

        # Update SAR
        prev_sar = self._sar

        if self._is_rising:
            self._sar = prev_sar + self._af * (self._ep - prev_sar)
            # SAR cannot be above prior two lows
            self._sar = min(self._sar, self._previous_bar.low, bar.low)

            if bar.low < self._sar:
                # Trend reversal
                self._is_rising = False
                self._sar = self._ep
                self._ep = bar.low
                self._af = self._af_start
            else:
                if bar.high > self._ep:
                    self._ep = bar.high
                    self._af = min(self._af + self._af_increment, self._af_max)
        else:
            self._sar = prev_sar - self._af * (prev_sar - self._ep)
            # SAR cannot be below prior two highs
            self._sar = max(self._sar, self._previous_bar.high, bar.high)

            if bar.high > self._sar:
                # Trend reversal
                self._is_rising = True
                self._sar = self._ep
                self._ep = bar.high
                self._af = self._af_start
            else:
                if bar.low < self._ep:
                    self._ep = bar.low
                    self._af = min(self._af + self._af_increment, self._af_max)

        self._previous_bar = bar
        self._current = IndicatorDataPoint(bar.time, self._sar)
        return True

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._sar

    def reset(self) -> None:
        super().reset()
        self._af = self._af_start
        self._ep = 0.0
        self._sar = 0.0
        self._is_rising = True
        self._initialized = False
        self._previous_bar = None


class Aroon(Indicator):
    """
    Aroon Indicator - Developed by Tushar Chande.

    Measures time since highest high and lowest low.

    Aroon Up = ((Period - Days Since Highest High) / Period) * 100
    Aroon Down = ((Period - Days Since Lowest Low) / Period) * 100
    Aroon Oscillator = Aroon Up - Aroon Down
    """

    def __init__(self, period: int = 25, name: Optional[str] = None):
        name = name or f"AROON({period})"
        super().__init__(name, period)
        self._highs = deque(maxlen=period)
        self._lows = deque(maxlen=period)

        self._aroon_up = 0.0
        self._aroon_down = 0.0
        self._oscillator = 0.0

    @property
    def aroon_up(self) -> float:
        return self._aroon_up

    @property
    def aroon_down(self) -> float:
        return self._aroon_down

    @property
    def oscillator(self) -> float:
        return self._oscillator

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        if len(self._highs) < self._period:
            self._current = IndicatorDataPoint(bar.time, 0.0)
            return False

        # Find periods since highest high and lowest low
        high_list = list(self._highs)
        low_list = list(self._lows)

        max_high = max(high_list)
        min_low = min(low_list)

        periods_since_high = self._period - 1 - high_list.index(max_high)
        periods_since_low = self._period - 1 - low_list.index(min_low)

        self._aroon_up = ((self._period - periods_since_high) / self._period) * 100
        self._aroon_down = ((self._period - periods_since_low) / self._period) * 100
        self._oscillator = self._aroon_up - self._aroon_down

        self._current = IndicatorDataPoint(bar.time, self._oscillator)
        return True

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._oscillator

    def reset(self) -> None:
        super().reset()
        self._highs.clear()
        self._lows.clear()
        self._aroon_up = 0.0
        self._aroon_down = 0.0
        self._oscillator = 0.0


class IchimokuCloud(Indicator):
    """
    Ichimoku Kinko Hyo (Ichimoku Cloud).

    Japanese trend indicator with multiple components:
    - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    - Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    - Chikou Span (Lagging Span): Close plotted 26 periods behind
    """

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26,
                 senkou_b_period: int = 52, name: Optional[str] = None):
        name = name or f"ICHIMOKU({tenkan_period},{kijun_period},{senkou_b_period})"
        super().__init__(name, senkou_b_period)
        self._tenkan_period = tenkan_period
        self._kijun_period = kijun_period
        self._senkou_b_period = senkou_b_period

        self._highs = deque(maxlen=senkou_b_period)
        self._lows = deque(maxlen=senkou_b_period)

        self._tenkan = 0.0
        self._kijun = 0.0
        self._senkou_a = 0.0
        self._senkou_b = 0.0

    @property
    def tenkan_sen(self) -> float:
        """Conversion Line"""
        return self._tenkan

    @property
    def kijun_sen(self) -> float:
        """Base Line"""
        return self._kijun

    @property
    def senkou_span_a(self) -> float:
        """Leading Span A"""
        return self._senkou_a

    @property
    def senkou_span_b(self) -> float:
        """Leading Span B"""
        return self._senkou_b

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        # Calculate Tenkan-sen (uses tenkan_period)
        if len(self._highs) >= self._tenkan_period:
            recent_highs = list(self._highs)[-self._tenkan_period:]
            recent_lows = list(self._lows)[-self._tenkan_period:]
            self._tenkan = (max(recent_highs) + min(recent_lows)) / 2

        # Calculate Kijun-sen (uses kijun_period)
        if len(self._highs) >= self._kijun_period:
            recent_highs = list(self._highs)[-self._kijun_period:]
            recent_lows = list(self._lows)[-self._kijun_period:]
            self._kijun = (max(recent_highs) + min(recent_lows)) / 2

        # Calculate Senkou Span A
        self._senkou_a = (self._tenkan + self._kijun) / 2

        # Calculate Senkou Span B (uses senkou_b_period)
        if len(self._highs) >= self._senkou_b_period:
            self._senkou_b = (max(self._highs) + min(self._lows)) / 2

        self._current = IndicatorDataPoint(bar.time, self._tenkan)
        return len(self._highs) >= self._senkou_b_period

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._tenkan

    def reset(self) -> None:
        super().reset()
        self._highs.clear()
        self._lows.clear()
        self._tenkan = 0.0
        self._kijun = 0.0
        self._senkou_a = 0.0
        self._senkou_b = 0.0


class SuperTrend(Indicator):
    """
    SuperTrend Indicator.

    Trend-following indicator based on ATR.

    Upper Band = (High + Low) / 2 + (Multiplier * ATR)
    Lower Band = (High + Low) / 2 - (Multiplier * ATR)

    SuperTrend = Lower Band during uptrend, Upper Band during downtrend
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0,
                 name: Optional[str] = None):
        name = name or f"ST({period},{multiplier})"
        super().__init__(name, period + 1)
        self._multiplier = multiplier
        self._atr = WilderMovingAverage(period)

        self._upper_band = 0.0
        self._lower_band = 0.0
        self._super_trend = 0.0
        self._is_uptrend = True
        self._previous_close: Optional[float] = None
        self._previous_super_trend = 0.0

    @property
    def upper_band(self) -> float:
        return self._upper_band

    @property
    def lower_band(self) -> float:
        return self._lower_band

    @property
    def is_uptrend(self) -> bool:
        return self._is_uptrend

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1

        # Calculate True Range
        if self._previous_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._previous_close),
                abs(bar.low - self._previous_close)
            )

        self._atr.update(bar.time, tr)

        if not self._atr.is_ready:
            self._previous_close = bar.close
            return False

        hl2 = (bar.high + bar.low) / 2
        atr_mult = self._multiplier * self._atr.value

        basic_upper = hl2 + atr_mult
        basic_lower = hl2 - atr_mult

        # Final bands with smoothing
        if self._previous_super_trend == 0:
            self._upper_band = basic_upper
            self._lower_band = basic_lower
        else:
            if self._previous_close is not None:
                # Upper band
                if basic_upper < self._upper_band or self._previous_close > self._upper_band:
                    self._upper_band = basic_upper
                # Lower band
                if basic_lower > self._lower_band or self._previous_close < self._lower_band:
                    self._lower_band = basic_lower

        # Determine trend and SuperTrend value
        if self._previous_super_trend == 0:
            self._super_trend = self._lower_band
            self._is_uptrend = True
        else:
            if self._previous_super_trend == self._previous_super_trend:  # Was lower band
                if bar.close <= self._lower_band:
                    self._super_trend = self._upper_band
                    self._is_uptrend = False
                else:
                    self._super_trend = self._lower_band
                    self._is_uptrend = True

        self._previous_close = bar.close
        self._previous_super_trend = self._super_trend
        self._current = IndicatorDataPoint(bar.time, self._super_trend)

        return True

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._super_trend

    def reset(self) -> None:
        super().reset()
        self._atr.reset()
        self._upper_band = 0.0
        self._lower_band = 0.0
        self._super_trend = 0.0
        self._is_uptrend = True
        self._previous_close = None
        self._previous_super_trend = 0.0
