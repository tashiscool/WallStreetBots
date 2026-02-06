"""
Advanced Oscillator Indicators

ElderRay, DeMarker, ConnorsRSI, FisherTransform, ChandeMomentum, KlingerVolume.
Inspired by Jesse and Backtrader indicator libraries.
"""

import math
from collections import deque
from datetime import datetime

import numpy as np

from .base import Indicator, IndicatorDataPoint, TradeBar, WindowIndicator


class ElderRayBull(Indicator):
    """Elder Ray Bull Power = High - EMA(Close, period)."""

    def __init__(self, period: int = 13):
        super().__init__("ElderRayBull", period)
        self._ema = 0.0
        self._multiplier = 2.0 / (period + 1)
        self._last_high = 0.0

    def update_bar(self, bar: TradeBar) -> bool:
        self._last_high = bar.high
        return self.update(bar.time, bar.close)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        if self._samples == 1:
            self._ema = input_point.value
        else:
            self._ema += self._multiplier * (input_point.value - self._ema)
        if not self.is_ready:
            return 0.0
        return self._last_high - self._ema


class ElderRayBear(Indicator):
    """Elder Ray Bear Power = Low - EMA(Close, period)."""

    def __init__(self, period: int = 13):
        super().__init__("ElderRayBear", period)
        self._ema = 0.0
        self._multiplier = 2.0 / (period + 1)
        self._last_low = 0.0

    def update_bar(self, bar: TradeBar) -> bool:
        self._last_low = bar.low
        return self.update(bar.time, bar.close)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        if self._samples == 1:
            self._ema = input_point.value
        else:
            self._ema += self._multiplier * (input_point.value - self._ema)
        if not self.is_ready:
            return 0.0
        return self._last_low - self._ema


class DeMarker(WindowIndicator):
    """DeMarker oscillator (0-1 range, like RSI but uses highs/lows)."""

    def __init__(self, period: int = 14):
        super().__init__("DeMarker", period)
        self._highs = deque(maxlen=period + 1)
        self._lows = deque(maxlen=period + 1)

    def update_bar(self, bar: TradeBar) -> bool:
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        return self.update(bar.time, bar.close)

    def _compute_window_value(self) -> float:
        if len(self._highs) < 2:
            return 0.5

        de_max_sum = 0.0
        de_min_sum = 0.0
        for i in range(1, len(self._highs)):
            de_max = max(self._highs[i] - self._highs[i - 1], 0)
            de_min = max(self._lows[i - 1] - self._lows[i], 0)
            de_max_sum += de_max
            de_min_sum += de_min

        denom = de_max_sum + de_min_sum
        if denom == 0:
            return 0.5
        return de_max_sum / denom


class ConnorsRSI(Indicator):
    """ConnorsRSI = (RSI + Streak RSI + PercentRank) / 3."""

    def __init__(
        self,
        rsi_period: int = 3,
        streak_period: int = 2,
        rank_period: int = 100,
    ):
        super().__init__("ConnorsRSI", max(rsi_period, streak_period, rank_period))
        self._rsi_period = rsi_period
        self._streak_period = streak_period
        self._rank_period = rank_period
        self._closes = deque(maxlen=rank_period + 1)
        self._gains = deque(maxlen=rsi_period)
        self._losses = deque(maxlen=rsi_period)
        self._streak = 0
        self._streak_values = deque(maxlen=streak_period)
        self._streak_gains = deque(maxlen=streak_period)
        self._streak_losses = deque(maxlen=streak_period)
        self._pct_changes = deque(maxlen=rank_period)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._closes.append(input_point.value)
        if len(self._closes) < 2:
            return 50.0

        # Price change
        change = self._closes[-1] - self._closes[-2]

        # Streak
        if change > 0:
            self._streak = max(1, self._streak + 1) if self._streak > 0 else 1
        elif change < 0:
            self._streak = min(-1, self._streak - 1) if self._streak < 0 else -1
        else:
            self._streak = 0

        # RSI of close
        gain = max(change, 0)
        loss = max(-change, 0)
        self._gains.append(gain)
        self._losses.append(loss)
        rsi = self._calc_rsi(self._gains, self._losses)

        # RSI of streak
        self._streak_values.append(float(self._streak))
        if len(self._streak_values) >= 2:
            s_change = self._streak_values[-1] - self._streak_values[-2]
            self._streak_gains.append(max(s_change, 0))
            self._streak_losses.append(max(-s_change, 0))
        streak_rsi = self._calc_rsi(self._streak_gains, self._streak_losses)

        # Percent rank
        pct_change = change / self._closes[-2] if self._closes[-2] != 0 else 0
        self._pct_changes.append(pct_change)
        if len(self._pct_changes) > 1:
            rank = sum(1 for p in list(self._pct_changes)[:-1] if p < pct_change)
            percent_rank = rank / (len(self._pct_changes) - 1) * 100
        else:
            percent_rank = 50.0

        if not self.is_ready:
            return 50.0

        return (rsi + streak_rsi + percent_rank) / 3

    @staticmethod
    def _calc_rsi(gains, losses) -> float:
        if not gains:
            return 50.0
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class FisherTransform(WindowIndicator):
    """Fisher Transform: normalizes price to Gaussian distribution."""

    def __init__(self, period: int = 10):
        super().__init__("FisherTransform", period)
        self._fisher = 0.0
        self._prev_fisher = 0.0
        self._highs = deque(maxlen=period)
        self._lows = deque(maxlen=period)

    def update_bar(self, bar: TradeBar) -> bool:
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        return self.update(bar.time, (bar.high + bar.low) / 2)

    def _compute_window_value(self) -> float:
        if not self._highs:
            return 0.0

        highest = max(self._highs)
        lowest = min(self._lows)
        r = highest - lowest
        if r == 0:
            return self._fisher

        mid = (self._values[-1] if self._values else 0)
        value = 0.33 * 2 * ((mid - lowest) / r - 0.5) + 0.67 * (
            self._fisher if self._fisher != 0 else 0
        )
        value = max(-0.999, min(0.999, value))

        self._prev_fisher = self._fisher
        self._fisher = 0.5 * math.log((1 + value) / (1 - value))
        return self._fisher


class ChandeMomentumOscillator(WindowIndicator):
    """Chande Momentum Oscillator: measures momentum on both sides."""

    def __init__(self, period: int = 14):
        super().__init__("ChandeMO", period)
        self._prev_value: float = 0.0

    def _compute_window_value(self) -> float:
        values = self.window
        if len(values) < 2:
            return 0.0

        sum_up = 0.0
        sum_down = 0.0
        for i in range(1, len(values)):
            diff = values[i] - values[i - 1]
            if diff > 0:
                sum_up += diff
            else:
                sum_down += abs(diff)

        denom = sum_up + sum_down
        if denom == 0:
            return 0.0
        return ((sum_up - sum_down) / denom) * 100


class KlingerVolumeOscillator(Indicator):
    """Klinger Volume Oscillator: volume-based trend confirmation."""

    def __init__(self, fast_period: int = 34, slow_period: int = 55):
        super().__init__("KlingerVolume", slow_period)
        self._fast_mult = 2.0 / (fast_period + 1)
        self._slow_mult = 2.0 / (slow_period + 1)
        self._fast_ema = 0.0
        self._slow_ema = 0.0
        self._prev_hlc = 0.0
        self._last_volume = 0.0

    def update_bar(self, bar: TradeBar) -> bool:
        hlc = bar.high + bar.low + bar.close
        trend = 1 if hlc > self._prev_hlc else -1
        self._prev_hlc = hlc
        self._last_volume = bar.volume * trend
        return self.update(bar.time, self._last_volume)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        vf = input_point.value
        if self._samples == 1:
            self._fast_ema = vf
            self._slow_ema = vf
        else:
            self._fast_ema += self._fast_mult * (vf - self._fast_ema)
            self._slow_ema += self._slow_mult * (vf - self._slow_ema)

        if not self.is_ready:
            return 0.0
        return self._fast_ema - self._slow_ema
