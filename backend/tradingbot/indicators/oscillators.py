"""
Oscillator Indicators - Ported from QuantConnect LEAN.

Includes: RSI, Stochastic, CCI, Williams %R, MFI

Original: https://github.com/QuantConnect/Lean/blob/master/Indicators/
License: Apache 2.0
"""

from collections import deque
from datetime import datetime
from typing import Optional

from .base import Indicator, IndicatorDataPoint, TradeBar, MovingAverageType
from .moving_averages import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WilderMovingAverage,
)


class RelativeStrengthIndex(Indicator):
    """
    Relative Strength Index (RSI) - Developed by J. Welles Wilder.

    Measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss

    Values above 70 typically indicate overbought conditions.
    Values below 30 typically indicate oversold conditions.
    """

    def __init__(self, period: int = 14,
                 moving_average_type: MovingAverageType = MovingAverageType.WILDERS,
                 name: Optional[str] = None):
        name = name or f"RSI({period})"
        super().__init__(name, period + 1)
        self._ma_type = moving_average_type
        self._previous_input: Optional[IndicatorDataPoint] = None

        # Create moving averages for gains and losses
        if moving_average_type == MovingAverageType.WILDERS:
            self._average_gain = WilderMovingAverage(period, f"{name}Gain")
            self._average_loss = WilderMovingAverage(period, f"{name}Loss")
        elif moving_average_type == MovingAverageType.SIMPLE:
            self._average_gain = SimpleMovingAverage(period, f"{name}Gain")
            self._average_loss = SimpleMovingAverage(period, f"{name}Loss")
        else:
            self._average_gain = ExponentialMovingAverage(period, name=f"{name}Gain")
            self._average_loss = ExponentialMovingAverage(period, name=f"{name}Loss")

    @property
    def average_gain(self) -> Indicator:
        return self._average_gain

    @property
    def average_loss(self) -> Indicator:
        return self._average_loss

    @property
    def is_ready(self) -> bool:
        return self._average_gain.is_ready and self._average_loss.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        if self._previous_input is not None:
            change = input_point.value - self._previous_input.value
            if change >= 0:
                self._average_gain.update(input_point.time, change)
                self._average_loss.update(input_point.time, 0.0)
            else:
                self._average_gain.update(input_point.time, 0.0)
                self._average_loss.update(input_point.time, abs(change))

        self._previous_input = input_point

        # Ensure positive averages (some MAs can go negative like DEMA)
        avg_loss = max(0, self._average_loss.value)
        avg_gain = max(0, self._average_gain.value)

        # Handle edge case of zero average loss
        if round(avg_loss, 10) == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1 + rs)

    def reset(self) -> None:
        super().reset()
        self._previous_input = None
        self._average_gain.reset()
        self._average_loss.reset()


class Stochastic(Indicator):
    """
    Stochastic Oscillator - Developed by George Lane.

    Compares closing price to price range over a period.

    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA of %K (signal line)

    Values above 80 typically indicate overbought.
    Values below 20 typically indicate oversold.
    """

    def __init__(self, period: int = 14, k_period: int = 3, d_period: int = 3,
                 name: Optional[str] = None):
        name = name or f"STOCH({period},{k_period},{d_period})"
        super().__init__(name, period)
        self._k_period = k_period
        self._d_period = d_period

        self._highs = deque(maxlen=period)
        self._lows = deque(maxlen=period)
        self._raw_k_values = deque(maxlen=k_period)

        self._stoch_k = 0.0
        self._stoch_d = 0.0
        self._k_sma = SimpleMovingAverage(k_period)
        self._d_sma = SimpleMovingAverage(d_period)

    @property
    def stoch_k(self) -> float:
        """Fast stochastic %K (smoothed)."""
        return self._stoch_k

    @property
    def stoch_d(self) -> float:
        """Slow stochastic %D (signal line)."""
        return self._stoch_d

    @property
    def fast_stoch(self) -> float:
        """Raw %K before smoothing."""
        return self._raw_k_values[-1] if self._raw_k_values else 0.0

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._samples += 1

        if len(self._highs) < self._period:
            return False

        highest_high = max(self._highs)
        lowest_low = min(self._lows)
        range_hl = highest_high - lowest_low

        if range_hl == 0:
            raw_k = 50.0  # Midpoint when no range
        else:
            raw_k = ((bar.close - lowest_low) / range_hl) * 100

        self._raw_k_values.append(raw_k)
        self._k_sma.update(bar.time, raw_k)
        self._stoch_k = self._k_sma.value

        self._d_sma.update(bar.time, self._stoch_k)
        self._stoch_d = self._d_sma.value

        self._current = IndicatorDataPoint(bar.time, self._stoch_k)
        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        # For simple price input, use as close
        return self._stoch_k

    def reset(self) -> None:
        super().reset()
        self._highs.clear()
        self._lows.clear()
        self._raw_k_values.clear()
        self._stoch_k = 0.0
        self._stoch_d = 0.0
        self._k_sma.reset()
        self._d_sma.reset()


class StochasticRSI(Indicator):
    """
    Stochastic RSI - Developed by Tushar Chande and Stanley Kroll.

    Applies Stochastic formula to RSI values instead of price.
    More sensitive than regular RSI.

    StochRSI = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI)
    """

    def __init__(self, rsi_period: int = 14, stoch_period: int = 14,
                 k_period: int = 3, d_period: int = 3,
                 name: Optional[str] = None):
        name = name or f"STOCHRSI({rsi_period},{stoch_period},{k_period},{d_period})"
        super().__init__(name, rsi_period + stoch_period)
        self._rsi = RelativeStrengthIndex(rsi_period)
        self._stoch_period = stoch_period
        self._rsi_values = deque(maxlen=stoch_period)
        self._k_sma = SimpleMovingAverage(k_period)
        self._d_sma = SimpleMovingAverage(d_period)
        self._stoch_rsi_k = 0.0
        self._stoch_rsi_d = 0.0

    @property
    def k(self) -> float:
        return self._stoch_rsi_k

    @property
    def d(self) -> float:
        return self._stoch_rsi_d

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._rsi.update(input_point.time, input_point.value)

        if not self._rsi.is_ready:
            return 0.0

        self._rsi_values.append(self._rsi.value)

        if len(self._rsi_values) < self._stoch_period:
            return 0.0

        highest_rsi = max(self._rsi_values)
        lowest_rsi = min(self._rsi_values)
        rsi_range = highest_rsi - lowest_rsi

        if rsi_range == 0:
            raw_stoch_rsi = 50.0
        else:
            raw_stoch_rsi = ((self._rsi.value - lowest_rsi) / rsi_range) * 100

        self._k_sma.update(input_point.time, raw_stoch_rsi)
        self._stoch_rsi_k = self._k_sma.value

        self._d_sma.update(input_point.time, self._stoch_rsi_k)
        self._stoch_rsi_d = self._d_sma.value

        return self._stoch_rsi_k

    def reset(self) -> None:
        super().reset()
        self._rsi.reset()
        self._rsi_values.clear()
        self._k_sma.reset()
        self._d_sma.reset()
        self._stoch_rsi_k = 0.0
        self._stoch_rsi_d = 0.0


class CommodityChannelIndex(Indicator):
    """
    Commodity Channel Index (CCI) - Developed by Donald Lambert.

    Measures current price level relative to average price level
    over a given period. Used to identify cyclical trends.

    CCI = (Typical Price - SMA of TP) / (0.015 * Mean Deviation)

    Values above +100 suggest overbought.
    Values below -100 suggest oversold.
    """

    def __init__(self, period: int = 20, name: Optional[str] = None):
        name = name or f"CCI({period})"
        super().__init__(name, period)
        self._tp_window = deque(maxlen=period)
        self._tp_sma = SimpleMovingAverage(period)
        self._constant = 0.015

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        typical_price = bar.typical_price
        return self.update(bar.time, typical_price)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._tp_window.append(input_point.value)
        self._tp_sma.update(input_point.time, input_point.value)

        if not self._tp_sma.is_ready:
            return 0.0

        # Calculate mean deviation
        mean_tp = self._tp_sma.value
        mean_deviation = sum(abs(tp - mean_tp) for tp in self._tp_window) / len(self._tp_window)

        if mean_deviation == 0:
            return 0.0

        return (input_point.value - mean_tp) / (self._constant * mean_deviation)

    def reset(self) -> None:
        super().reset()
        self._tp_window.clear()
        self._tp_sma.reset()


class WilliamsPercentR(Indicator):
    """
    Williams %R - Developed by Larry Williams.

    Similar to Stochastic but on inverted scale.
    Measures where closing price falls within the high-low range.

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Values from 0 to -20 suggest overbought.
    Values from -80 to -100 suggest oversold.
    """

    def __init__(self, period: int = 14, name: Optional[str] = None):
        name = name or f"WILLR({period})"
        super().__init__(name, period)
        self._highs = deque(maxlen=period)
        self._lows = deque(maxlen=period)
        self._current_close = 0.0

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._current_close = bar.close
        self._samples += 1

        if len(self._highs) < self._period:
            self._current = IndicatorDataPoint(bar.time, 0.0)
            return False

        highest_high = max(self._highs)
        lowest_low = min(self._lows)
        range_hl = highest_high - lowest_low

        if range_hl == 0:
            value = -50.0
        else:
            value = ((highest_high - bar.close) / range_hl) * -100

        self._current = IndicatorDataPoint(bar.time, value)
        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._highs.clear()
        self._lows.clear()
        self._current_close = 0.0


class MoneyFlowIndex(Indicator):
    """
    Money Flow Index (MFI) - Developed by Gene Quong and Avrum Soudack.

    Volume-weighted RSI. Measures buying and selling pressure
    using price and volume.

    Values above 80 suggest overbought.
    Values below 20 suggest oversold.
    """

    def __init__(self, period: int = 14, name: Optional[str] = None):
        name = name or f"MFI({period})"
        super().__init__(name, period + 1)
        self._positive_flow = deque(maxlen=period)
        self._negative_flow = deque(maxlen=period)
        self._previous_typical_price = None

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data with volume."""
        typical_price = bar.typical_price
        raw_money_flow = typical_price * bar.volume

        self._samples += 1

        if self._previous_typical_price is not None:
            if typical_price > self._previous_typical_price:
                self._positive_flow.append(raw_money_flow)
                self._negative_flow.append(0.0)
            elif typical_price < self._previous_typical_price:
                self._positive_flow.append(0.0)
                self._negative_flow.append(raw_money_flow)
            else:
                self._positive_flow.append(0.0)
                self._negative_flow.append(0.0)

        self._previous_typical_price = typical_price

        if len(self._positive_flow) < self._period - 1:
            self._current = IndicatorDataPoint(bar.time, 0.0)
            return False

        positive_flow_sum = sum(self._positive_flow)
        negative_flow_sum = sum(self._negative_flow)

        if negative_flow_sum == 0:
            mfi = 100.0
        else:
            money_ratio = positive_flow_sum / negative_flow_sum
            mfi = 100.0 - (100.0 / (1 + money_ratio))

        self._current = IndicatorDataPoint(bar.time, mfi)
        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._positive_flow.clear()
        self._negative_flow.clear()
        self._previous_typical_price = None


class UltimateOscillator(Indicator):
    """
    Ultimate Oscillator - Developed by Larry Williams.

    Uses weighted averages of three different time periods
    to reduce false signals.

    Typical settings: 7, 14, 28 periods with weights 4, 2, 1
    """

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28,
                 name: Optional[str] = None):
        name = name or f"UO({period1},{period2},{period3})"
        super().__init__(name, period3 + 1)
        self._period1 = period1
        self._period2 = period2
        self._period3 = period3

        self._bp = deque(maxlen=period3)  # Buying Pressure
        self._tr = deque(maxlen=period3)  # True Range
        self._previous_close = None

    def update_bar(self, bar: TradeBar) -> bool:
        """Update with OHLC bar data."""
        self._samples += 1

        if self._previous_close is None:
            self._previous_close = bar.close
            return False

        # Buying Pressure = Close - Min(Low, Previous Close)
        bp = bar.close - min(bar.low, self._previous_close)

        # True Range = Max(High, Previous Close) - Min(Low, Previous Close)
        tr = max(bar.high, self._previous_close) - min(bar.low, self._previous_close)

        self._bp.append(bp)
        self._tr.append(tr)
        self._previous_close = bar.close

        if len(self._bp) < self._period3:
            self._current = IndicatorDataPoint(bar.time, 0.0)
            return False

        # Calculate averages for each period
        bp_list = list(self._bp)
        tr_list = list(self._tr)

        avg1 = sum(bp_list[-self._period1:]) / sum(tr_list[-self._period1:]) if sum(tr_list[-self._period1:]) != 0 else 0
        avg2 = sum(bp_list[-self._period2:]) / sum(tr_list[-self._period2:]) if sum(tr_list[-self._period2:]) != 0 else 0
        avg3 = sum(bp_list) / sum(tr_list) if sum(tr_list) != 0 else 0

        # Weighted average (4, 2, 1)
        uo = ((4 * avg1) + (2 * avg2) + avg3) / 7 * 100

        self._current = IndicatorDataPoint(bar.time, uo)
        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def reset(self) -> None:
        super().reset()
        self._bp.clear()
        self._tr.clear()
        self._previous_close = None
