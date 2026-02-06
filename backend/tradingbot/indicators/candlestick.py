"""
Candlestick Pattern Indicators

Detects candlestick patterns from OHLCV data.
Inspired by Jesse and Backtrader candlestick indicators.
"""

from datetime import datetime
from typing import List, Optional

from .base import Indicator, IndicatorDataPoint, TradeBar


class CandlestickPatternIndicator(Indicator):
    """Base for candlestick pattern detection from bars."""

    def __init__(self, name: str, lookback: int = 3):
        super().__init__(name, period=lookback)
        self._bars: List[TradeBar] = []
        self._lookback = lookback

    def update_bar(self, bar: TradeBar) -> bool:
        self._bars.append(bar)
        if len(self._bars) > self._lookback + 2:
            self._bars = self._bars[-(self._lookback + 2):]
        self._samples += 1
        self._previous = self._current
        detected = self._detect()
        self._current = IndicatorDataPoint(bar.time, 1.0 if detected else 0.0)
        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value

    def _detect(self) -> bool:
        raise NotImplementedError

    @staticmethod
    def _body(bar: TradeBar) -> float:
        return abs(bar.close - bar.open)

    @staticmethod
    def _upper_shadow(bar: TradeBar) -> float:
        return bar.high - max(bar.open, bar.close)

    @staticmethod
    def _lower_shadow(bar: TradeBar) -> float:
        return min(bar.open, bar.close) - bar.low

    @staticmethod
    def _is_bullish(bar: TradeBar) -> bool:
        return bar.close > bar.open

    @staticmethod
    def _is_bearish(bar: TradeBar) -> bool:
        return bar.close < bar.open

    @staticmethod
    def _range(bar: TradeBar) -> float:
        return bar.high - bar.low


class Doji(CandlestickPatternIndicator):
    """Doji: open and close nearly equal."""

    def __init__(self, body_pct: float = 0.05):
        super().__init__("Doji", lookback=1)
        self.body_pct = body_pct

    def _detect(self) -> bool:
        if len(self._bars) < 1:
            return False
        bar = self._bars[-1]
        r = self._range(bar)
        if r == 0:
            return True
        return self._body(bar) / r < self.body_pct


class Hammer(CandlestickPatternIndicator):
    """Hammer: small body at top, long lower shadow (bullish reversal)."""

    def __init__(self, shadow_ratio: float = 2.0):
        super().__init__("Hammer", lookback=1)
        self.shadow_ratio = shadow_ratio

    def _detect(self) -> bool:
        if len(self._bars) < 1:
            return False
        bar = self._bars[-1]
        body = self._body(bar)
        if body == 0:
            return False
        lower = self._lower_shadow(bar)
        upper = self._upper_shadow(bar)
        return lower >= body * self.shadow_ratio and upper < body


class ShootingStar(CandlestickPatternIndicator):
    """Shooting Star: small body at bottom, long upper shadow (bearish reversal)."""

    def __init__(self, shadow_ratio: float = 2.0):
        super().__init__("ShootingStar", lookback=1)
        self.shadow_ratio = shadow_ratio

    def _detect(self) -> bool:
        if len(self._bars) < 1:
            return False
        bar = self._bars[-1]
        body = self._body(bar)
        if body == 0:
            return False
        upper = self._upper_shadow(bar)
        lower = self._lower_shadow(bar)
        return upper >= body * self.shadow_ratio and lower < body


class Engulfing(CandlestickPatternIndicator):
    """Bullish/Bearish Engulfing: current body engulfs previous.
    Value: 1.0 bullish, -1.0 bearish, 0.0 none.
    """

    def __init__(self):
        super().__init__("Engulfing", lookback=2)

    def _detect(self) -> bool:
        # Override to set directional value
        if len(self._bars) < 2:
            return False
        prev, curr = self._bars[-2], self._bars[-1]
        if self._is_bearish(prev) and self._is_bullish(curr):
            if curr.open <= prev.close and curr.close >= prev.open:
                self._current = IndicatorDataPoint(curr.time, 1.0)
                return True
        if self._is_bullish(prev) and self._is_bearish(curr):
            if curr.open >= prev.close and curr.close <= prev.open:
                self._current = IndicatorDataPoint(curr.time, -1.0)
                return True
        return False


class MorningStar(CandlestickPatternIndicator):
    """Morning Star: 3-bar bullish reversal."""

    def __init__(self):
        super().__init__("MorningStar", lookback=3)

    def _detect(self) -> bool:
        if len(self._bars) < 3:
            return False
        first, second, third = self._bars[-3], self._bars[-2], self._bars[-1]
        return (
            self._is_bearish(first)
            and self._body(first) > self._range(first) * 0.3
            and self._body(second) < self._range(first) * 0.3
            and self._is_bullish(third)
            and third.close > (first.open + first.close) / 2
        )


class EveningStar(CandlestickPatternIndicator):
    """Evening Star: 3-bar bearish reversal."""

    def __init__(self):
        super().__init__("EveningStar", lookback=3)

    def _detect(self) -> bool:
        if len(self._bars) < 3:
            return False
        first, second, third = self._bars[-3], self._bars[-2], self._bars[-1]
        return (
            self._is_bullish(first)
            and self._body(first) > self._range(first) * 0.3
            and self._body(second) < self._range(first) * 0.3
            and self._is_bearish(third)
            and third.close < (first.open + first.close) / 2
        )


class Harami(CandlestickPatternIndicator):
    """Harami: small bar inside previous bar's body.
    Value: 1.0 bullish, -1.0 bearish.
    """

    def __init__(self):
        super().__init__("Harami", lookback=2)

    def _detect(self) -> bool:
        if len(self._bars) < 2:
            return False
        prev, curr = self._bars[-2], self._bars[-1]
        prev_top = max(prev.open, prev.close)
        prev_bot = min(prev.open, prev.close)
        curr_top = max(curr.open, curr.close)
        curr_bot = min(curr.open, curr.close)

        inside = curr_top < prev_top and curr_bot > prev_bot
        if not inside:
            return False

        if self._is_bearish(prev) and self._is_bullish(curr):
            self._current = IndicatorDataPoint(curr.time, 1.0)
            return True
        if self._is_bullish(prev) and self._is_bearish(curr):
            self._current = IndicatorDataPoint(curr.time, -1.0)
            return True
        return False


class ThreeWhiteSoldiers(CandlestickPatternIndicator):
    """Three White Soldiers: 3 consecutive bullish bars."""

    def __init__(self):
        super().__init__("ThreeWhiteSoldiers", lookback=3)

    def _detect(self) -> bool:
        if len(self._bars) < 3:
            return False
        bars = self._bars[-3:]
        return all(
            self._is_bullish(b) and self._body(b) > self._range(b) * 0.3
            for b in bars
        ) and bars[1].close > bars[0].close and bars[2].close > bars[1].close


class ThreeCrows(CandlestickPatternIndicator):
    """Three Black Crows: 3 consecutive bearish bars."""

    def __init__(self):
        super().__init__("ThreeCrows", lookback=3)

    def _detect(self) -> bool:
        if len(self._bars) < 3:
            return False
        bars = self._bars[-3:]
        return all(
            self._is_bearish(b) and self._body(b) > self._range(b) * 0.3
            for b in bars
        ) and bars[1].close < bars[0].close and bars[2].close < bars[1].close


class SpinningTop(CandlestickPatternIndicator):
    """Spinning Top: small body with long shadows on both sides."""

    def __init__(self, body_pct: float = 0.25, shadow_ratio: float = 1.0):
        super().__init__("SpinningTop", lookback=1)
        self.body_pct = body_pct
        self.shadow_ratio = shadow_ratio

    def _detect(self) -> bool:
        if len(self._bars) < 1:
            return False
        bar = self._bars[-1]
        r = self._range(bar)
        if r == 0:
            return False
        body = self._body(bar)
        upper = self._upper_shadow(bar)
        lower = self._lower_shadow(bar)
        return (
            body / r < self.body_pct
            and upper >= body * self.shadow_ratio
            and lower >= body * self.shadow_ratio
        )
