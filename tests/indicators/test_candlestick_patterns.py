"""Tests for Candlestick Pattern Indicators."""
from datetime import datetime
import pytest

from backend.tradingbot.indicators.base import TradeBar
from backend.tradingbot.indicators.candlestick import (
    Doji, Hammer, ShootingStar, Engulfing, MorningStar,
    EveningStar, Harami, ThreeWhiteSoldiers, ThreeCrows, SpinningTop,
)


def make_bar(o, h, l, c, t=None):
    return TradeBar(time=t or datetime.now(), open=o, high=h, low=l, close=c, volume=1000)


class TestDoji:
    def test_detects_doji(self):
        indicator = Doji()
        bar = make_bar(100, 102, 98, 100.01)
        indicator.update_bar(bar)
        assert indicator.value == 1.0

    def test_rejects_non_doji(self):
        indicator = Doji()
        bar = make_bar(100, 110, 95, 108)
        indicator.update_bar(bar)
        assert indicator.value == 0.0


class TestHammer:
    def test_detects_hammer(self):
        indicator = Hammer()
        # body=0.5, lower_shadow=100-94=6, upper_shadow=101-100.5=0.5
        # Need lower >= body*2 (6 >= 1.0) and upper < body (0.5 < 0.5 fails)
        # Fix: upper must be < body, so use high=100.49
        bar = make_bar(100, 100.49, 94, 100.3)  # body=0.3, lower=6, upper=0.19
        indicator.update_bar(bar)
        assert indicator.value == 1.0

    def test_rejects_non_hammer(self):
        indicator = Hammer()
        bar = make_bar(100, 106, 99, 105)  # Long upper shadow
        indicator.update_bar(bar)
        assert indicator.value == 0.0


class TestShootingStar:
    def test_detects_shooting_star(self):
        indicator = ShootingStar()
        # body=0.3, upper=106-100.3=5.7, lower=100-99.8=0.2
        # upper >= body*2 (5.7 >= 0.6) and lower < body (0.2 < 0.3)
        bar = make_bar(100, 106, 99.8, 100.3)
        indicator.update_bar(bar)
        assert indicator.value == 1.0


class TestEngulfing:
    def test_bullish_engulfing(self):
        indicator = Engulfing()
        bar1 = make_bar(102, 103, 99, 100)  # Bearish
        bar2 = make_bar(99, 104, 98, 103)   # Bullish engulfing
        indicator.update_bar(bar1)
        indicator.update_bar(bar2)
        assert indicator.value == 1.0

    def test_bearish_engulfing(self):
        indicator = Engulfing()
        bar1 = make_bar(100, 103, 99, 102)  # Bullish
        bar2 = make_bar(103, 104, 98, 99)   # Bearish engulfing
        indicator.update_bar(bar1)
        indicator.update_bar(bar2)
        # Detected as pattern (update_bar sets value to 1.0 for detected)
        assert indicator.value == 1.0


class TestMorningStar:
    def test_detects_morning_star(self):
        indicator = MorningStar()
        bar1 = make_bar(106, 107, 100, 101)  # Big bearish
        bar2 = make_bar(101, 101.5, 100.5, 101.2)  # Small body
        bar3 = make_bar(101, 105, 100.5, 104)  # Big bullish, close > midpoint of bar1
        indicator.update_bar(bar1)
        indicator.update_bar(bar2)
        indicator.update_bar(bar3)
        assert indicator.value == 1.0


class TestEveningStar:
    def test_detects_evening_star(self):
        indicator = EveningStar()
        bar1 = make_bar(100, 106, 99, 105)   # Big bullish
        bar2 = make_bar(105, 105.5, 104.5, 105.2)  # Small body
        bar3 = make_bar(105, 105.5, 100, 101)  # Big bearish, close < midpoint of bar1
        indicator.update_bar(bar1)
        indicator.update_bar(bar2)
        indicator.update_bar(bar3)
        assert indicator.value == 1.0


class TestHarami:
    def test_bullish_harami(self):
        indicator = Harami()
        bar1 = make_bar(105, 106, 99, 100)   # Big bearish
        bar2 = make_bar(101, 103, 100.5, 102)  # Small bullish inside
        indicator.update_bar(bar1)
        indicator.update_bar(bar2)
        assert indicator.value == 1.0


class TestThreeWhiteSoldiers:
    def test_detects_pattern(self):
        indicator = ThreeWhiteSoldiers()
        bars = [
            make_bar(100, 104, 99, 103),
            make_bar(103, 107, 102, 106),
            make_bar(106, 110, 105, 109),
        ]
        for b in bars:
            indicator.update_bar(b)
        assert indicator.value == 1.0


class TestThreeCrows:
    def test_detects_pattern(self):
        indicator = ThreeCrows()
        bars = [
            make_bar(110, 111, 106, 107),
            make_bar(107, 108, 103, 104),
            make_bar(104, 105, 100, 101),
        ]
        for b in bars:
            indicator.update_bar(b)
        assert indicator.value == 1.0


class TestSpinningTop:
    def test_detects_spinning_top(self):
        indicator = SpinningTop()
        bar = make_bar(100, 103, 97, 100.2)  # Small body, equal shadows
        indicator.update_bar(bar)
        assert indicator.value == 1.0
