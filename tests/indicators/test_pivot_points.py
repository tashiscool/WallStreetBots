"""Tests for Pivot Point Indicators."""
from datetime import datetime
import pytest

from backend.tradingbot.indicators.base import TradeBar
from backend.tradingbot.indicators.pivot_points import (
    ClassicPivotPoints, FibonacciPivotPoints, WoodiePivotPoints, CamarillaPivotPoints,
)


def make_bar(h=110, low=90, c=100):
    return TradeBar(time=datetime.now(), open=95, high=h, low=low, close=c, volume=1000)


class TestClassicPivotPoints:
    def test_pivot_calculation(self):
        pp = ClassicPivotPoints()
        bar = make_bar(h=110, low=90, c=100)
        pp.update_bar(bar)
        assert pp.levels is not None
        expected_pivot = (110 + 90 + 100) / 3
        assert abs(pp.levels.pivot - expected_pivot) < 0.01

    def test_resistance_above_pivot(self):
        pp = ClassicPivotPoints()
        pp.update_bar(make_bar())
        assert pp.levels.r1 > pp.levels.pivot
        assert pp.levels.r2 > pp.levels.r1
        assert pp.levels.r3 > pp.levels.r2

    def test_support_below_pivot(self):
        pp = ClassicPivotPoints()
        pp.update_bar(make_bar())
        assert pp.levels.s1 < pp.levels.pivot
        assert pp.levels.s2 < pp.levels.s1
        assert pp.levels.s3 < pp.levels.s2


class TestFibonacciPivotPoints:
    def test_fibonacci_levels(self):
        pp = FibonacciPivotPoints()
        pp.update_bar(make_bar())
        assert pp.levels.r1 < pp.levels.r2 < pp.levels.r3
        assert pp.levels.s1 > pp.levels.s2 > pp.levels.s3


class TestWoodiePivotPoints:
    def test_close_weighting(self):
        pp_classic = ClassicPivotPoints()
        pp_woodie = WoodiePivotPoints()
        bar = make_bar(h=110, low=90, c=105)
        pp_classic.update_bar(bar)
        pp_woodie.update_bar(bar)
        # Woodie weights close more, so with close > median, Woodie pivot > Classic
        assert pp_woodie.levels.pivot > pp_classic.levels.pivot


class TestCamarillaPivotPoints:
    def test_tight_levels(self):
        pp = CamarillaPivotPoints()
        pp.update_bar(make_bar())
        # Camarilla levels are tighter than classic
        assert pp.levels.r1 < pp.levels.r2 < pp.levels.r3
        spread = pp.levels.r3 - pp.levels.s3
        assert spread > 0
