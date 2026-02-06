"""
Pivot Point Indicators

Classic, Fibonacci, Woodie, and Camarilla pivot point calculations.
Inspired by Jesse and Backtrader pivot indicators.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .base import Indicator, IndicatorDataPoint, TradeBar


@dataclass
class PivotLevels:
    """Calculated pivot levels."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


class PivotPointBase(Indicator):
    """Base for pivot point indicators."""

    def __init__(self, name: str):
        super().__init__(name, period=1)
        self._last_bar: Optional[TradeBar] = None
        self.levels: Optional[PivotLevels] = None

    def update_bar(self, bar: TradeBar) -> bool:
        self._last_bar = bar
        self._samples += 1
        self._previous = self._current
        self.levels = self._calculate_levels(bar)
        self._current = IndicatorDataPoint(bar.time, self.levels.pivot)
        return self.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        return self._current.value if self._current else 0.0

    def _calculate_levels(self, bar: TradeBar) -> PivotLevels:
        raise NotImplementedError


class ClassicPivotPoints(PivotPointBase):
    """Classic (Floor Trader) pivot points."""

    def __init__(self):
        super().__init__("ClassicPivot")

    def _calculate_levels(self, bar: TradeBar) -> PivotLevels:
        p = (bar.high + bar.low + bar.close) / 3
        return PivotLevels(
            pivot=p,
            r1=2 * p - bar.low,
            r2=p + (bar.high - bar.low),
            r3=bar.high + 2 * (p - bar.low),
            s1=2 * p - bar.high,
            s2=p - (bar.high - bar.low),
            s3=bar.low - 2 * (bar.high - p),
        )


class FibonacciPivotPoints(PivotPointBase):
    """Fibonacci pivot points."""

    def __init__(self):
        super().__init__("FibonacciPivot")

    def _calculate_levels(self, bar: TradeBar) -> PivotLevels:
        p = (bar.high + bar.low + bar.close) / 3
        r = bar.high - bar.low
        return PivotLevels(
            pivot=p,
            r1=p + 0.382 * r,
            r2=p + 0.618 * r,
            r3=p + 1.000 * r,
            s1=p - 0.382 * r,
            s2=p - 0.618 * r,
            s3=p - 1.000 * r,
        )


class WoodiePivotPoints(PivotPointBase):
    """Woodie's pivot points (weights close more)."""

    def __init__(self):
        super().__init__("WoodiePivot")

    def _calculate_levels(self, bar: TradeBar) -> PivotLevels:
        p = (bar.high + bar.low + 2 * bar.close) / 4
        return PivotLevels(
            pivot=p,
            r1=2 * p - bar.low,
            r2=p + (bar.high - bar.low),
            r3=bar.high + 2 * (p - bar.low),
            s1=2 * p - bar.high,
            s2=p - (bar.high - bar.low),
            s3=bar.low - 2 * (bar.high - p),
        )


class CamarillaPivotPoints(PivotPointBase):
    """Camarilla pivot points (tight intraday levels)."""

    def __init__(self):
        super().__init__("CamarillaPivot")

    def _calculate_levels(self, bar: TradeBar) -> PivotLevels:
        p = (bar.high + bar.low + bar.close) / 3
        r = bar.high - bar.low
        return PivotLevels(
            pivot=p,
            r1=bar.close + r * 1.1 / 12,
            r2=bar.close + r * 1.1 / 6,
            r3=bar.close + r * 1.1 / 4,
            s1=bar.close - r * 1.1 / 12,
            s2=bar.close - r * 1.1 / 6,
            s3=bar.close - r * 1.1 / 4,
        )
