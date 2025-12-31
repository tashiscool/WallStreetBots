"""
Base Indicator Classes - Ported from QuantConnect LEAN.

Provides the foundational classes for all technical indicators.

Original: https://github.com/QuantConnect/Lean/blob/master/Indicators/
License: Apache 2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional, Union
from collections import deque
import numpy as np


class MovingAverageType(Enum):
    """Types of moving averages."""
    SIMPLE = "simple"
    EXPONENTIAL = "exponential"
    WILDERS = "wilders"  # Wilder's smoothing (used in RSI)
    DOUBLE_EXPONENTIAL = "double_exponential"
    TRIPLE_EXPONENTIAL = "triple_exponential"
    TRIANGULAR = "triangular"
    KAMA = "kaufman_adaptive"
    HULL = "hull"
    ALMA = "arnaud_legoux"


@dataclass
class IndicatorDataPoint:
    """Represents a single data point for an indicator."""
    time: datetime
    value: float

    def __float__(self) -> float:
        return self.value

    def __lt__(self, other: Union["IndicatorDataPoint", float]) -> bool:
        other_val = other.value if isinstance(other, IndicatorDataPoint) else other
        return self.value < other_val

    def __gt__(self, other: Union["IndicatorDataPoint", float]) -> bool:
        other_val = other.value if isinstance(other, IndicatorDataPoint) else other
        return self.value > other_val

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, IndicatorDataPoint):
            return self.value == other.value
        return self.value == other


@dataclass
class TradeBar:
    """OHLCV bar data."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def typical_price(self) -> float:
        """(High + Low + Close) / 3"""
        return (self.high + self.low + self.close) / 3

    @property
    def median_price(self) -> float:
        """(High + Low) / 2"""
        return (self.high + self.low) / 2

    @property
    def true_range(self) -> float:
        """True Range for this bar (requires previous close)."""
        # This is typically calculated with previous close
        return self.high - self.low


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.

    Ported from QuantConnect LEAN's IndicatorBase<T>.
    """

    def __init__(self, name: str, period: int = 0):
        self._name = name
        self._period = period
        self._samples = 0
        self._current = IndicatorDataPoint(datetime.now(), 0.0)
        self._previous = None
        self._window: deque = deque(maxlen=period if period > 0 else None)

    @property
    def name(self) -> str:
        """Indicator name."""
        return self._name

    @property
    def period(self) -> int:
        """Lookback period."""
        return self._period

    @property
    def samples(self) -> int:
        """Number of samples processed."""
        return self._samples

    @property
    def current(self) -> IndicatorDataPoint:
        """Current indicator value."""
        return self._current

    @property
    def previous(self) -> Optional[IndicatorDataPoint]:
        """Previous indicator value."""
        return self._previous

    @property
    def value(self) -> float:
        """Current value as float."""
        return self._current.value

    @property
    def is_ready(self) -> bool:
        """Whether indicator has enough data to produce valid values."""
        return self._samples >= self._period

    @property
    def warm_up_period(self) -> int:
        """Number of data points needed before indicator is ready."""
        return self._period

    def update(self, time: datetime, value: float) -> bool:
        """
        Update the indicator with a new data point.

        Args:
            time: Timestamp of the data point
            value: Value of the data point

        Returns:
            True if the indicator is ready
        """
        input_point = IndicatorDataPoint(time, value)
        self._samples += 1
        self._previous = self._current

        # Store in window
        self._window.append(input_point)

        # Compute new value
        new_value = self._compute_next_value(input_point)
        self._current = IndicatorDataPoint(time, new_value)

        return self.is_ready

    def update_bar(self, bar: TradeBar) -> bool:
        """Update indicator with a TradeBar (uses close price by default)."""
        return self.update(bar.time, bar.close)

    @abstractmethod
    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        """
        Compute the next indicator value.

        Must be implemented by subclasses.
        """
        pass

    def reset(self) -> None:
        """Reset indicator to initial state."""
        self._samples = 0
        self._current = IndicatorDataPoint(datetime.now(), 0.0)
        self._previous = None
        self._window.clear()

    def __float__(self) -> float:
        return self.value

    def __lt__(self, other: Union["Indicator", float]) -> bool:
        other_val = other.value if isinstance(other, Indicator) else other
        return self.value < other_val

    def __gt__(self, other: Union["Indicator", float]) -> bool:
        other_val = other.value if isinstance(other, Indicator) else other
        return self.value > other_val

    def __repr__(self) -> str:
        return f"{self._name}: {self.value:.4f} (ready={self.is_ready})"


class WindowIndicator(Indicator):
    """
    Indicator that uses a fixed-size rolling window.
    """

    def __init__(self, name: str, period: int):
        super().__init__(name, period)
        self._values = deque(maxlen=period)

    @property
    def window(self) -> list:
        """Get the current window of values."""
        return list(self._values)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._values.append(input_point.value)
        if len(self._values) < self._period:
            return 0.0
        return self._compute_window_value()

    @abstractmethod
    def _compute_window_value(self) -> float:
        """Compute indicator value from the current window."""
        pass

    def reset(self) -> None:
        super().reset()
        self._values.clear()


class CompositeIndicator(Indicator):
    """
    Indicator composed of multiple sub-indicators.
    """

    def __init__(self, name: str, left: Indicator, right: Indicator,
                 composer: Callable[[float, float], float]):
        period = max(left.period, right.period)
        super().__init__(name, period)
        self._left = left
        self._right = right
        self._composer = composer

    @property
    def left(self) -> Indicator:
        return self._left

    @property
    def right(self) -> Indicator:
        return self._right

    @property
    def is_ready(self) -> bool:
        return self._left.is_ready and self._right.is_ready

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        self._left.update(input_point.time, input_point.value)
        self._right.update(input_point.time, input_point.value)
        if not self.is_ready:
            return 0.0
        return self._composer(self._left.value, self._right.value)

    def reset(self) -> None:
        super().reset()
        self._left.reset()
        self._right.reset()


def get_moving_average(period: int, ma_type: MovingAverageType = MovingAverageType.SIMPLE) -> Indicator:
    """
    Factory function to create moving average indicator of specified type.

    Args:
        period: Lookback period
        ma_type: Type of moving average

    Returns:
        Appropriate moving average indicator
    """
    from .moving_averages import (
        SimpleMovingAverage,
        ExponentialMovingAverage,
        WilderMovingAverage,
        DoubleExponentialMovingAverage,
        TripleExponentialMovingAverage,
        KaufmanAdaptiveMovingAverage,
    )

    ma_map = {
        MovingAverageType.SIMPLE: SimpleMovingAverage,
        MovingAverageType.EXPONENTIAL: ExponentialMovingAverage,
        MovingAverageType.WILDERS: WilderMovingAverage,
        MovingAverageType.DOUBLE_EXPONENTIAL: DoubleExponentialMovingAverage,
        MovingAverageType.TRIPLE_EXPONENTIAL: TripleExponentialMovingAverage,
        MovingAverageType.KAMA: KaufmanAdaptiveMovingAverage,
    }

    indicator_class = ma_map.get(ma_type, SimpleMovingAverage)
    return indicator_class(period)
