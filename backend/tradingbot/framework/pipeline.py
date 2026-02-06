"""
Factor Pipeline for Universe Screening

Systematic factor-based screening inspired by Zipline's Pipeline API.
Compute factors across all symbols, then screen/rank for universe selection.

Usage:
    pipe = Pipeline()
    pipe.add(AverageDollarVolume(window=20), 'adv')
    pipe.add(Returns(window=60), 'momentum')
    pipe.set_screen(pipe.get('adv') > 1_000_000)

    results = pipe.run(data)
    # Returns DataFrame: symbols x factors, filtered by screen
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


class Factor(ABC):
    """Base class for pipeline factors."""

    def __init__(self, window: int = 1, name: str = ""):
        self.window = window
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, data: Dict[str, Dict[str, Any]]) -> pd.Series:
        """Compute factor values for all symbols.

        Args:
            data: {symbol: {close: [...], volume: [...], ...}}

        Returns:
            Series indexed by symbol with factor values.
        """
        pass

    def __gt__(self, other):
        return ScreenFilter(self, ">", other)

    def __lt__(self, other):
        return ScreenFilter(self, "<", other)

    def __ge__(self, other):
        return ScreenFilter(self, ">=", other)

    def __le__(self, other):
        return ScreenFilter(self, "<=", other)

    def top(self, n: int):
        """Select top N by this factor."""
        return TopFilter(self, n)

    def bottom(self, n: int):
        """Select bottom N by this factor."""
        return BottomFilter(self, n)

    def percentile_between(self, low: float, high: float):
        """Select symbols between percentile range."""
        return PercentileFilter(self, low, high)


class AverageDollarVolume(Factor):
    """Average daily dollar volume over a window."""

    def __init__(self, window: int = 20):
        super().__init__(window, "AverageDollarVolume")

    def compute(self, data: Dict[str, Dict[str, Any]]) -> pd.Series:
        results = {}
        for symbol, sym_data in data.items():
            closes = np.array(sym_data.get("close", []))
            volumes = np.array(sym_data.get("volume", []))
            if len(closes) >= self.window and len(volumes) >= self.window:
                dv = closes[-self.window:] * volumes[-self.window:]
                results[symbol] = np.mean(dv)
        return pd.Series(results)


class Returns(Factor):
    """Total return over a window (momentum factor)."""

    def __init__(self, window: int = 20):
        super().__init__(window, f"Returns_{window}")

    def compute(self, data: Dict[str, Dict[str, Any]]) -> pd.Series:
        results = {}
        for symbol, sym_data in data.items():
            closes = np.array(sym_data.get("close", []))
            if len(closes) > self.window and closes[-self.window - 1] != 0:
                ret = (closes[-1] - closes[-self.window - 1]) / closes[-self.window - 1]
                results[symbol] = ret
        return pd.Series(results)


class Volatility(Factor):
    """Annualized volatility over a window."""

    def __init__(self, window: int = 20):
        super().__init__(window, f"Volatility_{window}")

    def compute(self, data: Dict[str, Dict[str, Any]]) -> pd.Series:
        results = {}
        for symbol, sym_data in data.items():
            closes = np.array(sym_data.get("close", []))
            if len(closes) > self.window:
                daily_rets = np.diff(closes[-self.window - 1:]) / closes[-self.window - 1:-1]
                results[symbol] = np.std(daily_rets) * np.sqrt(252)
        return pd.Series(results)


class MeanReversion(Factor):
    """Mean reversion factor: distance from N-day mean."""

    def __init__(self, window: int = 20):
        super().__init__(window, f"MeanReversion_{window}")

    def compute(self, data: Dict[str, Dict[str, Any]]) -> pd.Series:
        results = {}
        for symbol, sym_data in data.items():
            closes = np.array(sym_data.get("close", []))
            if len(closes) >= self.window:
                mean = np.mean(closes[-self.window:])
                if mean != 0:
                    results[symbol] = (closes[-1] - mean) / mean
        return pd.Series(results)


# --- Screen Filters ---

class ScreenFilter:
    """Binary filter for pipeline screening."""

    def __init__(self, factor: Factor, op: str, value: Any):
        self.factor = factor
        self.op = op
        self.value = value

    def apply(self, values: pd.Series) -> pd.Series:
        if self.op == ">":
            return values > self.value
        elif self.op == "<":
            return values < self.value
        elif self.op == ">=":
            return values >= self.value
        elif self.op == "<=":
            return values <= self.value
        return pd.Series(True, index=values.index)

    def __and__(self, other):
        return CombinedFilter(self, other, "and")

    def __or__(self, other):
        return CombinedFilter(self, other, "or")


class CombinedFilter:
    """Combine two filters with AND/OR."""

    def __init__(self, left: Any, right: Any, op: str):
        self.left = left
        self.right = right
        self.op = op

    def apply(self, values_dict: Dict[str, pd.Series]) -> pd.Series:
        left_mask = self._apply_one(self.left, values_dict)
        right_mask = self._apply_one(self.right, values_dict)
        if self.op == "and":
            return left_mask & right_mask
        return left_mask | right_mask

    def _apply_one(self, filt, values_dict):
        if isinstance(filt, ScreenFilter):
            vals = values_dict.get(filt.factor.name, pd.Series(dtype=float))
            return filt.apply(vals)
        elif isinstance(filt, CombinedFilter):
            return filt.apply(values_dict)
        return pd.Series(True, dtype=bool)


class TopFilter:
    def __init__(self, factor: Factor, n: int):
        self.factor = factor
        self.n = n

    def apply(self, values: pd.Series) -> pd.Series:
        threshold = values.nlargest(self.n).min()
        return values >= threshold


class BottomFilter:
    def __init__(self, factor: Factor, n: int):
        self.factor = factor
        self.n = n

    def apply(self, values: pd.Series) -> pd.Series:
        threshold = values.nsmallest(self.n).max()
        return values <= threshold


class PercentileFilter:
    def __init__(self, factor: Factor, low: float, high: float):
        self.factor = factor
        self.low = low
        self.high = high

    def apply(self, values: pd.Series) -> pd.Series:
        lo = np.percentile(values.dropna(), self.low)
        hi = np.percentile(values.dropna(), self.high)
        return (values >= lo) & (values <= hi)


# --- Pipeline ---

class Pipeline:
    """Factor pipeline for systematic universe screening."""

    def __init__(self):
        self._factors: Dict[str, Factor] = {}
        self._screen: Optional[Any] = None

    def add(self, factor: Factor, name: Optional[str] = None) -> None:
        """Add a factor to the pipeline."""
        key = name or factor.name
        self._factors[key] = factor

    def get(self, name: str) -> Factor:
        """Get a factor by name (for building screens)."""
        return self._factors[name]

    def set_screen(self, screen) -> None:
        """Set the screening filter."""
        self._screen = screen

    def run(self, data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Run the pipeline on data.

        Args:
            data: {symbol: {close: [...], volume: [...], ...}}

        Returns:
            DataFrame with symbols as index and factors as columns,
            filtered by screen if set.
        """
        # Compute all factors
        factor_values = {}
        for name, factor in self._factors.items():
            factor_values[name] = factor.compute(data)

        # Build DataFrame
        df = pd.DataFrame(factor_values)
        df = df.dropna(how="all")

        # Apply screen
        if self._screen is not None:
            if isinstance(self._screen, ScreenFilter):
                vals = factor_values.get(self._screen.factor.name, pd.Series(dtype=float))
                mask = self._screen.apply(vals)
                df = df[mask.reindex(df.index, fill_value=False)]
            elif isinstance(self._screen, CombinedFilter):
                mask = self._screen.apply(factor_values)
                df = df[mask.reindex(df.index, fill_value=False)]
            elif isinstance(self._screen, (TopFilter, BottomFilter, PercentileFilter)):
                vals = factor_values.get(self._screen.factor.name, pd.Series(dtype=float))
                mask = self._screen.apply(vals)
                df = df[mask.reindex(df.index, fill_value=False)]

        return df
