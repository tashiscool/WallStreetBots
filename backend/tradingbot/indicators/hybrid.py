"""
Hybrid Indicators

Volume Profile, Market Profile, VWMA, McGinley Dynamic, ZigZag.
Inspired by Jesse and Backtrader indicator libraries.
"""

from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import Indicator, IndicatorDataPoint, TradeBar, WindowIndicator


class VolumeWeightedMovingAverage(Indicator):
    """VWMA: Volume-weighted moving average."""

    def __init__(self, period: int = 20):
        super().__init__("VWMA", period)
        self._price_vol = deque(maxlen=period)
        self._volumes = deque(maxlen=period)

    def update_bar(self, bar: TradeBar) -> bool:
        self._price_vol.append(bar.close * bar.volume)
        self._volumes.append(bar.volume)
        return self.update(bar.time, bar.close)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        if len(self._volumes) < self._period:
            return 0.0
        total_vol = sum(self._volumes)
        if total_vol == 0:
            return input_point.value
        return sum(self._price_vol) / total_vol


class McGinleyDynamic(Indicator):
    """McGinley Dynamic: adaptive moving average that adjusts to market speed."""

    def __init__(self, period: int = 14):
        super().__init__("McGinleyDynamic", period)
        self._md = 0.0

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        price = input_point.value
        if self._samples == 1:
            self._md = price
            return price

        if self._md == 0:
            self._md = price
            return price

        ratio = price / self._md
        self._md += (price - self._md) / (self._period * ratio ** 4)
        return self._md


class ZigZag(Indicator):
    """ZigZag: identifies significant price swings filtering out noise."""

    def __init__(self, deviation: float = 5.0):
        super().__init__("ZigZag", period=1)
        self._deviation = deviation / 100.0
        self._last_pivot = 0.0
        self._last_pivot_type = 0  # 1=high, -1=low
        self._highest_since = 0.0
        self._lowest_since = float("inf")

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        price = input_point.value

        if self._samples == 1:
            self._last_pivot = price
            self._highest_since = price
            self._lowest_since = price
            return 0.0

        self._highest_since = max(self._highest_since, price)
        self._lowest_since = min(self._lowest_since, price)

        if self._last_pivot_type >= 0:
            # Looking for a low
            if price <= self._highest_since * (1 - self._deviation):
                self._last_pivot = self._highest_since
                self._last_pivot_type = -1
                self._lowest_since = price
                return -1.0  # Swing high confirmed
        if self._last_pivot_type <= 0:
            # Looking for a high
            if price >= self._lowest_since * (1 + self._deviation):
                self._last_pivot = self._lowest_since
                self._last_pivot_type = 1
                self._highest_since = price
                return 1.0  # Swing low confirmed

        return 0.0


class VolumeProfile(Indicator):
    """Volume Profile: distribution of volume at price levels."""

    def __init__(self, period: int = 50, n_bins: int = 20):
        super().__init__("VolumeProfile", period)
        self._n_bins = n_bins
        self._prices = deque(maxlen=period)
        self._volumes = deque(maxlen=period)
        self.poc: float = 0.0  # Point of Control (highest volume price)
        self.value_area_high: float = 0.0
        self.value_area_low: float = 0.0

    def update_bar(self, bar: TradeBar) -> bool:
        self._prices.append(bar.close)
        self._volumes.append(bar.volume)
        return self.update(bar.time, bar.close)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        if len(self._prices) < self._period:
            return 0.0

        prices = np.array(self._prices)
        volumes = np.array(self._volumes)

        # Create volume-at-price histogram
        price_min, price_max = prices.min(), prices.max()
        if price_max == price_min:
            self.poc = price_min
            self.value_area_high = price_max
            self.value_area_low = price_min
            return self.poc

        bins = np.linspace(price_min, price_max, self._n_bins + 1)
        indices = np.digitize(prices, bins) - 1
        indices = np.clip(indices, 0, self._n_bins - 1)

        vol_at_price = np.zeros(self._n_bins)
        for i, vol in zip(indices, volumes):
            vol_at_price[i] += vol

        # Point of Control
        poc_idx = np.argmax(vol_at_price)
        self.poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # Value Area (70% of volume)
        total_vol = vol_at_price.sum()
        target = total_vol * 0.7
        sorted_idx = np.argsort(vol_at_price)[::-1]
        cumulative = 0.0
        va_indices = []
        for idx in sorted_idx:
            cumulative += vol_at_price[idx]
            va_indices.append(idx)
            if cumulative >= target:
                break

        va_min = min(va_indices)
        va_max = max(va_indices)
        self.value_area_low = bins[va_min]
        self.value_area_high = bins[va_max + 1]

        return self.poc


class MarketProfile(Indicator):
    """Market Profile: time-price opportunity analysis (TPO)."""

    def __init__(self, period: int = 30, n_levels: int = 20):
        super().__init__("MarketProfile", period)
        self._n_levels = n_levels
        self._bars: List[TradeBar] = []
        self.poc: float = 0.0
        self.value_area_high: float = 0.0
        self.value_area_low: float = 0.0

    def update_bar(self, bar: TradeBar) -> bool:
        self._bars.append(bar)
        if len(self._bars) > self._period:
            self._bars = self._bars[-self._period:]
        return self.update(bar.time, bar.close)

    def _compute_next_value(self, input_point: IndicatorDataPoint) -> float:
        if len(self._bars) < self._period:
            return 0.0

        # Find price range
        all_highs = [b.high for b in self._bars]
        all_lows = [b.low for b in self._bars]
        price_min = min(all_lows)
        price_max = max(all_highs)

        if price_max == price_min:
            self.poc = price_min
            return self.poc

        # Build TPO (time at price level)
        levels = np.linspace(price_min, price_max, self._n_levels + 1)
        tpo_count = np.zeros(self._n_levels)

        for bar in self._bars:
            for i in range(self._n_levels):
                if bar.low <= levels[i + 1] and bar.high >= levels[i]:
                    tpo_count[i] += 1

        # POC
        poc_idx = np.argmax(tpo_count)
        self.poc = (levels[poc_idx] + levels[poc_idx + 1]) / 2

        # Value Area (70%)
        total_tpo = tpo_count.sum()
        target = total_tpo * 0.7
        sorted_idx = np.argsort(tpo_count)[::-1]
        cumulative = 0.0
        va_indices = []
        for idx in sorted_idx:
            cumulative += tpo_count[idx]
            va_indices.append(idx)
            if cumulative >= target:
                break

        va_min = min(va_indices)
        va_max = max(va_indices)
        self.value_area_low = levels[va_min]
        self.value_area_high = levels[va_max + 1]

        return self.poc
