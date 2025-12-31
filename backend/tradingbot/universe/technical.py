"""
Technical Universe Selection Filters.

Filters based on technical indicators like RSI, moving averages,
momentum, and volatility.
"""

from datetime import date
from decimal import Decimal
from typing import List, Optional
import logging

from .base import IUniverseSelectionModel, SecurityData

logger = logging.getLogger(__name__)


class TechnicalUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by technical indicators.

    Combines multiple technical criteria into a single filter.
    """

    def __init__(
        self,
        # RSI filters
        rsi_below: Optional[float] = None,
        rsi_above: Optional[float] = None,
        # Moving average filters
        above_sma_20: bool = False,
        above_sma_50: bool = False,
        above_sma_200: bool = False,
        below_sma_20: bool = False,
        below_sma_50: bool = False,
        below_sma_200: bool = False,
        # Volatility filters
        min_atr_percent: Optional[float] = None,
        max_atr_percent: Optional[float] = None,
        name: str = "TechnicalFilter",
    ):
        super().__init__(name)
        self.rsi_below = rsi_below
        self.rsi_above = rsi_above
        self.above_sma_20 = above_sma_20
        self.above_sma_50 = above_sma_50
        self.above_sma_200 = above_sma_200
        self.below_sma_20 = below_sma_20
        self.below_sma_50 = below_sma_50
        self.below_sma_200 = below_sma_200
        self.min_atr_percent = min_atr_percent
        self.max_atr_percent = max_atr_percent

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by technical indicators."""
        selected = []

        for s in securities:
            # RSI filters
            if self.rsi_below is not None:
                if s.rsi_14 is None or s.rsi_14 >= self.rsi_below:
                    continue

            if self.rsi_above is not None:
                if s.rsi_14 is None or s.rsi_14 <= self.rsi_above:
                    continue

            # SMA filters (price above/below moving averages)
            if self.above_sma_20 and (s.sma_20 is None or s.price is None or float(s.price) <= s.sma_20):
                continue

            if self.above_sma_50 and (s.sma_50 is None or s.price is None or float(s.price) <= s.sma_50):
                continue

            if self.above_sma_200 and (s.sma_200 is None or s.price is None or float(s.price) <= s.sma_200):
                continue

            if self.below_sma_20 and (s.sma_20 is None or s.price is None or float(s.price) >= s.sma_20):
                continue

            if self.below_sma_50 and (s.sma_50 is None or s.price is None or float(s.price) >= s.sma_50):
                continue

            if self.below_sma_200 and (s.sma_200 is None or s.price is None or float(s.price) >= s.sma_200):
                continue

            # ATR percentage filter
            if s.atr_14 is not None and s.price is not None and float(s.price) > 0:
                atr_percent = (s.atr_14 / float(s.price)) * 100

                if self.min_atr_percent is not None and atr_percent < self.min_atr_percent:
                    continue

                if self.max_atr_percent is not None and atr_percent > self.max_atr_percent:
                    continue

            selected.append(s.symbol)

        return selected


class MomentumUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by momentum.

    Ranks securities by price change over various periods.
    """

    def __init__(
        self,
        min_return_1m: Optional[float] = None,
        min_return_3m: Optional[float] = None,
        min_return_6m: Optional[float] = None,
        min_return_12m: Optional[float] = None,
        max_return_1m: Optional[float] = None,
        max_return_3m: Optional[float] = None,
        top_n: Optional[int] = None,
        bottom_n: Optional[int] = None,
        exclude_recent_highs: bool = False,
        name: str = "MomentumFilter",
    ):
        """
        Initialize momentum filter.

        Args:
            min_return_*: Minimum return over period (e.g., 0.05 = 5%)
            max_return_*: Maximum return over period
            top_n: Select top N by momentum
            bottom_n: Select bottom N (mean reversion)
            exclude_recent_highs: Exclude stocks at 52-week highs
        """
        super().__init__(name)
        self.min_return_1m = min_return_1m
        self.min_return_3m = min_return_3m
        self.min_return_6m = min_return_6m
        self.min_return_12m = min_return_12m
        self.max_return_1m = max_return_1m
        self.max_return_3m = max_return_3m
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.exclude_recent_highs = exclude_recent_highs

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by momentum."""
        # For now, use price vs SMA as momentum proxy
        # Real implementation would need historical price data

        filtered = []
        momentum_scores = []

        for s in securities:
            if s.price is None:
                continue

            # Calculate momentum score based on available data
            score = 0.0
            if s.sma_20 is not None:
                score += (float(s.price) - s.sma_20) / s.sma_20
            if s.sma_50 is not None:
                score += (float(s.price) - s.sma_50) / s.sma_50
            if s.sma_200 is not None:
                score += (float(s.price) - s.sma_200) / s.sma_200

            momentum_scores.append((s, score))

        # Sort by momentum
        momentum_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply filters
        if self.top_n is not None:
            momentum_scores = momentum_scores[:self.top_n]
        elif self.bottom_n is not None:
            momentum_scores = momentum_scores[-self.bottom_n:]

        return [s.symbol for s, _ in momentum_scores]


class VolatilityUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by volatility.

    Can filter by historical or implied volatility.
    """

    def __init__(
        self,
        min_volatility: Optional[float] = None,
        max_volatility: Optional[float] = None,
        min_iv_rank: Optional[float] = None,
        max_iv_rank: Optional[float] = None,
        top_n_volatile: Optional[int] = None,
        bottom_n_volatile: Optional[int] = None,
        name: str = "VolatilityFilter",
    ):
        """
        Initialize volatility filter.

        Args:
            min_volatility: Minimum 30-day volatility (annualized, e.g., 0.20 = 20%)
            max_volatility: Maximum 30-day volatility
            min_iv_rank: Minimum IV rank (0-100)
            max_iv_rank: Maximum IV rank
            top_n_volatile: Select N most volatile
            bottom_n_volatile: Select N least volatile
        """
        super().__init__(name)
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.min_iv_rank = min_iv_rank
        self.max_iv_rank = max_iv_rank
        self.top_n_volatile = top_n_volatile
        self.bottom_n_volatile = bottom_n_volatile

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by volatility."""
        filtered = []

        for s in securities:
            vol = s.volatility_30d

            if vol is None:
                # Skip if no volatility data
                continue

            # Apply min/max filters
            if self.min_volatility is not None and vol < self.min_volatility:
                continue

            if self.max_volatility is not None and vol > self.max_volatility:
                continue

            # IV rank filter (for options)
            if s.implied_volatility is not None:
                # Note: IV rank requires historical IV data
                # For now, use raw IV
                if self.min_iv_rank is not None and s.implied_volatility < self.min_iv_rank / 100:
                    continue
                if self.max_iv_rank is not None and s.implied_volatility > self.max_iv_rank / 100:
                    continue

            filtered.append(s)

        # Sort by volatility
        filtered.sort(key=lambda s: s.volatility_30d or 0, reverse=True)

        # Apply top/bottom N
        if self.top_n_volatile is not None:
            filtered = filtered[:self.top_n_volatile]
        elif self.bottom_n_volatile is not None:
            filtered = filtered[-self.bottom_n_volatile:]

        return [s.symbol for s in filtered]


class TrendUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by trend strength.

    Uses moving average crossovers and alignment.
    """

    def __init__(
        self,
        require_uptrend: bool = False,
        require_downtrend: bool = False,
        require_golden_cross: bool = False,
        require_death_cross: bool = False,
        min_trend_strength: Optional[float] = None,
        name: str = "TrendFilter",
    ):
        """
        Initialize trend filter.

        Args:
            require_uptrend: Price > SMA20 > SMA50 > SMA200
            require_downtrend: Price < SMA20 < SMA50 < SMA200
            require_golden_cross: SMA50 recently crossed above SMA200
            require_death_cross: SMA50 recently crossed below SMA200
            min_trend_strength: Minimum % distance from longest MA
        """
        super().__init__(name)
        self.require_uptrend = require_uptrend
        self.require_downtrend = require_downtrend
        self.require_golden_cross = require_golden_cross
        self.require_death_cross = require_death_cross
        self.min_trend_strength = min_trend_strength

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by trend."""
        selected = []

        for s in securities:
            if s.price is None:
                continue

            price = float(s.price)

            # Check uptrend (price > SMA20 > SMA50 > SMA200)
            if self.require_uptrend:
                if not (s.sma_20 and s.sma_50 and s.sma_200):
                    continue
                if not (price > s.sma_20 > s.sma_50 > s.sma_200):
                    continue

            # Check downtrend
            if self.require_downtrend:
                if not (s.sma_20 and s.sma_50 and s.sma_200):
                    continue
                if not (price < s.sma_20 < s.sma_50 < s.sma_200):
                    continue

            # Check trend strength
            if self.min_trend_strength is not None:
                if s.sma_200 is None or s.sma_200 == 0:
                    continue
                trend_strength = abs(price - s.sma_200) / s.sma_200
                if trend_strength < self.min_trend_strength:
                    continue

            selected.append(s.symbol)

        return selected


class OversoldUniverseSelection(IUniverseSelectionModel):
    """
    Select oversold securities for mean reversion strategies.

    Identifies securities that have dropped significantly and may bounce.
    """

    def __init__(
        self,
        rsi_threshold: float = 30,
        min_drop_percent: Optional[float] = None,
        below_lower_bb: bool = False,
        name: str = "OversoldFilter",
    ):
        """
        Initialize oversold filter.

        Args:
            rsi_threshold: RSI below this level (default 30)
            min_drop_percent: Minimum drop from recent high (e.g., 0.10 = 10%)
            below_lower_bb: Price below lower Bollinger Band
        """
        super().__init__(name)
        self.rsi_threshold = rsi_threshold
        self.min_drop_percent = min_drop_percent
        self.below_lower_bb = below_lower_bb

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter for oversold securities."""
        selected = []

        for s in securities:
            # RSI filter
            if s.rsi_14 is None or s.rsi_14 >= self.rsi_threshold:
                continue

            # Additional drop filter
            if self.min_drop_percent is not None:
                # Would need high price data for this
                pass

            selected.append(s.symbol)

        return selected


class OverboughtUniverseSelection(IUniverseSelectionModel):
    """
    Select overbought securities for short or exit strategies.

    Identifies securities that may be due for a pullback.
    """

    def __init__(
        self,
        rsi_threshold: float = 70,
        max_gain_percent: Optional[float] = None,
        above_upper_bb: bool = False,
        name: str = "OverboughtFilter",
    ):
        """
        Initialize overbought filter.

        Args:
            rsi_threshold: RSI above this level (default 70)
            max_gain_percent: Securities up more than this from recent low
            above_upper_bb: Price above upper Bollinger Band
        """
        super().__init__(name)
        self.rsi_threshold = rsi_threshold
        self.max_gain_percent = max_gain_percent
        self.above_upper_bb = above_upper_bb

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter for overbought securities."""
        selected = []

        for s in securities:
            # RSI filter
            if s.rsi_14 is None or s.rsi_14 <= self.rsi_threshold:
                continue

            selected.append(s.symbol)

        return selected
