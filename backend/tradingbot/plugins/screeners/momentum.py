"""
Momentum-based Stock Screeners.

Screen stocks based on price movement and momentum metrics.
"""

from decimal import Decimal
from typing import List, Optional

from .base import IScreener, StockData


class MomentumScreener(IScreener):
    """
    Filter stocks by price change percentage.

    Find gainers or losers based on daily performance.
    """

    def __init__(
        self,
        min_change_pct: Optional[float] = None,
        max_change_pct: Optional[float] = None,
        direction: Optional[str] = None,  # 'up', 'down', or None for both
        name: str = "MomentumScreener",
    ):
        """
        Initialize momentum screener.

        Args:
            min_change_pct: Minimum change percentage (e.g., 5.0 for 5%)
            max_change_pct: Maximum change percentage
            direction: Filter direction ('up', 'down', or None)
        """
        super().__init__(name=name)
        self.min_change_pct = min_change_pct
        self.max_change_pct = max_change_pct
        self.direction = direction

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by momentum."""
        filtered = []

        for stock in stocks:
            # Direction filter
            if self.direction == "up" and stock.change_pct <= 0:
                continue
            if self.direction == "down" and stock.change_pct >= 0:
                continue

            # Use absolute value for range comparison
            change = abs(stock.change_pct)

            if self.min_change_pct is not None and change < self.min_change_pct:
                continue

            if self.max_change_pct is not None and change > self.max_change_pct:
                continue

            filtered.append(stock)

        return filtered


class GapScreener(IScreener):
    """
    Filter stocks by gap percentage (open vs previous close).

    Find gap up or gap down stocks.
    """

    def __init__(
        self,
        min_gap_pct: float = 2.0,
        max_gap_pct: Optional[float] = None,
        direction: Optional[str] = None,  # 'up', 'down', or None
        name: str = "GapScreener",
    ):
        """
        Initialize gap screener.

        Args:
            min_gap_pct: Minimum gap percentage
            max_gap_pct: Maximum gap percentage (optional)
            direction: Gap direction ('up', 'down', or None for both)
        """
        super().__init__(name=name)
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.direction = direction

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by gap."""
        filtered = []

        for stock in stocks:
            gap = stock.gap_pct

            # Direction filter
            if self.direction == "up" and gap <= 0:
                continue
            if self.direction == "down" and gap >= 0:
                continue

            # Use absolute value for range comparison
            abs_gap = abs(gap)

            if abs_gap < self.min_gap_pct:
                continue

            if self.max_gap_pct is not None and abs_gap > self.max_gap_pct:
                continue

            filtered.append(stock)

        return filtered


class BreakoutScreener(IScreener):
    """
    Find stocks breaking out of recent ranges.

    Identifies stocks trading near their highs or lows.
    """

    def __init__(
        self,
        high_threshold: float = 0.98,  # Within 2% of high
        low_threshold: float = 0.02,   # Within 2% of low
        breakout_type: str = "high",   # 'high', 'low', or 'both'
        name: str = "BreakoutScreener",
    ):
        """
        Initialize breakout screener.

        Args:
            high_threshold: Percentage of high to consider breakout (0.98 = 98%)
            low_threshold: Percentage above low to consider breakdown
            breakout_type: Type of breakout to find
        """
        super().__init__(name=name)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.breakout_type = breakout_type

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Find breakout stocks."""
        filtered = []

        for stock in stocks:
            if stock.high is None or stock.low is None:
                continue

            high = float(stock.high)
            low = float(stock.low)
            price = float(stock.price)

            if high == low:
                continue

            # Calculate position in range
            range_position = (price - low) / (high - low)

            is_high_breakout = range_position >= self.high_threshold
            is_low_breakout = range_position <= self.low_threshold

            if self.breakout_type == "high" and is_high_breakout:
                filtered.append(stock)
            elif self.breakout_type == "low" and is_low_breakout:
                filtered.append(stock)
            elif self.breakout_type == "both" and (is_high_breakout or is_low_breakout):
                filtered.append(stock)

        return filtered


class IntradayMomentumScreener(IScreener):
    """
    Filter stocks by intraday momentum.

    Compares current price to open price for intraday moves.
    """

    def __init__(
        self,
        min_intraday_pct: float = 2.0,
        max_intraday_pct: Optional[float] = None,
        direction: Optional[str] = None,
        name: str = "IntradayMomentumScreener",
    ):
        """
        Initialize intraday momentum screener.

        Args:
            min_intraday_pct: Minimum intraday change from open
            max_intraday_pct: Maximum intraday change (optional)
            direction: Direction filter ('up', 'down', or None)
        """
        super().__init__(name=name)
        self.min_intraday_pct = min_intraday_pct
        self.max_intraday_pct = max_intraday_pct
        self.direction = direction

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by intraday momentum."""
        filtered = []

        for stock in stocks:
            if stock.open is None or stock.open == 0:
                continue

            intraday_change = (
                (float(stock.price) - float(stock.open)) / float(stock.open)
            ) * 100

            # Direction filter
            if self.direction == "up" and intraday_change <= 0:
                continue
            if self.direction == "down" and intraday_change >= 0:
                continue

            abs_change = abs(intraday_change)

            if abs_change < self.min_intraday_pct:
                continue

            if self.max_intraday_pct and abs_change > self.max_intraday_pct:
                continue

            filtered.append(stock)

        return filtered


class ReversalScreener(IScreener):
    """
    Find stocks showing reversal patterns.

    Identifies stocks that gapped in one direction but are
    trading in the opposite direction.
    """

    def __init__(
        self,
        min_gap_pct: float = 1.0,
        min_reversal_pct: float = 0.5,
        name: str = "ReversalScreener",
    ):
        """
        Initialize reversal screener.

        Args:
            min_gap_pct: Minimum gap to consider
            min_reversal_pct: Minimum reversal from open
        """
        super().__init__(name=name)
        self.min_gap_pct = min_gap_pct
        self.min_reversal_pct = min_reversal_pct

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Find reversal stocks."""
        filtered = []

        for stock in stocks:
            if stock.open is None or stock.open == 0:
                continue

            gap = stock.gap_pct
            intraday_change = (
                (float(stock.price) - float(stock.open)) / float(stock.open)
            ) * 100

            # Check for reversal (gap and intraday move in opposite directions)
            is_gap_up_reversal = (
                gap >= self.min_gap_pct and
                intraday_change <= -self.min_reversal_pct
            )

            is_gap_down_reversal = (
                gap <= -self.min_gap_pct and
                intraday_change >= self.min_reversal_pct
            )

            if is_gap_up_reversal or is_gap_down_reversal:
                filtered.append(stock)

        return filtered

