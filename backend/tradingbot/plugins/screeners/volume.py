"""
Volume-based Stock Screeners.

Screen stocks based on trading volume metrics.
"""

from decimal import Decimal
from typing import List, Optional

from .base import IScreener, StockData


class VolumeScreener(IScreener):
    """
    Filter stocks by absolute trading volume.

    Use to find highly traded stocks.
    """

    def __init__(
        self,
        min_volume: int = 100_000,
        max_volume: Optional[int] = None,
        name: str = "VolumeScreener",
    ):
        """
        Initialize volume screener.

        Args:
            min_volume: Minimum trading volume
            max_volume: Maximum trading volume (optional)
        """
        super().__init__(name=name)
        self.min_volume = min_volume
        self.max_volume = max_volume

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by volume."""
        filtered = []

        for stock in stocks:
            if stock.volume < self.min_volume:
                continue

            if self.max_volume and stock.volume > self.max_volume:
                continue

            filtered.append(stock)

        return filtered


class DollarVolumeScreener(IScreener):
    """
    Filter stocks by dollar volume (price * volume).

    Better liquidity measure than volume alone.
    """

    def __init__(
        self,
        min_dollar_volume: Decimal = Decimal("1000000"),  # $1M
        max_dollar_volume: Optional[Decimal] = None,
        name: str = "DollarVolumeScreener",
    ):
        """
        Initialize dollar volume screener.

        Args:
            min_dollar_volume: Minimum dollar volume
            max_dollar_volume: Maximum dollar volume (optional)
        """
        super().__init__(name=name)
        self.min_dollar_volume = min_dollar_volume
        self.max_dollar_volume = max_dollar_volume

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by dollar volume."""
        filtered = []

        for stock in stocks:
            # Calculate dollar volume if not provided
            dv = stock.dollar_volume
            if dv is None:
                dv = stock.price * stock.volume

            if dv < self.min_dollar_volume:
                continue

            if self.max_dollar_volume and dv > self.max_dollar_volume:
                continue

            filtered.append(stock)

        return filtered


class RelativeVolumeScreener(IScreener):
    """
    Filter stocks by relative volume (RVOL).

    RVOL = Current Volume / Average Volume
    High RVOL indicates unusual activity.
    """

    def __init__(
        self,
        min_rvol: float = 1.5,  # 150% of average volume
        max_rvol: Optional[float] = None,
        name: str = "RelativeVolumeScreener",
    ):
        """
        Initialize relative volume screener.

        Args:
            min_rvol: Minimum RVOL (1.5 = 150% of average)
            max_rvol: Maximum RVOL (optional)
        """
        super().__init__(name=name)
        self.min_rvol = min_rvol
        self.max_rvol = max_rvol

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by relative volume."""
        filtered = []

        for stock in stocks:
            rvol = stock.relative_volume

            if rvol < self.min_rvol:
                continue

            if self.max_rvol and rvol > self.max_rvol:
                continue

            filtered.append(stock)

        return filtered


class VolumeSpike(IScreener):
    """
    Find stocks with sudden volume spikes.

    Useful for finding unusual activity that might precede moves.
    """

    def __init__(
        self,
        spike_multiplier: float = 3.0,  # 3x average volume
        min_volume: int = 50_000,  # Minimum to avoid low-float noise
        name: str = "VolumeSpike",
    ):
        """
        Initialize volume spike screener.

        Args:
            spike_multiplier: Required multiple of average volume
            min_volume: Minimum absolute volume
        """
        super().__init__(name=name)
        self.spike_multiplier = spike_multiplier
        self.min_volume = min_volume

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Find volume spikes."""
        filtered = []

        for stock in stocks:
            if stock.volume < self.min_volume:
                continue

            if stock.relative_volume >= self.spike_multiplier:
                filtered.append(stock)

        return filtered
