"""
Coarse Universe Selection Filters.

Fast filters based on price, volume, and basic data.
These are designed to quickly narrow down the universe before
applying more expensive fine selection filters.
"""

from datetime import date
from decimal import Decimal
from typing import List, Optional
import logging

from .base import IUniverseSelectionModel, SecurityData

logger = logging.getLogger(__name__)


class CoarseSelectionFilter(IUniverseSelectionModel):
    """
    Generic coarse selection filter.

    Combines multiple coarse criteria into a single filter.
    """

    def __init__(
        self,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        min_volume: Optional[int] = None,
        min_dollar_volume: Optional[Decimal] = None,
        has_fundamental_data: bool = False,
        name: str = "CoarseFilter",
    ):
        super().__init__(name)
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.min_dollar_volume = min_dollar_volume
        self.has_fundamental_data = has_fundamental_data

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter securities based on coarse criteria."""
        selected = []

        for security in securities:
            # Price filter
            if self.min_price is not None:
                if security.price is None or security.price < self.min_price:
                    continue

            if self.max_price is not None:
                if security.price is None or security.price > self.max_price:
                    continue

            # Volume filter
            if self.min_volume is not None:
                if security.volume is None or security.volume < self.min_volume:
                    continue

            # Dollar volume filter
            if self.min_dollar_volume is not None:
                if security.dollar_volume is None or security.dollar_volume < self.min_dollar_volume:
                    continue

            # Fundamental data check
            if self.has_fundamental_data:
                if security.market_cap is None:
                    continue

            selected.append(security.symbol)

        logger.debug(
            f"{self.name}: {len(selected)}/{len(securities)} securities passed"
        )
        return selected


class VolumeUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by trading volume.

    Can select by minimum volume or top N by volume.
    """

    def __init__(
        self,
        min_volume: Optional[int] = None,
        top_n: Optional[int] = None,
        name: str = "VolumeFilter",
    ):
        """
        Initialize volume filter.

        Args:
            min_volume: Minimum average daily volume
            top_n: Select top N by volume (after min_volume filter if both set)
        """
        super().__init__(name)
        self.min_volume = min_volume
        self.top_n = top_n

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by volume."""
        # Filter by minimum volume
        filtered = securities
        if self.min_volume is not None:
            filtered = [s for s in filtered if s.volume and s.volume >= self.min_volume]

        # Sort by volume and take top N
        if self.top_n is not None:
            filtered = sorted(
                filtered,
                key=lambda s: s.volume or 0,
                reverse=True,
            )[:self.top_n]

        return [s.symbol for s in filtered]


class PriceUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by price range.

    Useful for filtering out penny stocks or very expensive stocks.
    """

    def __init__(
        self,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        name: str = "PriceFilter",
    ):
        """
        Initialize price filter.

        Args:
            min_price: Minimum price (e.g., $5 to avoid penny stocks)
            max_price: Maximum price
        """
        super().__init__(name)
        self.min_price = Decimal(str(min_price)) if min_price else None
        self.max_price = Decimal(str(max_price)) if max_price else None

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by price range."""
        selected = []

        for s in securities:
            if s.price is None:
                continue

            if self.min_price is not None and s.price < self.min_price:
                continue

            if self.max_price is not None and s.price > self.max_price:
                continue

            selected.append(s.symbol)

        return selected


class DollarVolumeUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by dollar volume (price * volume).

    Dollar volume is a better liquidity measure than volume alone
    as it accounts for share price.
    """

    def __init__(
        self,
        min_dollar_volume: Optional[Decimal] = None,
        top_n: Optional[int] = None,
        lookback_days: int = 20,
        name: str = "DollarVolumeFilter",
    ):
        """
        Initialize dollar volume filter.

        Args:
            min_dollar_volume: Minimum average daily dollar volume (e.g., $10M)
            top_n: Select top N by dollar volume
            lookback_days: Days to average volume over
        """
        super().__init__(name)
        self.min_dollar_volume = Decimal(str(min_dollar_volume)) if min_dollar_volume else None
        self.top_n = top_n
        self.lookback_days = lookback_days

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by dollar volume."""
        # Calculate dollar volume if not provided
        for s in securities:
            if s.dollar_volume is None and s.price and s.volume:
                s.dollar_volume = s.price * s.volume

        # Filter by minimum dollar volume
        filtered = securities
        if self.min_dollar_volume is not None:
            filtered = [
                s for s in filtered
                if s.dollar_volume and s.dollar_volume >= self.min_dollar_volume
            ]

        # Sort by dollar volume and take top N
        if self.top_n is not None:
            filtered = sorted(
                filtered,
                key=lambda s: s.dollar_volume or Decimal("0"),
                reverse=True,
            )[:self.top_n]

        return [s.symbol for s in filtered]


class SpreadUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by bid-ask spread.

    Tight spreads indicate good liquidity and lower trading costs.
    """

    def __init__(
        self,
        max_spread_percent: float = 0.5,
        name: str = "SpreadFilter",
    ):
        """
        Initialize spread filter.

        Args:
            max_spread_percent: Maximum spread as percentage of price (e.g., 0.5%)
        """
        super().__init__(name)
        self.max_spread_percent = max_spread_percent

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by spread."""
        # Note: This requires bid/ask data which may not be in SecurityData
        # For now, this is a placeholder that passes all securities
        # Real implementation would need market data integration
        logger.warning(
            f"{self.name}: Spread filtering requires market data integration"
        )
        return [s.symbol for s in securities]


class OptionsableUniverseSelection(IUniverseSelectionModel):
    """
    Select only securities that have options available.

    Critical for options trading strategies.
    """

    def __init__(
        self,
        min_option_volume: Optional[int] = None,
        require_weeklies: bool = False,
        name: str = "OptionsableFilter",
    ):
        """
        Initialize optionable filter.

        Args:
            min_option_volume: Minimum daily option volume
            require_weeklies: Require weekly options availability
        """
        super().__init__(name)
        self.min_option_volume = min_option_volume
        self.require_weeklies = require_weeklies

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter to optionable securities."""
        filtered = [s for s in securities if s.has_options]

        if self.min_option_volume is not None:
            filtered = [
                s for s in filtered
                if s.option_volume and s.option_volume >= self.min_option_volume
            ]

        return [s.symbol for s in filtered]


class ETFUniverseSelection(IUniverseSelectionModel):
    """
    Select only ETFs or exclude ETFs.
    """

    def __init__(
        self,
        include_etfs: bool = True,
        etf_only: bool = False,
        name: str = "ETFFilter",
    ):
        """
        Initialize ETF filter.

        Args:
            include_etfs: Include ETFs in selection
            etf_only: Select only ETFs (ignore include_etfs if True)
        """
        super().__init__(name)
        self.include_etfs = include_etfs
        self.etf_only = etf_only

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by ETF status."""
        from .base import SecurityType

        if self.etf_only:
            return [s.symbol for s in securities if s.security_type == SecurityType.ETF]

        if not self.include_etfs:
            return [s.symbol for s in securities if s.security_type != SecurityType.ETF]

        return [s.symbol for s in securities]
