"""
Fundamental Analysis-based Stock Screeners.

Screen stocks based on fundamental metrics and company data.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Set

from .base import IScreener, StockData


class FundamentalScreener(IScreener):
    """
    Multi-criteria fundamental screener.

    Combines market cap, P/E, and other fundamental filters.
    """

    def __init__(
        self,
        min_market_cap: Optional[Decimal] = None,
        max_market_cap: Optional[Decimal] = None,
        min_pe_ratio: Optional[float] = None,
        max_pe_ratio: Optional[float] = None,
        sectors: Optional[List[str]] = None,
        exclude_sectors: Optional[List[str]] = None,
        industries: Optional[List[str]] = None,
        require_dividend: bool = False,
        min_dividend_yield: Optional[float] = None,
        name: str = "FundamentalScreener",
    ):
        """
        Initialize fundamental screener.

        Args:
            min_market_cap/max_market_cap: Market cap range
            min_pe_ratio/max_pe_ratio: P/E ratio range
            sectors: Allowed sectors (None = all)
            exclude_sectors: Sectors to exclude
            industries: Allowed industries (None = all)
            require_dividend: Require dividend-paying stocks
            min_dividend_yield: Minimum dividend yield
        """
        super().__init__(name=name)
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.min_pe_ratio = min_pe_ratio
        self.max_pe_ratio = max_pe_ratio
        self.sectors = set(sectors) if sectors else None
        self.exclude_sectors = set(exclude_sectors) if exclude_sectors else None
        self.industries = set(industries) if industries else None
        self.require_dividend = require_dividend
        self.min_dividend_yield = min_dividend_yield

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by fundamentals."""
        filtered = []

        for stock in stocks:
            # Market cap filter
            if stock.market_cap is not None:
                if self.min_market_cap and stock.market_cap < self.min_market_cap:
                    continue
                if self.max_market_cap and stock.market_cap > self.max_market_cap:
                    continue

            # P/E filter
            if stock.pe_ratio is not None:
                if self.min_pe_ratio and stock.pe_ratio < self.min_pe_ratio:
                    continue
                if self.max_pe_ratio and stock.pe_ratio > self.max_pe_ratio:
                    continue

            # Sector filter
            if self.sectors and stock.sector not in self.sectors:
                continue

            if self.exclude_sectors and stock.sector in self.exclude_sectors:
                continue

            # Industry filter
            if self.industries and stock.industry not in self.industries:
                continue

            # Dividend filter
            if self.require_dividend:
                if stock.dividend_yield is None or stock.dividend_yield <= 0:
                    continue

            if self.min_dividend_yield is not None:
                if stock.dividend_yield is None:
                    continue
                if stock.dividend_yield < self.min_dividend_yield:
                    continue

            filtered.append(stock)

        return filtered


class MarketCapScreener(IScreener):
    """
    Filter stocks by market capitalization.

    Supports tier-based or range-based filtering.
    """

    # Market cap tiers (USD)
    TIERS = {
        "mega": (Decimal("200_000_000_000"), None),      # $200B+
        "large": (Decimal("10_000_000_000"), Decimal("200_000_000_000")),   # $10B-$200B
        "mid": (Decimal("2_000_000_000"), Decimal("10_000_000_000")),       # $2B-$10B
        "small": (Decimal("300_000_000"), Decimal("2_000_000_000")),        # $300M-$2B
        "micro": (Decimal("50_000_000"), Decimal("300_000_000")),           # $50M-$300M
        "nano": (None, Decimal("50_000_000")),           # <$50M
    }

    def __init__(
        self,
        tier: Optional[str] = None,  # mega, large, mid, small, micro, nano
        min_market_cap: Optional[Decimal] = None,
        max_market_cap: Optional[Decimal] = None,
        name: str = "MarketCapScreener",
    ):
        """
        Initialize market cap screener.

        Args:
            tier: Market cap tier (overrides min/max)
            min_market_cap: Minimum market cap
            max_market_cap: Maximum market cap
        """
        super().__init__(name=name)

        if tier and tier in self.TIERS:
            self.min_market_cap, self.max_market_cap = self.TIERS[tier]
        else:
            self.min_market_cap = min_market_cap
            self.max_market_cap = max_market_cap

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by market cap."""
        filtered = []

        for stock in stocks:
            if stock.market_cap is None:
                continue

            if self.min_market_cap and stock.market_cap < self.min_market_cap:
                continue
            if self.max_market_cap and stock.market_cap > self.max_market_cap:
                continue

            filtered.append(stock)

        return filtered


class EarningsScreener(IScreener):
    """
    Filter stocks by earnings-related criteria.

    Find stocks with upcoming earnings or specific P/E ranges.
    """

    def __init__(
        self,
        earnings_within_days: Optional[int] = None,  # Earnings in next N days
        avoid_earnings_days: Optional[int] = None,   # Avoid if earnings in N days
        min_pe: Optional[float] = None,
        max_pe: Optional[float] = None,
        require_positive_pe: bool = False,
        name: str = "EarningsScreener",
    ):
        """
        Initialize earnings screener.

        Args:
            earnings_within_days: Include if earnings within N days
            avoid_earnings_days: Exclude if earnings within N days
            min_pe/max_pe: P/E ratio range
            require_positive_pe: Only include profitable companies
        """
        super().__init__(name=name)
        self.earnings_within_days = earnings_within_days
        self.avoid_earnings_days = avoid_earnings_days
        self.min_pe = min_pe
        self.max_pe = max_pe
        self.require_positive_pe = require_positive_pe

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by earnings criteria."""
        filtered = []
        now = datetime.now()

        for stock in stocks:
            # Earnings date filter
            if stock.earnings_date is not None:
                days_until = (stock.earnings_date - now).days

                if self.earnings_within_days is not None:
                    if days_until < 0 or days_until > self.earnings_within_days:
                        continue

                if self.avoid_earnings_days is not None:
                    if 0 <= days_until <= self.avoid_earnings_days:
                        continue

            # P/E filters
            if self.require_positive_pe:
                if stock.pe_ratio is None or stock.pe_ratio <= 0:
                    continue

            if stock.pe_ratio is not None:
                if self.min_pe is not None and stock.pe_ratio < self.min_pe:
                    continue
                if self.max_pe is not None and stock.pe_ratio > self.max_pe:
                    continue

            filtered.append(stock)

        return filtered


class DividendScreener(IScreener):
    """
    Filter stocks by dividend characteristics.

    Find dividend-paying stocks with specific yields.
    """

    def __init__(
        self,
        min_yield: Optional[float] = None,
        max_yield: Optional[float] = None,
        require_dividend: bool = True,
        exclude_dividend: bool = False,
        name: str = "DividendScreener",
    ):
        """
        Initialize dividend screener.

        Args:
            min_yield: Minimum dividend yield (e.g., 2.0 for 2%)
            max_yield: Maximum dividend yield
            require_dividend: Only include dividend payers
            exclude_dividend: Exclude dividend payers
        """
        super().__init__(name=name)
        self.min_yield = min_yield
        self.max_yield = max_yield
        self.require_dividend = require_dividend
        self.exclude_dividend = exclude_dividend

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by dividend criteria."""
        filtered = []

        for stock in stocks:
            has_dividend = (
                stock.dividend_yield is not None and stock.dividend_yield > 0
            )

            # Dividend requirement filters
            if self.require_dividend and not has_dividend:
                continue

            if self.exclude_dividend and has_dividend:
                continue

            # Yield range filters
            if has_dividend:
                if self.min_yield and stock.dividend_yield < self.min_yield:
                    continue
                if self.max_yield and stock.dividend_yield > self.max_yield:
                    continue

            filtered.append(stock)

        return filtered


class SectorScreener(IScreener):
    """
    Filter stocks by sector or industry.

    GICS sector classification support.
    """

    # GICS Sectors
    GICS_SECTORS = {
        "technology": "Information Technology",
        "healthcare": "Health Care",
        "financials": "Financials",
        "consumer_discretionary": "Consumer Discretionary",
        "consumer_staples": "Consumer Staples",
        "industrials": "Industrials",
        "energy": "Energy",
        "utilities": "Utilities",
        "real_estate": "Real Estate",
        "materials": "Materials",
        "communication": "Communication Services",
    }

    def __init__(
        self,
        sectors: Optional[List[str]] = None,
        exclude_sectors: Optional[List[str]] = None,
        industries: Optional[List[str]] = None,
        name: str = "SectorScreener",
    ):
        """
        Initialize sector screener.

        Args:
            sectors: Allowed sectors
            exclude_sectors: Sectors to exclude
            industries: Allowed industries
        """
        super().__init__(name=name)

        # Normalize sector names
        self.sectors = self._normalize_sectors(sectors) if sectors else None
        self.exclude_sectors = (
            self._normalize_sectors(exclude_sectors)
            if exclude_sectors
            else None
        )
        self.industries = set(industries) if industries else None

    def _normalize_sectors(self, sectors: List[str]) -> Set[str]:
        """Normalize sector names to GICS format."""
        normalized = set()
        for sector in sectors:
            lower = sector.lower().replace(" ", "_")
            if lower in self.GICS_SECTORS:
                normalized.add(self.GICS_SECTORS[lower])
            else:
                normalized.add(sector)
        return normalized

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by sector/industry."""
        filtered = []

        for stock in stocks:
            # Sector filter
            if self.sectors:
                if stock.sector not in self.sectors:
                    continue

            if self.exclude_sectors:
                if stock.sector in self.exclude_sectors:
                    continue

            # Industry filter
            if self.industries:
                if stock.industry not in self.industries:
                    continue

            filtered.append(stock)

        return filtered


class OptionsScreener(IScreener):
    """
    Filter stocks by options characteristics.

    Find stocks with options, high IV rank, etc.
    """

    def __init__(
        self,
        require_options: bool = True,
        min_iv_rank: Optional[float] = None,
        max_iv_rank: Optional[float] = None,
        min_put_call_ratio: Optional[float] = None,
        max_put_call_ratio: Optional[float] = None,
        name: str = "OptionsScreener",
    ):
        """
        Initialize options screener.

        Args:
            require_options: Only include optionable stocks
            min_iv_rank/max_iv_rank: IV rank range (0-100)
            min_put_call_ratio/max_put_call_ratio: P/C ratio range
        """
        super().__init__(name=name)
        self.require_options = require_options
        self.min_iv_rank = min_iv_rank
        self.max_iv_rank = max_iv_rank
        self.min_put_call_ratio = min_put_call_ratio
        self.max_put_call_ratio = max_put_call_ratio

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by options criteria."""
        filtered = []

        for stock in stocks:
            # Options availability
            if self.require_options and not stock.has_options:
                continue

            # IV rank filter
            if stock.iv_rank is not None:
                if self.min_iv_rank and stock.iv_rank < self.min_iv_rank:
                    continue
                if self.max_iv_rank and stock.iv_rank > self.max_iv_rank:
                    continue

            # Put/Call ratio filter
            if stock.put_call_ratio is not None:
                if self.min_put_call_ratio:
                    if stock.put_call_ratio < self.min_put_call_ratio:
                        continue
                if self.max_put_call_ratio:
                    if stock.put_call_ratio > self.max_put_call_ratio:
                        continue

            filtered.append(stock)

        return filtered

