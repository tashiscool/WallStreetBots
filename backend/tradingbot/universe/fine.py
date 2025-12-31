"""
Fine Universe Selection Filters.

Detailed filters based on fundamentals, sector, and company data.
These are more expensive to compute and should be used after
coarse filters have narrowed down the universe.
"""

from datetime import date
from decimal import Decimal
from typing import List, Optional, Set
import logging

from .base import IUniverseSelectionModel, SecurityData

logger = logging.getLogger(__name__)


class FineSelectionFilter(IUniverseSelectionModel):
    """
    Generic fine selection filter.

    Combines multiple fundamental criteria into a single filter.
    """

    def __init__(
        self,
        min_market_cap: Optional[Decimal] = None,
        max_market_cap: Optional[Decimal] = None,
        sectors: Optional[List[str]] = None,
        exclude_sectors: Optional[List[str]] = None,
        min_pe: Optional[float] = None,
        max_pe: Optional[float] = None,
        min_dividend_yield: Optional[float] = None,
        name: str = "FineFilter",
    ):
        super().__init__(name)
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.sectors = set(s.lower() for s in sectors) if sectors else None
        self.exclude_sectors = set(s.lower() for s in exclude_sectors) if exclude_sectors else None
        self.min_pe = min_pe
        self.max_pe = max_pe
        self.min_dividend_yield = min_dividend_yield

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter securities based on fine criteria."""
        selected = []

        for security in securities:
            # Market cap filter
            if self.min_market_cap is not None:
                if security.market_cap is None or security.market_cap < self.min_market_cap:
                    continue

            if self.max_market_cap is not None:
                if security.market_cap is None or security.market_cap > self.max_market_cap:
                    continue

            # Sector filter
            if self.sectors is not None:
                if security.sector is None or security.sector.lower() not in self.sectors:
                    continue

            if self.exclude_sectors is not None:
                if security.sector and security.sector.lower() in self.exclude_sectors:
                    continue

            # P/E ratio filter
            if self.min_pe is not None:
                if security.pe_ratio is None or security.pe_ratio < self.min_pe:
                    continue

            if self.max_pe is not None:
                if security.pe_ratio is None or security.pe_ratio > self.max_pe:
                    continue

            # Dividend yield filter
            if self.min_dividend_yield is not None:
                if security.dividend_yield is None or security.dividend_yield < self.min_dividend_yield:
                    continue

            selected.append(security.symbol)

        logger.debug(
            f"{self.name}: {len(selected)}/{len(securities)} securities passed"
        )
        return selected


class MarketCapUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by market capitalization.

    Can select by range or top/bottom N by market cap.
    """

    # Market cap tiers
    MEGA_CAP = Decimal("200e9")      # $200B+
    LARGE_CAP = Decimal("10e9")      # $10B-$200B
    MID_CAP = Decimal("2e9")         # $2B-$10B
    SMALL_CAP = Decimal("300e6")     # $300M-$2B
    MICRO_CAP = Decimal("50e6")      # $50M-$300M

    def __init__(
        self,
        min_market_cap: Optional[Decimal] = None,
        max_market_cap: Optional[Decimal] = None,
        tiers: Optional[List[str]] = None,  # ["mega", "large", "mid", "small", "micro"]
        top_n: Optional[int] = None,
        bottom_n: Optional[int] = None,
        name: str = "MarketCapFilter",
    ):
        """
        Initialize market cap filter.

        Args:
            min_market_cap: Minimum market cap
            max_market_cap: Maximum market cap
            tiers: List of tiers to include (mega, large, mid, small, micro)
            top_n: Select top N by market cap
            bottom_n: Select bottom N by market cap
        """
        super().__init__(name)
        self.min_market_cap = Decimal(str(min_market_cap)) if min_market_cap else None
        self.max_market_cap = Decimal(str(max_market_cap)) if max_market_cap else None
        self.tiers = [t.lower() for t in tiers] if tiers else None
        self.top_n = top_n
        self.bottom_n = bottom_n

    def _get_tier(self, market_cap: Decimal) -> str:
        """Get tier name for a market cap value."""
        if market_cap >= self.MEGA_CAP:
            return "mega"
        elif market_cap >= self.LARGE_CAP:
            return "large"
        elif market_cap >= self.MID_CAP:
            return "mid"
        elif market_cap >= self.SMALL_CAP:
            return "small"
        elif market_cap >= self.MICRO_CAP:
            return "micro"
        else:
            return "nano"

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by market cap."""
        # Filter by range
        filtered = []
        for s in securities:
            if s.market_cap is None:
                continue

            if self.min_market_cap is not None and s.market_cap < self.min_market_cap:
                continue

            if self.max_market_cap is not None and s.market_cap > self.max_market_cap:
                continue

            if self.tiers is not None:
                tier = self._get_tier(s.market_cap)
                if tier not in self.tiers:
                    continue

            filtered.append(s)

        # Sort by market cap
        filtered = sorted(filtered, key=lambda s: s.market_cap or 0, reverse=True)

        # Take top N
        if self.top_n is not None:
            filtered = filtered[:self.top_n]

        # Take bottom N
        if self.bottom_n is not None:
            filtered = filtered[-self.bottom_n:]

        return [s.symbol for s in filtered]


class SectorUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by sector.

    Uses GICS sectors or custom sector definitions.
    """

    # Standard GICS sectors
    GICS_SECTORS = {
        "technology": ["Information Technology", "Technology"],
        "healthcare": ["Health Care", "Healthcare"],
        "financials": ["Financials", "Financial Services"],
        "consumer_discretionary": ["Consumer Discretionary"],
        "consumer_staples": ["Consumer Staples"],
        "industrials": ["Industrials"],
        "energy": ["Energy"],
        "utilities": ["Utilities"],
        "materials": ["Materials", "Basic Materials"],
        "real_estate": ["Real Estate"],
        "communication": ["Communication Services", "Telecommunications"],
    }

    def __init__(
        self,
        include_sectors: Optional[List[str]] = None,
        exclude_sectors: Optional[List[str]] = None,
        max_per_sector: Optional[int] = None,
        name: str = "SectorFilter",
    ):
        """
        Initialize sector filter.

        Args:
            include_sectors: Sectors to include (normalized names)
            exclude_sectors: Sectors to exclude
            max_per_sector: Maximum securities per sector
        """
        super().__init__(name)
        self.include_sectors = set(s.lower() for s in include_sectors) if include_sectors else None
        self.exclude_sectors = set(s.lower() for s in exclude_sectors) if exclude_sectors else None
        self.max_per_sector = max_per_sector

    def _normalize_sector(self, sector: str) -> str:
        """Normalize sector name to standard form."""
        sector_lower = sector.lower()
        for normalized, variants in self.GICS_SECTORS.items():
            if sector_lower == normalized or sector in variants:
                return normalized
        return sector_lower

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by sector."""
        sector_counts: dict = {}
        selected = []

        for s in securities:
            if s.sector is None:
                continue

            normalized = self._normalize_sector(s.sector)

            # Check include/exclude lists
            if self.include_sectors is not None and normalized not in self.include_sectors:
                continue

            if self.exclude_sectors is not None and normalized in self.exclude_sectors:
                continue

            # Check per-sector limit
            if self.max_per_sector is not None:
                current_count = sector_counts.get(normalized, 0)
                if current_count >= self.max_per_sector:
                    continue
                sector_counts[normalized] = current_count + 1

            selected.append(s.symbol)

        return selected


class FundamentalUniverseSelection(IUniverseSelectionModel):
    """
    Select securities by fundamental metrics.

    Supports value, growth, quality, and dividend factors.
    """

    def __init__(
        self,
        # Value metrics
        min_pe: Optional[float] = None,
        max_pe: Optional[float] = None,
        # Growth metrics
        min_revenue_growth: Optional[float] = None,
        min_eps_growth: Optional[float] = None,
        # Quality metrics
        min_profit_margin: Optional[float] = None,
        min_roe: Optional[float] = None,
        # Dividend metrics
        min_dividend_yield: Optional[float] = None,
        max_payout_ratio: Optional[float] = None,
        # Debt metrics
        max_debt_to_equity: Optional[float] = None,
        name: str = "FundamentalFilter",
    ):
        super().__init__(name)
        self.min_pe = min_pe
        self.max_pe = max_pe
        self.min_revenue_growth = min_revenue_growth
        self.min_eps_growth = min_eps_growth
        self.min_profit_margin = min_profit_margin
        self.min_roe = min_roe
        self.min_dividend_yield = min_dividend_yield
        self.max_payout_ratio = max_payout_ratio
        self.max_debt_to_equity = max_debt_to_equity

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Filter by fundamental metrics."""
        selected = []

        for s in securities:
            # P/E filter
            if self.min_pe is not None:
                if s.pe_ratio is None or s.pe_ratio < self.min_pe:
                    continue
            if self.max_pe is not None:
                if s.pe_ratio is None or s.pe_ratio > self.max_pe:
                    continue

            # Dividend filter
            if self.min_dividend_yield is not None:
                if s.dividend_yield is None or s.dividend_yield < self.min_dividend_yield:
                    continue

            # Note: Other fundamental metrics would need to be added to SecurityData
            # For now, we filter on what's available

            selected.append(s.symbol)

        return selected


class DividendAristocratsSelection(IUniverseSelectionModel):
    """
    Select dividend aristocrats (25+ years of dividend increases).

    These are high-quality, stable companies ideal for income strategies.
    """

    # Well-known dividend aristocrats (sample list)
    DIVIDEND_ARISTOCRATS = {
        "JNJ", "PG", "KO", "PEP", "MCD", "MMM", "ABT", "ADP", "AFL",
        "APD", "BDX", "BEN", "CAH", "CINF", "CLX", "CTAS", "CVX",
        "DOV", "ECL", "ED", "EMR", "ESS", "FRT", "GD", "GPC", "GWW",
        "HRL", "ITW", "KMB", "LEG", "LIN", "LOW", "MKC", "NDSN", "NEE",
        "NUE", "O", "PBCT", "PNR", "PPG", "ROP", "SBUX", "SHW", "SPGI",
        "SWK", "SYY", "T", "TGT", "TROW", "VFC", "WBA", "WMT", "XOM",
    }

    def __init__(
        self,
        custom_list: Optional[Set[str]] = None,
        name: str = "DividendAristocrats",
    ):
        super().__init__(name)
        self.aristocrats = custom_list or self.DIVIDEND_ARISTOCRATS

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Select only dividend aristocrats."""
        return [s.symbol for s in securities if s.symbol in self.aristocrats]


class SP500UniverseSelection(IUniverseSelectionModel):
    """
    Select only S&P 500 components.

    Uses a static list or integrates with point-in-time membership.
    """

    # Sample S&P 500 list (top holdings by weight)
    SP500_SAMPLE = {
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK.B",
        "TSLA", "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX",
        "MRK", "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "MCD", "TMO",
        "WMT", "CSCO", "ACN", "ABT", "DHR", "BAC", "CRM", "PFE", "ADBE",
        "NKE", "DIS", "NFLX", "CMCSA", "VZ", "INTC", "TXN", "PM", "NEE",
        "UPS", "RTX", "BMY", "MS", "HON", "QCOM", "UNP", "T", "ORCL",
        "LOW", "IBM", "AMD", "SPGI", "GS", "ELV", "AMAT", "SBUX", "INTU",
        "CAT", "DE", "BLK", "GILD", "PLD", "AXP", "MDT", "CVS", "AMT",
    }

    def __init__(
        self,
        use_point_in_time: bool = False,
        custom_list: Optional[Set[str]] = None,
        name: str = "SP500Filter",
    ):
        super().__init__(name)
        self.use_point_in_time = use_point_in_time
        self.custom_list = custom_list or self.SP500_SAMPLE

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """Select only S&P 500 components."""
        return [s.symbol for s in securities if s.symbol in self.custom_list]
