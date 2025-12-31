"""
Base classes for Stock Screener plugins.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SortOrder(Enum):
    """Sort order for screener results."""
    ASCENDING = "asc"
    DESCENDING = "desc"


@dataclass
class StockData:
    """
    Data for a stock being screened.

    Combines price, volume, and fundamental data for screening decisions.
    """
    symbol: str

    # Price data
    price: Decimal
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    prev_close: Optional[Decimal] = None

    # Volume data
    volume: int = 0
    avg_volume: int = 0
    dollar_volume: Optional[Decimal] = None

    # Change metrics
    change_pct: float = 0.0
    gap_pct: float = 0.0

    # Technical indicators
    rsi: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    atr: Optional[float] = None
    volatility: Optional[float] = None

    # Fundamentals
    market_cap: Optional[Decimal] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    earnings_date: Optional[datetime] = None

    # Options data
    has_options: bool = False
    iv_rank: Optional[float] = None
    put_call_ratio: Optional[float] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: Optional[str] = None

    @property
    def is_up(self) -> bool:
        """True if stock is up on the day."""
        return self.change_pct > 0

    @property
    def is_down(self) -> bool:
        """True if stock is down on the day."""
        return self.change_pct < 0

    @property
    def relative_volume(self) -> float:
        """Volume relative to average (RVOL)."""
        if self.avg_volume > 0:
            return self.volume / self.avg_volume
        return 1.0

    @property
    def above_sma_20(self) -> bool:
        """True if price is above 20 SMA."""
        return self.sma_20 is not None and float(self.price) > self.sma_20

    @property
    def above_sma_50(self) -> bool:
        """True if price is above 50 SMA."""
        return self.sma_50 is not None and float(self.price) > self.sma_50

    @property
    def above_sma_200(self) -> bool:
        """True if price is above 200 SMA."""
        return self.sma_200 is not None and float(self.price) > self.sma_200


@dataclass
class ScreenerResult:
    """Result from a screening operation."""
    stocks: List[StockData]
    screener_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of stocks in result."""
        return len(self.stocks)

    @property
    def symbols(self) -> List[str]:
        """List of symbols in result."""
        return [s.symbol for s in self.stocks]


class IScreener(ABC):
    """
    Abstract interface for stock screeners.

    Screeners filter stocks based on specific criteria.
    They can be chained together in a pipeline.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        refresh_interval: int = 60,  # seconds
    ):
        """
        Initialize screener.

        Args:
            name: Screener name (default: class name)
            enabled: Whether screener is active
            refresh_interval: How often to refresh (seconds)
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.refresh_interval = refresh_interval
        self._last_run: Optional[datetime] = None
        self._cached_result: Optional[ScreenerResult] = None

    @abstractmethod
    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """
        Screen stocks based on criteria.

        Args:
            stocks: Input stocks to filter

        Returns:
            Filtered list of stocks
        """
        pass

    def should_refresh(self) -> bool:
        """Check if screener should refresh."""
        if self._last_run is None:
            return True
        elapsed = (datetime.now() - self._last_run).total_seconds()
        return elapsed >= self.refresh_interval

    async def run(
        self,
        stocks: List[StockData],
        force_refresh: bool = False,
    ) -> ScreenerResult:
        """
        Run the screener.

        Args:
            stocks: Input stocks
            force_refresh: Force refresh even if cached

        Returns:
            ScreenerResult with filtered stocks
        """
        if not self.enabled:
            return ScreenerResult(
                stocks=stocks,
                screener_name=self.name,
                metadata={"enabled": False},
            )

        if not force_refresh and not self.should_refresh():
            if self._cached_result:
                return self._cached_result

        filtered = await self.screen(stocks)
        self._last_run = datetime.now()

        result = ScreenerResult(
            stocks=filtered,
            screener_name=self.name,
            metadata={
                "input_count": len(stocks),
                "output_count": len(filtered),
                "filter_ratio": len(filtered) / len(stocks) if stocks else 0,
            }
        )

        self._cached_result = result
        logger.info(
            f"{self.name}: {len(filtered)}/{len(stocks)} stocks passed"
        )

        return result


class CompositeScreener(IScreener):
    """
    Combines multiple screeners with AND logic.

    All screeners must pass for a stock to be included.
    """

    def __init__(
        self,
        screeners: Optional[List[IScreener]] = None,
        name: str = "CompositeScreener",
    ):
        super().__init__(name=name)
        self.screeners = screeners or []

    def add_screener(self, screener: IScreener) -> "CompositeScreener":
        """Add a screener to the composite."""
        self.screeners.append(screener)
        return self

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Apply all screeners in sequence."""
        current = stocks

        for screener in self.screeners:
            if not screener.enabled:
                continue

            current = await screener.screen(current)

            if not current:
                logger.info(f"{screener.name} filtered out all stocks")
                break

        return current


class SortScreener(IScreener):
    """
    Sorts stocks by a specified field.

    Not a filter, but can be used in a pipeline to order results.
    """

    def __init__(
        self,
        sort_by: str = "volume",
        order: SortOrder = SortOrder.DESCENDING,
        limit: Optional[int] = None,
        name: str = "SortScreener",
    ):
        """
        Initialize sort screener.

        Args:
            sort_by: Field to sort by (volume, change_pct, market_cap, etc.)
            order: Sort order (ascending or descending)
            limit: Limit number of results
        """
        super().__init__(name=name)
        self.sort_by = sort_by
        self.order = order
        self.limit = limit

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Sort stocks by specified field."""
        def get_sort_key(stock: StockData):
            value = getattr(stock, self.sort_by, 0)
            if value is None:
                return 0
            if isinstance(value, Decimal):
                return float(value)
            return value

        reverse = self.order == SortOrder.DESCENDING
        sorted_stocks = sorted(stocks, key=get_sort_key, reverse=reverse)

        if self.limit:
            sorted_stocks = sorted_stocks[:self.limit]

        return sorted_stocks
