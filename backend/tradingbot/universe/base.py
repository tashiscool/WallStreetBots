"""
Base classes for Universe Selection.

Ported from QuantConnect/LEAN, adapted for stock/options trading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class SecurityType(Enum):
    """Types of securities."""
    EQUITY = "equity"
    OPTION = "option"
    ETF = "etf"
    INDEX = "index"


@dataclass
class SecurityData:
    """
    Data for a security used in universe selection.

    This is the equivalent of LEAN's CoarseFundamental/FineFundamental.
    """
    symbol: str
    security_type: SecurityType = SecurityType.EQUITY

    # Price data (coarse)
    price: Optional[Decimal] = None
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    volume: Optional[int] = None
    dollar_volume: Optional[Decimal] = None

    # Fundamental data (fine)
    market_cap: Optional[Decimal] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[Decimal] = None

    # Technical indicators
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi_14: Optional[float] = None
    atr_14: Optional[float] = None
    volatility_30d: Optional[float] = None

    # Options-specific
    has_options: bool = False
    option_volume: Optional[int] = None
    implied_volatility: Optional[float] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: Optional[str] = None

    def __post_init__(self):
        """Convert types if needed."""
        if self.price is not None and not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if self.dollar_volume is not None and not isinstance(self.dollar_volume, Decimal):
            self.dollar_volume = Decimal(str(self.dollar_volume))
        if self.market_cap is not None and not isinstance(self.market_cap, Decimal):
            self.market_cap = Decimal(str(self.market_cap))


@dataclass
class UniverseSelectionResult:
    """Result of a universe selection operation."""
    symbols: List[str]
    added: Set[str] = field(default_factory=set)
    removed: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)
    filter_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of symbols in the result."""
        return len(self.symbols)


class IUniverseSelectionModel(ABC):
    """
    Base interface for universe selection models.

    Inspired by LEAN's IUniverseSelectionModel.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the selection model.

        Args:
            name: Name for this filter (for logging/debugging)
        """
        self.name = name or self.__class__.__name__
        self._previous_selection: Set[str] = set()

    @abstractmethod
    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """
        Select securities that pass this filter.

        Args:
            securities: List of security data to filter
            current_date: Date of selection (for point-in-time)

        Returns:
            List of selected symbols
        """
        pass

    def get_changes(self, new_selection: List[str]) -> UniverseSelectionResult:
        """
        Compare new selection with previous and return changes.

        Args:
            new_selection: New list of symbols

        Returns:
            UniverseSelectionResult with added/removed symbols
        """
        new_set = set(new_selection)
        added = new_set - self._previous_selection
        removed = self._previous_selection - new_set

        result = UniverseSelectionResult(
            symbols=new_selection,
            added=added,
            removed=removed,
            filter_name=self.name,
        )

        self._previous_selection = new_set
        return result


class CompositeUniverseSelection(IUniverseSelectionModel):
    """
    Combines multiple selection models with AND logic.

    A symbol must pass ALL filters to be selected.
    """

    def __init__(
        self,
        filters: Optional[List[IUniverseSelectionModel]] = None,
        name: str = "CompositeFilter",
    ):
        super().__init__(name)
        self.filters = filters or []

    def add_filter(self, filter_model: IUniverseSelectionModel) -> None:
        """Add a filter to the composite."""
        self.filters.append(filter_model)

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """
        Apply all filters in sequence.

        Args:
            securities: List of security data to filter
            current_date: Date of selection

        Returns:
            List of symbols that pass ALL filters
        """
        if not self.filters:
            return [s.symbol for s in securities]

        # Start with all securities
        current_securities = securities

        for filter_model in self.filters:
            # Get symbols that pass this filter
            selected = await filter_model.select(current_securities, current_date)
            selected_set = set(selected)

            # Filter securities for next iteration
            current_securities = [s for s in current_securities if s.symbol in selected_set]

            if not current_securities:
                logger.info(f"Filter {filter_model.name} eliminated all securities")
                break

        return [s.symbol for s in current_securities]


class UnionUniverseSelection(IUniverseSelectionModel):
    """
    Combines multiple selection models with OR logic.

    A symbol passes if it passes ANY filter.
    """

    def __init__(
        self,
        filters: Optional[List[IUniverseSelectionModel]] = None,
        name: str = "UnionFilter",
    ):
        super().__init__(name)
        self.filters = filters or []

    def add_filter(self, filter_model: IUniverseSelectionModel) -> None:
        """Add a filter to the union."""
        self.filters.append(filter_model)

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """
        Apply all filters and union results.

        Args:
            securities: List of security data to filter
            current_date: Date of selection

        Returns:
            List of symbols that pass ANY filter
        """
        if not self.filters:
            return [s.symbol for s in securities]

        selected_set: Set[str] = set()

        for filter_model in self.filters:
            selected = await filter_model.select(securities, current_date)
            selected_set.update(selected)

        # Return in original order
        return [s.symbol for s in securities if s.symbol in selected_set]


class ScheduledUniverseSelection(IUniverseSelectionModel):
    """
    Universe selection that only runs on specific days/times.

    Useful for reducing computation by not re-selecting every tick.
    """

    def __init__(
        self,
        inner_filter: IUniverseSelectionModel,
        selection_days: Optional[List[int]] = None,  # 0=Monday, 6=Sunday
        selection_hour: int = 9,  # Default to market open
        selection_minute: int = 30,
        name: str = "ScheduledFilter",
    ):
        super().__init__(name)
        self.inner_filter = inner_filter
        self.selection_days = selection_days or [0]  # Default to Monday only
        self.selection_hour = selection_hour
        self.selection_minute = selection_minute
        self._last_selection: List[str] = []
        self._last_run_date: Optional[date] = None

    async def select(
        self,
        securities: List[SecurityData],
        current_date: Optional[date] = None,
    ) -> List[str]:
        """
        Select securities only on scheduled days.

        Args:
            securities: List of security data to filter
            current_date: Date of selection

        Returns:
            Last selection if not scheduled day, new selection otherwise
        """
        today = current_date or date.today()
        now = datetime.now()

        # Check if it's time to run selection
        is_selection_day = today.weekday() in self.selection_days
        is_past_selection_time = (
            now.hour > self.selection_hour or
            (now.hour == self.selection_hour and now.minute >= self.selection_minute)
        )

        if is_selection_day and is_past_selection_time and self._last_run_date != today:
            self._last_selection = await self.inner_filter.select(securities, current_date)
            self._last_run_date = today
            logger.info(
                f"Scheduled selection ran, selected {len(self._last_selection)} symbols"
            )

        return self._last_selection
