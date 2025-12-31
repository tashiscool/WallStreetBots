"""
Universe Selection Models - Ported from QuantConnect LEAN.

Universe Selection determines which securities to trade.

Original: https://github.com/QuantConnect/Lean/tree/master/Algorithm.Framework/Selection
License: Apache 2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum


class UniverseChangeType(Enum):
    """Type of universe change."""
    ADDED = "added"
    REMOVED = "removed"


@dataclass
class UniverseChange:
    """Represents a change in the universe."""
    symbol: str
    change_type: UniverseChangeType
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""

    def __repr__(self) -> str:
        return f"UniverseChange({self.symbol}: {self.change_type.name})"


@dataclass
class SecurityData:
    """Data about a security for universe selection."""
    symbol: str
    price: float = 0.0
    volume: float = 0.0
    market_cap: Optional[float] = None
    sector: str = ""
    industry: str = ""
    pe_ratio: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    avg_volume_30d: Optional[float] = None
    dollar_volume: Optional[float] = None
    is_tradable: bool = True

    @property
    def dollar_volume_calc(self) -> float:
        """Calculate dollar volume."""
        if self.dollar_volume is not None:
            return self.dollar_volume
        return self.price * self.volume


class UniverseSelectionModel(ABC):
    """
    Abstract base class for Universe Selection Models.

    Universe Selection Models determine which securities should be
    included in the trading universe at any given time.
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__
        self._current_universe: Set[str] = set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def current_universe(self) -> Set[str]:
        """Currently selected symbols."""
        return self._current_universe.copy()

    @abstractmethod
    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """
        Select securities for the universe.

        Args:
            securities: Available securities to choose from
            timestamp: Current timestamp

        Returns:
            List of selected symbols
        """
        pass

    def get_changes(self, new_universe: List[str]) -> List[UniverseChange]:
        """Calculate changes from current to new universe."""
        changes = []
        new_set = set(new_universe)

        # Added
        for symbol in new_set - self._current_universe:
            changes.append(UniverseChange(
                symbol=symbol,
                change_type=UniverseChangeType.ADDED,
            ))

        # Removed
        for symbol in self._current_universe - new_set:
            changes.append(UniverseChange(
                symbol=symbol,
                change_type=UniverseChangeType.REMOVED,
            ))

        self._current_universe = new_set
        return changes

    def __repr__(self) -> str:
        return f"{self._name}(size={len(self._current_universe)})"


class ManualUniverseSelectionModel(UniverseSelectionModel):
    """
    Manual Universe Selection Model.

    Uses a fixed list of symbols. Simple and predictable.
    """

    def __init__(self, symbols: List[str], name: Optional[str] = None):
        super().__init__(name or "ManualUniverse")
        self._symbols = list(symbols)

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Return the fixed symbol list."""
        return self._symbols

    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the universe."""
        if symbol not in self._symbols:
            self._symbols.append(symbol)

    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the universe."""
        if symbol in self._symbols:
            self._symbols.remove(symbol)


class ScheduledUniverseSelectionModel(UniverseSelectionModel):
    """
    Scheduled Universe Selection Model.

    Runs selection at specific times (e.g., daily at market open).
    """

    def __init__(self, selection_func: Callable[[List[SecurityData], datetime], List[str]],
                 rebalance_time: time = time(9, 30),
                 rebalance_days: Optional[List[int]] = None,
                 name: Optional[str] = None):
        """
        Args:
            selection_func: Function that performs selection
            rebalance_time: Time of day to rebalance
            rebalance_days: Days of week to rebalance (0=Mon, 4=Fri), None = daily
        """
        super().__init__(name or "ScheduledUniverse")
        self._selection_func = selection_func
        self._rebalance_time = rebalance_time
        self._rebalance_days = rebalance_days
        self._last_rebalance_date: Optional[datetime] = None
        self._cached_selection: List[str] = []

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Select if at rebalance time."""
        current_time = timestamp.time() if hasattr(timestamp, 'time') else time(0, 0)
        current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp

        # Check if we should rebalance
        should_rebalance = False

        if self._last_rebalance_date is None:
            should_rebalance = True
        elif current_date != self._last_rebalance_date:
            # Check day of week
            day_ok = (self._rebalance_days is None or
                      current_date.weekday() in self._rebalance_days)
            # Check time
            time_ok = current_time >= self._rebalance_time
            should_rebalance = day_ok and time_ok

        if should_rebalance:
            self._cached_selection = self._selection_func(securities, timestamp)
            self._last_rebalance_date = current_date

        return self._cached_selection


class QC500UniverseModel(UniverseSelectionModel):
    """
    QC500 Universe Model - Inspired by QuantConnect's QC500.

    Selects top 500 most liquid US equities based on dollar volume.
    Rebalances monthly.
    """

    def __init__(self, num_securities: int = 500,
                 min_price: float = 5.0,
                 min_volume: float = 1000000,
                 name: Optional[str] = None):
        super().__init__(name or f"QC500({num_securities})")
        self._num_securities = num_securities
        self._min_price = min_price
        self._min_volume = min_volume
        self._last_month: Optional[int] = None
        self._cached_selection: List[str] = []

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Select top securities by dollar volume."""
        current_month = timestamp.month

        # Rebalance monthly
        if self._last_month == current_month and self._cached_selection:
            return self._cached_selection

        # Filter by minimum criteria
        filtered = [
            s for s in securities
            if s.is_tradable
            and s.price >= self._min_price
            and s.volume >= self._min_volume
        ]

        # Sort by dollar volume
        filtered.sort(key=lambda s: s.dollar_volume_calc, reverse=True)

        # Select top N
        self._cached_selection = [s.symbol for s in filtered[:self._num_securities]]
        self._last_month = current_month

        return self._cached_selection


class FundamentalUniverseModel(UniverseSelectionModel):
    """
    Fundamental Universe Model.

    Selects securities based on fundamental criteria like P/E ratio,
    market cap, sector, etc.
    """

    def __init__(self,
                 min_market_cap: Optional[float] = None,
                 max_market_cap: Optional[float] = None,
                 sectors: Optional[List[str]] = None,
                 min_pe: Optional[float] = None,
                 max_pe: Optional[float] = None,
                 min_dividend_yield: Optional[float] = None,
                 max_securities: int = 100,
                 name: Optional[str] = None):
        super().__init__(name or "FundamentalUniverse")
        self._min_market_cap = min_market_cap
        self._max_market_cap = max_market_cap
        self._sectors = sectors
        self._min_pe = min_pe
        self._max_pe = max_pe
        self._min_dividend = min_dividend_yield
        self._max_securities = max_securities

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Filter by fundamental criteria."""
        filtered = []

        for s in securities:
            if not s.is_tradable:
                continue

            # Market cap filter
            if self._min_market_cap and s.market_cap is not None:
                if s.market_cap < self._min_market_cap:
                    continue
            if self._max_market_cap and s.market_cap is not None:
                if s.market_cap > self._max_market_cap:
                    continue

            # Sector filter
            if self._sectors and s.sector not in self._sectors:
                continue

            # P/E ratio filter
            if self._min_pe and s.pe_ratio is not None:
                if s.pe_ratio < self._min_pe:
                    continue
            if self._max_pe and s.pe_ratio is not None:
                if s.pe_ratio > self._max_pe:
                    continue

            # Dividend yield filter
            if self._min_dividend and s.dividend_yield is not None:
                if s.dividend_yield < self._min_dividend:
                    continue

            filtered.append(s)

        # Sort by market cap (largest first)
        filtered.sort(
            key=lambda s: s.market_cap if s.market_cap else 0,
            reverse=True
        )

        return [s.symbol for s in filtered[:self._max_securities]]


class LiquidityUniverseModel(UniverseSelectionModel):
    """
    Liquidity Universe Model.

    Selects securities based on liquidity (volume, dollar volume).
    Good for strategies that need easy entry/exit.
    """

    def __init__(self,
                 num_securities: int = 100,
                 min_price: float = 5.0,
                 min_avg_volume: float = 500000,
                 min_dollar_volume: float = 5000000,
                 name: Optional[str] = None):
        super().__init__(name or f"LiquidityUniverse({num_securities})")
        self._num_securities = num_securities
        self._min_price = min_price
        self._min_avg_volume = min_avg_volume
        self._min_dollar_volume = min_dollar_volume

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Select most liquid securities."""
        filtered = [
            s for s in securities
            if s.is_tradable
            and s.price >= self._min_price
            and (s.avg_volume_30d or s.volume) >= self._min_avg_volume
            and s.dollar_volume_calc >= self._min_dollar_volume
        ]

        # Sort by dollar volume
        filtered.sort(key=lambda s: s.dollar_volume_calc, reverse=True)

        return [s.symbol for s in filtered[:self._num_securities]]


class SectorUniverseModel(UniverseSelectionModel):
    """
    Sector Universe Model.

    Selects top securities from each sector for sector rotation.
    """

    def __init__(self,
                 securities_per_sector: int = 5,
                 sectors: Optional[List[str]] = None,
                 min_price: float = 5.0,
                 name: Optional[str] = None):
        super().__init__(name or f"SectorUniverse({securities_per_sector})")
        self._per_sector = securities_per_sector
        self._sectors = sectors
        self._min_price = min_price

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Select top securities per sector."""
        # Group by sector
        by_sector: Dict[str, List[SecurityData]] = {}
        for s in securities:
            if not s.is_tradable or s.price < self._min_price:
                continue
            if self._sectors and s.sector not in self._sectors:
                continue
            if s.sector not in by_sector:
                by_sector[s.sector] = []
            by_sector[s.sector].append(s)

        selected = []
        for sector, sector_securities in by_sector.items():
            # Sort by market cap or dollar volume
            sector_securities.sort(
                key=lambda s: s.market_cap if s.market_cap else s.dollar_volume_calc,
                reverse=True
            )
            selected.extend([s.symbol for s in sector_securities[:self._per_sector]])

        return selected


class MomentumUniverseModel(UniverseSelectionModel):
    """
    Momentum Universe Model.

    Selects securities with strongest momentum (price performance).
    Requires historical price data to be passed in.
    """

    def __init__(self,
                 num_securities: int = 50,
                 lookback_days: int = 126,
                 min_price: float = 5.0,
                 min_volume: float = 500000,
                 name: Optional[str] = None):
        super().__init__(name or f"MomentumUniverse({num_securities})")
        self._num_securities = num_securities
        self._lookback = lookback_days
        self._min_price = min_price
        self._min_volume = min_volume
        self._returns: Dict[str, float] = {}

    def set_return(self, symbol: str, return_pct: float) -> None:
        """Set momentum return for a symbol."""
        self._returns[symbol] = return_pct

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Select highest momentum securities."""
        filtered = [
            s for s in securities
            if s.is_tradable
            and s.price >= self._min_price
            and s.volume >= self._min_volume
            and s.symbol in self._returns
        ]

        # Sort by return
        filtered.sort(key=lambda s: self._returns.get(s.symbol, 0), reverse=True)

        return [s.symbol for s in filtered[:self._num_securities]]


class VolatilityUniverseModel(UniverseSelectionModel):
    """
    Volatility Universe Model.

    Selects securities based on volatility (can select high or low vol).
    """

    def __init__(self,
                 num_securities: int = 50,
                 select_high_volatility: bool = False,
                 min_price: float = 5.0,
                 name: Optional[str] = None):
        super().__init__(name or f"VolatilityUniverse({num_securities})")
        self._num_securities = num_securities
        self._high_vol = select_high_volatility
        self._min_price = min_price
        self._volatilities: Dict[str, float] = {}

    def set_volatility(self, symbol: str, volatility: float) -> None:
        """Set volatility for a symbol."""
        self._volatilities[symbol] = volatility

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Select by volatility."""
        filtered = [
            s for s in securities
            if s.is_tradable
            and s.price >= self._min_price
            and s.symbol in self._volatilities
        ]

        # Sort by volatility
        filtered.sort(
            key=lambda s: self._volatilities.get(s.symbol, 0),
            reverse=self._high_vol
        )

        return [s.symbol for s in filtered[:self._num_securities]]


class ETFConstituentsUniverseModel(UniverseSelectionModel):
    """
    ETF Constituents Universe Model.

    Uses ETF holdings as the universe. Useful for tracking
    SPY, QQQ, or sector ETFs.
    """

    def __init__(self, etf_symbol: str,
                 holdings: Optional[List[str]] = None,
                 name: Optional[str] = None):
        super().__init__(name or f"ETFUniverse({etf_symbol})")
        self._etf_symbol = etf_symbol
        self._holdings = holdings or []

    def set_holdings(self, holdings: List[str]) -> None:
        """Update ETF holdings."""
        self._holdings = holdings

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Return ETF holdings that are tradable."""
        available = {s.symbol for s in securities if s.is_tradable}
        return [h for h in self._holdings if h in available]


class CompositeUniverseModel(UniverseSelectionModel):
    """
    Composite Universe Model.

    Combines multiple universe models. Can use union or intersection.
    """

    def __init__(self, models: List[UniverseSelectionModel],
                 use_intersection: bool = False,
                 name: Optional[str] = None):
        """
        Args:
            models: List of universe models to combine
            use_intersection: If True, symbol must be in ALL models.
                            If False, symbol can be in ANY model.
        """
        super().__init__(name or "CompositeUniverse")
        self._models = models
        self._use_intersection = use_intersection

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Combine selections from all models."""
        if not self._models:
            return []

        # Get selections from each model
        selections = [set(model.select(securities, timestamp))
                     for model in self._models]

        if self._use_intersection:
            # Intersection - must be in all
            result = selections[0]
            for s in selections[1:]:
                result = result.intersection(s)
        else:
            # Union - can be in any
            result = set()
            for s in selections:
                result = result.union(s)

        return list(result)


class NullUniverseModel(UniverseSelectionModel):
    """
    Null Universe Model.

    Returns empty universe. Useful when universe is managed externally.
    """

    def __init__(self):
        super().__init__("Null")

    def select(self, securities: List[SecurityData],
               timestamp: datetime) -> List[str]:
        """Return empty list."""
        return []
