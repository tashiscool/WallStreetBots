"""
Universe Manager - Coordinates universe selection.

Provides a high-level interface for building and managing
dynamic stock universes with multiple selection criteria.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio
import logging

from .base import (
    IUniverseSelectionModel,
    UniverseSelectionResult,
    SecurityData,
    CompositeUniverseSelection,
)

logger = logging.getLogger(__name__)


class UniverseManager:
    """
    Manages universe selection and changes.

    Coordinates coarse and fine selection filters, tracks changes,
    and provides callbacks for universe updates.

    Usage:
        manager = UniverseManager(data_client)
        manager.add_filter(VolumeUniverseSelection(min_volume=1_000_000))
        manager.add_filter(MarketCapUniverseSelection(min_market_cap=1e9))

        # Get selected symbols
        symbols = await manager.select()

        # Or with change tracking
        result = await manager.select_with_changes()
        print(f"Added: {result.added}, Removed: {result.removed}")
    """

    def __init__(
        self,
        data_client=None,
        selection_interval: timedelta = timedelta(days=1),
        cache_duration: timedelta = timedelta(hours=1),
    ):
        """
        Initialize Universe Manager.

        Args:
            data_client: Data client for fetching security data
            selection_interval: How often to re-run selection
            cache_duration: How long to cache security data
        """
        self.data_client = data_client
        self.selection_interval = selection_interval
        self.cache_duration = cache_duration

        self._composite = CompositeUniverseSelection()
        self._previous_selection: Set[str] = set()
        self._last_selection_time: Optional[datetime] = None
        self._cached_result: Optional[UniverseSelectionResult] = None

        # Callbacks
        self._on_securities_changed: List[Callable] = []
        self._on_security_added: List[Callable] = []
        self._on_security_removed: List[Callable] = []

        # Base universe (starting symbols to filter from)
        self._base_universe: Optional[List[str]] = None

    def set_base_universe(self, symbols: List[str]) -> None:
        """
        Set the base universe of symbols to filter from.

        If not set, will try to get all symbols from data client.

        Args:
            symbols: List of ticker symbols
        """
        self._base_universe = symbols
        logger.info(f"Base universe set with {len(symbols)} symbols")

    def add_filter(self, filter_model: IUniverseSelectionModel) -> "UniverseManager":
        """
        Add a filter to the selection pipeline.

        Filters are applied in the order they are added (AND logic).

        Args:
            filter_model: Filter to add

        Returns:
            self for chaining
        """
        self._composite.add_filter(filter_model)
        logger.info(f"Added filter: {filter_model.name}")
        return self

    def clear_filters(self) -> "UniverseManager":
        """Remove all filters."""
        self._composite.filters.clear()
        return self

    def on_securities_changed(self, callback: Callable) -> "UniverseManager":
        """
        Register callback for when securities change.

        Callback receives: (added: Set[str], removed: Set[str])
        """
        self._on_securities_changed.append(callback)
        return self

    def on_security_added(self, callback: Callable) -> "UniverseManager":
        """
        Register callback for when a security is added.

        Callback receives: symbol: str
        """
        self._on_security_added.append(callback)
        return self

    def on_security_removed(self, callback: Callable) -> "UniverseManager":
        """
        Register callback for when a security is removed.

        Callback receives: symbol: str
        """
        self._on_security_removed.append(callback)
        return self

    async def _fetch_securities_data(
        self,
        symbols: Optional[List[str]] = None,
    ) -> List[SecurityData]:
        """
        Fetch security data for universe selection.

        Args:
            symbols: Specific symbols to fetch, or None for all

        Returns:
            List of SecurityData objects
        """
        target_symbols = symbols or self._base_universe

        if target_symbols is None:
            if self.data_client is not None:
                # Try to get all tradeable symbols
                try:
                    target_symbols = await self.data_client.get_all_symbols()
                except Exception as e:
                    logger.error(f"Failed to get symbols from data client: {e}")
                    return []
            else:
                logger.warning("No base universe or data client configured")
                return []

        securities = []

        for symbol in target_symbols:
            try:
                # Get basic price/volume data
                data = SecurityData(symbol=symbol)

                if self.data_client is not None:
                    # Fetch price data
                    quote = await self.data_client.get_quote(symbol)
                    if quote:
                        data.price = Decimal(str(quote.get("price", 0)))
                        data.volume = quote.get("volume")
                        if data.price and data.volume:
                            data.dollar_volume = data.price * data.volume

                    # Fetch fundamental data if available
                    fundamentals = await self.data_client.get_fundamentals(symbol)
                    if fundamentals:
                        data.market_cap = Decimal(str(fundamentals.get("market_cap", 0)))
                        data.sector = fundamentals.get("sector")
                        data.industry = fundamentals.get("industry")
                        data.pe_ratio = fundamentals.get("pe_ratio")
                        data.dividend_yield = fundamentals.get("dividend_yield")

                    # Fetch technical indicators if available
                    technicals = await self.data_client.get_technicals(symbol)
                    if technicals:
                        data.sma_20 = technicals.get("sma_20")
                        data.sma_50 = technicals.get("sma_50")
                        data.sma_200 = technicals.get("sma_200")
                        data.rsi_14 = technicals.get("rsi_14")
                        data.atr_14 = technicals.get("atr_14")
                        data.volatility_30d = technicals.get("volatility_30d")

                securities.append(data)

            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue

        logger.info(f"Fetched data for {len(securities)} securities")
        return securities

    async def select(
        self,
        current_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> List[str]:
        """
        Run universe selection and return selected symbols.

        Args:
            current_date: Date for selection (default: today)
            force_refresh: Force re-selection even if cached

        Returns:
            List of selected symbols
        """
        result = await self.select_with_changes(current_date, force_refresh)
        return result.symbols

    async def select_with_changes(
        self,
        current_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> UniverseSelectionResult:
        """
        Run universe selection and return result with changes.

        Args:
            current_date: Date for selection (default: today)
            force_refresh: Force re-selection even if cached

        Returns:
            UniverseSelectionResult with symbols and changes
        """
        now = datetime.now()

        # Check if we can use cached result
        if (
            not force_refresh
            and self._cached_result is not None
            and self._last_selection_time is not None
            and now - self._last_selection_time < self.selection_interval
        ):
            return self._cached_result

        # Fetch security data
        securities = await self._fetch_securities_data()

        if not securities:
            logger.warning("No securities data available for selection")
            return UniverseSelectionResult(symbols=[], filter_name="UniverseManager")

        # Run selection
        selected = await self._composite.select(securities, current_date)

        # Calculate changes
        new_set = set(selected)
        added = new_set - self._previous_selection
        removed = self._previous_selection - new_set

        result = UniverseSelectionResult(
            symbols=selected,
            added=added,
            removed=removed,
            filter_name="UniverseManager",
            metadata={
                "total_securities": len(securities),
                "filters_applied": len(self._composite.filters),
            }
        )

        # Update state
        self._previous_selection = new_set
        self._last_selection_time = now
        self._cached_result = result

        # Fire callbacks
        await self._fire_callbacks(added, removed)

        logger.info(
            f"Universe selection complete: {len(selected)} symbols "
            f"(+{len(added)} added, -{len(removed)} removed)"
        )

        return result

    async def _fire_callbacks(self, added: Set[str], removed: Set[str]) -> None:
        """Fire registered callbacks for universe changes."""
        if added or removed:
            for callback in self._on_securities_changed:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(added, removed)
                    else:
                        callback(added, removed)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        for symbol in added:
            for callback in self._on_security_added:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol)
                    else:
                        callback(symbol)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        for symbol in removed:
            for callback in self._on_security_removed:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol)
                    else:
                        callback(symbol)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    @property
    def current_universe(self) -> List[str]:
        """Get current universe (from last selection)."""
        return list(self._previous_selection)

    @property
    def filter_count(self) -> int:
        """Number of filters in the pipeline."""
        return len(self._composite.filters)

    def __len__(self) -> int:
        """Return size of current universe."""
        return len(self._previous_selection)


class PresetUniverseManager:
    """
    Factory for common universe presets.

    Provides pre-configured universe managers for common use cases.
    """

    @staticmethod
    def liquid_large_caps(data_client=None) -> UniverseManager:
        """
        Create a universe of liquid large-cap stocks.

        - Market cap > $10B
        - Daily dollar volume > $10M
        - Price > $5
        """
        from .coarse import DollarVolumeUniverseSelection, PriceUniverseSelection
        from .fine import MarketCapUniverseSelection

        manager = UniverseManager(data_client)
        manager.add_filter(PriceUniverseSelection(min_price=Decimal("5")))
        manager.add_filter(DollarVolumeUniverseSelection(min_dollar_volume=Decimal("10000000")))
        manager.add_filter(MarketCapUniverseSelection(min_market_cap=Decimal("10e9")))
        return manager

    @staticmethod
    def high_momentum(data_client=None, top_n: int = 50) -> UniverseManager:
        """
        Create a universe of high-momentum stocks.

        - Liquid stocks (dollar volume > $5M)
        - Strong uptrend (above all major SMAs)
        - Top N by momentum score
        """
        from .coarse import DollarVolumeUniverseSelection
        from .technical import MomentumUniverseSelection, TrendUniverseSelection

        manager = UniverseManager(data_client)
        manager.add_filter(DollarVolumeUniverseSelection(min_dollar_volume=Decimal("5000000")))
        manager.add_filter(TrendUniverseSelection(require_uptrend=True))
        manager.add_filter(MomentumUniverseSelection(top_n=top_n))
        return manager

    @staticmethod
    def oversold_bounces(data_client=None) -> UniverseManager:
        """
        Create a universe of oversold stocks for mean reversion.

        - Liquid stocks
        - RSI < 30
        - Large caps only (for safety)
        """
        from .coarse import DollarVolumeUniverseSelection
        from .fine import MarketCapUniverseSelection
        from .technical import OversoldUniverseSelection

        manager = UniverseManager(data_client)
        manager.add_filter(DollarVolumeUniverseSelection(min_dollar_volume=Decimal("5000000")))
        manager.add_filter(MarketCapUniverseSelection(min_market_cap=Decimal("2e9")))
        manager.add_filter(OversoldUniverseSelection(rsi_threshold=30))
        return manager

    @staticmethod
    def dividend_income(data_client=None) -> UniverseManager:
        """
        Create a universe for dividend income strategies.

        - Dividend aristocrats
        - Or high-yield stocks with decent fundamentals
        """
        from .coarse import DollarVolumeUniverseSelection
        from .fine import DividendAristocratsSelection, FundamentalUniverseSelection

        manager = UniverseManager(data_client)
        manager.add_filter(DollarVolumeUniverseSelection(min_dollar_volume=Decimal("1000000")))
        manager.add_filter(DividendAristocratsSelection())
        return manager

    @staticmethod
    def options_universe(data_client=None) -> UniverseManager:
        """
        Create a universe suitable for options trading.

        - Must have options
        - Sufficient option volume
        - Liquid underlying
        """
        from .coarse import DollarVolumeUniverseSelection, OptionsableUniverseSelection

        manager = UniverseManager(data_client)
        manager.add_filter(DollarVolumeUniverseSelection(min_dollar_volume=Decimal("5000000")))
        manager.add_filter(OptionsableUniverseSelection(min_option_volume=1000))
        return manager

    @staticmethod
    def sp500(data_client=None) -> UniverseManager:
        """
        Create universe of S&P 500 stocks.
        """
        from .fine import SP500UniverseSelection

        manager = UniverseManager(data_client)
        manager.add_filter(SP500UniverseSelection())
        return manager

    @staticmethod
    def tech_growth(data_client=None) -> UniverseManager:
        """
        Create universe of technology growth stocks.

        - Technology sector
        - Market cap > $1B
        - Strong momentum
        """
        from .coarse import DollarVolumeUniverseSelection
        from .fine import MarketCapUniverseSelection, SectorUniverseSelection
        from .technical import TrendUniverseSelection

        manager = UniverseManager(data_client)
        manager.add_filter(DollarVolumeUniverseSelection(min_dollar_volume=Decimal("5000000")))
        manager.add_filter(SectorUniverseSelection(include_sectors=["technology"]))
        manager.add_filter(MarketCapUniverseSelection(min_market_cap=Decimal("1e9")))
        manager.add_filter(TrendUniverseSelection(require_uptrend=True))
        return manager

    @staticmethod
    def high_volatility(data_client=None, top_n: int = 30) -> UniverseManager:
        """
        Create universe of high-volatility stocks for options strategies.

        - High implied volatility
        - Liquid options
        """
        from .coarse import DollarVolumeUniverseSelection, OptionsableUniverseSelection
        from .technical import VolatilityUniverseSelection

        manager = UniverseManager(data_client)
        manager.add_filter(DollarVolumeUniverseSelection(min_dollar_volume=Decimal("5000000")))
        manager.add_filter(OptionsableUniverseSelection())
        manager.add_filter(VolatilityUniverseSelection(top_n_volatile=top_n))
        return manager
