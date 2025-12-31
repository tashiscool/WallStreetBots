"""
Screener Manager Module.

Coordinates multiple screeners and provides data fetching integration.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from .base import (
    CompositeScreener,
    IScreener,
    ScreenerResult,
    SortOrder,
    SortScreener,
    StockData,
)

logger = logging.getLogger(__name__)


@dataclass
class ScreenerPipeline:
    """Configuration for a screening pipeline."""

    name: str
    screeners: List[IScreener]
    sort_by: Optional[str] = None
    sort_order: SortOrder = SortOrder.DESCENDING
    limit: Optional[int] = None
    enabled: bool = True


class ScreenerManager:
    """
    Manages stock screener pipelines.

    Coordinates data fetching, screening, and result aggregation.
    """

    def __init__(
        self,
        data_client: Optional[Any] = None,
        default_symbols: Optional[List[str]] = None,
        cache_ttl: int = 60,  # seconds
    ):
        """
        Initialize screener manager.

        Args:
            data_client: Data client for fetching stock data
            default_symbols: Default symbols to screen
            cache_ttl: Cache time-to-live in seconds
        """
        self.data_client = data_client
        self.default_symbols = default_symbols or []
        self.cache_ttl = cache_ttl

        self._screeners: List[IScreener] = []
        self._pipelines: Dict[str, ScreenerPipeline] = {}
        self._cache: Dict[str, ScreenerResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._callbacks: List[Callable[[ScreenerResult], None]] = []

    def add_screener(self, screener: IScreener) -> "ScreenerManager":
        """Add a screener to the default pipeline."""
        self._screeners.append(screener)
        return self

    def remove_screener(self, name: str) -> bool:
        """Remove a screener by name."""
        for i, s in enumerate(self._screeners):
            if s.name == name:
                self._screeners.pop(i)
                return True
        return False

    def add_pipeline(
        self,
        name: str,
        screeners: List[IScreener],
        sort_by: Optional[str] = None,
        sort_order: SortOrder = SortOrder.DESCENDING,
        limit: Optional[int] = None,
    ) -> "ScreenerManager":
        """Add a named screening pipeline."""
        self._pipelines[name] = ScreenerPipeline(
            name=name,
            screeners=screeners,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
        )
        return self

    def on_result(
        self,
        callback: Callable[[ScreenerResult], None],
    ) -> "ScreenerManager":
        """Register callback for screening results."""
        self._callbacks.append(callback)
        return self

    async def fetch_stock_data(
        self,
        symbols: Optional[List[str]] = None,
    ) -> List[StockData]:
        """
        Fetch stock data for screening.

        Override this method to integrate with your data source.
        """
        symbols = symbols or self.default_symbols

        if self.data_client is None:
            logger.warning("No data client configured")
            return []

        stocks = []

        # Fetch data from client
        try:
            if hasattr(self.data_client, "get_quotes_batch"):
                quotes = await self.data_client.get_quotes_batch(symbols)
                for symbol, quote in quotes.items():
                    stocks.append(self._quote_to_stock_data(symbol, quote))

            elif hasattr(self.data_client, "get_quote"):
                for symbol in symbols:
                    quote = await self.data_client.get_quote(symbol)
                    if quote:
                        stocks.append(self._quote_to_stock_data(symbol, quote))

        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")

        return stocks

    def _quote_to_stock_data(
        self,
        symbol: str,
        quote: Dict[str, Any],
    ) -> StockData:
        """Convert quote data to StockData object."""
        return StockData(
            symbol=symbol,
            price=Decimal(str(quote.get("price", 0))),
            open=Decimal(str(quote["open"])) if quote.get("open") else None,
            high=Decimal(str(quote["high"])) if quote.get("high") else None,
            low=Decimal(str(quote["low"])) if quote.get("low") else None,
            close=Decimal(str(quote["close"])) if quote.get("close") else None,
            prev_close=Decimal(str(quote["prev_close"])) if quote.get("prev_close") else None,
            volume=quote.get("volume", 0),
            avg_volume=quote.get("avg_volume", 0),
            change_pct=quote.get("change_pct", 0.0),
            gap_pct=quote.get("gap_pct", 0.0),
            market_cap=Decimal(str(quote["market_cap"])) if quote.get("market_cap") else None,
            sector=quote.get("sector"),
            industry=quote.get("industry"),
            pe_ratio=quote.get("pe_ratio"),
            dividend_yield=quote.get("dividend_yield"),
            has_options=quote.get("has_options", False),
            iv_rank=quote.get("iv_rank"),
            rsi=quote.get("rsi"),
            sma_20=quote.get("sma_20"),
            sma_50=quote.get("sma_50"),
            sma_200=quote.get("sma_200"),
            atr=quote.get("atr"),
            volatility=quote.get("volatility"),
        )

    async def screen(
        self,
        symbols: Optional[List[str]] = None,
        stocks: Optional[List[StockData]] = None,
        force_refresh: bool = False,
    ) -> ScreenerResult:
        """
        Run the default screening pipeline.

        Args:
            symbols: Symbols to fetch (if stocks not provided)
            stocks: Pre-fetched stock data
            force_refresh: Force refresh even if cached

        Returns:
            ScreenerResult with filtered stocks
        """
        # Check cache
        cache_key = "default"
        if not force_refresh and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Get stock data
        if stocks is None:
            stocks = await self.fetch_stock_data(symbols)

        if not stocks:
            return ScreenerResult(
                stocks=[],
                screener_name="ScreenerManager",
                metadata={"error": "No stock data"},
            )

        # Apply screeners
        filtered = stocks
        for screener in self._screeners:
            if screener.enabled:
                filtered = await screener.screen(filtered)
                if not filtered:
                    break

        result = ScreenerResult(
            stocks=filtered,
            screener_name="ScreenerManager",
            metadata={
                "input_count": len(stocks),
                "output_count": len(filtered),
                "screeners_applied": len(self._screeners),
            },
        )

        # Cache result
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return result

    async def run_pipeline(
        self,
        pipeline_name: str,
        symbols: Optional[List[str]] = None,
        stocks: Optional[List[StockData]] = None,
        force_refresh: bool = False,
    ) -> ScreenerResult:
        """
        Run a named screening pipeline.

        Args:
            pipeline_name: Name of pipeline to run
            symbols: Symbols to fetch
            stocks: Pre-fetched stock data
            force_refresh: Force refresh

        Returns:
            ScreenerResult from pipeline
        """
        pipeline = self._pipelines.get(pipeline_name)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_name}")

        if not pipeline.enabled:
            return ScreenerResult(
                stocks=[],
                screener_name=pipeline_name,
                metadata={"enabled": False},
            )

        # Check cache
        cache_key = f"pipeline:{pipeline_name}"
        if not force_refresh and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Get stock data
        if stocks is None:
            stocks = await self.fetch_stock_data(symbols)

        # Apply pipeline screeners
        filtered = stocks
        for screener in pipeline.screeners:
            if screener.enabled:
                filtered = await screener.screen(filtered)
                if not filtered:
                    break

        # Apply sorting
        if pipeline.sort_by:
            sort_screener = SortScreener(
                sort_by=pipeline.sort_by,
                order=pipeline.sort_order,
                limit=pipeline.limit,
            )
            filtered = await sort_screener.screen(filtered)

        result = ScreenerResult(
            stocks=filtered,
            screener_name=pipeline_name,
            metadata={
                "input_count": len(stocks),
                "output_count": len(filtered),
                "pipeline": pipeline_name,
            },
        )

        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()

        return result

    async def run_all_pipelines(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, ScreenerResult]:
        """Run all pipelines concurrently."""
        stocks = await self.fetch_stock_data(symbols)

        tasks = []
        pipeline_names = []

        for name, pipeline in self._pipelines.items():
            if pipeline.enabled:
                tasks.append(self.run_pipeline(name, stocks=stocks))
                pipeline_names.append(name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            name: result
            for name, result in zip(pipeline_names, results)
            if not isinstance(result, Exception)
        }

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self._cache:
            return False

        timestamp = self._cache_timestamps.get(key)
        if timestamp is None:
            return False

        elapsed = (datetime.now() - timestamp).total_seconds()
        return elapsed < self.cache_ttl

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._cache_timestamps.clear()

    @property
    def screeners(self) -> List[IScreener]:
        """Get list of active screeners."""
        return self._screeners.copy()

    @property
    def pipelines(self) -> Dict[str, ScreenerPipeline]:
        """Get all pipelines."""
        return self._pipelines.copy()


class PresetScreenerManager:
    """
    Factory for common screening presets.

    Provides ready-to-use screening configurations.
    """

    @staticmethod
    def day_trading(
        data_client: Optional[Any] = None,
    ) -> ScreenerManager:
        """
        Day trading preset.

        Finds volatile, liquid stocks with momentum.
        """
        from .momentum import MomentumScreener
        from .volume import DollarVolumeScreener, RelativeVolumeScreener

        manager = ScreenerManager(data_client=data_client)

        manager.add_screener(DollarVolumeScreener(
            min_dollar_volume=Decimal("5000000"),  # $5M+ daily
        ))
        manager.add_screener(RelativeVolumeScreener(
            min_rvol=1.5,  # 150%+ of average
        ))
        manager.add_screener(MomentumScreener(
            min_change_pct=2.0,  # 2%+ move
        ))

        return manager

    @staticmethod
    def swing_trading(
        data_client: Optional[Any] = None,
    ) -> ScreenerManager:
        """
        Swing trading preset.

        Finds trending stocks with good technicals.
        """
        from .technical import MovingAverageScreener, RSIScreener
        from .volume import DollarVolumeScreener

        manager = ScreenerManager(data_client=data_client)

        manager.add_screener(DollarVolumeScreener(
            min_dollar_volume=Decimal("1000000"),  # $1M+ daily
        ))
        manager.add_screener(MovingAverageScreener(
            trend="bullish",
        ))
        manager.add_screener(RSIScreener(
            min_rsi=40.0,
            max_rsi=70.0,
        ))

        return manager

    @staticmethod
    def options_premium(
        data_client: Optional[Any] = None,
    ) -> ScreenerManager:
        """
        Options premium selling preset.

        Finds stocks with high IV for premium collection.
        """
        from .fundamental import MarketCapScreener, OptionsScreener
        from .volume import DollarVolumeScreener

        manager = ScreenerManager(data_client=data_client)

        manager.add_screener(MarketCapScreener(
            tier="large",  # Large caps
        ))
        manager.add_screener(DollarVolumeScreener(
            min_dollar_volume=Decimal("10000000"),  # $10M+ daily
        ))
        manager.add_screener(OptionsScreener(
            require_options=True,
            min_iv_rank=50.0,  # IV rank > 50
        ))

        return manager

    @staticmethod
    def dividend_income(
        data_client: Optional[Any] = None,
    ) -> ScreenerManager:
        """
        Dividend income preset.

        Finds quality dividend-paying stocks.
        """
        from .fundamental import DividendScreener, MarketCapScreener
        from .technical import MovingAverageScreener

        manager = ScreenerManager(data_client=data_client)

        manager.add_screener(MarketCapScreener(
            tier="large",
        ))
        manager.add_screener(DividendScreener(
            min_yield=2.0,  # 2%+ yield
            max_yield=10.0,  # Cap at 10% (avoid traps)
        ))
        manager.add_screener(MovingAverageScreener(
            trend="bullish",
            use_sma_20=False,  # Less strict for dividend stocks
        ))

        return manager

    @staticmethod
    def gap_scanner(
        data_client: Optional[Any] = None,
    ) -> ScreenerManager:
        """
        Gap scanner preset.

        Finds stocks gapping up or down.
        """
        from .momentum import GapScreener
        from .volume import DollarVolumeScreener, RelativeVolumeScreener

        manager = ScreenerManager(data_client=data_client)

        manager.add_screener(DollarVolumeScreener(
            min_dollar_volume=Decimal("2000000"),  # $2M+ daily
        ))
        manager.add_screener(GapScreener(
            min_gap_pct=3.0,  # 3%+ gap
        ))
        manager.add_screener(RelativeVolumeScreener(
            min_rvol=2.0,  # 200%+ volume
        ))

        return manager

    @staticmethod
    def oversold_bounce(
        data_client: Optional[Any] = None,
    ) -> ScreenerManager:
        """
        Oversold bounce preset.

        Finds oversold stocks starting to recover.
        """
        from .momentum import MomentumScreener
        from .technical import RSIScreener
        from .volume import DollarVolumeScreener

        manager = ScreenerManager(data_client=data_client)

        manager.add_screener(DollarVolumeScreener(
            min_dollar_volume=Decimal("5000000"),
        ))
        manager.add_screener(RSIScreener(
            max_rsi=35.0,  # Oversold
        ))
        manager.add_screener(MomentumScreener(
            direction="up",  # Starting to bounce
            min_change_pct=1.0,
        ))

        return manager

    @staticmethod
    def earnings_play(
        data_client: Optional[Any] = None,
    ) -> ScreenerManager:
        """
        Earnings play preset.

        Finds stocks with upcoming earnings and high IV.
        """
        from .fundamental import EarningsScreener, OptionsScreener
        from .volume import DollarVolumeScreener

        manager = ScreenerManager(data_client=data_client)

        manager.add_screener(DollarVolumeScreener(
            min_dollar_volume=Decimal("10000000"),
        ))
        manager.add_screener(EarningsScreener(
            earnings_within_days=14,  # Earnings in 2 weeks
        ))
        manager.add_screener(OptionsScreener(
            require_options=True,
            min_iv_rank=30.0,
        ))

        return manager

    @staticmethod
    def sector_rotation(
        data_client: Optional[Any] = None,
        sectors: Optional[List[str]] = None,
    ) -> ScreenerManager:
        """
        Sector rotation preset.

        Finds leaders in specified sectors.
        """
        from .fundamental import SectorScreener
        from .technical import MovingAverageScreener
        from .volume import DollarVolumeScreener

        manager = ScreenerManager(data_client=data_client)

        manager.add_screener(DollarVolumeScreener(
            min_dollar_volume=Decimal("5000000"),
        ))

        if sectors:
            manager.add_screener(SectorScreener(
                sectors=sectors,
            ))

        manager.add_screener(MovingAverageScreener(
            trend="bullish",
        ))

        return manager

