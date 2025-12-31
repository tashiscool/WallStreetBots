"""
Data Source Base Classes.

Provides abstract interface for market data providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataResolution(Enum):
    """Data resolution/timeframe."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


class AssetType(Enum):
    """Type of asset."""
    STOCK = "stock"
    OPTION = "option"
    ETF = "etf"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    INDEX = "index"


@dataclass
class OHLCV:
    """OHLCV bar data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    symbol: str = ""
    resolution: DataResolution = DataResolution.DAILY

    # Optional fields
    adjusted_close: Optional[Decimal] = None
    vwap: Optional[Decimal] = None
    trade_count: Optional[int] = None

    @property
    def typical_price(self) -> Decimal:
        """HLC average."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> Decimal:
        """High - Low."""
        return self.high - self.low

    @property
    def body(self) -> Decimal:
        """Absolute body size."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Close > Open."""
        return self.close > self.open


@dataclass
class Quote:
    """Real-time quote data."""
    symbol: str
    timestamp: datetime
    bid: Decimal
    bid_size: int
    ask: Decimal
    ask_size: int
    last: Optional[Decimal] = None
    last_size: Optional[int] = None
    volume: int = 0

    @property
    def spread(self) -> Decimal:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def mid(self) -> Decimal:
        """Mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class Trade:
    """Individual trade/tick data."""
    symbol: str
    timestamp: datetime
    price: Decimal
    size: int
    exchange: Optional[str] = None
    conditions: Optional[List[str]] = None


@dataclass
class OptionChain:
    """Options chain data."""
    symbol: str  # Underlying symbol
    expiration: date
    calls: List[Dict[str, Any]] = field(default_factory=list)
    puts: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FundamentalData:
    """Fundamental/company data."""
    symbol: str
    name: str
    exchange: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[Decimal] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    shares_outstanding: Optional[int] = None
    avg_volume: Optional[int] = None
    high_52w: Optional[Decimal] = None
    low_52w: Optional[Decimal] = None


class IDataSource(ABC):
    """
    Abstract interface for market data sources.

    All data providers implement this interface.
    """

    def __init__(self, name: str = "DataSource"):
        """Initialize data source."""
        self.name = name
        self._connected = False
        self._rate_limit_remaining: Optional[int] = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to data source."""
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the data source.

        Returns:
            True if connected successfully
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        pass

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """
        Get historical OHLCV bars.

        Args:
            symbol: Stock symbol
            resolution: Bar resolution
            start: Start datetime
            end: End datetime (default: now)
            limit: Maximum bars to return

        Returns:
            List of OHLCV bars
        """
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get real-time quote.

        Args:
            symbol: Stock symbol

        Returns:
            Quote data or None
        """
        pass

    @abstractmethod
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dict of symbol -> Quote
        """
        pass

    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """
        Get historical trades/ticks.

        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime
            limit: Maximum trades

        Returns:
            List of trades
        """
        raise NotImplementedError("Trades not supported by this data source")

    async def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None,
    ) -> Optional[OptionChain]:
        """
        Get options chain for a symbol.

        Args:
            symbol: Underlying symbol
            expiration: Specific expiration date (or None for all)

        Returns:
            OptionChain data or None
        """
        raise NotImplementedError("Options not supported by this data source")

    async def get_expirations(
        self,
        symbol: str,
    ) -> List[date]:
        """
        Get available option expiration dates.

        Args:
            symbol: Underlying symbol

        Returns:
            List of expiration dates
        """
        raise NotImplementedError("Options not supported by this data source")

    async def get_fundamentals(
        self,
        symbol: str,
    ) -> Optional[FundamentalData]:
        """
        Get fundamental data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            FundamentalData or None
        """
        raise NotImplementedError("Fundamentals not supported by this data source")

    async def search_symbols(
        self,
        query: str,
        asset_type: Optional[AssetType] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for symbols matching query.

        Args:
            query: Search query
            asset_type: Filter by asset type
            limit: Maximum results

        Returns:
            List of matching symbols with metadata
        """
        raise NotImplementedError("Symbol search not supported")


class CachedDataSource(IDataSource):
    """
    Data source wrapper with caching.

    Caches responses to reduce API calls.
    """

    def __init__(
        self,
        source: IDataSource,
        cache_ttl: int = 60,  # seconds
    ):
        """
        Initialize cached data source.

        Args:
            source: Underlying data source
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(f"Cached_{source.name}")
        self._source = source
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}  # key -> (value, timestamp)

    def _cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        return ":".join(str(a) for a in args)

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = (value, datetime.now())

    async def connect(self) -> bool:
        """Connect to underlying source."""
        result = await self._source.connect()
        self._connected = result
        return result

    async def disconnect(self) -> None:
        """Disconnect and clear cache."""
        await self._source.disconnect()
        self._cache.clear()
        self._connected = False

    async def get_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """Get bars with caching."""
        key = self._cache_key("bars", symbol, resolution.value, start, end, limit)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        bars = await self._source.get_bars(symbol, resolution, start, end, limit)
        self._set_cached(key, bars)
        return bars

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get quote with caching."""
        key = self._cache_key("quote", symbol)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        quote = await self._source.get_quote(symbol)
        if quote:
            self._set_cached(key, quote)
        return quote

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes with caching."""
        result = {}
        uncached = []

        for symbol in symbols:
            key = self._cache_key("quote", symbol)
            cached = self._get_cached(key)
            if cached:
                result[symbol] = cached
            else:
                uncached.append(symbol)

        if uncached:
            quotes = await self._source.get_quotes(uncached)
            for symbol, quote in quotes.items():
                key = self._cache_key("quote", symbol)
                self._set_cached(key, quote)
                result[symbol] = quote

        return result

