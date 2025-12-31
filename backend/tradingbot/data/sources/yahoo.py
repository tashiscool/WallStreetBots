"""
Yahoo Finance Data Source.

Provides market data from Yahoo Finance API.
Free tier with rate limits.
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
import logging

from .base import (
    AssetType,
    DataResolution,
    FundamentalData,
    IDataSource,
    OHLCV,
    OptionChain,
    Quote,
)

logger = logging.getLogger(__name__)

# Resolution mapping to yfinance intervals
RESOLUTION_MAP = {
    DataResolution.MINUTE: "1m",
    DataResolution.MINUTE_5: "5m",
    DataResolution.MINUTE_15: "15m",
    DataResolution.MINUTE_30: "30m",
    DataResolution.HOUR: "1h",
    DataResolution.HOUR_4: "4h",
    DataResolution.DAILY: "1d",
    DataResolution.WEEKLY: "1wk",
    DataResolution.MONTHLY: "1mo",
}


class YahooDataSource(IDataSource):
    """
    Yahoo Finance data source.

    Uses yfinance library for data access.
    Free but rate-limited.
    """

    def __init__(self):
        """Initialize Yahoo data source."""
        super().__init__("Yahoo")
        self._yf = None

    async def connect(self) -> bool:
        """Connect to Yahoo Finance."""
        try:
            import yfinance as yf
            self._yf = yf
            self._connected = True
            logger.info("Connected to Yahoo Finance")
            return True
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Yahoo Finance."""
        self._connected = False
        logger.info("Disconnected from Yahoo Finance")

    def _convert_resolution(self, resolution: DataResolution) -> str:
        """Convert resolution to yfinance interval."""
        return RESOLUTION_MAP.get(resolution, "1d")

    async def get_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """Get historical bars from Yahoo Finance."""
        if not self._connected:
            raise ConnectionError("Not connected to Yahoo Finance")

        try:
            ticker = self._yf.Ticker(symbol)
            interval = self._convert_resolution(resolution)

            # yfinance uses different period names
            if end is None:
                end = datetime.now()

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                )
            )

            if df.empty:
                return []

            bars = []
            for idx, row in df.iterrows():
                bar = OHLCV(
                    timestamp=idx.to_pydatetime(),
                    open=Decimal(str(row["Open"])),
                    high=Decimal(str(row["High"])),
                    low=Decimal(str(row["Low"])),
                    close=Decimal(str(row["Close"])),
                    volume=int(row["Volume"]),
                    symbol=symbol,
                    resolution=resolution,
                )
                bars.append(bar)

            if limit:
                bars = bars[-limit:]

            return bars

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return []

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote from Yahoo Finance."""
        if not self._connected:
            raise ConnectionError("Not connected to Yahoo Finance")

        try:
            ticker = self._yf.Ticker(symbol)

            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                lambda: ticker.info
            )

            if not info:
                return None

            # Extract bid/ask from info
            bid = Decimal(str(info.get("bid", 0) or 0))
            ask = Decimal(str(info.get("ask", 0) or 0))
            bid_size = int(info.get("bidSize", 0) or 0)
            ask_size = int(info.get("askSize", 0) or 0)
            last = Decimal(str(info.get("regularMarketPrice", 0) or 0))
            volume = int(info.get("regularMarketVolume", 0) or 0)

            return Quote(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=bid,
                bid_size=bid_size,
                ask=ask,
                ask_size=ask_size,
                last=last,
                volume=volume,
            )

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols."""
        quotes = {}
        # Fetch in parallel with limited concurrency
        semaphore = asyncio.Semaphore(5)

        async def fetch_quote(symbol: str):
            async with semaphore:
                quote = await self.get_quote(symbol)
                if quote:
                    quotes[symbol] = quote

        await asyncio.gather(*[fetch_quote(s) for s in symbols])
        return quotes

    async def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None,
    ) -> Optional[OptionChain]:
        """Get options chain from Yahoo Finance."""
        if not self._connected:
            raise ConnectionError("Not connected to Yahoo Finance")

        try:
            ticker = self._yf.Ticker(symbol)

            loop = asyncio.get_event_loop()

            # Get expirations first
            expirations = await loop.run_in_executor(
                None,
                lambda: ticker.options
            )

            if not expirations:
                return None

            # Use specified expiration or first available
            if expiration:
                exp_str = expiration.strftime("%Y-%m-%d")
            else:
                exp_str = expirations[0]
                expiration = datetime.strptime(exp_str, "%Y-%m-%d").date()

            # Get options data
            opt_chain = await loop.run_in_executor(
                None,
                lambda: ticker.option_chain(exp_str)
            )

            calls = []
            for _, row in opt_chain.calls.iterrows():
                calls.append({
                    "strike": float(row["strike"]),
                    "last": float(row.get("lastPrice", 0)),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "volume": int(row.get("volume", 0) or 0),
                    "open_interest": int(row.get("openInterest", 0) or 0),
                    "implied_volatility": float(row.get("impliedVolatility", 0) or 0),
                    "in_the_money": bool(row.get("inTheMoney", False)),
                })

            puts = []
            for _, row in opt_chain.puts.iterrows():
                puts.append({
                    "strike": float(row["strike"]),
                    "last": float(row.get("lastPrice", 0)),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "volume": int(row.get("volume", 0) or 0),
                    "open_interest": int(row.get("openInterest", 0) or 0),
                    "implied_volatility": float(row.get("impliedVolatility", 0) or 0),
                    "in_the_money": bool(row.get("inTheMoney", False)),
                })

            return OptionChain(
                symbol=symbol,
                expiration=expiration,
                calls=calls,
                puts=puts,
            )

        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return None

    async def get_expirations(self, symbol: str) -> List[date]:
        """Get available option expiration dates."""
        if not self._connected:
            raise ConnectionError("Not connected to Yahoo Finance")

        try:
            ticker = self._yf.Ticker(symbol)

            loop = asyncio.get_event_loop()
            expirations = await loop.run_in_executor(
                None,
                lambda: ticker.options
            )

            if not expirations:
                return []

            return [
                datetime.strptime(exp, "%Y-%m-%d").date()
                for exp in expirations
            ]

        except Exception as e:
            logger.error(f"Error fetching expirations for {symbol}: {e}")
            return []

    async def get_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data from Yahoo Finance."""
        if not self._connected:
            raise ConnectionError("Not connected to Yahoo Finance")

        try:
            ticker = self._yf.Ticker(symbol)

            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                lambda: ticker.info
            )

            if not info:
                return None

            return FundamentalData(
                symbol=symbol,
                name=info.get("longName", symbol),
                exchange=info.get("exchange", ""),
                sector=info.get("sector"),
                industry=info.get("industry"),
                market_cap=Decimal(str(info.get("marketCap", 0) or 0)),
                pe_ratio=info.get("trailingPE"),
                eps=info.get("trailingEps"),
                dividend_yield=info.get("dividendYield"),
                beta=info.get("beta"),
                shares_outstanding=info.get("sharesOutstanding"),
                avg_volume=info.get("averageVolume"),
                high_52w=Decimal(str(info.get("fiftyTwoWeekHigh", 0) or 0)),
                low_52w=Decimal(str(info.get("fiftyTwoWeekLow", 0) or 0)),
            )

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return None

    async def search_symbols(
        self,
        query: str,
        asset_type: Optional[AssetType] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for symbols matching query."""
        if not self._connected:
            raise ConnectionError("Not connected to Yahoo Finance")

        try:
            # Use yfinance's search functionality
            loop = asyncio.get_event_loop()

            # yfinance doesn't have native search, so we'll use a workaround
            # by trying to get info for the symbol
            ticker = self._yf.Ticker(query)
            info = await loop.run_in_executor(
                None,
                lambda: ticker.info
            )

            if info and info.get("symbol"):
                return [{
                    "symbol": info.get("symbol"),
                    "name": info.get("longName", ""),
                    "exchange": info.get("exchange", ""),
                    "type": info.get("quoteType", ""),
                }]

            return []

        except Exception as e:
            logger.error(f"Error searching for {query}: {e}")
            return []

