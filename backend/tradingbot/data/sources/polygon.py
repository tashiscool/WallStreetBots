"""
Polygon.io Data Source.

Provides market data from Polygon.io API.
Requires API key with appropriate subscription.
"""

import asyncio
import aiohttp
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
    Trade,
)

logger = logging.getLogger(__name__)

# Resolution mapping to Polygon timespan/multiplier
RESOLUTION_MAP = {
    DataResolution.MINUTE: ("minute", 1),
    DataResolution.MINUTE_5: ("minute", 5),
    DataResolution.MINUTE_15: ("minute", 15),
    DataResolution.MINUTE_30: ("minute", 30),
    DataResolution.HOUR: ("hour", 1),
    DataResolution.HOUR_4: ("hour", 4),
    DataResolution.DAILY: ("day", 1),
    DataResolution.WEEKLY: ("week", 1),
    DataResolution.MONTHLY: ("month", 1),
}


class PolygonDataSource(IDataSource):
    """
    Polygon.io data source.

    Professional-grade market data API.
    Requires paid subscription for real-time data.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        """
        Initialize Polygon data source.

        Args:
            api_key: Polygon.io API key
        """
        super().__init__("Polygon")
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Connect to Polygon API."""
        try:
            self._session = aiohttp.ClientSession()

            # Test connection with a simple request
            url = f"{self.BASE_URL}/v1/marketstatus/now"
            params = {"apiKey": self.api_key}

            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    self._connected = True
                    logger.info("Connected to Polygon.io")
                    return True
                else:
                    error = await resp.text()
                    logger.error(f"Polygon connection failed: {error}")
                    return False

        except Exception as e:
            logger.error(f"Error connecting to Polygon: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Polygon API."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Disconnected from Polygon.io")

    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make API request to Polygon."""
        if not self._session:
            raise ConnectionError("Not connected to Polygon")

        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        url = f"{self.BASE_URL}{endpoint}"

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error = await resp.text()
                    logger.error(f"Polygon API error: {resp.status} - {error}")
                    return None

        except Exception as e:
            logger.error(f"Polygon request error: {e}")
            return None

    async def get_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """Get historical bars from Polygon."""
        if not self._connected:
            raise ConnectionError("Not connected to Polygon")

        timespan, multiplier = RESOLUTION_MAP.get(resolution, ("day", 1))

        if end is None:
            end = datetime.now()

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"

        params = {
            "adjusted": "true",
            "sort": "asc",
        }
        if limit:
            params["limit"] = limit

        data = await self._request(endpoint, params)

        if not data or "results" not in data:
            return []

        bars = []
        for r in data["results"]:
            timestamp = datetime.fromtimestamp(r["t"] / 1000)
            bar = OHLCV(
                timestamp=timestamp,
                open=Decimal(str(r["o"])),
                high=Decimal(str(r["h"])),
                low=Decimal(str(r["l"])),
                close=Decimal(str(r["c"])),
                volume=int(r["v"]),
                symbol=symbol,
                resolution=resolution,
                vwap=Decimal(str(r.get("vw", 0))) if r.get("vw") else None,
                trade_count=r.get("n"),
            )
            bars.append(bar)

        return bars

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get real-time quote from Polygon."""
        if not self._connected:
            raise ConnectionError("Not connected to Polygon")

        # Get last quote
        endpoint = f"/v2/last/nbbo/{symbol}"
        data = await self._request(endpoint)

        if not data or "results" not in data:
            return None

        r = data["results"]

        # Also get last trade for last price
        trade_endpoint = f"/v2/last/trade/{symbol}"
        trade_data = await self._request(trade_endpoint)
        last_price = None
        last_size = None
        if trade_data and "results" in trade_data:
            last_price = Decimal(str(trade_data["results"]["p"]))
            last_size = int(trade_data["results"]["s"])

        return Quote(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(r["t"] / 1000000000),  # nanoseconds
            bid=Decimal(str(r["p"])),  # bid price
            bid_size=int(r["s"]),  # bid size
            ask=Decimal(str(r["P"])),  # ask price
            ask_size=int(r["S"]),  # ask size
            last=last_price,
            last_size=last_size,
        )

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols."""
        # Use snapshot endpoint for batch quotes
        endpoint = "/v2/snapshot/locale/us/markets/stocks/tickers"

        params = {"tickers": ",".join(symbols)}
        data = await self._request(endpoint, params)

        if not data or "tickers" not in data:
            return {}

        quotes = {}
        for ticker_data in data["tickers"]:
            symbol = ticker_data["ticker"]

            if "lastQuote" not in ticker_data:
                continue

            lq = ticker_data["lastQuote"]
            lt = ticker_data.get("lastTrade", {})

            quotes[symbol] = Quote(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=Decimal(str(lq.get("p", 0))),
                bid_size=int(lq.get("s", 0)),
                ask=Decimal(str(lq.get("P", 0))),
                ask_size=int(lq.get("S", 0)),
                last=Decimal(str(lt.get("p", 0))) if lt.get("p") else None,
                last_size=int(lt.get("s", 0)) if lt.get("s") else None,
                volume=int(ticker_data.get("day", {}).get("v", 0)),
            )

        return quotes

    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """Get historical trades from Polygon."""
        if not self._connected:
            raise ConnectionError("Not connected to Polygon")

        date_str = start.strftime("%Y-%m-%d")
        endpoint = f"/v3/trades/{symbol}"

        params = {"timestamp": date_str}
        if limit:
            params["limit"] = limit

        data = await self._request(endpoint, params)

        if not data or "results" not in data:
            return []

        trades = []
        for t in data["results"]:
            trades.append(Trade(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(t["sip_timestamp"] / 1000000000),
                price=Decimal(str(t["price"])),
                size=int(t["size"]),
                exchange=t.get("exchange"),
                conditions=t.get("conditions"),
            ))

        return trades

    async def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None,
    ) -> Optional[OptionChain]:
        """Get options chain from Polygon."""
        if not self._connected:
            raise ConnectionError("Not connected to Polygon")

        endpoint = "/v3/reference/options/contracts"

        params = {
            "underlying_ticker": symbol,
            "expired": "false",
            "limit": 250,
        }

        if expiration:
            params["expiration_date"] = expiration.strftime("%Y-%m-%d")

        data = await self._request(endpoint, params)

        if not data or "results" not in data:
            return None

        calls = []
        puts = []

        exp_date = None
        for contract in data["results"]:
            opt_type = contract.get("contract_type", "").lower()
            exp_str = contract.get("expiration_date", "")

            if exp_date is None and exp_str:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()

            option_data = {
                "contract_id": contract.get("ticker"),
                "strike": float(contract.get("strike_price", 0)),
                "expiration": exp_str,
            }

            if opt_type == "call":
                calls.append(option_data)
            elif opt_type == "put":
                puts.append(option_data)

        return OptionChain(
            symbol=symbol,
            expiration=exp_date or date.today(),
            calls=calls,
            puts=puts,
        )

    async def get_expirations(self, symbol: str) -> List[date]:
        """Get available option expiration dates."""
        if not self._connected:
            raise ConnectionError("Not connected to Polygon")

        endpoint = "/v3/reference/options/contracts"
        params = {
            "underlying_ticker": symbol,
            "expired": "false",
            "limit": 1000,
        }

        data = await self._request(endpoint, params)

        if not data or "results" not in data:
            return []

        expirations = set()
        for contract in data["results"]:
            exp_str = contract.get("expiration_date")
            if exp_str:
                expirations.add(datetime.strptime(exp_str, "%Y-%m-%d").date())

        return sorted(expirations)

    async def get_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """Get fundamental data from Polygon."""
        if not self._connected:
            raise ConnectionError("Not connected to Polygon")

        endpoint = f"/v3/reference/tickers/{symbol}"
        data = await self._request(endpoint)

        if not data or "results" not in data:
            return None

        r = data["results"]

        return FundamentalData(
            symbol=symbol,
            name=r.get("name", symbol),
            exchange=r.get("primary_exchange", ""),
            sector=r.get("sic_description"),
            market_cap=Decimal(str(r.get("market_cap", 0) or 0)),
            shares_outstanding=r.get("share_class_shares_outstanding"),
        )

    async def search_symbols(
        self,
        query: str,
        asset_type: Optional[AssetType] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for symbols matching query."""
        if not self._connected:
            raise ConnectionError("Not connected to Polygon")

        endpoint = "/v3/reference/tickers"
        params = {
            "search": query,
            "active": "true",
            "limit": limit,
        }

        if asset_type == AssetType.STOCK:
            params["type"] = "CS"
        elif asset_type == AssetType.ETF:
            params["type"] = "ETF"

        data = await self._request(endpoint, params)

        if not data or "results" not in data:
            return []

        results = []
        for r in data["results"]:
            results.append({
                "symbol": r.get("ticker"),
                "name": r.get("name"),
                "exchange": r.get("primary_exchange"),
                "type": r.get("type"),
                "market": r.get("market"),
            })

        return results

