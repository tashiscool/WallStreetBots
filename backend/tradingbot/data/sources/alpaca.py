"""
Alpaca Markets Data Source.

Provides market data from Alpaca's data API.
Free tier available for IEX data.
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

# Resolution mapping to Alpaca timeframe
RESOLUTION_MAP = {
    DataResolution.MINUTE: "1Min",
    DataResolution.MINUTE_5: "5Min",
    DataResolution.MINUTE_15: "15Min",
    DataResolution.MINUTE_30: "30Min",
    DataResolution.HOUR: "1Hour",
    DataResolution.HOUR_4: "4Hour",
    DataResolution.DAILY: "1Day",
    DataResolution.WEEKLY: "1Week",
    DataResolution.MONTHLY: "1Month",
}


class AlpacaDataSource(IDataSource):
    """
    Alpaca Markets data source.

    Free tier provides IEX data.
    Paid subscription for SIP data.
    """

    DATA_URL = "https://data.alpaca.markets"
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        feed: str = "iex",  # 'iex' or 'sip'
    ):
        """
        Initialize Alpaca data source.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: Use paper trading API
            feed: Data feed ('iex' free, 'sip' paid)
        """
        super().__init__("Alpaca")
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.feed = feed
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def _headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    @property
    def _trading_url(self) -> str:
        """Get trading API URL."""
        return self.PAPER_URL if self.paper else self.LIVE_URL

    async def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            self._session = aiohttp.ClientSession(headers=self._headers)

            # Test connection with account endpoint
            url = f"{self._trading_url}/v2/account"

            async with self._session.get(url) as resp:
                if resp.status == 200:
                    self._connected = True
                    logger.info("Connected to Alpaca Markets")
                    return True
                else:
                    error = await resp.text()
                    logger.error(f"Alpaca connection failed: {error}")
                    return False

        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Disconnected from Alpaca Markets")

    async def _data_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make API request to Alpaca data API."""
        if not self._session:
            raise ConnectionError("Not connected to Alpaca")

        url = f"{self.DATA_URL}{endpoint}"

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error = await resp.text()
                    logger.error(f"Alpaca data API error: {resp.status} - {error}")
                    return None

        except Exception as e:
            logger.error(f"Alpaca request error: {e}")
            return None

    async def get_bars(
        self,
        symbol: str,
        resolution: DataResolution,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCV]:
        """Get historical bars from Alpaca."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        timeframe = RESOLUTION_MAP.get(resolution, "1Day")

        if end is None:
            end = datetime.now()

        # Format timestamps in RFC3339
        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        endpoint = f"/v2/stocks/{symbol}/bars"

        params = {
            "timeframe": timeframe,
            "start": start_str,
            "end": end_str,
            "feed": self.feed,
            "adjustment": "all",  # Include splits and dividends
        }

        if limit:
            params["limit"] = limit

        data = await self._data_request(endpoint, params)

        if not data or "bars" not in data:
            return []

        bars = []
        for bar in data["bars"]:
            timestamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
            bars.append(OHLCV(
                timestamp=timestamp,
                open=Decimal(str(bar["o"])),
                high=Decimal(str(bar["h"])),
                low=Decimal(str(bar["l"])),
                close=Decimal(str(bar["c"])),
                volume=int(bar["v"]),
                symbol=symbol,
                resolution=resolution,
                vwap=Decimal(str(bar.get("vw", 0))) if bar.get("vw") else None,
                trade_count=bar.get("n"),
            ))

        return bars

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote from Alpaca."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        endpoint = f"/v2/stocks/{symbol}/quotes/latest"
        params = {"feed": self.feed}

        data = await self._data_request(endpoint, params)

        if not data or "quote" not in data:
            return None

        q = data["quote"]
        timestamp = datetime.fromisoformat(q["t"].replace("Z", "+00:00"))

        # Also get latest trade
        trade_endpoint = f"/v2/stocks/{symbol}/trades/latest"
        trade_data = await self._data_request(trade_endpoint, params)

        last_price = None
        last_size = None
        if trade_data and "trade" in trade_data:
            last_price = Decimal(str(trade_data["trade"]["p"]))
            last_size = int(trade_data["trade"]["s"])

        return Quote(
            symbol=symbol,
            timestamp=timestamp,
            bid=Decimal(str(q["bp"])),
            bid_size=int(q["bs"]),
            ask=Decimal(str(q["ap"])),
            ask_size=int(q["as"]),
            last=last_price,
            last_size=last_size,
        )

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        # Use multi-symbol endpoint
        endpoint = "/v2/stocks/quotes/latest"
        params = {
            "symbols": ",".join(symbols),
            "feed": self.feed,
        }

        data = await self._data_request(endpoint, params)

        if not data or "quotes" not in data:
            return {}

        quotes = {}
        for symbol, q in data["quotes"].items():
            timestamp = datetime.fromisoformat(q["t"].replace("Z", "+00:00"))
            quotes[symbol] = Quote(
                symbol=symbol,
                timestamp=timestamp,
                bid=Decimal(str(q["bp"])),
                bid_size=int(q["bs"]),
                ask=Decimal(str(q["ap"])),
                ask_size=int(q["as"]),
            )

        # Get latest trades as well
        trade_endpoint = "/v2/stocks/trades/latest"
        trade_data = await self._data_request(trade_endpoint, params)

        if trade_data and "trades" in trade_data:
            for symbol, t in trade_data["trades"].items():
                if symbol in quotes:
                    quotes[symbol].last = Decimal(str(t["p"]))
                    quotes[symbol].last_size = int(t["s"])

        return quotes

    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """Get historical trades from Alpaca."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        if end is None:
            end = datetime.now()

        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        endpoint = f"/v2/stocks/{symbol}/trades"
        params = {
            "start": start_str,
            "end": end_str,
            "feed": self.feed,
        }

        if limit:
            params["limit"] = limit

        data = await self._data_request(endpoint, params)

        if not data or "trades" not in data:
            return []

        trades = []
        for t in data["trades"]:
            timestamp = datetime.fromisoformat(t["t"].replace("Z", "+00:00"))
            trades.append(Trade(
                symbol=symbol,
                timestamp=timestamp,
                price=Decimal(str(t["p"])),
                size=int(t["s"]),
                exchange=t.get("x"),
                conditions=t.get("c"),
            ))

        return trades

    async def get_option_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None,
    ) -> Optional[OptionChain]:
        """
        Get options chain from Alpaca.

        Note: Requires options data subscription.
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        # Alpaca options data endpoint
        endpoint = f"/v1beta1/options/snapshots/{symbol}"

        params = {}
        if expiration:
            params["expiration_date"] = expiration.strftime("%Y-%m-%d")

        data = await self._data_request(endpoint, params)

        if not data or "snapshots" not in data:
            logger.warning("Options data may require subscription")
            return None

        calls = []
        puts = []
        exp_date = None

        for contract_symbol, snapshot in data["snapshots"].items():
            # Parse contract symbol for strike and type
            # Format: O:AAPL230120C00150000
            try:
                parts = contract_symbol.split(":")
                if len(parts) < 2:
                    continue

                contract = parts[1]
                opt_type = "call" if "C" in contract else "put"

                option_data = {
                    "contract": contract_symbol,
                    "bid": float(snapshot.get("latestQuote", {}).get("bp", 0)),
                    "ask": float(snapshot.get("latestQuote", {}).get("ap", 0)),
                    "last": float(snapshot.get("latestTrade", {}).get("p", 0)),
                    "volume": int(snapshot.get("dailyBar", {}).get("v", 0)),
                    "implied_volatility": float(snapshot.get("impliedVolatility", 0)),
                }

                if opt_type == "call":
                    calls.append(option_data)
                else:
                    puts.append(option_data)

            except Exception as e:
                logger.debug(f"Error parsing option contract: {e}")
                continue

        return OptionChain(
            symbol=symbol,
            expiration=exp_date or expiration or date.today(),
            calls=calls,
            puts=puts,
        )

    async def get_expirations(self, symbol: str) -> List[date]:
        """Get available option expiration dates."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        # Get option contracts to extract expirations
        endpoint = "/v1beta1/options/contracts"
        params = {
            "underlying_symbol": symbol,
            "status": "active",
        }

        data = await self._data_request(endpoint, params)

        if not data or "option_contracts" not in data:
            return []

        expirations = set()
        for contract in data["option_contracts"]:
            exp_str = contract.get("expiration_date")
            if exp_str:
                expirations.add(datetime.strptime(exp_str, "%Y-%m-%d").date())

        return sorted(expirations)

    async def get_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """Get asset info from Alpaca."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        # Use trading API for asset info
        url = f"{self._trading_url}/v2/assets/{symbol}"

        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

        except Exception as e:
            logger.error(f"Error fetching asset info: {e}")
            return None

        return FundamentalData(
            symbol=symbol,
            name=data.get("name", symbol),
            exchange=data.get("exchange", ""),
        )

    async def search_symbols(
        self,
        query: str,
        asset_type: Optional[AssetType] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for tradable assets."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        # Get all assets and filter
        url = f"{self._trading_url}/v2/assets"
        params = {"status": "active"}

        if asset_type == AssetType.STOCK:
            params["asset_class"] = "us_equity"

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

        except Exception as e:
            logger.error(f"Error searching assets: {e}")
            return []

        # Filter by query
        query_lower = query.lower()
        results = []

        for asset in data:
            symbol = asset.get("symbol", "")
            name = asset.get("name", "")

            if query_lower in symbol.lower() or query_lower in name.lower():
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "exchange": asset.get("exchange"),
                    "type": asset.get("class"),
                    "tradable": asset.get("tradable"),
                })

                if len(results) >= limit:
                    break

        return results

