"""Real Data Providers
Replaces hardcoded values with live market data.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import aiohttp


@dataclass
class MarketData:
    """Standardized market data structure."""

    ticker: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open_price: float
    previous_close: float
    timestamp: datetime


@dataclass
class OptionsData:
    """Options chain data."""

    ticker: str
    expiry_date: str
    strike: float
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class EarningsEvent:
    """Earnings event data."""

    ticker: str
    earnings_date: datetime
    time: str  # 'AMC' or 'BMO'
    expected_move: float
    actual_eps: float | None = None
    estimated_eps: float | None = None
    surprise: float | None = None


class IEXDataProvider:
    """IEX Cloud data provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https: //cloud.iexapis.com / stable"
        self.logger = logging.getLogger(__name__)

    async def get_quote(self, ticker: str) -> MarketData:
        """Get real - time quote."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/stock / {ticker}/quote"
                params = {"token": self.api_key}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return MarketData(
                            ticker=ticker,
                            price=float(data.get("latestPrice", 0)),
                            change=float(data.get("change", 0)),
                            change_percent=float(data.get("changePercent", 0)),
                            volume=int(data.get("volume", 0)),
                            high=float(data.get("high", 0)),
                            low=float(data.get("low", 0)),
                            open_price=float(data.get("open", 0)),
                            previous_close=float(data.get("previousClose", 0)),
                            timestamp=datetime.now(),
                        )
                    else:
                        self.logger.error(f"IEX API error: {response.status}")
                        return self._get_fallback_data(ticker)
        except Exception as e:
            self.logger.error(f"Error fetching IEX data for {ticker}: {e}")
            return self._get_fallback_data(ticker)

    def _get_fallback_data(self, ticker: str) -> MarketData:
        """Fallback data when API fails."""
        return MarketData(
            ticker=ticker,
            price=0.0,
            change=0.0,
            change_percent=0.0,
            volume=0,
            high=0.0,
            low=0.0,
            open_price=0.0,
            previous_close=0.0,
            timestamp=datetime.now(),
        )


class PolygonDataProvider:
    """Polygon.io data provider for options and real - time data."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https: //api.polygon.io"
        self.logger = logging.getLogger(__name__)

    async def get_options_chain(
        self, ticker: str, expiry_date: str | None = None
    ) -> list[OptionsData]:
        """Get options chain data."""
        try:
            async with aiohttp.ClientSession() as session:
                if expiry_date:
                    url = f"{self.base_url}/v3 / reference / options / contracts"
                    params = {
                        "underlying_ticker": ticker,
                        "expiration_date": expiry_date,
                        "apikey": self.api_key,
                    }
                else:
                    url = f"{self.base_url}/v3 / reference / options / contracts"
                    params = {"underlying_ticker": ticker, "apikey": self.api_key}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        options = []

                        for contract in data.get("results", []):
                            options.append(
                                OptionsData(
                                    ticker=ticker,
                                    expiry_date=contract.get("expiration_date", ""),
                                    strike=float(contract.get("strike_price", 0)),
                                    option_type=contract.get("contract_type", "call"),
                                    bid=0.0,  # Would need separate API call for quotes
                                    ask=0.0,
                                    last_price=0.0,
                                    volume=0,
                                    open_interest=0,
                                    implied_volatility=0.0,
                                    delta=0.0,
                                    gamma=0.0,
                                    theta=0.0,
                                    vega=0.0,
                                )
                            )

                        return options
                    else:
                        self.logger.error(f"Polygon API error: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error fetching Polygon options data for {ticker}: {e}")
            return []

    async def get_real_time_quote(self, ticker: str) -> MarketData:
        """Get real - time quote from Polygon."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/v2 / last / trade / {ticker}"
                params = {"apikey": self.api_key}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return MarketData(
                            ticker=ticker,
                            price=float(data.get("results", {}).get("p", 0)),
                            change=0.0,  # Would need additional data
                            change_percent=0.0,
                            volume=0,
                            high=0.0,
                            low=0.0,
                            open_price=0.0,
                            previous_close=0.0,
                            timestamp=datetime.now(),
                        )
                    else:
                        self.logger.error(f"Polygon API error: {response.status}")
                        return self._get_fallback_data(ticker)
        except Exception as e:
            self.logger.error(f"Error fetching Polygon data for {ticker}: {e}")
            return self._get_fallback_data(ticker)

    def _get_fallback_data(self, ticker: str) -> MarketData:
        """Fallback data when API fails."""
        return MarketData(
            ticker=ticker,
            price=0.0,
            change=0.0,
            change_percent=0.0,
            volume=0,
            high=0.0,
            low=0.0,
            open_price=0.0,
            previous_close=0.0,
            timestamp=datetime.now(),
        )


class EarningsDataProvider:
    """Real earnings data provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https: //financialmodelingprep.com / api / v3"
        self.logger = logging.getLogger(__name__)

    async def get_upcoming_earnings(self, days_ahead: int = 7) -> list[EarningsEvent]:
        """Get upcoming earnings events."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/earning_calendar"
                params = {
                    "apikey": self.api_key,
                    "from": datetime.now().strftime("%Y-%m-%d"),
                    "to": (datetime.now() + timedelta(days=days_ahead)).strftime(
                        "%Y-%m-%d"
                    ),
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        events = []

                        for event in data:
                            events.append(
                                EarningsEvent(
                                    ticker=event.get("symbol", ""),
                                    earnings_date=datetime.strptime(
                                        event.get("date", ""), "%Y-%m-%d"
                                    ),
                                    time=event.get("time", "AMC"),
                                    expected_move=0.0,  # Would need additional calculation
                                    estimated_eps=float(event.get("epsEstimated", 0))
                                    if event.get("epsEstimated")
                                    else None,
                                )
                            )

                        return events
                    else:
                        self.logger.error(f"FMP API error: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error fetching earnings data: {e}")
            return []

    async def get_earnings_history(
        self, ticker: str, limit: int = 4
    ) -> list[EarningsEvent]:
        """Get historical earnings data."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/historical / earning_calendar / {ticker}"
                params = {"apikey": self.api_key, "limit": limit}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        events = []

                        for event in data:
                            events.append(
                                EarningsEvent(
                                    ticker=ticker,
                                    earnings_date=datetime.strptime(
                                        event.get("date", ""), "%Y-%m-%d"
                                    ),
                                    time=event.get("time", "AMC"),
                                    expected_move=0.0,
                                    actual_eps=float(event.get("epsActual", 0))
                                    if event.get("epsActual")
                                    else None,
                                    estimated_eps=float(event.get("epsEstimated", 0))
                                    if event.get("epsEstimated")
                                    else None,
                                    surprise=float(event.get("epsSurprise", 0))
                                    if event.get("epsSurprise")
                                    else None,
                                )
                            )

                        return events
                    else:
                        self.logger.error(f"FMP API error: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error fetching earnings history for {ticker}: {e}")
            return []


class NewsDataProvider:
    """News and sentiment data provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https: //newsapi.org / v2"
        self.logger = logging.getLogger(__name__)

    async def get_ticker_news(
        self, ticker: str, days_back: int = 1
    ) -> list[dict[str, Any]]:
        """Get recent news for ticker."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/everything"
                params = {
                    "q": ticker,
                    "from": (datetime.now() - timedelta(days=days_back)).strftime(
                        "%Y-%m-%d"
                    ),
                    "sortBy": "publishedAt",
                    "apiKey": self.api_key,
                    "pageSize": 20,
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("articles", [])
                    else:
                        self.logger.error(f"News API error: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    async def analyze_sentiment(self, ticker: str) -> dict[str, float]:
        """Analyze news sentiment for ticker."""
        try:
            news = await self.get_ticker_news(ticker)
            if not news:
                return {"score": 0.0, "confidence": 0.0}

            # Simple sentiment analysis (in production, use proper NLP)
            positive_words = [
                "bullish",
                "growth",
                "beat",
                "exceed",
                "strong",
                "positive",
            ]
            negative_words = [
                "bearish",
                "decline",
                "miss",
                "weak",
                "negative",
                "concern",
            ]

            total_score = 0.0
            total_articles = len(news)

            for article in news:
                title = article.get("title", "").lower()
                description = article.get("description", "").lower()
                text = f"{title} {description}"

                positive_count = sum(1 for word in positive_words if word in text)
                negative_count = sum(1 for word in negative_words if word in text)

                article_score = (positive_count - negative_count) / max(
                    positive_count + negative_count, 1
                )
                total_score += article_score

            avg_score = total_score / total_articles if total_articles > 0 else 0.0

            return {
                "score": avg_score,
                "confidence": min(
                    total_articles / 10.0, 1.0
                ),  # Confidence based on article count
            }
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {ticker}: {e}")
            return {"score": 0.0, "confidence": 0.0}


class UnifiedDataProvider:
    """Unified data provider that aggregates multiple sources."""

    def __init__(self, config: dict[str, str]):
        self.iex = IEXDataProvider(config.get("iex_api_key", ""))
        self.polygon = PolygonDataProvider(config.get("polygon_api_key", ""))
        self.earnings = EarningsDataProvider(config.get("fmp_api_key", ""))
        self.news = NewsDataProvider(config.get("news_api_key", ""))
        self.logger = logging.getLogger(__name__)

    async def get_market_data(self, ticker: str) -> MarketData:
        """Get market data from best available source."""
        try:
            # Try IEX first
            data = await self.iex.get_quote(ticker)
            if data.price > 0:
                return data

            # Fallback to Polygon
            data = await self.polygon.get_real_time_quote(ticker)
            if data.price > 0:
                return data

            # Return fallback data
            return MarketData(
                ticker=ticker,
                price=0.0,
                change=0.0,
                change_percent=0.0,
                volume=0,
                high=0.0,
                low=0.0,
                open_price=0.0,
                previous_close=0.0,
                timestamp=datetime.now(),
            )
        except Exception as e:
            self.logger.error(f"Error getting market data for {ticker}: {e}")
            return MarketData(
                ticker=ticker,
                price=0.0,
                change=0.0,
                change_percent=0.0,
                volume=0,
                high=0.0,
                low=0.0,
                open_price=0.0,
                previous_close=0.0,
                timestamp=datetime.now(),
            )

    async def get_options_data(
        self, ticker: str, expiry_date: str | None = None
    ) -> list[OptionsData]:
        """Get options data from Polygon."""
        return await self.polygon.get_options_chain(ticker, expiry_date)

    async def get_earnings_data(
        self, ticker: str, days_ahead: int = 7
    ) -> list[EarningsEvent]:
        """Get earnings data from FMP."""
        return await self.earnings.get_upcoming_earnings(days_ahead)

    async def get_sentiment_data(self, ticker: str) -> dict[str, float]:
        """Get sentiment data from news analysis."""
        return await self.news.analyze_sentiment(ticker)


# Factory function for easy initialization
def create_data_provider(config: dict[str, str]) -> UnifiedDataProvider:
    """Create unified data provider with configuration."""
    return UnifiedDataProvider(config)
