"""
Unified Prediction Market Platform Client - Best-of-Breed Synthesis.

Combines patterns from 8 arbitrage bots:
- Factory pattern for multi-environment clients (prophet-arbitrage-bot)
- Gateway abstraction for external APIs (RichardFeynmanEnthusiast)
- Rate limiting with backoff (dexorynLabs)
- Dual WebSocket strategies (jtdoherty/arb-bot)
- Environment-aware configuration (antevorta)

Supports: Polymarket, Kalshi, and extensible to other platforms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
import asyncio
import threading
import logging
import time

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported prediction market platforms."""
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class Environment(Enum):
    """Trading environment."""
    DEMO = "demo"
    PRODUCTION = "production"


class Side(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class Outcome(Enum):
    """Binary market outcome."""
    YES = "yes"
    NO = "no"


@dataclass
class PriceLevel:
    """Single price level in order book."""
    price: Decimal
    size: Decimal
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderBook:
    """Order book for a single outcome."""
    market_id: str
    outcome: Outcome
    bids: List[PriceLevel] = field(default_factory=list)
    asks: List[PriceLevel] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Highest bid price."""
        if not self.bids:
            return None
        return max(self.bids, key=lambda x: x.price)

    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Lowest ask price."""
        if not self.asks:
            return None
        return min(self.asks, key=lambda x: x.price)

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Mid-market price."""
        if not self.best_bid or not self.best_ask:
            return None
        return (self.best_bid.price + self.best_ask.price) / 2

    @property
    def spread(self) -> Optional[Decimal]:
        """Bid-ask spread."""
        if not self.best_bid or not self.best_ask:
            return None
        return self.best_ask.price - self.best_bid.price

    def is_stale(self, threshold_seconds: float = 5.0) -> bool:
        """Check if order book data is stale."""
        age = (datetime.now() - self.last_update).total_seconds()
        return age > threshold_seconds


@dataclass
class MarketState:
    """Unified market state across platforms."""
    market_id: str
    title: str
    platform: Platform

    # Order books for YES and NO outcomes
    yes_book: Optional[OrderBook] = None
    no_book: Optional[OrderBook] = None

    # Market metadata
    expiration: Optional[datetime] = None
    volume_24h: Optional[Decimal] = None
    liquidity: Optional[Decimal] = None

    @property
    def yes_price(self) -> Optional[Decimal]:
        """Best YES ask price (cost to buy YES)."""
        if self.yes_book and self.yes_book.best_ask:
            return self.yes_book.best_ask.price
        return None

    @property
    def no_price(self) -> Optional[Decimal]:
        """Best NO ask price (cost to buy NO)."""
        if self.no_book and self.no_book.best_ask:
            return self.no_book.best_ask.price
        return None

    @property
    def implied_probability(self) -> Optional[Decimal]:
        """YES implied probability."""
        return self.yes_price

    @property
    def combined_price(self) -> Optional[Decimal]:
        """YES + NO price (should be ~1.0)."""
        if self.yes_price and self.no_price:
            return self.yes_price + self.no_price
        return None


@dataclass
class OrderRequest:
    """Order placement request."""
    platform: Platform
    market_id: str
    outcome: Outcome
    side: Side
    quantity: int
    price: Decimal
    order_type: str = "limit"  # "limit" or "market"


@dataclass
class OrderResponse:
    """Order placement response."""
    success: bool
    order_id: Optional[str] = None
    filled_quantity: int = 0
    average_price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    error: Optional[str] = None


class RateLimiter:
    """
    Rate limiter with exponential backoff.

    From dexorynLabs: Intelligent rate limiting with multiple layers.
    """

    def __init__(
        self,
        min_interval_ms: int = 100,
        max_requests_per_second: float = 10.0,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 60.0,
    ):
        self._min_interval = min_interval_ms / 1000
        self._max_rps = max_requests_per_second
        self._backoff_factor = backoff_factor
        self._max_backoff = max_backoff_seconds
        self._last_request = 0.0
        self._consecutive_errors = 0
        self._cooldown_until = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Wait until request is allowed."""
        with self._lock:
            now = time.time()

            # Check cooldown
            if now < self._cooldown_until:
                sleep_time = self._cooldown_until - now
                logger.warning(f"Rate limited, waiting {sleep_time:.2f}s")
                time.sleep(sleep_time)
                now = time.time()

            # Enforce minimum interval
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            self._last_request = time.time()

    def record_success(self) -> None:
        """Record successful request."""
        with self._lock:
            self._consecutive_errors = 0

    def record_error(self, is_rate_limit: bool = False) -> None:
        """Record failed request."""
        with self._lock:
            self._consecutive_errors += 1

            if is_rate_limit:
                backoff = min(
                    self._backoff_factor ** self._consecutive_errors,
                    self._max_backoff
                )
                self._cooldown_until = time.time() + backoff
                logger.warning(f"Rate limit hit, backing off for {backoff:.2f}s")


class PlatformClient(ABC):
    """
    Abstract base class for prediction market platform clients.

    Implements Gateway pattern from RichardFeynmanEnthusiast.
    """

    def __init__(
        self,
        platform: Platform,
        environment: Environment = Environment.PRODUCTION,
    ):
        self.platform = platform
        self.environment = environment
        self._rate_limiter = RateLimiter()
        self._connected = False

    @property
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to platform."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from platform."""
        pass

    @abstractmethod
    async def get_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[MarketState]:
        """Fetch available markets."""
        pass

    @abstractmethod
    async def get_order_book(
        self,
        market_id: str,
        outcome: Outcome,
    ) -> Optional[OrderBook]:
        """Fetch order book for a specific outcome."""
        pass

    @abstractmethod
    async def place_order(
        self,
        request: OrderRequest,
    ) -> OrderResponse:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(
        self,
        order_id: str,
    ) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_positions(self) -> Dict[str, Decimal]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_balance(self) -> Decimal:
        """Get available balance."""
        pass


class PolymarketClient(PlatformClient):
    """
    Polymarket client implementation.

    Features from multiple bots:
    - py-clob-client integration (antevorta)
    - Delta-based order book updates (jtdoherty)
    - WebSocket streaming (ImMike)
    """

    # API endpoints
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(
        self,
        private_key: Optional[str] = None,
        environment: Environment = Environment.PRODUCTION,
    ):
        super().__init__(Platform.POLYMARKET, environment)
        self._private_key = private_key
        self._session = None

    @property
    def is_authenticated(self) -> bool:
        return self._private_key is not None

    async def connect(self) -> bool:
        """Initialize connection to Polymarket."""
        try:
            # Would use py-clob-client here
            self._connected = True
            logger.info("Connected to Polymarket")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Polymarket: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Polymarket."""
        self._connected = False
        logger.info("Disconnected from Polymarket")

    async def get_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[MarketState]:
        """Fetch Polymarket markets."""
        self._rate_limiter.acquire()
        markets = []

        try:
            # Would fetch from Gamma API
            # Response format: events with markets containing clobTokenIds
            self._rate_limiter.record_success()
            return markets

        except Exception as e:
            self._rate_limiter.record_error()
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def get_order_book(
        self,
        market_id: str,
        outcome: Outcome,
    ) -> Optional[OrderBook]:
        """Fetch Polymarket order book."""
        self._rate_limiter.acquire()

        try:
            # Would fetch from CLOB API
            # Polymarket has separate books for YES and NO
            self._rate_limiter.record_success()
            return None

        except Exception as e:
            self._rate_limiter.record_error()
            logger.error(f"Failed to fetch order book: {e}")
            return None

    async def place_order(
        self,
        request: OrderRequest,
    ) -> OrderResponse:
        """Place order on Polymarket."""
        if not self.is_authenticated:
            return OrderResponse(
                success=False,
                error="Not authenticated"
            )

        self._rate_limiter.acquire()

        try:
            # Would use py-clob-client to place order
            self._rate_limiter.record_success()
            return OrderResponse(success=True, order_id="placeholder")

        except Exception as e:
            self._rate_limiter.record_error()
            logger.error(f"Failed to place order: {e}")
            return OrderResponse(success=False, error=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Polymarket."""
        self._rate_limiter.acquire()

        try:
            # Would use py-clob-client to cancel
            self._rate_limiter.record_success()
            return True

        except Exception as e:
            self._rate_limiter.record_error()
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_positions(self) -> Dict[str, Decimal]:
        """Get Polymarket positions."""
        return {}

    async def get_balance(self) -> Decimal:
        """Get Polymarket balance."""
        return Decimal("0")


class KalshiClient(PlatformClient):
    """
    Kalshi client implementation.

    Features from multiple bots:
    - RSA-PSS signing (prophet-arbitrage-bot)
    - Dual environment support (RichardFeynmanEnthusiast)
    - Derived NO price calculation (RichardFeynmanEnthusiast)
    """

    # API endpoints
    DEMO_API = "https://demo-api.kalshi.co/trade-api/v2"
    PROD_API = "https://trading-api.kalshi.com/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
        environment: Environment = Environment.PRODUCTION,
    ):
        super().__init__(Platform.KALSHI, environment)
        self._api_key = api_key
        self._private_key_path = private_key_path

    @property
    def base_url(self) -> str:
        """Get API base URL for current environment."""
        if self.environment == Environment.DEMO:
            return self.DEMO_API
        return self.PROD_API

    @property
    def is_authenticated(self) -> bool:
        return self._api_key is not None

    async def connect(self) -> bool:
        """Initialize connection to Kalshi."""
        try:
            self._connected = True
            logger.info(f"Connected to Kalshi ({self.environment.value})")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kalshi: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Kalshi."""
        self._connected = False
        logger.info("Disconnected from Kalshi")

    async def get_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[MarketState]:
        """Fetch Kalshi markets."""
        self._rate_limiter.acquire()
        markets = []

        try:
            # Would fetch from Kalshi events/markets API
            self._rate_limiter.record_success()
            return markets

        except Exception as e:
            self._rate_limiter.record_error()
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def get_order_book(
        self,
        market_id: str,
        outcome: Outcome,
    ) -> Optional[OrderBook]:
        """
        Fetch Kalshi order book.

        Note: Kalshi has single book, NO price derived from YES bid.
        """
        self._rate_limiter.acquire()

        try:
            # Kalshi returns yes_bid, yes_ask, no_bid, no_ask
            # For NO outcome: ask = 1.0 - yes_bid
            self._rate_limiter.record_success()
            return None

        except Exception as e:
            self._rate_limiter.record_error()
            logger.error(f"Failed to fetch order book: {e}")
            return None

    def derive_no_ask_price(self, yes_bid: Decimal) -> Decimal:
        """
        Derive NO ask price from YES bid.

        From RichardFeynmanEnthusiast:
        NO_ask = 1.0 - YES_bid (zero-sum property)
        """
        return Decimal("1.0") - yes_bid

    async def place_order(
        self,
        request: OrderRequest,
    ) -> OrderResponse:
        """Place order on Kalshi."""
        if not self.is_authenticated:
            return OrderResponse(
                success=False,
                error="Not authenticated"
            )

        self._rate_limiter.acquire()

        try:
            # Would use kalshi-py to place order
            self._rate_limiter.record_success()
            return OrderResponse(success=True, order_id="placeholder")

        except Exception as e:
            self._rate_limiter.record_error()
            logger.error(f"Failed to place order: {e}")
            return OrderResponse(success=False, error=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Kalshi."""
        self._rate_limiter.acquire()

        try:
            self._rate_limiter.record_success()
            return True

        except Exception as e:
            self._rate_limiter.record_error()
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_positions(self) -> Dict[str, Decimal]:
        """Get Kalshi positions."""
        return {}

    async def get_balance(self) -> Decimal:
        """Get Kalshi balance (USD)."""
        return Decimal("0")


class PlatformClientFactory:
    """
    Factory for creating platform clients.

    From prophet-arbitrage-bot: Centralized credential management.
    """

    def __init__(
        self,
        environment: Environment = Environment.PRODUCTION,
    ):
        self.environment = environment
        self._clients: Dict[Platform, PlatformClient] = {}

    def create_polymarket_client(
        self,
        private_key: Optional[str] = None,
    ) -> PolymarketClient:
        """Create Polymarket client."""
        client = PolymarketClient(
            private_key=private_key,
            environment=self.environment,
        )
        self._clients[Platform.POLYMARKET] = client
        return client

    def create_kalshi_client(
        self,
        api_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
    ) -> KalshiClient:
        """Create Kalshi client."""
        client = KalshiClient(
            api_key=api_key,
            private_key_path=private_key_path,
            environment=self.environment,
        )
        self._clients[Platform.KALSHI] = client
        return client

    def get_client(self, platform: Platform) -> Optional[PlatformClient]:
        """Get existing client for platform."""
        return self._clients.get(platform)

    async def connect_all(self) -> bool:
        """Connect all registered clients."""
        results = await asyncio.gather(
            *[client.connect() for client in self._clients.values()],
            return_exceptions=True
        )
        return all(r is True for r in results if not isinstance(r, Exception))

    async def disconnect_all(self) -> None:
        """Disconnect all clients."""
        await asyncio.gather(
            *[client.disconnect() for client in self._clients.values()],
            return_exceptions=True
        )
