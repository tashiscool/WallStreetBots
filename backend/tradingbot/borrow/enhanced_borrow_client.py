"""
Enhanced Borrow Client

Dynamic borrow rate tracking, availability monitoring, and HTB detection
for short selling operations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class BorrowDifficulty(Enum):
    """Difficulty level for borrowing shares."""
    EASY_TO_BORROW = "easy"  # General collateral (GC)
    MODERATE = "moderate"  # Slightly elevated rates
    HARD_TO_BORROW = "htb"  # HTB - high rates
    VERY_HARD = "very_hard"  # Very limited availability
    NO_BORROW = "no_borrow"  # Cannot borrow


class ShortSqueezeRisk(Enum):
    """Short squeeze risk levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class EnhancedLocateQuote:
    """Enhanced locate quote with additional data."""
    symbol: str
    available: bool
    shares_available: int
    borrow_rate_bps: float  # Annualized basis points
    borrow_rate_pct: float  # Annual percentage
    difficulty: BorrowDifficulty
    timestamp: datetime

    # Additional info
    reason: Optional[str] = None
    utilization_pct: Optional[float] = None  # % of float shorted
    days_to_cover: Optional[float] = None  # Short interest / avg volume
    cost_estimate_daily: Optional[Decimal] = None
    squeeze_risk: Optional[ShortSqueezeRisk] = None

    @property
    def is_htb(self) -> bool:
        """Check if this is a hard-to-borrow security."""
        return self.difficulty in (
            BorrowDifficulty.HARD_TO_BORROW,
            BorrowDifficulty.VERY_HARD,
        )

    def daily_cost_per_share(self, current_price: Decimal) -> Decimal:
        """Calculate daily borrow cost per share."""
        return current_price * Decimal(str(self.borrow_rate_pct / 100 / 365))


@dataclass
class ShortPosition:
    """Active short position with borrow tracking."""
    symbol: str
    qty: int
    entry_price: Decimal
    entry_date: datetime
    borrow_rate_bps: float
    current_price: Optional[Decimal] = None
    mark_to_market: Optional[Decimal] = None
    accrued_borrow_cost: Decimal = Decimal("0")
    last_rate_update: Optional[datetime] = None

    @property
    def days_held(self) -> int:
        return (datetime.now() - self.entry_date).days

    @property
    def unrealized_pnl(self) -> Optional[Decimal]:
        if self.current_price:
            return (self.entry_price - self.current_price) * self.qty - self.accrued_borrow_cost
        return None

    @property
    def total_cost(self) -> Decimal:
        """Total cost including borrow fees."""
        return self.accrued_borrow_cost


@dataclass
class BorrowRateHistory:
    """Historical borrow rate data."""
    symbol: str
    rate_bps: float
    availability: int
    timestamp: datetime


# Sample HTB list (in production, this would come from broker API)
KNOWN_HTB_STOCKS = {
    "GME": {"base_rate": 150.0, "volatility": 0.8},
    "AMC": {"base_rate": 100.0, "volatility": 0.7},
    "BBBY": {"base_rate": 200.0, "volatility": 0.9},
    "KOSS": {"base_rate": 300.0, "volatility": 0.95},
    "SPCE": {"base_rate": 80.0, "volatility": 0.5},
    "CVNA": {"base_rate": 120.0, "volatility": 0.6},
    "RIVN": {"base_rate": 90.0, "volatility": 0.5},
}

# Default rates by market cap tier
DEFAULT_RATES = {
    "mega_cap": 15.0,  # >$200B
    "large_cap": 25.0,  # $10B-$200B
    "mid_cap": 40.0,  # $2B-$10B
    "small_cap": 80.0,  # $300M-$2B
    "micro_cap": 200.0,  # <$300M
}


class EnhancedBorrowClient:
    """
    Enhanced borrow client with dynamic rate tracking.

    Features:
    - Dynamic borrow rate estimation
    - Availability tracking
    - Short squeeze risk detection
    - Accrued cost calculation
    - HTB identification
    """

    def __init__(
        self,
        broker_client: Optional[Any] = None,
        use_real_rates: bool = False,
    ):
        """
        Initialize enhanced borrow client.

        Args:
            broker_client: Optional broker client for real rates
            use_real_rates: Use real borrow rates from broker
        """
        self.broker = broker_client
        self.use_real_rates = use_real_rates and broker_client is not None

        # Caches
        self._rate_cache: Dict[str, EnhancedLocateQuote] = {}
        self._rate_history: Dict[str, List[BorrowRateHistory]] = {}
        self._positions: Dict[str, ShortPosition] = {}
        self._cache_expiry = timedelta(minutes=15)

    async def get_locate_quote(
        self,
        symbol: str,
        qty: int,
        refresh: bool = False,
    ) -> EnhancedLocateQuote:
        """
        Get locate quote for a symbol.

        Args:
            symbol: Stock symbol
            qty: Quantity to short
            refresh: Force refresh from source

        Returns:
            EnhancedLocateQuote with availability and rate
        """
        symbol = symbol.upper()

        # Check cache
        if not refresh and symbol in self._rate_cache:
            cached = self._rate_cache[symbol]
            if datetime.now() - cached.timestamp < self._cache_expiry:
                return cached

        # Get quote
        if self.use_real_rates:
            quote = await self._get_real_rate(symbol, qty)
        else:
            quote = self._estimate_rate(symbol, qty)

        # Cache result
        self._rate_cache[symbol] = quote

        # Store history
        if symbol not in self._rate_history:
            self._rate_history[symbol] = []
        self._rate_history[symbol].append(BorrowRateHistory(
            symbol=symbol,
            rate_bps=quote.borrow_rate_bps,
            availability=quote.shares_available,
            timestamp=quote.timestamp,
        ))

        return quote

    async def _get_real_rate(self, symbol: str, qty: int) -> EnhancedLocateQuote:
        """Get real borrow rate from broker."""
        try:
            if hasattr(self.broker, 'get_borrow_rate'):
                result = await self.broker.get_borrow_rate(symbol, qty)
                return EnhancedLocateQuote(
                    symbol=symbol,
                    available=result.get('available', False),
                    shares_available=result.get('shares_available', 0),
                    borrow_rate_bps=result.get('rate_bps', 30.0),
                    borrow_rate_pct=result.get('rate_bps', 30.0) / 100,
                    difficulty=self._classify_difficulty(result.get('rate_bps', 30.0)),
                    timestamp=datetime.now(),
                )
        except Exception as e:
            logger.warning(f"Failed to get real rate for {symbol}: {e}")

        # Fallback to estimate
        return self._estimate_rate(symbol, qty)

    def _estimate_rate(self, symbol: str, qty: int) -> EnhancedLocateQuote:
        """Estimate borrow rate based on known data."""
        symbol = symbol.upper()

        # Check if known HTB
        if symbol in KNOWN_HTB_STOCKS:
            htb_data = KNOWN_HTB_STOCKS[symbol]
            base_rate = htb_data["base_rate"]
            # Add some randomness to simulate market conditions
            import random
            volatility = htb_data["volatility"]
            rate_variation = random.uniform(-volatility * 20, volatility * 30)
            rate_bps = max(50, base_rate + rate_variation)

            difficulty = BorrowDifficulty.HARD_TO_BORROW
            if rate_bps > 200:
                difficulty = BorrowDifficulty.VERY_HARD
            elif rate_bps > 100:
                difficulty = BorrowDifficulty.HARD_TO_BORROW

            squeeze_risk = ShortSqueezeRisk.HIGH if rate_bps > 150 else ShortSqueezeRisk.MODERATE

            return EnhancedLocateQuote(
                symbol=symbol,
                available=True,
                shares_available=max(1000, 100000 - int(rate_bps * 100)),
                borrow_rate_bps=rate_bps,
                borrow_rate_pct=rate_bps / 100,
                difficulty=difficulty,
                timestamp=datetime.now(),
                utilization_pct=min(95, rate_bps / 3),
                days_to_cover=rate_bps / 50,
                squeeze_risk=squeeze_risk,
            )

        # Estimate based on symbol characteristics
        rate_bps = self._estimate_rate_by_characteristics(symbol)

        return EnhancedLocateQuote(
            symbol=symbol,
            available=True,
            shares_available=1000000,  # Assume good availability
            borrow_rate_bps=rate_bps,
            borrow_rate_pct=rate_bps / 100,
            difficulty=self._classify_difficulty(rate_bps),
            timestamp=datetime.now(),
            squeeze_risk=ShortSqueezeRisk.LOW,
        )

    def _estimate_rate_by_characteristics(self, symbol: str) -> float:
        """Estimate rate based on symbol characteristics."""
        # Mega caps
        mega_caps = ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "BRK.A", "BRK.B", "TSM"]
        if symbol in mega_caps:
            return DEFAULT_RATES["mega_cap"]

        # Large caps (simplified - in production use market cap data)
        large_caps = ["JPM", "V", "MA", "XOM", "CVX", "PG", "JNJ", "HD", "BAC", "DIS"]
        if symbol in large_caps:
            return DEFAULT_RATES["large_cap"]

        # Symbol length heuristic (longer symbols often smaller companies)
        if len(symbol) <= 3:
            return DEFAULT_RATES["large_cap"]
        elif len(symbol) == 4:
            return DEFAULT_RATES["mid_cap"]
        else:
            return DEFAULT_RATES["small_cap"]

    def _classify_difficulty(self, rate_bps: float) -> BorrowDifficulty:
        """Classify borrow difficulty based on rate."""
        if rate_bps <= 30:
            return BorrowDifficulty.EASY_TO_BORROW
        elif rate_bps <= 75:
            return BorrowDifficulty.MODERATE
        elif rate_bps <= 200:
            return BorrowDifficulty.HARD_TO_BORROW
        elif rate_bps <= 500:
            return BorrowDifficulty.VERY_HARD
        else:
            return BorrowDifficulty.NO_BORROW

    async def open_short_position(
        self,
        symbol: str,
        qty: int,
        entry_price: Decimal,
    ) -> ShortPosition:
        """
        Track a new short position.

        Args:
            symbol: Stock symbol
            qty: Quantity shorted
            entry_price: Entry price per share

        Returns:
            ShortPosition object
        """
        symbol = symbol.upper()

        # Get current rate
        quote = await self.get_locate_quote(symbol, qty)

        position = ShortPosition(
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            entry_date=datetime.now(),
            borrow_rate_bps=quote.borrow_rate_bps,
            current_price=entry_price,
            last_rate_update=datetime.now(),
        )

        self._positions[symbol] = position
        logger.info(
            f"Opened short position: {symbol} x{qty} @ ${entry_price} "
            f"(borrow rate: {quote.borrow_rate_bps}bps)"
        )

        return position

    async def update_position_costs(
        self,
        symbol: str,
        current_price: Optional[Decimal] = None,
    ) -> Optional[ShortPosition]:
        """
        Update position with current costs.

        Args:
            symbol: Stock symbol
            current_price: Current market price

        Returns:
            Updated ShortPosition or None
        """
        symbol = symbol.upper()

        if symbol not in self._positions:
            return None

        position = self._positions[symbol]

        if current_price:
            position.current_price = current_price
            position.mark_to_market = (position.entry_price - current_price) * position.qty

        # Update borrow rate if stale
        if (not position.last_rate_update or
                datetime.now() - position.last_rate_update > timedelta(hours=1)):
            quote = await self.get_locate_quote(symbol, position.qty, refresh=True)
            position.borrow_rate_bps = quote.borrow_rate_bps
            position.last_rate_update = datetime.now()

        # Calculate accrued cost
        days_held = position.days_held
        daily_rate = Decimal(str(position.borrow_rate_bps / 100 / 365))
        position_value = position.entry_price * position.qty
        position.accrued_borrow_cost = position_value * daily_rate * days_held

        return position

    def close_position(self, symbol: str) -> Optional[ShortPosition]:
        """Close and return a short position."""
        symbol = symbol.upper()
        return self._positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[ShortPosition]:
        """Get a tracked short position."""
        return self._positions.get(symbol.upper())

    def get_all_positions(self) -> List[ShortPosition]:
        """Get all tracked short positions."""
        return list(self._positions.values())

    async def get_total_borrow_costs(self) -> Decimal:
        """Calculate total accrued borrow costs across all positions."""
        total = Decimal("0")
        for symbol, position in self._positions.items():
            await self.update_position_costs(symbol)
            total += position.accrued_borrow_cost
        return total

    def get_rate_history(
        self,
        symbol: str,
        days: int = 30,
    ) -> List[BorrowRateHistory]:
        """Get historical borrow rates for a symbol."""
        symbol = symbol.upper()
        if symbol not in self._rate_history:
            return []

        cutoff = datetime.now() - timedelta(days=days)
        return [h for h in self._rate_history[symbol] if h.timestamp >= cutoff]

    async def scan_for_htb_opportunities(
        self,
        symbols: List[str],
    ) -> List[EnhancedLocateQuote]:
        """
        Scan for HTB securities (potential squeeze candidates or premium shorts).

        Args:
            symbols: List of symbols to scan

        Returns:
            List of HTB quotes sorted by rate
        """
        htb_quotes = []

        for symbol in symbols:
            try:
                quote = await self.get_locate_quote(symbol, 100)
                if quote.is_htb:
                    htb_quotes.append(quote)
            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")

        # Sort by rate (highest first)
        return sorted(htb_quotes, key=lambda q: q.borrow_rate_bps, reverse=True)

    def detect_squeeze_risk(self, quote: EnhancedLocateQuote) -> ShortSqueezeRisk:
        """Detect short squeeze risk for a position."""
        risk_score = 0

        # High borrow rate
        if quote.borrow_rate_bps > 200:
            risk_score += 3
        elif quote.borrow_rate_bps > 100:
            risk_score += 2
        elif quote.borrow_rate_bps > 50:
            risk_score += 1

        # High utilization
        if quote.utilization_pct:
            if quote.utilization_pct > 80:
                risk_score += 3
            elif quote.utilization_pct > 50:
                risk_score += 2
            elif quote.utilization_pct > 25:
                risk_score += 1

        # Days to cover
        if quote.days_to_cover:
            if quote.days_to_cover > 5:
                risk_score += 3
            elif quote.days_to_cover > 3:
                risk_score += 2
            elif quote.days_to_cover > 1:
                risk_score += 1

        # Classify risk
        if risk_score >= 7:
            return ShortSqueezeRisk.EXTREME
        elif risk_score >= 5:
            return ShortSqueezeRisk.HIGH
        elif risk_score >= 3:
            return ShortSqueezeRisk.MODERATE
        else:
            return ShortSqueezeRisk.LOW


def create_enhanced_borrow_client(
    broker_client: Optional[Any] = None,
) -> EnhancedBorrowClient:
    """Factory function to create enhanced borrow client."""
    return EnhancedBorrowClient(broker_client)
