"""
Arbitrage Detection - Inspired by Polymarket-Kalshi Arbitrage Bot.

Scans for arbitrage opportunities across prediction markets:
- Cross-platform YES + NO arbitrage
- Same-platform mispricing detection
- Fee-aware profit calculation

Concepts from: https://github.com/terauss/Polymarket-Kalshi-Arbitrage-bot
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import asyncio
import threading
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Type of arbitrage opportunity."""
    CROSS_PLATFORM = "cross_platform"  # YES on platform A + NO on platform B
    SAME_PLATFORM = "same_platform"    # Mispriced YES + NO on same platform
    TEMPORAL = "temporal"              # Price discrepancy over time


class OpportunityStatus(Enum):
    """Status of an arbitrage opportunity."""
    DETECTED = "detected"
    VALIDATED = "validated"
    EXECUTING = "executing"
    EXECUTED = "executed"
    EXPIRED = "expired"
    MISSED = "missed"


@dataclass
class MarketQuote:
    """Quote for a prediction market."""
    symbol: str
    platform: str
    yes_price: float  # 0-1
    no_price: float   # 0-1
    yes_size: int     # Available quantity
    no_size: int
    timestamp: datetime = field(default_factory=datetime.now)
    fee_rate: float = 0.0  # Platform fee rate

    @property
    def spread(self) -> float:
        """YES + NO spread (should be ~1.0)."""
        return self.yes_price + self.no_price

    @property
    def implied_probability(self) -> float:
        """Implied probability of YES outcome."""
        return self.yes_price / self.spread if self.spread > 0 else 0.5


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""
    opportunity_id: str
    arb_type: ArbitrageType

    # Market details
    yes_quote: MarketQuote
    no_quote: MarketQuote

    # Profit calculation
    total_cost: float      # Cost to enter (YES price + NO price)
    guaranteed_payout: float  # Always $1 for matched YES+NO
    gross_profit: float    # Payout - cost
    total_fees: float      # All platform fees
    net_profit: float      # Gross - fees
    profit_percent: float  # Net profit / cost as percentage

    # Execution details
    max_quantity: int      # Limited by available size
    total_profit_potential: float  # net_profit * max_quantity

    # Status
    status: OpportunityStatus = OpportunityStatus.DETECTED
    detected_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    executed_quantity: int = 0
    actual_profit: float = 0.0

    @property
    def is_profitable(self) -> bool:
        return self.net_profit > 0

    @property
    def edge(self) -> float:
        """Edge in cents (1 = $0.01 profit per contract)."""
        return self.net_profit * 100

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class DetectorConfig:
    """Configuration for arbitrage detection."""
    # Profit thresholds
    min_profit_cents: float = 1.0  # Minimum 1 cent per contract
    min_profit_percent: float = 0.5  # Minimum 0.5% return
    min_quantity: int = 10  # Minimum contracts to be worth it

    # Scanning
    scan_interval_ms: int = 100  # How often to scan
    quote_staleness_ms: int = 5000  # Max age of quotes

    # Risk limits
    max_exposure_per_opp: float = 1000.0  # Max dollars per opportunity
    max_daily_opportunities: int = 100  # Max opportunities per day

    # Validation
    require_both_platforms: bool = True  # Must have quotes from both
    validate_liquidity: bool = True  # Check size availability


class ArbitrageDetector:
    """
    Detects arbitrage opportunities in prediction markets.

    Core strategy: Buy YES on Platform A + Buy NO on Platform B
    If YES_A + NO_B < $1, guaranteed profit = $1 - (YES_A + NO_B)
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        on_opportunity: Optional[Callable[[ArbitrageOpportunity], None]] = None,
    ):
        self._config = config or DetectorConfig()
        self._on_opportunity = on_opportunity

        # Quote storage: {market_id: {platform: MarketQuote}}
        self._quotes: Dict[str, Dict[str, MarketQuote]] = {}
        self._lock = threading.RLock()

        # Opportunity tracking
        self._opportunities: Dict[str, ArbitrageOpportunity] = {}
        self._opportunity_history: deque = deque(maxlen=1000)
        self._daily_count = 0
        self._last_reset = datetime.now().date()

        # Statistics
        self._stats = {
            "scans": 0,
            "opportunities_found": 0,
            "opportunities_executed": 0,
            "total_profit": 0.0,
        }

    def update_quote(self, quote: MarketQuote) -> Optional[ArbitrageOpportunity]:
        """
        Update a market quote and check for arbitrage.

        Returns an opportunity if detected.
        """
        with self._lock:
            self._check_daily_reset()

            # Normalize market ID (strip platform suffix if present)
            market_id = self._normalize_market_id(quote.symbol)

            # Store quote
            if market_id not in self._quotes:
                self._quotes[market_id] = {}
            self._quotes[market_id][quote.platform] = quote

            # Check for arbitrage
            return self._scan_market(market_id)

    def scan_all(self) -> List[ArbitrageOpportunity]:
        """Scan all markets for arbitrage opportunities."""
        opportunities = []

        with self._lock:
            self._check_daily_reset()
            self._stats["scans"] += 1

            for market_id in self._quotes:
                opp = self._scan_market(market_id)
                if opp:
                    opportunities.append(opp)

        return opportunities

    def _scan_market(self, market_id: str) -> Optional[ArbitrageOpportunity]:
        """Scan a single market for arbitrage."""
        quotes = self._quotes.get(market_id, {})

        if len(quotes) < 2 and self._config.require_both_platforms:
            return None

        # Get all platform quotes
        platforms = list(quotes.keys())

        # Check cross-platform arbitrage
        for i, platform_a in enumerate(platforms):
            for platform_b in platforms[i+1:]:
                opp = self._check_cross_platform(
                    market_id,
                    quotes[platform_a],
                    quotes[platform_b],
                )
                if opp and opp.is_profitable:
                    return self._register_opportunity(opp)

        # Check same-platform arbitrage (if spread < 1)
        for platform, quote in quotes.items():
            opp = self._check_same_platform(market_id, quote)
            if opp and opp.is_profitable:
                return self._register_opportunity(opp)

        return None

    def _check_cross_platform(
        self,
        market_id: str,
        quote_a: MarketQuote,
        quote_b: MarketQuote,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check for cross-platform arbitrage.

        Strategy: Buy YES on cheaper platform + Buy NO on cheaper platform
        """
        # Check quote freshness
        if self._is_stale(quote_a) or self._is_stale(quote_b):
            return None

        # Find best YES and NO prices across platforms
        # Option 1: YES from A, NO from B
        cost_1 = quote_a.yes_price + quote_b.no_price
        fees_1 = (quote_a.yes_price * quote_a.fee_rate +
                  quote_b.no_price * quote_b.fee_rate)

        # Option 2: YES from B, NO from A
        cost_2 = quote_b.yes_price + quote_a.no_price
        fees_2 = (quote_b.yes_price * quote_b.fee_rate +
                  quote_a.no_price * quote_a.fee_rate)

        # Pick the better option
        if cost_1 + fees_1 < cost_2 + fees_2:
            total_cost = cost_1
            total_fees = fees_1
            yes_quote = quote_a
            no_quote = quote_b
            max_qty = min(quote_a.yes_size, quote_b.no_size)
        else:
            total_cost = cost_2
            total_fees = fees_2
            yes_quote = quote_b
            no_quote = quote_a
            max_qty = min(quote_b.yes_size, quote_a.no_size)

        # Calculate profit
        payout = 1.0  # $1 guaranteed payout
        gross_profit = payout - total_cost
        net_profit = gross_profit - total_fees

        # Check if profitable
        if net_profit <= 0:
            return None

        # Check minimum thresholds
        profit_cents = net_profit * 100
        profit_percent = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        if profit_cents < self._config.min_profit_cents:
            return None
        if profit_percent < self._config.min_profit_percent:
            return None
        if max_qty < self._config.min_quantity:
            return None

        # Calculate max quantity based on exposure limit
        if total_cost > 0:
            max_by_exposure = int(self._config.max_exposure_per_opp / total_cost)
            max_qty = min(max_qty, max_by_exposure)

        opp_id = f"{market_id}-{yes_quote.platform}-{no_quote.platform}-{datetime.now().timestamp():.0f}"

        return ArbitrageOpportunity(
            opportunity_id=opp_id,
            arb_type=ArbitrageType.CROSS_PLATFORM,
            yes_quote=yes_quote,
            no_quote=no_quote,
            total_cost=total_cost,
            guaranteed_payout=payout,
            gross_profit=gross_profit,
            total_fees=total_fees,
            net_profit=net_profit,
            profit_percent=profit_percent,
            max_quantity=max_qty,
            total_profit_potential=net_profit * max_qty,
            expires_at=datetime.now() + timedelta(seconds=self._config.quote_staleness_ms / 1000),
        )

    def _check_same_platform(
        self,
        market_id: str,
        quote: MarketQuote,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check for same-platform arbitrage (mispriced spread).

        If YES + NO < 1.0 on the same platform, there's arbitrage.
        """
        if self._is_stale(quote):
            return None

        total_cost = quote.yes_price + quote.no_price

        # Must be profitable after spread
        if total_cost >= 1.0:
            return None

        # Calculate fees
        total_fees = total_cost * quote.fee_rate

        payout = 1.0
        gross_profit = payout - total_cost
        net_profit = gross_profit - total_fees

        if net_profit <= 0:
            return None

        profit_cents = net_profit * 100
        profit_percent = (net_profit / total_cost) * 100 if total_cost > 0 else 0

        if profit_cents < self._config.min_profit_cents:
            return None
        if profit_percent < self._config.min_profit_percent:
            return None

        max_qty = min(quote.yes_size, quote.no_size)
        if max_qty < self._config.min_quantity:
            return None

        opp_id = f"{market_id}-{quote.platform}-same-{datetime.now().timestamp():.0f}"

        return ArbitrageOpportunity(
            opportunity_id=opp_id,
            arb_type=ArbitrageType.SAME_PLATFORM,
            yes_quote=quote,
            no_quote=quote,
            total_cost=total_cost,
            guaranteed_payout=payout,
            gross_profit=gross_profit,
            total_fees=total_fees,
            net_profit=net_profit,
            profit_percent=profit_percent,
            max_quantity=max_qty,
            total_profit_potential=net_profit * max_qty,
            expires_at=datetime.now() + timedelta(seconds=self._config.quote_staleness_ms / 1000),
        )

    def _register_opportunity(
        self,
        opp: ArbitrageOpportunity,
    ) -> ArbitrageOpportunity:
        """Register a new opportunity."""
        # Check daily limit
        if self._daily_count >= self._config.max_daily_opportunities:
            opp.status = OpportunityStatus.MISSED
            logger.warning("Daily opportunity limit reached")
            return opp

        opp.status = OpportunityStatus.VALIDATED
        self._opportunities[opp.opportunity_id] = opp
        self._daily_count += 1
        self._stats["opportunities_found"] += 1

        logger.info(
            f"Arbitrage opportunity: {opp.arb_type.value} | "
            f"Edge: {opp.edge:.2f}¢ | "
            f"Max qty: {opp.max_quantity} | "
            f"Potential: ${opp.total_profit_potential:.2f}"
        )

        # Callback
        if self._on_opportunity:
            self._on_opportunity(opp)

        return opp

    def mark_executing(self, opportunity_id: str) -> None:
        """Mark an opportunity as being executed."""
        with self._lock:
            if opportunity_id in self._opportunities:
                self._opportunities[opportunity_id].status = OpportunityStatus.EXECUTING

    def mark_executed(
        self,
        opportunity_id: str,
        quantity: int,
        actual_profit: float,
    ) -> None:
        """Mark an opportunity as executed."""
        with self._lock:
            if opportunity_id in self._opportunities:
                opp = self._opportunities[opportunity_id]
                opp.status = OpportunityStatus.EXECUTED
                opp.executed_quantity = quantity
                opp.actual_profit = actual_profit

                self._stats["opportunities_executed"] += 1
                self._stats["total_profit"] += actual_profit

                # Move to history
                self._opportunity_history.append(opp)
                del self._opportunities[opportunity_id]

    def cleanup_expired(self) -> int:
        """Remove expired opportunities."""
        expired_count = 0

        with self._lock:
            expired_ids = [
                opp_id for opp_id, opp in self._opportunities.items()
                if opp.is_expired
            ]

            for opp_id in expired_ids:
                opp = self._opportunities.pop(opp_id)
                opp.status = OpportunityStatus.EXPIRED
                self._opportunity_history.append(opp)
                expired_count += 1

        return expired_count

    def get_active_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get all active opportunities."""
        with self._lock:
            return [
                opp for opp in self._opportunities.values()
                if not opp.is_expired and opp.status in (
                    OpportunityStatus.DETECTED,
                    OpportunityStatus.VALIDATED,
                )
            ]

    def get_best_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """Get the best current opportunity by profit potential."""
        opportunities = self.get_active_opportunities()
        if not opportunities:
            return None

        return max(opportunities, key=lambda o: o.total_profit_potential)

    def _is_stale(self, quote: MarketQuote) -> bool:
        """Check if a quote is too old."""
        age_ms = (datetime.now() - quote.timestamp).total_seconds() * 1000
        return age_ms > self._config.quote_staleness_ms

    def _normalize_market_id(self, symbol: str) -> str:
        """Normalize market ID across platforms."""
        # Strip common platform suffixes
        for suffix in ["-polymarket", "-kalshi", "-yes", "-no"]:
            if symbol.lower().endswith(suffix):
                symbol = symbol[:-len(suffix)]
        return symbol.lower()

    def _check_daily_reset(self) -> None:
        """Reset daily counters if needed."""
        today = datetime.now().date()
        if self._last_reset < today:
            self._daily_count = 0
            self._last_reset = today

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        with self._lock:
            active = len(self.get_active_opportunities())

            return {
                **self._stats,
                "active_opportunities": active,
                "daily_count": self._daily_count,
                "markets_tracked": len(self._quotes),
                "history_size": len(self._opportunity_history),
            }


class ArbitrageScanner:
    """
    Continuous scanner for arbitrage opportunities.

    Runs in background and emits opportunities as detected.
    """

    def __init__(
        self,
        detector: ArbitrageDetector,
        quote_fetchers: List[Callable[[], List[MarketQuote]]],
        scan_interval_ms: int = 100,
    ):
        self._detector = detector
        self._fetchers = quote_fetchers
        self._interval = scan_interval_ms / 1000
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the scanner."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        logger.info("Arbitrage scanner started")

    def stop(self) -> None:
        """Stop the scanner."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Arbitrage scanner stopped")

    def _scan_loop(self) -> None:
        """Main scanning loop."""
        import time

        while self._running:
            try:
                # Fetch quotes from all sources
                for fetcher in self._fetchers:
                    try:
                        quotes = fetcher()
                        for quote in quotes:
                            self._detector.update_quote(quote)
                    except Exception as e:
                        logger.error(f"Quote fetcher error: {e}")

                # Cleanup expired
                self._detector.cleanup_expired()

                # Sleep
                time.sleep(self._interval)

            except Exception as e:
                logger.error(f"Scanner error: {e}")
                time.sleep(1.0)


# Global detector instance
_global_detector: Optional[ArbitrageDetector] = None


def get_arbitrage_detector() -> ArbitrageDetector:
    """Get or create the global arbitrage detector."""
    global _global_detector
    if _global_detector is None:
        _global_detector = ArbitrageDetector()
    return _global_detector


def configure_arbitrage_detector(config: DetectorConfig) -> ArbitrageDetector:
    """Configure the global arbitrage detector."""
    global _global_detector
    _global_detector = ArbitrageDetector(config)
    return _global_detector
