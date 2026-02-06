"""
Adverse Selection & Toxic Flow Detection.

Detects when order fills are adversely selected (price moves against
you immediately after fill), indicating informed counterparties.

Implements:
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Post-fill price impact tracking
- Toxic flow scoring per symbol/time-of-day

References:
- Easley, López de Prado & O'Hara (2012) - Flow Toxicity and Liquidity
- Easley, López de Prado & O'Hara (2011) - The Microstructure of the Flash Crash
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade observation for adverse selection analysis."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    fill_price: float
    mid_price_at_fill: float
    mid_price_1s: Optional[float] = None  # Mid 1 second after
    mid_price_5s: Optional[float] = None  # Mid 5 seconds after
    mid_price_30s: Optional[float] = None  # Mid 30 seconds after
    mid_price_60s: Optional[float] = None  # Mid 60 seconds after
    mid_price_300s: Optional[float] = None  # Mid 5 minutes after
    is_maker: bool = False


@dataclass
class AdverseSelectionMetrics:
    """Adverse selection metrics for a symbol or portfolio."""
    symbol: str
    period_start: datetime
    period_end: datetime
    num_trades: int

    # Immediate adverse selection (fill vs mid at fill)
    avg_effective_spread_bps: float  # Effective spread (fill vs mid)
    avg_realized_spread_bps: float  # Realized spread (fill vs mid after delay)

    # Post-fill price impact at various horizons
    impact_1s_bps: float
    impact_5s_bps: float
    impact_30s_bps: float
    impact_60s_bps: float
    impact_300s_bps: float

    # Adverse selection cost
    adverse_selection_cost_bps: float  # realized - effective spread
    toxicity_score: float  # 0-1 overall toxicity

    # Breakdown by side
    buy_adverse_bps: float
    sell_adverse_bps: float


@dataclass
class VPINResult:
    """VPIN calculation result."""
    vpin: float  # 0-1, probability of informed trading
    num_buckets: int
    bucket_volume: float
    buy_volume_imbalance: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_toxic(self) -> bool:
        """VPIN > 0.7 considered toxic (high informed trading)."""
        return self.vpin > 0.7

    @property
    def toxicity_level(self) -> str:
        if self.vpin < 0.3:
            return 'low'
        elif self.vpin < 0.5:
            return 'moderate'
        elif self.vpin < 0.7:
            return 'elevated'
        else:
            return 'toxic'


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading.

    VPIN estimates the proportion of toxic (informed) flow in the
    market by measuring volume imbalance across fixed-volume buckets.

    High VPIN indicates that informed traders are present, and your
    fills are more likely to be adversely selected.

    Example:
        vpin = VPINCalculator(bucket_volume=10000, num_buckets=50)
        vpin.add_trade(price=100.0, volume=500, side='buy')
        result = vpin.calculate()
        if result.is_toxic:
            reduce_position_size()
    """

    def __init__(
        self,
        bucket_volume: float = 10_000.0,
        num_buckets: int = 50,
    ):
        """
        Args:
            bucket_volume: Volume per bucket (in shares)
            num_buckets: Number of buckets for rolling VPIN calculation
        """
        self.bucket_volume = bucket_volume
        self.num_buckets = num_buckets

        # Current bucket accumulation
        self._current_buy_volume: float = 0.0
        self._current_sell_volume: float = 0.0
        self._current_total_volume: float = 0.0

        # Completed buckets: list of (buy_volume, sell_volume)
        self._buckets: Deque[Tuple[float, float]] = deque(maxlen=num_buckets)

        # Trade classification state
        self._last_price: Optional[float] = None

    def add_trade(
        self,
        price: float,
        volume: float,
        side: Optional[str] = None,
    ) -> Optional[VPINResult]:
        """
        Add a trade observation.

        If side is not provided, uses tick rule (uptick = buy, downtick = sell).

        Args:
            price: Trade price
            volume: Trade volume (shares)
            side: 'buy' or 'sell', or None for tick classification

        Returns:
            VPINResult if a bucket was just completed, else None
        """
        # Classify trade direction
        if side is None:
            side = self._classify_tick(price)

        if side == 'buy':
            self._current_buy_volume += volume
        else:
            self._current_sell_volume += volume
        self._current_total_volume += volume

        self._last_price = price

        # Check if bucket is full
        result = None
        while self._current_total_volume >= self.bucket_volume:
            overflow = self._current_total_volume - self.bucket_volume

            # Proportion of overflow that belongs to current classification
            if self._current_total_volume > 0:
                buy_ratio = self._current_buy_volume / self._current_total_volume
            else:
                buy_ratio = 0.5

            bucket_buy = self._current_buy_volume - overflow * buy_ratio
            bucket_sell = self._current_sell_volume - overflow * (1 - buy_ratio)

            self._buckets.append((max(0, bucket_buy), max(0, bucket_sell)))

            # Start new bucket with overflow
            self._current_buy_volume = overflow * buy_ratio
            self._current_sell_volume = overflow * (1 - buy_ratio)
            self._current_total_volume = overflow

            result = self.calculate()

        return result

    def calculate(self) -> VPINResult:
        """Calculate current VPIN from completed buckets."""
        if not self._buckets:
            return VPINResult(
                vpin=0.0,
                num_buckets=0,
                bucket_volume=self.bucket_volume,
                buy_volume_imbalance=0.0,
            )

        # VPIN = mean(|V_buy - V_sell|) / bucket_volume
        imbalances = []
        for buy_vol, sell_vol in self._buckets:
            imbalances.append(abs(buy_vol - sell_vol))

        total_imbalance = sum(imbalances)
        n = len(self._buckets)
        vpin = total_imbalance / (n * self.bucket_volume) if n > 0 else 0.0

        # Clamp to [0, 1]
        vpin = max(0.0, min(1.0, vpin))

        # Buy-side imbalance (positive = more buys)
        total_buy = sum(b for b, _ in self._buckets)
        total_sell = sum(s for _, s in self._buckets)
        total = total_buy + total_sell
        buy_imbalance = (total_buy - total_sell) / total if total > 0 else 0.0

        return VPINResult(
            vpin=vpin,
            num_buckets=n,
            bucket_volume=self.bucket_volume,
            buy_volume_imbalance=buy_imbalance,
        )

    def _classify_tick(self, price: float) -> str:
        """Classify trade as buy/sell using tick rule."""
        if self._last_price is None:
            return 'buy'  # Default first trade
        if price > self._last_price:
            return 'buy'
        elif price < self._last_price:
            return 'sell'
        else:
            return 'buy'  # Zero tick defaults to previous classification

    def reset(self) -> None:
        """Reset all state."""
        self._current_buy_volume = 0.0
        self._current_sell_volume = 0.0
        self._current_total_volume = 0.0
        self._buckets.clear()
        self._last_price = None


class ToxicFlowDetector:
    """
    Detects toxic (adversely selected) order flow.

    Tracks post-fill price movements to determine if your fills
    are systematically followed by adverse price moves, indicating
    you are trading against informed counterparties.

    Example:
        detector = ToxicFlowDetector()
        detector.record_fill("AAPL", "buy", 100, 150.00, mid=150.01)
        # ... later, after price updates ...
        detector.update_post_fill_prices("AAPL", mid_now=149.50, horizon_seconds=60)
        metrics = detector.get_metrics("AAPL")
    """

    def __init__(
        self,
        lookback_trades: int = 500,
        toxic_threshold_bps: float = 5.0,
    ):
        """
        Args:
            lookback_trades: Number of recent trades to analyze
            toxic_threshold_bps: Adverse move threshold to flag as toxic
        """
        self.lookback_trades = lookback_trades
        self.toxic_threshold_bps = toxic_threshold_bps
        self._trades: Dict[str, Deque[TradeRecord]] = {}
        self._pending_updates: Dict[str, List[TradeRecord]] = {}

    def record_fill(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        mid_price: float,
        is_maker: bool = False,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a fill for adverse selection tracking.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Filled quantity
            fill_price: Actual fill price
            mid_price: Mid price at time of fill
            is_maker: Whether this was a maker (limit) fill
            timestamp: Fill timestamp
        """
        ts = timestamp or datetime.now()
        record = TradeRecord(
            timestamp=ts,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            mid_price_at_fill=mid_price,
            is_maker=is_maker,
        )

        if symbol not in self._trades:
            self._trades[symbol] = deque(maxlen=self.lookback_trades)
        self._trades[symbol].append(record)

        if symbol not in self._pending_updates:
            self._pending_updates[symbol] = []
        self._pending_updates[symbol].append(record)

    def update_post_fill_prices(
        self,
        symbol: str,
        current_mid: float,
        current_time: Optional[datetime] = None,
    ) -> None:
        """
        Update post-fill mid prices for pending trades.

        Call this periodically with current market data to populate
        the post-fill price impact fields.

        Args:
            symbol: Trading symbol
            current_mid: Current mid price
            current_time: Current timestamp
        """
        now = current_time or datetime.now()
        pending = self._pending_updates.get(symbol, [])
        still_pending = []

        for trade in pending:
            elapsed = (now - trade.timestamp).total_seconds()

            if trade.mid_price_1s is None and elapsed >= 1:
                trade.mid_price_1s = current_mid
            if trade.mid_price_5s is None and elapsed >= 5:
                trade.mid_price_5s = current_mid
            if trade.mid_price_30s is None and elapsed >= 30:
                trade.mid_price_30s = current_mid
            if trade.mid_price_60s is None and elapsed >= 60:
                trade.mid_price_60s = current_mid
            if trade.mid_price_300s is None and elapsed >= 300:
                trade.mid_price_300s = current_mid

            # Keep in pending if not all horizons filled
            if trade.mid_price_300s is None:
                still_pending.append(trade)

        self._pending_updates[symbol] = still_pending

    def get_metrics(
        self,
        symbol: str,
        lookback_minutes: int = 60,
    ) -> Optional[AdverseSelectionMetrics]:
        """
        Calculate adverse selection metrics for a symbol.

        Args:
            symbol: Trading symbol
            lookback_minutes: Analysis window in minutes

        Returns:
            AdverseSelectionMetrics or None if insufficient data
        """
        trades = self._trades.get(symbol)
        if not trades or len(trades) < 5:
            return None

        cutoff = datetime.now() - timedelta(minutes=lookback_minutes)
        recent = [t for t in trades if t.timestamp >= cutoff]

        if len(recent) < 5:
            # Use all available trades if not enough recent ones
            recent = list(trades)

        effective_spreads = []
        realized_spreads = []
        impacts = {1: [], 5: [], 30: [], 60: [], 300: []}
        buy_adverse = []
        sell_adverse = []

        for trade in recent:
            mid = trade.mid_price_at_fill
            if mid <= 0:
                continue

            # Effective spread: how much you paid vs mid
            if trade.side == 'buy':
                eff_spread = ((trade.fill_price - mid) / mid) * 10000.0
            else:
                eff_spread = ((mid - trade.fill_price) / mid) * 10000.0
            effective_spreads.append(eff_spread)

            # Post-fill impacts at each horizon
            for horizon, mid_after in [
                (1, trade.mid_price_1s),
                (5, trade.mid_price_5s),
                (30, trade.mid_price_30s),
                (60, trade.mid_price_60s),
                (300, trade.mid_price_300s),
            ]:
                if mid_after is not None:
                    if trade.side == 'buy':
                        impact = ((mid_after - mid) / mid) * 10000.0
                    else:
                        impact = ((mid - mid_after) / mid) * 10000.0
                    impacts[horizon].append(impact)

            # Realized spread (using 30s horizon as standard)
            if trade.mid_price_30s is not None:
                if trade.side == 'buy':
                    real_spread = ((trade.fill_price - trade.mid_price_30s) / mid) * 10000.0
                else:
                    real_spread = ((trade.mid_price_30s - trade.fill_price) / mid) * 10000.0
                realized_spreads.append(real_spread)

                # Track per-side adverse selection
                adverse = -real_spread  # Negative realized = adverse
                if trade.side == 'buy':
                    buy_adverse.append(adverse)
                else:
                    sell_adverse.append(adverse)

        avg_effective = float(np.mean(effective_spreads)) if effective_spreads else 0.0
        avg_realized = float(np.mean(realized_spreads)) if realized_spreads else 0.0

        def _safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        # Adverse selection cost = effective - realized
        # Positive = you're paying more than the market moves (adverse)
        adverse_cost = avg_effective - avg_realized

        # Toxicity score: 0-1 based on adverse selection cost
        # Calibrated so 5bps adverse = 0.5 toxicity
        toxicity = min(1.0, max(0.0, adverse_cost / (2 * self.toxic_threshold_bps)))

        timestamps = [t.timestamp for t in recent]

        return AdverseSelectionMetrics(
            symbol=symbol,
            period_start=min(timestamps),
            period_end=max(timestamps),
            num_trades=len(recent),
            avg_effective_spread_bps=avg_effective,
            avg_realized_spread_bps=avg_realized,
            impact_1s_bps=_safe_mean(impacts[1]),
            impact_5s_bps=_safe_mean(impacts[5]),
            impact_30s_bps=_safe_mean(impacts[30]),
            impact_60s_bps=_safe_mean(impacts[60]),
            impact_300s_bps=_safe_mean(impacts[300]),
            adverse_selection_cost_bps=adverse_cost,
            toxicity_score=toxicity,
            buy_adverse_bps=_safe_mean(buy_adverse),
            sell_adverse_bps=_safe_mean(sell_adverse),
        )

    def get_all_metrics(self, lookback_minutes: int = 60) -> Dict[str, AdverseSelectionMetrics]:
        """Get metrics for all tracked symbols."""
        result = {}
        for symbol in self._trades:
            metrics = self.get_metrics(symbol, lookback_minutes)
            if metrics is not None:
                result[symbol] = metrics
        return result

    def should_reduce_size(self, symbol: str) -> Tuple[bool, str]:
        """
        Quick check: should we reduce position size due to toxicity?

        Returns:
            (should_reduce, reason)
        """
        metrics = self.get_metrics(symbol)
        if metrics is None:
            return False, "insufficient_data"

        if metrics.toxicity_score > 0.7:
            return True, f"high_toxicity ({metrics.toxicity_score:.2f})"

        if metrics.adverse_selection_cost_bps > self.toxic_threshold_bps:
            return True, f"high_adverse_cost ({metrics.adverse_selection_cost_bps:.1f}bps)"

        return False, "ok"

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset tracking for a symbol or all symbols."""
        if symbol:
            self._trades.pop(symbol, None)
            self._pending_updates.pop(symbol, None)
        else:
            self._trades.clear()
            self._pending_updates.clear()
