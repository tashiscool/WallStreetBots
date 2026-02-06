"""
Level 2 Order Book Module.

Provides order book data handling, analysis, and feature extraction
for market microstructure-aware execution. Supports real L2 data feeds
and simulated order books from L1 quotes when L2 is unavailable.

References:
- Cont, Stoikov & Talreja (2010) - Order book dynamics
- Cartea, Jaimungal & Penalva (2015) - Algorithmic and HF Trading
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    size: float
    num_orders: int = 1


@dataclass
class OrderBookSnapshot:
    """Point-in-time snapshot of the order book."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def best_bid_size(self) -> float:
        return self.bids[0].size if self.bids else 0.0

    @property
    def best_ask_size(self) -> float:
        return self.asks[0].size if self.asks else 0.0

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        mid = self.mid_price
        spread = self.spread
        if mid and mid > 0 and spread is not None:
            return (spread / mid) * 10000.0
        return None


@dataclass
class OrderBookFeatures:
    """Extracted features for ML models and execution decisions."""
    spread_bps: float
    mid_price: float
    imbalance: float  # -1 (ask heavy) to +1 (bid heavy)
    weighted_imbalance: float  # Size-weighted across levels
    depth_bid_5: float  # Total bid size within 5 levels
    depth_ask_5: float  # Total ask size within 5 levels
    depth_ratio: float  # bid_depth / (bid_depth + ask_depth)
    vwap_bid_5: float  # Volume-weighted avg bid price (5 levels)
    vwap_ask_5: float  # Volume-weighted avg ask price (5 levels)
    microprice: float  # Size-weighted mid price
    bid_slope: float  # Avg size increase per level (bid)
    ask_slope: float  # Avg size increase per level (ask)
    resilience_score: float  # 0-1, how quickly book replenishes
    timestamp: datetime = field(default_factory=datetime.now)


class OrderBook:
    """
    Level 2 order book with real-time updates and feature extraction.

    Supports two modes:
    1. Real L2 data: call update_book() with actual bid/ask levels
    2. Simulated: call update_from_quote() with L1 bid/ask/size data

    Example:
        book = OrderBook("AAPL", max_depth=10)
        book.update_from_quote(bid=150.00, ask=150.02, bid_size=100, ask_size=200)
        features = book.get_features()
    """

    def __init__(
        self,
        symbol: str,
        max_depth: int = 10,
        history_size: int = 100,
    ):
        self.symbol = symbol
        self.max_depth = max_depth
        self._bids: List[OrderBookLevel] = []
        self._asks: List[OrderBookLevel] = []
        self._last_update: Optional[datetime] = None
        self._snapshots: Deque[OrderBookSnapshot] = deque(maxlen=history_size)
        self._is_simulated = False

    @property
    def is_empty(self) -> bool:
        return not self._bids and not self._asks

    @property
    def last_update(self) -> Optional[datetime]:
        return self._last_update

    @property
    def is_simulated(self) -> bool:
        return self._is_simulated

    def update_book(
        self,
        bids: List[Tuple[float, float, int]],
        asks: List[Tuple[float, float, int]],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Update with real L2 data.

        Args:
            bids: List of (price, size, num_orders) sorted descending by price
            asks: List of (price, size, num_orders) sorted ascending by price
            timestamp: Data timestamp
        """
        ts = timestamp or datetime.now()
        self._bids = [
            OrderBookLevel(price=p, size=s, num_orders=n)
            for p, s, n in bids[:self.max_depth]
        ]
        self._asks = [
            OrderBookLevel(price=p, size=s, num_orders=n)
            for p, s, n in asks[:self.max_depth]
        ]
        self._last_update = ts
        self._is_simulated = False

        snap = OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=ts,
            bids=list(self._bids),
            asks=list(self._asks),
        )
        self._snapshots.append(snap)

    def update_from_quote(
        self,
        bid: float,
        ask: float,
        bid_size: float = 100.0,
        ask_size: float = 100.0,
        last_price: Optional[float] = None,
        avg_daily_volume: float = 1_000_000.0,
        volatility: float = 0.02,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Simulate L2 book from L1 quote data.

        Uses empirical microstructure models to generate realistic
        depth around the quoted bid/ask.

        Args:
            bid: Best bid price
            ask: Best ask price
            bid_size: Quoted bid size (shares)
            ask_size: Quoted ask size (shares)
            last_price: Last trade price (for calibration)
            avg_daily_volume: Average daily volume (for depth scaling)
            volatility: Recent realized volatility (for tick spacing)
            timestamp: Quote timestamp
        """
        ts = timestamp or datetime.now()
        spread = ask - bid
        tick = max(0.01, spread * 0.5)  # Min tick = 1 cent

        # Scale depth based on ADV
        depth_scale = max(1.0, avg_daily_volume / 500_000.0)

        bids = []
        asks = []
        for i in range(self.max_depth):
            # Sizes tend to increase away from BBO (empirical)
            level_scale = 1.0 + 0.3 * i
            bid_level_size = bid_size * level_scale * depth_scale
            ask_level_size = ask_size * level_scale * depth_scale

            # Add randomness for realism
            bid_level_size *= (0.8 + 0.4 * ((hash((self.symbol, i, 'b')) % 100) / 100.0))
            ask_level_size *= (0.8 + 0.4 * ((hash((self.symbol, i, 'a')) % 100) / 100.0))

            bid_price = bid - i * tick
            ask_price = ask + i * tick

            if bid_price > 0:
                bids.append(OrderBookLevel(
                    price=round(bid_price, 4),
                    size=round(bid_level_size, 0),
                    num_orders=max(1, int(3 + i * 2)),
                ))

            asks.append(OrderBookLevel(
                price=round(ask_price, 4),
                size=round(ask_level_size, 0),
                num_orders=max(1, int(3 + i * 2)),
            ))

        self._bids = bids
        self._asks = asks
        self._last_update = ts
        self._is_simulated = True

        snap = OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=ts,
            bids=list(self._bids),
            asks=list(self._asks),
        )
        self._snapshots.append(snap)

    def get_snapshot(self) -> Optional[OrderBookSnapshot]:
        """Get current order book snapshot."""
        if self.is_empty:
            return None
        return OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=self._last_update or datetime.now(),
            bids=list(self._bids),
            asks=list(self._asks),
        )

    def get_features(self) -> Optional[OrderBookFeatures]:
        """Extract ML-ready features from the current book state."""
        if self.is_empty:
            return None

        snap = self.get_snapshot()
        if snap is None or snap.mid_price is None:
            return None

        mid = snap.mid_price
        spread_bps = snap.spread_bps or 0.0

        # Top-of-book imbalance
        bb_size = snap.best_bid_size
        ba_size = snap.best_ask_size
        total_top = bb_size + ba_size
        imbalance = (bb_size - ba_size) / total_top if total_top > 0 else 0.0

        # Depth within 5 levels
        n_levels = min(5, len(self._bids), len(self._asks))
        depth_bid = sum(l.size for l in self._bids[:n_levels])
        depth_ask = sum(l.size for l in self._asks[:n_levels])
        total_depth = depth_bid + depth_ask
        depth_ratio = depth_bid / total_depth if total_depth > 0 else 0.5

        # Weighted imbalance across levels (exponential decay)
        w_bid = 0.0
        w_ask = 0.0
        for i in range(n_levels):
            weight = math.exp(-0.5 * i)  # Decay factor
            w_bid += self._bids[i].size * weight if i < len(self._bids) else 0.0
            w_ask += self._asks[i].size * weight if i < len(self._asks) else 0.0
        w_total = w_bid + w_ask
        weighted_imbalance = (w_bid - w_ask) / w_total if w_total > 0 else 0.0

        # VWAP of bid/ask sides (5 levels)
        bid_value = sum(l.price * l.size for l in self._bids[:n_levels])
        ask_value = sum(l.price * l.size for l in self._asks[:n_levels])
        vwap_bid = bid_value / depth_bid if depth_bid > 0 else (snap.best_bid or mid)
        vwap_ask = ask_value / depth_ask if depth_ask > 0 else (snap.best_ask or mid)

        # Microprice (size-weighted mid)
        microprice = mid
        if bb_size + ba_size > 0:
            microprice = (
                snap.best_bid * ba_size + snap.best_ask * bb_size
            ) / (bb_size + ba_size)

        # Slope: average size increase per level
        bid_slope = 0.0
        ask_slope = 0.0
        if n_levels >= 2:
            bid_sizes = [self._bids[i].size for i in range(n_levels)]
            ask_sizes = [self._asks[i].size for i in range(n_levels)]
            bid_slope = (bid_sizes[-1] - bid_sizes[0]) / (n_levels - 1)
            ask_slope = (ask_sizes[-1] - ask_sizes[0]) / (n_levels - 1)

        # Resilience score from snapshot history
        resilience = self._calculate_resilience()

        return OrderBookFeatures(
            spread_bps=spread_bps,
            mid_price=mid,
            imbalance=imbalance,
            weighted_imbalance=weighted_imbalance,
            depth_bid_5=depth_bid,
            depth_ask_5=depth_ask,
            depth_ratio=depth_ratio,
            vwap_bid_5=vwap_bid,
            vwap_ask_5=vwap_ask,
            microprice=microprice,
            bid_slope=bid_slope,
            ask_slope=ask_slope,
            resilience_score=resilience,
            timestamp=self._last_update or datetime.now(),
        )

    def estimate_market_impact(
        self,
        side: str,
        quantity: float,
    ) -> Dict[str, float]:
        """
        Estimate market impact of executing a given quantity.

        Walks the order book to calculate how many levels would be
        consumed and the resulting average fill price vs mid.

        Args:
            side: 'buy' or 'sell'
            quantity: Shares to execute

        Returns:
            Dict with impact_bps, levels_consumed, avg_fill_price, etc.
        """
        if self.is_empty:
            return {
                'impact_bps': 0.0,
                'levels_consumed': 0,
                'avg_fill_price': 0.0,
                'remaining_quantity': quantity,
            }

        snap = self.get_snapshot()
        mid = snap.mid_price if snap else 0.0
        if mid is None or mid <= 0:
            return {
                'impact_bps': 0.0,
                'levels_consumed': 0,
                'avg_fill_price': 0.0,
                'remaining_quantity': quantity,
            }

        levels = self._asks if side == 'buy' else self._bids
        remaining = quantity
        total_cost = 0.0
        levels_consumed = 0

        for level in levels:
            if remaining <= 0:
                break
            fill_qty = min(remaining, level.size)
            total_cost += fill_qty * level.price
            remaining -= fill_qty
            levels_consumed += 1

        filled = quantity - remaining
        avg_price = total_cost / filled if filled > 0 else mid

        if side == 'buy':
            impact_bps = ((avg_price - mid) / mid) * 10000.0 if mid > 0 else 0.0
        else:
            impact_bps = ((mid - avg_price) / mid) * 10000.0 if mid > 0 else 0.0

        return {
            'impact_bps': max(0.0, impact_bps),
            'levels_consumed': levels_consumed,
            'avg_fill_price': avg_price,
            'remaining_quantity': remaining,
            'filled_quantity': filled,
        }

    def get_liquidity_at_bps(self, bps_from_mid: float) -> Dict[str, float]:
        """
        Get total liquidity within N basis points of the mid price.

        Args:
            bps_from_mid: Distance from mid in basis points

        Returns:
            Dict with bid_liquidity, ask_liquidity, total_liquidity
        """
        snap = self.get_snapshot()
        if snap is None or snap.mid_price is None:
            return {'bid_liquidity': 0.0, 'ask_liquidity': 0.0, 'total_liquidity': 0.0}

        mid = snap.mid_price
        threshold = mid * (bps_from_mid / 10000.0)

        bid_liq = sum(
            l.size for l in self._bids
            if mid - l.price <= threshold
        )
        ask_liq = sum(
            l.size for l in self._asks
            if l.price - mid <= threshold
        )

        return {
            'bid_liquidity': bid_liq,
            'ask_liquidity': ask_liq,
            'total_liquidity': bid_liq + ask_liq,
        }

    def _calculate_resilience(self) -> float:
        """
        Calculate book resilience from snapshot history.

        Resilience measures how quickly the book recovers depth
        after a large order consumes liquidity. Returns 0-1 score.
        """
        if len(self._snapshots) < 3:
            return 0.5  # Default when insufficient history

        # Compare depth changes over recent snapshots
        depths = []
        for snap in list(self._snapshots)[-10:]:
            bid_depth = sum(l.size for l in snap.bids[:5])
            ask_depth = sum(l.size for l in snap.asks[:5])
            depths.append(bid_depth + ask_depth)

        if not depths or max(depths) == 0:
            return 0.5

        # Resilience = how stable depth is (low variation = high resilience)
        mean_depth = np.mean(depths)
        std_depth = np.std(depths)
        cv = std_depth / mean_depth if mean_depth > 0 else 1.0

        # Map coefficient of variation to 0-1 score
        # CV of 0 = perfect resilience (1.0), CV > 1 = poor (close to 0)
        return max(0.0, min(1.0, 1.0 - cv))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current state."""
        return {
            'symbol': self.symbol,
            'is_simulated': self._is_simulated,
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'num_bid_levels': len(self._bids),
            'num_ask_levels': len(self._asks),
            'best_bid': self._bids[0].price if self._bids else None,
            'best_ask': self._asks[0].price if self._asks else None,
        }


class OrderBookManager:
    """
    Manages order books for multiple symbols.

    Provides a unified interface for retrieving and updating books.

    Example:
        mgr = OrderBookManager()
        mgr.update_from_quote("AAPL", bid=150.00, ask=150.02)
        features = mgr.get_features("AAPL")
    """

    def __init__(self, max_depth: int = 10, history_size: int = 100):
        self._max_depth = max_depth
        self._history_size = history_size
        self._books: Dict[str, OrderBook] = {}

    def get_or_create(self, symbol: str) -> OrderBook:
        """Get existing order book or create a new one."""
        if symbol not in self._books:
            self._books[symbol] = OrderBook(
                symbol=symbol,
                max_depth=self._max_depth,
                history_size=self._history_size,
            )
        return self._books[symbol]

    def update_from_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: float = 100.0,
        ask_size: float = 100.0,
        **kwargs,
    ) -> None:
        """Update a symbol's book from L1 quote."""
        book = self.get_or_create(symbol)
        book.update_from_quote(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size, **kwargs)

    def update_book(
        self,
        symbol: str,
        bids: List[Tuple[float, float, int]],
        asks: List[Tuple[float, float, int]],
        **kwargs,
    ) -> None:
        """Update a symbol's book from L2 data."""
        book = self.get_or_create(symbol)
        book.update_book(bids=bids, asks=asks, **kwargs)

    def get_features(self, symbol: str) -> Optional[OrderBookFeatures]:
        """Get features for a symbol."""
        book = self._books.get(symbol)
        if book is None:
            return None
        return book.get_features()

    def get_all_features(self) -> Dict[str, OrderBookFeatures]:
        """Get features for all tracked symbols."""
        result = {}
        for symbol, book in self._books.items():
            features = book.get_features()
            if features is not None:
                result[symbol] = features
        return result

    def estimate_impact(self, symbol: str, side: str, quantity: float) -> Dict[str, float]:
        """Estimate market impact for a symbol."""
        book = self._books.get(symbol)
        if book is None:
            return {'impact_bps': 0.0, 'levels_consumed': 0, 'avg_fill_price': 0.0}
        return book.estimate_market_impact(side, quantity)

    @property
    def symbols(self) -> List[str]:
        return list(self._books.keys())

    def remove(self, symbol: str) -> None:
        """Remove a symbol's book."""
        self._books.pop(symbol, None)
