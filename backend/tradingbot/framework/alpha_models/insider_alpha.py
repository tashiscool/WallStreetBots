"""
Insider Alpha Model — Generates insights from insider transaction data.

Cluster buys are bullish; large insider sales (especially CEO) are bearish.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection

logger = logging.getLogger(__name__)

# Title weighting: higher = more informative signal
TITLE_WEIGHTS: Dict[str, float] = {
    "ceo": 1.0,
    "cfo": 0.9,
    "coo": 0.85,
    "president": 0.85,
    "cto": 0.8,
    "evp": 0.7,
    "svp": 0.65,
    "vp": 0.6,
    "director": 0.5,
    "officer": 0.5,
    "controller": 0.4,
    "secretary": 0.3,
}


class InsiderAlphaModel(AlphaModel):
    """
    Alpha model that converts insider transaction data into directional insights.

    Expects insider data in ``data["insider"]`` keyed by symbol::

        data["insider"] = {
            "AAPL": [InsiderTransaction(...), ...],
        }

    Signals:
    - Cluster buys (3+ insiders buying) → strong bullish
    - Large CEO/CFO buy → bullish
    - Large CEO sale → bearish
    - Director sales → weak bearish (often pre-planned)
    """

    def __init__(
        self,
        cluster_buy_threshold: int = 3,
        large_sale_shares: int = 100000,
        min_confidence: float = 0.3,
        lookback_days: int = 30,
        name: str = "InsiderAlpha",
    ) -> None:
        super().__init__(name)
        self.cluster_buy_threshold = cluster_buy_threshold
        self.large_sale_shares = large_sale_shares
        self.min_confidence = min_confidence
        self.lookback_days = lookback_days

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """Generate insights from insider transaction data."""
        insights: List[Insight] = []
        insider_data = data.get("insider", {})

        for symbol in symbols:
            transactions = insider_data.get(symbol, [])
            if not transactions:
                continue

            analysis = self._analyze_transactions(transactions)
            direction, confidence, metadata = self._determine_signal(analysis)

            if direction == InsightDirection.FLAT:
                continue

            if confidence < self.min_confidence:
                continue

            insights.append(Insight(
                symbol=symbol,
                direction=direction,
                magnitude=0.03,
                confidence=confidence,
                period=timedelta(days=30),
                source_model=self.name,
                metadata=metadata,
            ))

        return insights

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze_transactions(
        self,
        transactions: List[Any],
    ) -> Dict[str, Any]:
        """Aggregate insider transactions into summary metrics."""
        buys: List[Any] = []
        sells: List[Any] = []

        for txn in transactions:
            txn_type = getattr(txn, "transaction_type", "")
            if txn_type == "buy":
                buys.append(txn)
            elif txn_type == "sell":
                sells.append(txn)

        unique_buyers = {getattr(t, "insider_name", "") for t in buys}
        unique_sellers = {getattr(t, "insider_name", "") for t in sells}

        # Calculate weighted buy/sell pressure
        buy_weight = sum(self._get_title_weight(t) for t in buys)
        sell_weight = sum(self._get_title_weight(t) for t in sells)

        # Total shares
        buy_shares = sum(getattr(t, "shares", 0) for t in buys)
        sell_shares = sum(getattr(t, "shares", 0) for t in sells)

        # Check for large sales
        large_sales = [
            t for t in sells
            if getattr(t, "shares", 0) >= self.large_sale_shares
        ]

        return {
            "num_buys": len(buys),
            "num_sells": len(sells),
            "unique_buyers": len(unique_buyers),
            "unique_sellers": len(unique_sellers),
            "buy_weight": buy_weight,
            "sell_weight": sell_weight,
            "buy_shares": buy_shares,
            "sell_shares": sell_shares,
            "large_sales": len(large_sales),
            "is_cluster_buy": len(unique_buyers) >= self.cluster_buy_threshold,
        }

    def _determine_signal(
        self,
        analysis: Dict[str, Any],
    ) -> tuple:
        """Convert analysis into direction + confidence.

        Returns:
            ``(direction, confidence, metadata)``
        """
        direction = InsightDirection.FLAT
        confidence = 0.0
        signals: List[str] = []

        # Cluster buy — strongest bullish signal
        if analysis["is_cluster_buy"]:
            direction = InsightDirection.UP
            confidence = 0.8
            signals.append(f"cluster_buy:{analysis['unique_buyers']}_insiders")

        # Weighted buy pressure exceeds sell pressure
        elif analysis["buy_weight"] > analysis["sell_weight"] and analysis["num_buys"] > 0:
            direction = InsightDirection.UP
            confidence = min(0.3 + analysis["buy_weight"] * 0.1, 0.7)
            signals.append(f"buy_pressure:{analysis['buy_weight']:.2f}")

        # Large insider sales — bearish
        elif analysis["large_sales"] > 0:
            direction = InsightDirection.DOWN
            confidence = min(0.4 + analysis["large_sales"] * 0.15, 0.8)
            signals.append(f"large_sales:{analysis['large_sales']}")

        # Weighted sell pressure dominates
        elif analysis["sell_weight"] > analysis["buy_weight"] and analysis["num_sells"] > 0:
            direction = InsightDirection.DOWN
            confidence = min(0.3 + analysis["sell_weight"] * 0.08, 0.6)
            signals.append(f"sell_pressure:{analysis['sell_weight']:.2f}")

        metadata = {**analysis, "signals": signals}
        return direction, confidence, metadata

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_title_weight(transaction: Any) -> float:
        """Get importance weight for an insider's title."""
        title = getattr(transaction, "title", "").lower()
        for key, weight in TITLE_WEIGHTS.items():
            if key in title:
                return weight
        return 0.3  # Default for unknown titles
