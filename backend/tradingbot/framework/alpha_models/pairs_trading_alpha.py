"""
Pairs Trading Alpha Model

Cointegration-based spread z-score signals.
Inspired by QuantConnect LEAN's PairsTradingAlphaModel.
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class PairsTradingAlphaModel(AlphaModel):
    """
    Generates insights based on cointegration spread z-scores.

    When a pair's spread deviates significantly from its mean,
    generates opposing signals for convergence.
    """

    def __init__(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None,
        lookback_period: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        insight_period: timedelta = timedelta(days=10),
        name: str = "PairsTradingAlpha",
    ):
        super().__init__(name)
        self.pairs = pairs or []
        self.lookback_period = lookback_period
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.insight_period = insight_period

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        insights = []

        # Auto-discover pairs if not specified
        pairs = self.pairs or self._find_pairs(data, symbols)

        for sym_a, sym_b in pairs:
            data_a = data.get(sym_a)
            data_b = data.get(sym_b)
            if data_a is None or data_b is None:
                continue

            closes_a = np.array(data_a.get("close", []))
            closes_b = np.array(data_b.get("close", []))

            min_len = min(len(closes_a), len(closes_b))
            if min_len < self.lookback_period:
                continue

            closes_a = closes_a[-self.lookback_period:]
            closes_b = closes_b[-self.lookback_period:]

            # OLS hedge ratio
            beta = np.cov(closes_a, closes_b)[0, 1] / (np.var(closes_b) + 1e-10)
            spread = closes_a - beta * closes_b

            # Z-score of spread
            mean_spread = np.mean(spread)
            std_spread = np.std(spread)
            if std_spread < 1e-10:
                continue

            z_score = (spread[-1] - mean_spread) / std_spread

            if z_score > self.entry_z:
                # Spread too high: short A, long B
                insights.append(Insight(
                    symbol=sym_a,
                    direction=InsightDirection.DOWN,
                    magnitude=abs(z_score) * 0.01,
                    confidence=min(abs(z_score) / 4, 0.9),
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"pair": sym_b, "z_score": z_score, "beta": beta},
                ))
                insights.append(Insight(
                    symbol=sym_b,
                    direction=InsightDirection.UP,
                    magnitude=abs(z_score) * 0.01,
                    confidence=min(abs(z_score) / 4, 0.9),
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"pair": sym_a, "z_score": z_score, "beta": beta},
                ))

            elif z_score < -self.entry_z:
                # Spread too low: long A, short B
                insights.append(Insight(
                    symbol=sym_a,
                    direction=InsightDirection.UP,
                    magnitude=abs(z_score) * 0.01,
                    confidence=min(abs(z_score) / 4, 0.9),
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"pair": sym_b, "z_score": z_score, "beta": beta},
                ))
                insights.append(Insight(
                    symbol=sym_b,
                    direction=InsightDirection.DOWN,
                    magnitude=abs(z_score) * 0.01,
                    confidence=min(abs(z_score) / 4, 0.9),
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"pair": sym_a, "z_score": z_score, "beta": beta},
                ))

        return insights

    def _find_pairs(
        self,
        data: Dict[str, Any],
        symbols: List[str],
        top_n: int = 5,
    ) -> List[Tuple[str, str]]:
        """Find most correlated pairs from available symbols."""
        valid = []
        for s in symbols:
            closes = data.get(s, {}).get("close", [])
            if len(closes) >= self.lookback_period:
                valid.append((s, np.array(closes[-self.lookback_period:])))

        if len(valid) < 2:
            return []

        pairs_scores = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                sym_a, ca = valid[i]
                sym_b, cb = valid[j]
                corr = np.corrcoef(ca, cb)[0, 1]
                if corr > 0.7:
                    pairs_scores.append((sym_a, sym_b, corr))

        pairs_scores.sort(key=lambda x: x[2], reverse=True)
        return [(a, b) for a, b, _ in pairs_scores[:top_n]]
