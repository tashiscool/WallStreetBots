"""
ATR Reversion Alpha Model

Mean reversion at N*ATR from moving average.
Inspired by QuantConnect LEAN's alpha models.
"""

from datetime import timedelta
from typing import Any, Dict, List

import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class ATRReversionAlphaModel(AlphaModel):
    """
    Mean reversion triggered at extreme ATR distance from moving average.

    When price is N*ATR away from its SMA, expects reversion to mean.
    """

    def __init__(
        self,
        sma_period: int = 20,
        atr_period: int = 14,
        atr_threshold: float = 2.0,
        insight_period: timedelta = timedelta(days=5),
        name: str = "ATRReversionAlpha",
    ):
        super().__init__(name)
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        self.insight_period = insight_period

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        insights = []

        for symbol in symbols:
            symbol_data = data.get(symbol)
            if symbol_data is None:
                continue

            closes = np.array(symbol_data.get("close", []))
            highs = np.array(symbol_data.get("high", closes))
            lows = np.array(symbol_data.get("low", closes))

            min_len = max(self.sma_period, self.atr_period + 1)
            if len(closes) < min_len:
                continue

            sma = np.mean(closes[-self.sma_period:])

            # ATR
            if len(closes) < 2:
                continue
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1]),
                ),
            )
            atr = np.mean(tr[-self.atr_period:])

            if atr == 0:
                continue

            current = closes[-1]
            distance_atr = (current - sma) / atr

            if distance_atr < -self.atr_threshold:
                # Price below SMA by N*ATR -> expect reversion up
                confidence = min(0.5 + abs(distance_atr - self.atr_threshold) * 0.1, 0.9)
                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=abs(current - sma) / sma,
                    confidence=confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"distance_atr": distance_atr, "atr": atr, "sma": sma},
                ))
            elif distance_atr > self.atr_threshold:
                # Price above SMA by N*ATR -> expect reversion down
                confidence = min(0.5 + (distance_atr - self.atr_threshold) * 0.1, 0.9)
                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=abs(current - sma) / sma,
                    confidence=confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"distance_atr": distance_atr, "atr": atr, "sma": sma},
                ))

        return insights
