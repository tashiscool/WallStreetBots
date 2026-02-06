"""
Volatility Alpha Model

ATR/Keltner breakout signals.
Inspired by QuantConnect LEAN's alpha models.
"""

from datetime import timedelta
from typing import Any, Dict, List

import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class VolatilityAlphaModel(AlphaModel):
    """
    Volatility breakout alpha model.

    Generates signals when price breaks out of Keltner Channel
    (EMA +/- ATR multiplier), indicating strong directional moves.
    """

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        insight_period: timedelta = timedelta(days=5),
        name: str = "VolatilityAlpha",
    ):
        super().__init__(name)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
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

            min_len = max(self.ema_period, self.atr_period + 1)
            if len(closes) < min_len:
                continue

            # EMA
            ema = self._ema(closes, self.ema_period)

            # ATR
            atr = self._atr(highs, lows, closes, self.atr_period)
            if atr == 0:
                continue

            upper = ema + self.atr_multiplier * atr
            lower = ema - self.atr_multiplier * atr
            current = closes[-1]

            if current > upper:
                distance = (current - upper) / atr
                confidence = min(0.5 + distance * 0.1, 0.9)
                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=abs(current - ema) / ema,
                    confidence=confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"atr": atr, "ema": ema, "breakout": "upper"},
                ))
            elif current < lower:
                distance = (lower - current) / atr
                confidence = min(0.5 + distance * 0.1, 0.9)
                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=abs(ema - current) / ema,
                    confidence=confidence,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"atr": atr, "ema": ema, "breakout": "lower"},
                ))

        return insights

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        mult = 2.0 / (period + 1)
        ema = data[0]
        for val in data[1:]:
            ema = val * mult + ema * (1 - mult)
        return ema

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        if len(closes) < 2:
            return 0.0
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        return np.mean(tr[-period:])
