"""
Bollinger Bands Alpha Model

Generates signals based on Bollinger Band breakouts and mean reversion.
Inspired by QuantConnect LEAN's alpha models.
"""

from datetime import timedelta
from typing import Any, Dict, List

import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class BollingerAlphaModel(AlphaModel):
    """
    Bollinger Band alpha model.

    Mean reversion mode: buy at lower band, sell at upper band.
    Breakout mode: buy above upper band, sell below lower band.
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        mode: str = "mean_reversion",  # or "breakout"
        insight_period: timedelta = timedelta(days=5),
        name: str = "BollingerAlpha",
    ):
        super().__init__(name)
        self.period = period
        self.num_std = num_std
        self.mode = mode
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
            if len(closes) < self.period:
                continue

            window = closes[-self.period:]
            sma = np.mean(window)
            std = np.std(window)
            upper = sma + self.num_std * std
            lower = sma - self.num_std * std
            current = closes[-1]

            if std == 0:
                continue

            pct_b = (current - lower) / (upper - lower)

            if self.mode == "mean_reversion":
                if current < lower:
                    confidence = min(0.4 + abs(pct_b) * 0.3, 0.9)
                    insights.append(Insight(
                        symbol=symbol,
                        direction=InsightDirection.UP,
                        magnitude=abs(current - sma) / sma,
                        confidence=confidence,
                        period=self.insight_period,
                        source_model=self.name,
                        metadata={"pct_b": pct_b, "upper": upper, "lower": lower},
                    ))
                elif current > upper:
                    confidence = min(0.4 + abs(pct_b - 1) * 0.3, 0.9)
                    insights.append(Insight(
                        symbol=symbol,
                        direction=InsightDirection.DOWN,
                        magnitude=abs(current - sma) / sma,
                        confidence=confidence,
                        period=self.insight_period,
                        source_model=self.name,
                        metadata={"pct_b": pct_b, "upper": upper, "lower": lower},
                    ))
            else:  # breakout
                if current > upper:
                    confidence = min(0.4 + (pct_b - 1) * 0.2, 0.85)
                    insights.append(Insight(
                        symbol=symbol,
                        direction=InsightDirection.UP,
                        magnitude=abs(current - upper) / upper,
                        confidence=confidence,
                        period=self.insight_period,
                        source_model=self.name,
                        metadata={"pct_b": pct_b, "breakout": "upper"},
                    ))
                elif current < lower:
                    confidence = min(0.4 + abs(pct_b) * 0.2, 0.85)
                    insights.append(Insight(
                        symbol=symbol,
                        direction=InsightDirection.DOWN,
                        magnitude=abs(lower - current) / lower,
                        confidence=confidence,
                        period=self.insight_period,
                        source_model=self.name,
                        metadata={"pct_b": pct_b, "breakout": "lower"},
                    ))

        return insights
