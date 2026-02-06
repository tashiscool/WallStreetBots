"""
Stochastic Alpha Model

Stochastic oscillator crossover signals.
Inspired by QuantConnect LEAN's alpha models.
"""

from datetime import timedelta
from typing import Any, Dict, List

import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class StochasticAlphaModel(AlphaModel):
    """
    Stochastic crossover alpha model.

    Generates signals on %K/%D crossovers in oversold/overbought zones.
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        insight_period: timedelta = timedelta(days=5),
        name: str = "StochasticAlpha",
    ):
        super().__init__(name)
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought
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

            min_len = self.k_period + self.d_period
            if len(closes) < min_len:
                continue

            # Calculate %K values
            k_values = []
            for i in range(self.k_period - 1, len(closes)):
                window_high = np.max(highs[i - self.k_period + 1: i + 1])
                window_low = np.min(lows[i - self.k_period + 1: i + 1])
                if window_high == window_low:
                    k_values.append(50.0)
                else:
                    k_values.append(
                        (closes[i] - window_low) / (window_high - window_low) * 100
                    )

            if len(k_values) < self.d_period + 1:
                continue

            # %D is SMA of %K
            k = np.array(k_values)
            d = np.convolve(k, np.ones(self.d_period) / self.d_period, mode="valid")

            if len(d) < 2:
                continue

            curr_k = k[-1]
            curr_d = d[-1]
            prev_k = k[-2]
            prev_d = d[-2] if len(d) >= 2 else d[-1]

            # Bullish crossover in oversold zone
            if prev_k <= prev_d and curr_k > curr_d and curr_k < self.oversold:
                confidence = 0.5 + (self.oversold - curr_k) / self.oversold * 0.3
                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=0.03,
                    confidence=min(confidence, 0.85),
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"k": curr_k, "d": curr_d, "zone": "oversold"},
                ))

            # Bearish crossover in overbought zone
            elif prev_k >= prev_d and curr_k < curr_d and curr_k > self.overbought:
                confidence = 0.5 + (curr_k - self.overbought) / (100 - self.overbought) * 0.3
                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=0.03,
                    confidence=min(confidence, 0.85),
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={"k": curr_k, "d": curr_d, "zone": "overbought"},
                ))

        return insights
