"""
RSI Alpha Model

Generates insights based on RSI (Relative Strength Index) signals.
"""

from datetime import timedelta
from typing import Any, Dict, List
import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class RSIAlphaModel(AlphaModel):
    """
    Alpha model using RSI for overbought/oversold signals.

    Generates:
    - Long insight when RSI crosses below oversold threshold
    - Short insight when RSI crosses above overbought threshold
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        insight_period: timedelta = timedelta(days=5),
        name: str = "RSIAlpha",
    ):
        """
        Initialize RSI Alpha Model.

        Args:
            rsi_period: RSI calculation period
            oversold: RSI level below which is considered oversold
            overbought: RSI level above which is considered overbought
            insight_period: How long insights are valid
            name: Model name
        """
        super().__init__(name)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.insight_period = insight_period

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """Generate RSI-based insights."""
        insights = []

        for symbol in symbols:
            symbol_data = data.get(symbol)
            if symbol_data is None:
                continue

            closes = symbol_data.get('close', [])
            if len(closes) < self.rsi_period + 1:
                continue

            # Calculate RSI
            rsi = self._calculate_rsi(closes, self.rsi_period)
            if rsi is None:
                continue

            prev_rsi = self._calculate_rsi(closes[:-1], self.rsi_period)

            # Check for crossovers
            if rsi < self.oversold and (prev_rsi is None or prev_rsi >= self.oversold):
                # Oversold crossover - bullish signal
                confidence = min((self.oversold - rsi) / self.oversold, 1.0)
                magnitude = 0.02 + (self.oversold - rsi) / 100  # 2-3% expected move

                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=magnitude,
                    confidence=0.5 + confidence * 0.3,  # 50-80% confidence
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={'rsi': rsi, 'signal': 'oversold'},
                ))

            elif rsi > self.overbought and (prev_rsi is None or prev_rsi <= self.overbought):
                # Overbought crossover - bearish signal
                confidence = min((rsi - self.overbought) / (100 - self.overbought), 1.0)
                magnitude = 0.02 + (rsi - self.overbought) / 100

                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=magnitude,
                    confidence=0.5 + confidence * 0.3,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={'rsi': rsi, 'signal': 'overbought'},
                ))

        return insights

    def _calculate_rsi(self, closes: List[float], period: int) -> float:
        """Calculate RSI from close prices."""
        if len(closes) < period + 1:
            return None

        closes = np.array(closes[-period - 1:])
        deltas = np.diff(closes)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
