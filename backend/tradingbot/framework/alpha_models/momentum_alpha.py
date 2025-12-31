"""
Momentum Alpha Model

Generates insights based on price momentum.
"""

from datetime import timedelta
from typing import Any, Dict, List
import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class MomentumAlphaModel(AlphaModel):
    """
    Alpha model using price momentum for trend-following signals.

    Generates long insights for positive momentum, short for negative.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        momentum_threshold: float = 0.05,  # 5% minimum momentum
        insight_period: timedelta = timedelta(days=10),
        name: str = "MomentumAlpha",
    ):
        """
        Initialize Momentum Alpha Model.

        Args:
            lookback_period: Period to calculate momentum over
            momentum_threshold: Minimum momentum to generate signal
            insight_period: How long insights are valid
            name: Model name
        """
        super().__init__(name)
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.insight_period = insight_period

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """Generate momentum-based insights."""
        insights = []
        momentum_scores = []

        # Calculate momentum for all symbols
        for symbol in symbols:
            symbol_data = data.get(symbol)
            if symbol_data is None:
                continue

            closes = symbol_data.get('close', [])
            if len(closes) < self.lookback_period:
                continue

            # Calculate momentum (rate of change)
            current_price = closes[-1]
            past_price = closes[-self.lookback_period]

            if past_price == 0:
                continue

            momentum = (current_price - past_price) / past_price
            momentum_scores.append((symbol, momentum, symbol_data))

        # Sort by momentum (strongest first)
        momentum_scores.sort(key=lambda x: abs(x[1]), reverse=True)

        # Generate insights for top momentum stocks
        for symbol, momentum, symbol_data in momentum_scores:
            if abs(momentum) < self.momentum_threshold:
                continue

            # Calculate confidence based on momentum strength
            confidence = min(abs(momentum) / 0.20, 0.90)  # Max 90% at 20% momentum

            # Calculate volatility-adjusted magnitude
            closes = np.array(symbol_data.get('close', []))
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.02

            # Expected move based on momentum and volatility
            magnitude = min(abs(momentum) * 0.5, 0.10)  # Half momentum, max 10%

            direction = InsightDirection.UP if momentum > 0 else InsightDirection.DOWN

            insights.append(Insight(
                symbol=symbol,
                direction=direction,
                magnitude=magnitude,
                confidence=0.4 + confidence * 0.4,  # 40-80%
                period=self.insight_period,
                source_model=self.name,
                metadata={
                    'momentum': momentum,
                    'volatility': volatility,
                    'lookback_period': self.lookback_period,
                },
            ))

        return insights
