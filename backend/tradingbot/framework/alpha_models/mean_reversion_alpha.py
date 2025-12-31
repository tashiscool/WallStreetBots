"""
Mean Reversion Alpha Model

Generates insights based on deviation from moving average.
"""

from datetime import timedelta
from typing import Any, Dict, List
import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class MeanReversionAlphaModel(AlphaModel):
    """
    Alpha model betting on mean reversion.

    Generates:
    - Long insight when price is significantly below MA
    - Short insight when price is significantly above MA
    """

    def __init__(
        self,
        ma_period: int = 20,
        num_std_devs: float = 2.0,  # Bollinger-band style threshold
        insight_period: timedelta = timedelta(days=5),
        name: str = "MeanReversionAlpha",
    ):
        """
        Initialize Mean Reversion Alpha Model.

        Args:
            ma_period: Moving average period
            num_std_devs: Number of standard deviations for signal
            insight_period: How long insights are valid
            name: Model name
        """
        super().__init__(name)
        self.ma_period = ma_period
        self.num_std_devs = num_std_devs
        self.insight_period = insight_period

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """Generate mean reversion-based insights."""
        insights = []

        for symbol in symbols:
            symbol_data = data.get(symbol)
            if symbol_data is None:
                continue

            closes = np.array(symbol_data.get('close', []))
            if len(closes) < self.ma_period:
                continue

            # Calculate MA and standard deviation
            ma = np.mean(closes[-self.ma_period:])
            std = np.std(closes[-self.ma_period:])

            if std == 0:
                continue

            current_price = closes[-1]

            # Calculate z-score (number of std devs from mean)
            z_score = (current_price - ma) / std

            # Generate signal if beyond threshold
            if abs(z_score) >= self.num_std_devs:
                # Confidence increases with extremity
                confidence = min(abs(z_score) / 4.0, 0.9)

                # Expected reversion magnitude
                reversion_target = ma
                expected_move = (reversion_target - current_price) / current_price
                magnitude = min(abs(expected_move) * 0.5, 0.08)  # Expect 50% reversion, max 8%

                if z_score < -self.num_std_devs:
                    # Price below MA - expect reversion up
                    direction = InsightDirection.UP
                else:
                    # Price above MA - expect reversion down
                    direction = InsightDirection.DOWN

                insights.append(Insight(
                    symbol=symbol,
                    direction=direction,
                    magnitude=magnitude,
                    confidence=0.4 + confidence * 0.4,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={
                        'z_score': z_score,
                        'moving_average': ma,
                        'std_dev': std,
                        'current_price': current_price,
                    },
                ))

        return insights
