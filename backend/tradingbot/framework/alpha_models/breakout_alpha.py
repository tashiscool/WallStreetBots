"""
Breakout Alpha Model

Generates insights based on price breakouts from ranges.
"""

from datetime import timedelta
from typing import Any, Dict, List
import numpy as np

from ..alpha_model import AlphaModel
from ..insight import Insight, InsightDirection


class BreakoutAlphaModel(AlphaModel):
    """
    Alpha model detecting price breakouts.

    Generates:
    - Long insight on breakout above resistance
    - Short insight on breakdown below support
    """

    def __init__(
        self,
        lookback_period: int = 20,
        breakout_threshold: float = 0.02,  # 2% beyond high/low
        volume_confirmation: bool = True,
        volume_multiplier: float = 1.5,  # Volume must be 1.5x average
        insight_period: timedelta = timedelta(days=5),
        name: str = "BreakoutAlpha",
    ):
        """
        Initialize Breakout Alpha Model.

        Args:
            lookback_period: Period to define support/resistance
            breakout_threshold: Minimum % beyond range to signal
            volume_confirmation: Require volume spike for confirmation
            volume_multiplier: Required volume relative to average
            insight_period: How long insights are valid
            name: Model name
        """
        super().__init__(name)
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volume_confirmation = volume_confirmation
        self.volume_multiplier = volume_multiplier
        self.insight_period = insight_period

    def generate_insights(
        self,
        data: Dict[str, Any],
        symbols: List[str],
    ) -> List[Insight]:
        """Generate breakout-based insights."""
        insights = []

        for symbol in symbols:
            symbol_data = data.get(symbol)
            if symbol_data is None:
                continue

            closes = np.array(symbol_data.get('close', []))
            highs = np.array(symbol_data.get('high', closes))
            lows = np.array(symbol_data.get('low', closes))
            volumes = np.array(symbol_data.get('volume', [0] * len(closes)))

            if len(closes) < self.lookback_period + 1:
                continue

            # Define range (excluding current bar)
            range_highs = highs[-self.lookback_period - 1:-1]
            range_lows = lows[-self.lookback_period - 1:-1]

            resistance = np.max(range_highs)
            support = np.min(range_lows)

            current_close = closes[-1]
            current_high = highs[-1]
            current_low = lows[-1]
            current_volume = volumes[-1] if len(volumes) > 0 else 0

            # Average volume
            avg_volume = np.mean(volumes[-self.lookback_period - 1:-1]) if len(volumes) > self.lookback_period else 0

            # Check volume confirmation
            volume_confirmed = True
            if self.volume_confirmation and avg_volume > 0:
                volume_confirmed = current_volume >= avg_volume * self.volume_multiplier

            # Breakout above resistance
            breakout_pct = (current_close - resistance) / resistance if resistance > 0 else 0
            if breakout_pct >= self.breakout_threshold and volume_confirmed:
                confidence = min(breakout_pct / 0.05, 0.85)  # Max 85% at 5% breakout
                if self.volume_confirmation and avg_volume > 0:
                    volume_factor = min(current_volume / avg_volume / self.volume_multiplier, 1.5)
                    confidence = min(confidence * volume_factor, 0.90)

                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.UP,
                    magnitude=breakout_pct * 2,  # Double the breakout %
                    confidence=0.4 + confidence * 0.4,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={
                        'breakout_type': 'resistance',
                        'resistance': resistance,
                        'breakout_pct': breakout_pct,
                        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 0,
                    },
                ))

            # Breakdown below support
            breakdown_pct = (support - current_close) / support if support > 0 else 0
            if breakdown_pct >= self.breakout_threshold and volume_confirmed:
                confidence = min(breakdown_pct / 0.05, 0.85)
                if self.volume_confirmation and avg_volume > 0:
                    volume_factor = min(current_volume / avg_volume / self.volume_multiplier, 1.5)
                    confidence = min(confidence * volume_factor, 0.90)

                insights.append(Insight(
                    symbol=symbol,
                    direction=InsightDirection.DOWN,
                    magnitude=breakdown_pct * 2,
                    confidence=0.4 + confidence * 0.4,
                    period=self.insight_period,
                    source_model=self.name,
                    metadata={
                        'breakout_type': 'support',
                        'support': support,
                        'breakdown_pct': breakdown_pct,
                        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 0,
                    },
                ))

        return insights
