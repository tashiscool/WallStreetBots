"""Insight-Weighted Portfolio Model"""

from decimal import Decimal
from typing import Dict, List, Optional

from ..portfolio_model import PortfolioConstructionModel, PortfolioState
from ..portfolio_target import PortfolioTarget
from ..insight import Insight, InsightDirection


class InsightWeightedPortfolioModel(PortfolioConstructionModel):
    """
    Weight positions by insight confidence and magnitude.

    Higher confidence/magnitude insights get larger allocations.
    """

    def __init__(
        self,
        max_positions: int = 10,
        max_position_weight: float = 0.20,
        min_confidence: float = 0.4,
        total_weight: float = 0.95,  # Use 95% of portfolio
        name: str = "InsightWeighted",
    ):
        super().__init__(name)
        self.max_positions = max_positions
        self.max_position_weight = max_position_weight
        self.min_confidence = min_confidence
        self.total_weight = total_weight

    def create_targets(
        self,
        insights: List[Insight],
        portfolio_state: Optional[PortfolioState] = None,
    ) -> List[PortfolioTarget]:
        """Create confidence-weighted targets."""
        state = portfolio_state or self.portfolio_state

        # Filter by confidence
        insights = [i for i in insights if i.confidence >= self.min_confidence]

        # Sort and limit
        insights = sorted(
            insights,
            key=lambda x: x.confidence * (1 + x.magnitude),
            reverse=True
        )
        insights = insights[:self.max_positions]

        if not insights:
            return []

        # Calculate weighted scores
        scores = [i.confidence * (1 + i.magnitude) for i in insights]
        total_score = sum(scores)

        targets = []
        for insight, score in zip(insights, scores):
            # Weight proportional to score
            raw_weight = (score / total_score) * self.total_weight
            weight = min(raw_weight, self.max_position_weight)

            current_price = Decimal(str(insight.metadata.get('price', 100)))
            quantity = self.calculate_quantity_from_weight(
                insight.symbol,
                weight,
                current_price,
            )

            if insight.is_short:
                quantity = -quantity

            targets.append(PortfolioTarget(
                symbol=insight.symbol,
                quantity=quantity,
                target_weight=weight,
                source_insight_id=insight.id,
            ))

        return targets
