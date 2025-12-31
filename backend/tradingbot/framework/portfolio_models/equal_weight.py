"""Equal Weight Portfolio Model"""

from decimal import Decimal
from typing import Dict, List, Optional

from ..portfolio_model import PortfolioConstructionModel, PortfolioState
from ..portfolio_target import PortfolioTarget
from ..insight import Insight, InsightDirection


class EqualWeightPortfolioModel(PortfolioConstructionModel):
    """
    Equal weight allocation across all insights.

    Each position gets an equal share of the portfolio.
    """

    def __init__(
        self,
        max_positions: int = 10,
        max_position_weight: float = 0.15,  # Max 15% per position
        long_only: bool = False,
        name: str = "EqualWeight",
    ):
        super().__init__(name)
        self.max_positions = max_positions
        self.max_position_weight = max_position_weight
        self.long_only = long_only

    def create_targets(
        self,
        insights: List[Insight],
        portfolio_state: Optional[PortfolioState] = None,
    ) -> List[PortfolioTarget]:
        """Create equal-weight targets."""
        state = portfolio_state or self.portfolio_state

        # Filter insights
        if self.long_only:
            insights = [i for i in insights if i.is_long]

        # Sort by confidence and take top N
        insights = sorted(insights, key=lambda x: x.confidence, reverse=True)
        insights = insights[:self.max_positions]

        if not insights:
            return []

        # Calculate equal weight
        weight_per_position = min(
            1.0 / len(insights),
            self.max_position_weight
        )

        targets = []
        for insight in insights:
            # Get current price from metadata or estimate
            current_price = Decimal(str(insight.metadata.get('price', 100)))

            # Calculate quantity
            quantity = self.calculate_quantity_from_weight(
                insight.symbol,
                weight_per_position,
                current_price,
            )

            # Adjust for direction
            if insight.is_short:
                quantity = -quantity

            targets.append(PortfolioTarget(
                symbol=insight.symbol,
                quantity=quantity,
                target_weight=weight_per_position,
                source_insight_id=insight.id,
            ))

        return targets
