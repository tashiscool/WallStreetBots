"""Kelly Criterion Portfolio Model"""

from decimal import Decimal
from typing import Dict, List, Optional

from ..portfolio_model import PortfolioConstructionModel, PortfolioState
from ..portfolio_target import PortfolioTarget
from ..insight import Insight


class KellyPortfolioModel(PortfolioConstructionModel):
    """
    Kelly Criterion position sizing.

    Optimal bet size = (p * b - q) / b
    where p = win probability, q = 1-p, b = win/loss ratio
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,  # Use 25% of full Kelly (safer)
        max_position_weight: float = 0.15,
        min_edge: float = 0.05,  # Minimum expected edge to trade
        name: str = "Kelly",
    ):
        super().__init__(name)
        self.kelly_fraction = kelly_fraction
        self.max_position_weight = max_position_weight
        self.min_edge = min_edge

    def calculate_kelly_weight(
        self,
        win_probability: float,
        win_loss_ratio: float,
    ) -> float:
        """
        Calculate Kelly optimal weight.

        Args:
            win_probability: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss

        Returns:
            Optimal fraction of capital to bet
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        if win_loss_ratio <= 0:
            return 0.0

        p = win_probability
        q = 1 - p
        b = win_loss_ratio

        kelly = (p * b - q) / b

        # Apply fraction and cap
        kelly = kelly * self.kelly_fraction
        return max(0, min(kelly, self.max_position_weight))

    def create_targets(
        self,
        insights: List[Insight],
        portfolio_state: Optional[PortfolioState] = None,
    ) -> List[PortfolioTarget]:
        """Create Kelly-weighted targets."""
        state = portfolio_state or self.portfolio_state

        targets = []
        for insight in insights:
            # Use confidence as win probability proxy
            win_prob = insight.confidence

            # Estimate win/loss ratio from magnitude
            # Assume symmetric win/loss for simplicity
            win_loss_ratio = 1.5  # Default assumption: 1.5:1 reward:risk

            if insight.metadata.get('win_loss_ratio'):
                win_loss_ratio = insight.metadata['win_loss_ratio']

            # Calculate Kelly weight
            weight = self.calculate_kelly_weight(win_prob, win_loss_ratio)

            # Check minimum edge
            expected_edge = (win_prob * win_loss_ratio - (1 - win_prob)) / 1
            if expected_edge < self.min_edge:
                continue

            if weight <= 0:
                continue

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
                metadata={
                    'kelly_weight': weight,
                    'win_probability': win_prob,
                    'expected_edge': expected_edge,
                },
            ))

        return targets
