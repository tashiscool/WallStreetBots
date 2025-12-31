"""Risk Parity Portfolio Model"""

from decimal import Decimal
from typing import Dict, List, Optional
import numpy as np

from ..portfolio_model import PortfolioConstructionModel, PortfolioState
from ..portfolio_target import PortfolioTarget
from ..insight import Insight


class RiskParityPortfolioModel(PortfolioConstructionModel):
    """
    Risk parity allocation - each position contributes equal risk.

    Higher volatility assets get smaller allocations.
    """

    def __init__(
        self,
        max_positions: int = 15,
        volatility_lookback: int = 20,
        min_weight: float = 0.02,
        max_weight: float = 0.20,
        name: str = "RiskParity",
    ):
        super().__init__(name)
        self.max_positions = max_positions
        self.volatility_lookback = volatility_lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._volatilities: Dict[str, float] = {}

    def update_volatilities(self, returns_data: Dict[str, List[float]]) -> None:
        """Update volatility estimates."""
        for symbol, returns in returns_data.items():
            if len(returns) >= self.volatility_lookback:
                self._volatilities[symbol] = np.std(returns[-self.volatility_lookback:])

    def create_targets(
        self,
        insights: List[Insight],
        portfolio_state: Optional[PortfolioState] = None,
    ) -> List[PortfolioTarget]:
        """Create risk-parity weighted targets."""
        state = portfolio_state or self.portfolio_state

        # Filter to top insights
        insights = sorted(insights, key=lambda x: x.confidence, reverse=True)
        insights = insights[:self.max_positions]

        if not insights:
            return []

        # Get volatilities (use metadata or stored)
        vols = []
        for insight in insights:
            vol = insight.metadata.get(
                'volatility',
                self._volatilities.get(insight.symbol, 0.02)
            )
            vols.append(max(vol, 0.001))  # Minimum volatility

        # Inverse volatility weighting
        inv_vols = [1.0 / v for v in vols]
        total_inv_vol = sum(inv_vols)

        targets = []
        for insight, inv_vol in zip(insights, inv_vols):
            # Weight inversely proportional to volatility
            raw_weight = inv_vol / total_inv_vol
            weight = max(min(raw_weight, self.max_weight), self.min_weight)

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
                metadata={'volatility': vols[insights.index(insight)]},
            ))

        return targets
