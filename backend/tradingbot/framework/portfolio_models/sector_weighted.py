"""Sector-Weighted Portfolio Model"""

from collections import defaultdict
from decimal import Decimal
from typing import ClassVar, Dict, List, Optional

from ..portfolio_model import PortfolioConstructionModel, PortfolioState
from ..portfolio_target import PortfolioTarget
from ..insight import Insight


class SectorWeightedPortfolioModel(PortfolioConstructionModel):
    """
    Sector-constrained portfolio construction.

    Ensures diversification across sectors with configurable limits.
    """

    DEFAULT_SECTOR_WEIGHTS: ClassVar[dict] = {
        'Technology': 0.20,
        'Healthcare': 0.15,
        'Financials': 0.15,
        'Consumer': 0.15,
        'Industrial': 0.10,
        'Energy': 0.10,
        'Utilities': 0.05,
        'Materials': 0.05,
        'Other': 0.05,
    }

    def __init__(
        self,
        sector_weights: Optional[Dict[str, float]] = None,
        max_sector_weight: float = 0.25,
        max_position_weight: float = 0.10,
        name: str = "SectorWeighted",
    ):
        super().__init__(name)
        self.sector_weights = sector_weights or self.DEFAULT_SECTOR_WEIGHTS
        self.max_sector_weight = max_sector_weight
        self.max_position_weight = max_position_weight
        self._symbol_sectors: Dict[str, str] = {}

    def set_symbol_sectors(self, mapping: Dict[str, str]) -> None:
        """Set symbol to sector mapping."""
        self._symbol_sectors = mapping

    def create_targets(
        self,
        insights: List[Insight],
        portfolio_state: Optional[PortfolioState] = None,
    ) -> List[PortfolioTarget]:
        """Create sector-constrained targets."""
        state = portfolio_state or self.portfolio_state

        # Group insights by sector
        sector_insights: Dict[str, List[Insight]] = defaultdict(list)
        for insight in insights:
            sector = insight.metadata.get(
                'sector',
                self._symbol_sectors.get(insight.symbol, 'Other')
            )
            sector_insights[sector].append(insight)

        # Sort each sector by confidence
        for sector in sector_insights:
            sector_insights[sector].sort(key=lambda x: x.confidence, reverse=True)

        targets = []
        sector_allocated: Dict[str, float] = defaultdict(float)

        # Allocate within sector limits
        for sector, weight_limit in self.sector_weights.items():
            insights_in_sector = sector_insights.get(sector, [])
            sector_limit = min(weight_limit, self.max_sector_weight)
            remaining_sector_weight = sector_limit

            for insight in insights_in_sector:
                if remaining_sector_weight <= 0:
                    break

                # Weight based on confidence within sector allocation
                position_weight = min(
                    insight.confidence * sector_limit / len(insights_in_sector),
                    remaining_sector_weight,
                    self.max_position_weight,
                )

                if position_weight < 0.01:  # Minimum 1%
                    continue

                current_price = Decimal(str(insight.metadata.get('price', 100)))
                quantity = self.calculate_quantity_from_weight(
                    insight.symbol,
                    position_weight,
                    current_price,
                )

                if insight.is_short:
                    quantity = -quantity

                targets.append(PortfolioTarget(
                    symbol=insight.symbol,
                    quantity=quantity,
                    target_weight=position_weight,
                    source_insight_id=insight.id,
                    metadata={'sector': sector},
                ))

                remaining_sector_weight -= position_weight
                sector_allocated[sector] += position_weight

        return targets
