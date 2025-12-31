"""Sector Exposure Risk Model"""

from collections import defaultdict
from decimal import Decimal
from typing import Dict, List

from ..risk_model import (
    RiskManagementModel,
    RiskAssessment,
    RiskAction,
    PortfolioRiskMetrics,
)
from ..portfolio_target import PortfolioTarget


class SectorExposureRiskModel(RiskManagementModel):
    """
    Limit sector concentration to ensure diversification.
    """

    def __init__(
        self,
        max_sector_weight: float = 0.30,  # Max 30% in any sector
        sector_mapping: Dict[str, str] = None,  # symbol -> sector
        name: str = "SectorExposureRisk",
    ):
        super().__init__(name)
        self.max_sector_weight = max_sector_weight
        self.sector_mapping = sector_mapping or {}
        self._current_sector_weights: Dict[str, float] = defaultdict(float)

    def set_sector_mapping(self, mapping: Dict[str, str]) -> None:
        """Update symbol to sector mapping."""
        self.sector_mapping = mapping

    def set_current_sector_weights(self, weights: Dict[str, float]) -> None:
        """Update current sector weights."""
        self._current_sector_weights = defaultdict(float, weights)

    def manage_risk(
        self,
        targets: List[PortfolioTarget],
        risk_metrics: PortfolioRiskMetrics,
    ) -> List[RiskAssessment]:
        """Assess risk based on sector concentration."""
        assessments = []

        # Track proposed sector allocations
        proposed_weights = defaultdict(float, self._current_sector_weights)

        for target in targets:
            sector = (
                target.metadata.get('sector') or
                self.sector_mapping.get(target.symbol, 'Other')
            )

            current_sector_weight = proposed_weights[sector]
            target_weight = target.target_weight
            new_sector_weight = current_sector_weight + target_weight

            if new_sector_weight > self.max_sector_weight:
                # Reduce to fit within limit
                available = max(0, self.max_sector_weight - current_sector_weight)

                if available > 0.01:  # At least 1% allocation
                    scale = available / target_weight
                    adjusted_qty = target.quantity * Decimal(str(scale))
                    assessments.append(RiskAssessment(
                        target=target,
                        action=RiskAction.REDUCE,
                        reason=f"Sector {sector} would be {new_sector_weight:.1%}, max {self.max_sector_weight:.1%}",
                        adjusted_quantity=adjusted_qty,
                        risk_score=0.7,
                        metadata={
                            'sector': sector,
                            'current_sector_weight': current_sector_weight,
                            'available_weight': available,
                        },
                    ))
                    proposed_weights[sector] += available
                else:
                    # Reject - no room in sector
                    assessments.append(RiskAssessment(
                        target=target,
                        action=RiskAction.REJECT,
                        reason=f"Sector {sector} at {current_sector_weight:.1%}, no room",
                        risk_score=0.6,
                        metadata={'sector': sector},
                    ))
            else:
                # Allow
                proposed_weights[sector] = new_sector_weight
                assessments.append(RiskAssessment(
                    target=target,
                    action=RiskAction.ALLOW,
                    risk_score=new_sector_weight / self.max_sector_weight,
                    metadata={
                        'sector': sector,
                        'new_sector_weight': new_sector_weight,
                    },
                ))

        return assessments
