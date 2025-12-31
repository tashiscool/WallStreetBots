"""Position Limit Risk Model"""

from decimal import Decimal
from typing import Dict, List

from ..risk_model import (
    RiskManagementModel,
    RiskAssessment,
    RiskAction,
    PortfolioRiskMetrics,
)
from ..portfolio_target import PortfolioTarget


class PositionLimitRiskModel(RiskManagementModel):
    """
    Enforce position concentration limits.
    """

    def __init__(
        self,
        max_position_weight: float = 0.15,  # Max 15% in single position
        max_positions: int = 20,  # Maximum number of positions
        max_leverage: float = 1.0,  # No leverage
        name: str = "PositionLimitRisk",
    ):
        super().__init__(name)
        self.max_position_weight = max_position_weight
        self.max_positions = max_positions
        self.max_leverage = max_leverage
        self._current_positions: Dict[str, float] = {}  # symbol -> weight

    def set_current_positions(self, positions: Dict[str, float]) -> None:
        """Update current position weights."""
        self._current_positions = positions

    def manage_risk(
        self,
        targets: List[PortfolioTarget],
        risk_metrics: PortfolioRiskMetrics,
    ) -> List[RiskAssessment]:
        """Assess risk based on position limits."""
        assessments = []
        current_count = len(self._current_positions)

        for target in targets:
            risk_score = 0.0
            current_weight = self._current_positions.get(target.symbol, 0.0)
            target_weight = target.target_weight

            # Check position weight limit
            if target_weight > self.max_position_weight:
                # Reduce to max weight
                scale = self.max_position_weight / target_weight
                adjusted_qty = target.quantity * Decimal(str(scale))
                assessments.append(RiskAssessment(
                    target=target,
                    action=RiskAction.REDUCE,
                    reason=f"Weight {target_weight:.1%} exceeds max {self.max_position_weight:.1%}",
                    adjusted_quantity=adjusted_qty,
                    risk_score=0.7,
                ))
                continue

            # Check position count limit
            is_new_position = target.symbol not in self._current_positions
            if is_new_position and current_count >= self.max_positions:
                assessments.append(RiskAssessment(
                    target=target,
                    action=RiskAction.REJECT,
                    reason=f"Max positions {self.max_positions} reached",
                    risk_score=0.5,
                ))
                continue

            # Check leverage
            if risk_metrics.leverage > self.max_leverage:
                # Reduce to respect leverage
                scale = self.max_leverage / risk_metrics.leverage
                adjusted_qty = target.quantity * Decimal(str(scale))
                assessments.append(RiskAssessment(
                    target=target,
                    action=RiskAction.REDUCE,
                    reason=f"Leverage {risk_metrics.leverage:.1f}x exceeds max",
                    adjusted_quantity=adjusted_qty,
                    risk_score=0.8,
                ))
                continue

            # Allow
            risk_score = target_weight / self.max_position_weight
            assessments.append(RiskAssessment(
                target=target,
                action=RiskAction.ALLOW,
                risk_score=risk_score,
            ))

        return assessments
