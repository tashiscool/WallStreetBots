"""Maximum Drawdown Risk Model"""

from typing import List

from ..risk_model import (
    RiskManagementModel,
    RiskAssessment,
    RiskAction,
    PortfolioRiskMetrics,
)
from ..portfolio_target import PortfolioTarget


class MaxDrawdownRiskModel(RiskManagementModel):
    """
    Stop or reduce trading when drawdown exceeds threshold.
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,  # 10% max drawdown
        reduce_at: float = 0.05,  # Start reducing at 5%
        reduction_factor: float = 0.5,  # Reduce positions by 50%
        halt_trading: bool = True,  # Halt all trading at max
        name: str = "MaxDrawdownRisk",
    ):
        super().__init__(name)
        self.max_drawdown = max_drawdown
        self.reduce_at = reduce_at
        self.reduction_factor = reduction_factor
        self.halt_trading = halt_trading

    def manage_risk(
        self,
        targets: List[PortfolioTarget],
        risk_metrics: PortfolioRiskMetrics,
    ) -> List[RiskAssessment]:
        """Assess risk based on current drawdown."""
        current_dd = abs(risk_metrics.current_drawdown)
        assessments = []

        for target in targets:
            if current_dd >= self.max_drawdown:
                # Maximum drawdown exceeded
                if self.halt_trading:
                    assessments.append(RiskAssessment(
                        target=target,
                        action=RiskAction.HALT,
                        reason=f"Max drawdown {current_dd:.1%} >= {self.max_drawdown:.1%}",
                        risk_score=1.0,
                    ))
                else:
                    assessments.append(RiskAssessment(
                        target=target,
                        action=RiskAction.LIQUIDATE,
                        reason=f"Drawdown {current_dd:.1%} exceeds limit",
                        risk_score=1.0,
                    ))

            elif current_dd >= self.reduce_at:
                # Reduce position size
                reduced_qty = target.quantity * self.reduction_factor
                assessments.append(RiskAssessment(
                    target=target,
                    action=RiskAction.REDUCE,
                    reason=f"Drawdown {current_dd:.1%} - reducing position",
                    adjusted_quantity=reduced_qty,
                    risk_score=current_dd / self.max_drawdown,
                ))

            else:
                # Allow
                assessments.append(RiskAssessment(
                    target=target,
                    action=RiskAction.ALLOW,
                    risk_score=current_dd / self.max_drawdown,
                ))

        return assessments
