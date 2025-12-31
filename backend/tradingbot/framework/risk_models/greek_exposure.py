"""Greek Exposure Risk Model"""

from decimal import Decimal
from typing import List, Optional

from ..risk_model import (
    RiskManagementModel,
    RiskAssessment,
    RiskAction,
    PortfolioRiskMetrics,
)
from ..portfolio_target import PortfolioTarget


class GreekExposureRiskModel(RiskManagementModel):
    """
    Limit portfolio Greek exposures for options trading.
    """

    def __init__(
        self,
        max_delta: float = 0.50,  # Max 50% delta exposure
        max_gamma: float = 0.10,  # Max 10% gamma exposure
        max_theta: float = -500,  # Max $500/day theta burn
        max_vega: float = 1000,  # Max $1000 per 1% IV change
        name: str = "GreekExposureRisk",
    ):
        super().__init__(name)
        self.max_delta = max_delta
        self.max_gamma = max_gamma
        self.max_theta = max_theta
        self.max_vega = max_vega

    def manage_risk(
        self,
        targets: List[PortfolioTarget],
        risk_metrics: PortfolioRiskMetrics,
    ) -> List[RiskAssessment]:
        """Assess risk based on Greek exposures."""
        assessments = []

        # Get current portfolio Greeks
        portfolio_delta = risk_metrics.portfolio_delta or 0
        portfolio_gamma = risk_metrics.portfolio_gamma or 0
        portfolio_theta = risk_metrics.portfolio_theta or 0
        portfolio_vega = risk_metrics.portfolio_vega or 0

        for target in targets:
            # Get target's Greeks from metadata
            target_delta = target.metadata.get('delta', 0)
            target_gamma = target.metadata.get('gamma', 0)
            target_theta = target.metadata.get('theta', 0)
            target_vega = target.metadata.get('vega', 0)

            # Check if adding this position would breach limits
            new_delta = portfolio_delta + target_delta
            new_gamma = portfolio_gamma + target_gamma
            new_theta = portfolio_theta + target_theta
            new_vega = portfolio_vega + target_vega

            breaches = []

            if abs(new_delta) > self.max_delta:
                breaches.append(f"Delta {new_delta:.2f} > {self.max_delta}")

            if abs(new_gamma) > self.max_gamma:
                breaches.append(f"Gamma {new_gamma:.2f} > {self.max_gamma}")

            if new_theta < self.max_theta:  # Theta is usually negative
                breaches.append(f"Theta {new_theta:.0f} < {self.max_theta}")

            if abs(new_vega) > self.max_vega:
                breaches.append(f"Vega {new_vega:.0f} > {self.max_vega}")

            if breaches:
                # Try to find acceptable size
                scale_factors = []

                if abs(new_delta) > self.max_delta and target_delta != 0:
                    max_additional = self.max_delta - abs(portfolio_delta)
                    scale_factors.append(max_additional / abs(target_delta))

                if abs(new_gamma) > self.max_gamma and target_gamma != 0:
                    max_additional = self.max_gamma - abs(portfolio_gamma)
                    scale_factors.append(max_additional / abs(target_gamma))

                if scale_factors:
                    scale = max(0, min(scale_factors))
                    if scale > 0.1:  # Only reduce if meaningful size remains
                        adjusted_qty = target.quantity * Decimal(str(scale))
                        assessments.append(RiskAssessment(
                            target=target,
                            action=RiskAction.REDUCE,
                            reason="; ".join(breaches),
                            adjusted_quantity=adjusted_qty,
                            risk_score=0.8,
                            metadata={'scale_factor': scale},
                        ))
                        continue

                # Reject if can't reduce enough
                assessments.append(RiskAssessment(
                    target=target,
                    action=RiskAction.REJECT,
                    reason="; ".join(breaches),
                    risk_score=0.9,
                ))
            else:
                # Calculate risk score based on how close to limits
                delta_util = abs(new_delta) / self.max_delta if self.max_delta else 0
                gamma_util = abs(new_gamma) / self.max_gamma if self.max_gamma else 0
                risk_score = max(delta_util, gamma_util)

                assessments.append(RiskAssessment(
                    target=target,
                    action=RiskAction.ALLOW,
                    risk_score=risk_score,
                    metadata={
                        'delta_utilization': delta_util,
                        'gamma_utilization': gamma_util,
                    },
                ))

        return assessments
