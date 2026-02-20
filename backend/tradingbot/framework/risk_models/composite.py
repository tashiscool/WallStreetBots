"""Composite Risk Model"""

from typing import ClassVar, List

from ..risk_model import (
    RiskManagementModel,
    RiskAssessment,
    RiskAction,
    PortfolioRiskMetrics,
)
from ..portfolio_target import PortfolioTarget


class CompositeRiskModel(RiskManagementModel):
    """
    Combines multiple risk models.

    Uses the most restrictive action from all models.
    """

    ACTION_PRIORITY: ClassVar[dict] = {
        RiskAction.HALT: 5,
        RiskAction.LIQUIDATE: 4,
        RiskAction.REJECT: 3,
        RiskAction.REDUCE: 2,
        RiskAction.ALLOW: 1,
    }

    def __init__(
        self,
        risk_models: List[RiskManagementModel],
        name: str = "CompositeRisk",
    ):
        super().__init__(name)
        self.risk_models = risk_models

    def manage_risk(
        self,
        targets: List[PortfolioTarget],
        risk_metrics: PortfolioRiskMetrics,
    ) -> List[RiskAssessment]:
        """Run all risk models and combine results."""
        # Collect assessments from all models
        all_assessments = {}  # target_id -> list of assessments

        for model in self.risk_models:
            model_assessments = model.manage_risk(targets, risk_metrics)
            for assessment in model_assessments:
                target_id = assessment.target.id
                if target_id not in all_assessments:
                    all_assessments[target_id] = []
                all_assessments[target_id].append(assessment)

        # Combine assessments per target
        final_assessments = []

        for target in targets:
            assessments = all_assessments.get(target.id, [])
            if not assessments:
                final_assessments.append(RiskAssessment(
                    target=target,
                    action=RiskAction.ALLOW,
                ))
                continue

            # Find most restrictive action
            most_restrictive = max(
                assessments,
                key=lambda a: self.ACTION_PRIORITY[a.action]
            )

            # If reducing, use smallest adjusted quantity
            if most_restrictive.action == RiskAction.REDUCE:
                reduce_assessments = [
                    a for a in assessments
                    if a.action == RiskAction.REDUCE and a.adjusted_quantity is not None
                ]
                if reduce_assessments:
                    smallest = min(
                        reduce_assessments,
                        key=lambda a: a.adjusted_quantity
                    )
                    most_restrictive = smallest

            # Combine reasons
            reasons = [a.reason for a in assessments if a.reason]
            combined_reason = "; ".join(reasons)

            # Average risk scores
            avg_risk = sum(a.risk_score for a in assessments) / len(assessments)

            # Create combined assessment
            final_assessments.append(RiskAssessment(
                target=target,
                action=most_restrictive.action,
                reason=combined_reason,
                adjusted_quantity=most_restrictive.adjusted_quantity,
                risk_score=avg_risk,
                metadata={
                    'models_checked': len(assessments),
                    'restrictive_model': most_restrictive.target.metadata.get('model', 'unknown'),
                },
            ))

        return final_assessments

    def is_halted(self) -> bool:
        """Check if any sub-model has halted trading."""
        return any(m.is_halted() for m in self.risk_models)

    def get_halt_reason(self) -> str:
        """Get combined halt reasons."""
        reasons = [m.get_halt_reason() for m in self.risk_models if m.is_halted()]
        return "; ".join(reasons)

    def resume_trading(self) -> None:
        """Resume trading on all sub-models."""
        for model in self.risk_models:
            model.resume_trading()
