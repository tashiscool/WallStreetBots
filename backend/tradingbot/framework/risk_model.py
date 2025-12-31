"""
Risk Management Model Base Class

RiskManagementModels modify or reject portfolio targets based on risk limits.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import logging

from .portfolio_target import PortfolioTarget
from .insight import Insight

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Actions that risk model can take."""
    ALLOW = "allow"           # Allow target as-is
    REDUCE = "reduce"         # Reduce target quantity
    LIQUIDATE = "liquidate"   # Force liquidation
    REJECT = "reject"         # Reject target entirely
    HALT = "halt"            # Halt all trading


@dataclass
class RiskAssessment:
    """Result of risk assessment for a target."""
    target: PortfolioTarget
    action: RiskAction
    reason: str = ""
    adjusted_quantity: Optional[Decimal] = None
    risk_score: float = 0.0  # 0 = low risk, 1 = high risk
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics."""
    total_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")  # Long - Short
    gross_exposure: Decimal = Decimal("0")  # Long + Short
    leverage: float = 1.0
    var_95: Optional[float] = None  # Value at Risk 95%
    var_99: Optional[float] = None  # Value at Risk 99%
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    beta: Optional[float] = None
    correlation_to_market: Optional[float] = None

    # Greek exposures (for options)
    portfolio_delta: Optional[float] = None
    portfolio_gamma: Optional[float] = None
    portfolio_theta: Optional[float] = None
    portfolio_vega: Optional[float] = None


class RiskManagementModel(ABC):
    """
    Base class for risk management.

    RiskManagementModels examine portfolio targets and either:
    - Allow them to proceed
    - Reduce their size
    - Reject them entirely
    - Force liquidation of existing positions

    Override manage_risk() to implement your risk logic.

    Example:
        class MaxDrawdownRiskModel(RiskManagementModel):
            def __init__(self, max_drawdown=0.10):
                super().__init__("MaxDrawdown")
                self.max_drawdown = max_drawdown

            def manage_risk(self, targets, risk_metrics):
                if risk_metrics.current_drawdown >= self.max_drawdown:
                    # Force liquidation of all positions
                    return [
                        RiskAssessment(t, RiskAction.LIQUIDATE, "Max drawdown exceeded")
                        for t in targets
                    ]
                return [RiskAssessment(t, RiskAction.ALLOW) for t in targets]
    """

    def __init__(self, name: str = "RiskModel"):
        """
        Initialize RiskManagementModel.

        Args:
            name: Name of this model (for tracking)
        """
        self.name = name
        self._is_halted = False
        self._halt_reason = ""
        self._risk_events: List[Dict[str, Any]] = []

    @abstractmethod
    def manage_risk(
        self,
        targets: List[PortfolioTarget],
        risk_metrics: PortfolioRiskMetrics,
    ) -> List[RiskAssessment]:
        """
        Assess and potentially modify portfolio targets.

        This is the main method to override in subclasses.

        Args:
            targets: List of portfolio targets to assess
            risk_metrics: Current portfolio risk metrics

        Returns:
            List of RiskAssessment objects
        """
        pass

    def apply_risk_assessments(
        self,
        assessments: List[RiskAssessment],
    ) -> List[PortfolioTarget]:
        """
        Apply risk assessments to get final targets.

        Args:
            assessments: List of risk assessments

        Returns:
            List of approved/modified portfolio targets
        """
        final_targets = []

        for assessment in assessments:
            if assessment.action == RiskAction.ALLOW:
                final_targets.append(assessment.target)

            elif assessment.action == RiskAction.REDUCE:
                if assessment.adjusted_quantity is not None:
                    modified = assessment.target
                    modified.quantity = assessment.adjusted_quantity
                    final_targets.append(modified)

            elif assessment.action == RiskAction.LIQUIDATE:
                liquidation_target = PortfolioTarget.liquidate(
                    symbol=assessment.target.symbol,
                    source_insight_id=assessment.target.source_insight_id,
                )
                liquidation_target.metadata['liquidation_reason'] = assessment.reason
                final_targets.append(liquidation_target)

            elif assessment.action == RiskAction.REJECT:
                logger.warning(
                    f"{self.name}: Rejected target for {assessment.target.symbol}: "
                    f"{assessment.reason}"
                )
                self._log_risk_event("reject", assessment)

            elif assessment.action == RiskAction.HALT:
                self._is_halted = True
                self._halt_reason = assessment.reason
                logger.error(f"{self.name}: Trading HALTED - {assessment.reason}")
                self._log_risk_event("halt", assessment)

        return final_targets

    def _log_risk_event(self, event_type: str, assessment: RiskAssessment) -> None:
        """Log a risk event."""
        self._risk_events.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'symbol': assessment.target.symbol,
            'action': assessment.action.value,
            'reason': assessment.reason,
            'risk_score': assessment.risk_score,
        })

    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self._is_halted

    def get_halt_reason(self) -> str:
        """Get reason for trading halt."""
        return self._halt_reason

    def resume_trading(self) -> None:
        """Resume trading after halt."""
        logger.info(f"{self.name}: Trading resumed")
        self._is_halted = False
        self._halt_reason = ""

    def calculate_position_risk(
        self,
        symbol: str,
        quantity: Decimal,
        current_price: Decimal,
        portfolio_value: Decimal,
    ) -> float:
        """
        Calculate risk score for a single position.

        Returns:
            Risk score 0-1 (higher = more risky)
        """
        position_value = quantity * current_price
        concentration = float(abs(position_value) / portfolio_value)
        return min(concentration * 10, 1.0)  # 10% position = 1.0 risk

    def get_risk_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent risk events."""
        return self._risk_events[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            'name': self.name,
            'is_halted': self._is_halted,
            'halt_reason': self._halt_reason,
            'risk_events': len(self._risk_events),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, halted={self._is_halted})"
