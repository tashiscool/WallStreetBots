"""Black-Litterman Portfolio Model"""

from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np

from ..portfolio_model import PortfolioConstructionModel, PortfolioState
from ..portfolio_target import PortfolioTarget
from ..insight import Insight, InsightDirection


class BlackLittermanPortfolioModel(PortfolioConstructionModel):
    """
    Black-Litterman model.

    Combines market equilibrium returns with investor views (insights)
    to produce a posterior expected return that blends both sources.
    """

    def __init__(
        self,
        max_positions: int = 15,
        lookback: int = 60,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        min_weight: float = 0.02,
        max_weight: float = 0.30,
        name: str = "BlackLitterman",
    ):
        super().__init__(name)
        self.max_positions = max_positions
        self.lookback = lookback
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight

    def create_targets(
        self,
        insights: List[Insight],
        portfolio_state: Optional[PortfolioState] = None,
    ) -> List[PortfolioTarget]:
        state = portfolio_state or self.portfolio_state
        insights = sorted(insights, key=lambda x: x.confidence, reverse=True)
        insights = insights[:self.max_positions]

        if len(insights) < 2:
            return self._equal_weight(insights, state)

        returns_data = []
        valid_insights = []
        for ins in insights:
            rets = ins.metadata.get("returns")
            if rets is not None and len(rets) >= self.lookback:
                returns_data.append(np.array(rets[-self.lookback:]))
                valid_insights.append(ins)

        if len(valid_insights) < 2:
            return self._equal_weight(insights, state)

        n = len(valid_insights)
        returns_matrix = np.column_stack(returns_data)
        cov = np.cov(returns_matrix, rowvar=False)

        # Market-cap weights (use equal if not provided)
        w_mkt = np.ones(n) / n

        # Implied equilibrium returns: pi = delta * C * w_mkt
        pi = self.risk_aversion * cov @ w_mkt

        # Views from insights
        P = np.eye(n)  # Each insight is an absolute view
        Q = np.array([
            ins.magnitude * (1 if ins.direction == InsightDirection.UP else -1)
            for ins in valid_insights
        ])
        # Confidence in views
        omega_diag = np.array([
            (1 - ins.confidence) ** 2 * self.tau for ins in valid_insights
        ])
        omega = np.diag(omega_diag + 1e-10)

        # BL posterior: mu_bl = [(tau*C)^-1 + P'*omega^-1*P]^-1 * [(tau*C)^-1*pi + P'*omega^-1*Q]
        tau_cov = self.tau * cov + np.eye(n) * 1e-8
        try:
            inv_tau_cov = np.linalg.inv(tau_cov)
            inv_omega = np.linalg.inv(omega)
        except np.linalg.LinAlgError:
            return self._equal_weight(insights, state)

        M = inv_tau_cov + P.T @ inv_omega @ P
        try:
            inv_M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return self._equal_weight(insights, state)

        mu_bl = inv_M @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)

        # Optimal weights from posterior
        raw_weights = (1.0 / self.risk_aversion) * np.linalg.inv(
            cov + np.eye(n) * 1e-8
        ) @ mu_bl
        raw_weights = np.maximum(raw_weights, 0)

        if raw_weights.sum() == 0:
            return self._equal_weight(insights, state)

        weights = raw_weights / raw_weights.sum()
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights /= weights.sum()

        targets = []
        for ins, w in zip(valid_insights, weights):
            price = Decimal(str(ins.metadata.get("price", 100)))
            qty = self.calculate_quantity_from_weight(ins.symbol, w, price)
            if ins.is_short:
                qty = -qty
            targets.append(PortfolioTarget(
                symbol=ins.symbol, quantity=qty, target_weight=w,
                source_insight_id=ins.id,
            ))
        return targets

    def _equal_weight(self, insights, state):
        if not insights:
            return []
        w = 1.0 / len(insights)
        targets = []
        for ins in insights:
            price = Decimal(str(ins.metadata.get("price", 100)))
            qty = self.calculate_quantity_from_weight(ins.symbol, w, price)
            if ins.is_short:
                qty = -qty
            targets.append(PortfolioTarget(
                symbol=ins.symbol, quantity=qty, target_weight=w,
                source_insight_id=ins.id,
            ))
        return targets
