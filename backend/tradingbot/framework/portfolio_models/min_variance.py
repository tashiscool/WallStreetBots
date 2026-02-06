"""Minimum Variance Portfolio Model"""

from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np

from ..portfolio_model import PortfolioConstructionModel, PortfolioState
from ..portfolio_target import PortfolioTarget
from ..insight import Insight


class MinVariancePortfolioModel(PortfolioConstructionModel):
    """
    Minimum variance portfolio - minimizes total portfolio volatility.

    Solves: min w'Cw subject to sum(w)=1, w>=0
    Uses analytical solution via inverse covariance matrix.
    """

    def __init__(
        self,
        max_positions: int = 15,
        lookback: int = 60,
        min_weight: float = 0.02,
        max_weight: float = 0.30,
        name: str = "MinVariance",
    ):
        super().__init__(name)
        self.max_positions = max_positions
        self.lookback = lookback
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
            return self._simple_targets(insights, state)

        # Build covariance matrix from metadata returns
        returns_data = []
        valid_insights = []
        for ins in insights:
            rets = ins.metadata.get("returns")
            if rets is not None and len(rets) >= self.lookback:
                returns_data.append(np.array(rets[-self.lookback:]))
                valid_insights.append(ins)

        if len(valid_insights) < 2:
            return self._simple_targets(insights, state)

        returns_matrix = np.column_stack(returns_data)
        cov = np.cov(returns_matrix, rowvar=False)

        # Minimum variance weights via inverse covariance
        try:
            inv_cov = np.linalg.inv(cov + np.eye(len(valid_insights)) * 1e-8)
        except np.linalg.LinAlgError:
            return self._simple_targets(insights, state)

        ones = np.ones(len(valid_insights))
        raw_weights = inv_cov @ ones
        weights = raw_weights / raw_weights.sum()

        # Clip and renormalize
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights /= weights.sum()

        return self._build_targets(valid_insights, weights, state)

    def _simple_targets(self, insights, state):
        if not insights:
            return []
        w = 1.0 / len(insights)
        return self._build_targets(insights, [w] * len(insights), state)

    def _build_targets(self, insights, weights, state):
        targets = []
        for insight, weight in zip(insights, weights):
            price = Decimal(str(insight.metadata.get("price", 100)))
            qty = self.calculate_quantity_from_weight(insight.symbol, weight, price)
            if insight.is_short:
                qty = -qty
            targets.append(PortfolioTarget(
                symbol=insight.symbol,
                quantity=qty,
                target_weight=weight,
                source_insight_id=insight.id,
            ))
        return targets
