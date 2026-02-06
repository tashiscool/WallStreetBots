"""Maximum Sharpe Ratio Portfolio Model"""

from decimal import Decimal
from typing import List, Optional

import numpy as np

from ..portfolio_model import PortfolioConstructionModel, PortfolioState
from ..portfolio_target import PortfolioTarget
from ..insight import Insight


class MaxSharpePortfolioModel(PortfolioConstructionModel):
    """
    Maximum Sharpe ratio portfolio (tangency portfolio).

    Finds weights that maximize (mu - rf) / sigma_p.
    Uses closed-form solution when available.
    """

    def __init__(
        self,
        max_positions: int = 15,
        lookback: int = 60,
        risk_free_rate: float = 0.02,
        min_weight: float = 0.02,
        max_weight: float = 0.30,
        name: str = "MaxSharpe",
    ):
        super().__init__(name)
        self.max_positions = max_positions
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate / 252  # Daily
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

        returns_matrix = np.column_stack(returns_data)
        mu = np.mean(returns_matrix, axis=0)
        cov = np.cov(returns_matrix, rowvar=False)

        # Excess returns
        excess = mu - self.risk_free_rate

        # Tangency portfolio: w* = inv(C) * (mu - rf)
        try:
            inv_cov = np.linalg.inv(cov + np.eye(len(valid_insights)) * 1e-8)
        except np.linalg.LinAlgError:
            return self._equal_weight(insights, state)

        raw_weights = inv_cov @ excess
        # Only long positions where expected excess return > 0
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
