"""
Hierarchical Risk Parity (HRP) Portfolio Model

Lopez de Prado's HRP algorithm using hierarchical clustering
of the correlation matrix for robust portfolio allocation.
"""

from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np

from ..portfolio_model import PortfolioConstructionModel, PortfolioState
from ..portfolio_target import PortfolioTarget
from ..insight import Insight


class HierarchicalRiskParityModel(PortfolioConstructionModel):
    """
    Hierarchical Risk Parity (HRP).

    Uses hierarchical clustering on correlation matrix to build
    a tree-based allocation that is more robust to estimation error
    than mean-variance optimization.
    """

    def __init__(
        self,
        max_positions: int = 20,
        lookback: int = 60,
        min_weight: float = 0.01,
        max_weight: float = 0.30,
        name: str = "HRP",
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
            return self._equal_weight(insights, state)

        # Extract returns
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
        n = returns_matrix.shape[1]

        # Correlation and distance matrices
        corr = np.corrcoef(returns_matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        dist = np.sqrt(0.5 * (1 - corr))

        # Hierarchical clustering (single linkage)
        order = self._quasi_diag(dist, n)

        # Recursive bisection for weights
        cov = np.cov(returns_matrix, rowvar=False)
        weights = self._recursive_bisection(cov, order)

        # Clip and normalize
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights /= weights.sum()

        targets = []
        for i, idx in enumerate(order):
            if idx >= len(valid_insights):
                continue
            ins = valid_insights[idx]
            w = weights[i]
            price = Decimal(str(ins.metadata.get("price", 100)))
            qty = self.calculate_quantity_from_weight(ins.symbol, w, price)
            if ins.is_short:
                qty = -qty
            targets.append(PortfolioTarget(
                symbol=ins.symbol, quantity=qty, target_weight=w,
                source_insight_id=ins.id,
            ))
        return targets

    def _quasi_diag(self, dist: np.ndarray, n: int) -> List[int]:
        """Quasi-diagonalization via hierarchical clustering."""
        # Simple single-linkage clustering
        clusters = {i: [i] for i in range(n)}
        active = list(range(n))

        while len(active) > 1:
            # Find closest pair
            min_dist = float("inf")
            merge_a, merge_b = 0, 1
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    ci = active[i]
                    cj = active[j]
                    # Average linkage distance
                    d = np.mean([
                        dist[a][b]
                        for a in clusters[ci]
                        for b in clusters[cj]
                    ])
                    if d < min_dist:
                        min_dist = d
                        merge_a, merge_b = i, j

            ci = active[merge_a]
            cj = active[merge_b]
            new_id = max(clusters.keys()) + 1
            clusters[new_id] = clusters[ci] + clusters[cj]
            active.pop(merge_b)
            active.pop(merge_a)
            active.append(new_id)

        return clusters[active[0]]

    def _recursive_bisection(
        self, cov: np.ndarray, order: List[int]
    ) -> np.ndarray:
        """Allocate weights via recursive bisection."""
        n = len(order)
        weights = np.ones(n)

        clusters = [list(range(n))]
        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Cluster variances
                left_var = self._cluster_var(cov, [order[i] for i in left])
                right_var = self._cluster_var(cov, [order[i] for i in right])

                alpha = 1 - left_var / (left_var + right_var + 1e-10)

                for i in left:
                    weights[i] *= alpha
                for i in right:
                    weights[i] *= (1 - alpha)

                new_clusters.extend([left, right])
            clusters = [c for c in new_clusters if len(c) > 1]

        return weights

    @staticmethod
    def _cluster_var(cov: np.ndarray, indices: List[int]) -> float:
        """Compute cluster variance using inverse-variance weighting."""
        sub_cov = cov[np.ix_(indices, indices)]
        diag = np.diag(sub_cov)
        diag = np.maximum(diag, 1e-10)
        inv_diag = 1.0 / diag
        w = inv_diag / inv_diag.sum()
        return float(w @ sub_cov @ w)

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
