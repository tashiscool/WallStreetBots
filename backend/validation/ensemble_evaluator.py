"""Ensemble & Correlation Analysis with Fixed Penalty + Real Portfolio."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


def _sharpe(r: pd.Series) -> float:
    """Calculate Sharpe ratio safely."""
    r = r.dropna()
    if len(r) < 2:
        return 0.0
    std = r.std(ddof=1)
    if std < 1e-10:  # Use threshold for near-zero volatility
        return 0.0
    return float(np.sqrt(252) * r.mean() / std)


class EnsembleValidator:
    """Validates ensemble strategies with correlation analysis."""
    
    def __init__(self, corr_threshold: float = 0.7):
        self.corr_threshold = corr_threshold

    def analyze_strategy_correlations(self, strategy_returns_dict: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze correlations between strategies and optimize ensemble."""
        df = pd.DataFrame(strategy_returns_dict).dropna(how='all')
        df = df.dropna()  # strict align
        
        if df.empty or df.shape[1] < 2:
            return {
                'correlation_matrix': pd.DataFrame(),
                'strategy_clusters': [],
                'redundant_strategies': [],
                'ensemble_performance': {},
                'diversification_ratio': 0.0
            }

        corr = df.corr()
        redundant = self._find_redundant(corr)

        # Equal weight portfolio
        ew = np.ones(df.shape[1]) / df.shape[1]
        ew_series = (df * ew).sum(axis=1)

        # Optimized portfolio
        opt = self._optimize_weights(df)
        portfolio_series = (df * opt['weights']).sum(axis=1)

        return {
            'correlation_matrix': corr,
            'strategy_clusters': self._clusters_from_corr(corr),
            'redundant_strategies': redundant,
            'ensemble_performance': {
                'equal_weight_sharpe': _sharpe(ew_series),
                'optimized_sharpe': _sharpe(portfolio_series),
                'optimal_weights': dict(zip(df.columns, opt['weights'])),
                'optimized_series': portfolio_series
            },
            'diversification_ratio': float((df.std(ddof=1).mean()) / (df.sum(axis=1).std(ddof=1) + 1e-9))
        }

    def _optimize_weights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio weights with correlation penalty."""
        n = df.shape[1]

        def neg_sharpe(w):
            w = np.asarray(w)
            port = (df * w).sum(axis=1)
            return -_sharpe(port)

        def corr_penalty(w):
            # Compute average pairwise correlation for strategies with weight > 0.05
            sel = np.where(np.asarray(w) > 0.05)[0]
            if len(sel) < 2:
                return 0.0
            sub = df.iloc[:, sel].corr().values
            iu = np.triu_indices_from(sub, k=1)
            return float(max(0, sub[iu].mean() - 0.4))  # penalize avg corr above 0.4

        def objective(w): 
            return neg_sharpe(w) + 2.0 * corr_penalty(w)

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        bnds = tuple((0.0, 1.0) for _ in range(n))
        x0 = np.ones(n) / n
        
        try:
            res = minimize(objective, x0=x0, bounds=bnds, constraints=cons)
            w = res.x if res.success else x0
            return {'weights': w, 'success': bool(res.success)}
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            return {'weights': x0, 'success': False}

    @staticmethod
    def _find_redundant(corr: pd.DataFrame, thr: float = 0.9) -> List[str]:
        """Find redundant strategies based on correlation."""
        redundant = set()
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if corr.iloc[i, j] >= thr:
                    redundant.add(cols[j])
        return sorted(redundant)

    @staticmethod
    def _clusters_from_corr(corr: pd.DataFrame) -> List[List[str]]:
        """Simple threshold clustering."""
        thr = 0.7
        clusters = []
        seen = set()
        for c in corr.columns:
            if c in seen: 
                continue
            group = [c] + [k for k in corr.columns if k != c and corr.loc[c, k] >= thr]
            for g in group: 
                seen.add(g)
            clusters.append(sorted(set(group)))
        return clusters