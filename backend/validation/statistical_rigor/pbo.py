"""Combinatorially Symmetric Cross-Validation (CSCV) / PBO.

Bailey, Borwein, López de Prado & Zhu (2017).
Estimates the probability that the best in-sample strategy
underperforms out-of-sample when train/test partitions are reshuffled.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PBOResult:
    """Result from Probability of Backtest Overfitting analysis."""
    pbo: float  # Probability of Backtest Overfitting [0, 1]
    num_combinations: int
    logit_distribution: List[float]  # distribution of logit(rank) across combos
    mean_oos_rank: float
    degradation_ratio: float  # fraction of combos where IS-best underperforms OOS median


class CombinatorialPBO:
    """Combinatorially Symmetric Cross-Validation for PBO estimation.

    Given a (T x N) matrix of strategy returns (T periods, N strategies),
    splits T into S sub-periods, exhaustively partitions them into
    train/test halves, and measures how often the IS-best strategy
    underperforms OOS.
    """

    def __init__(self, n_splits: int = 8):
        """
        Parameters
        ----------
        n_splits : int
            Number of equal sub-periods to divide the data into.
            Must be even. Default 8 gives C(8,4)=70 combinations.
        """
        if n_splits % 2 != 0:
            raise ValueError("n_splits must be even")
        self.n_splits = n_splits

    def evaluate(self, returns_matrix: np.ndarray) -> PBOResult:
        """Run CSCV analysis.

        Parameters
        ----------
        returns_matrix : np.ndarray, shape (T, N)
            Each column is a strategy's return series.

        Returns
        -------
        PBOResult
        """
        T, N = returns_matrix.shape
        S = self.n_splits
        half = S // 2

        # Split into S equal sub-periods
        split_size = T // S
        if split_size < 2:
            raise ValueError(f"Not enough data: {T} rows for {S} splits")

        sub_returns = []
        for i in range(S):
            start = i * split_size
            end = start + split_size if i < S - 1 else T
            sub_returns.append(returns_matrix[start:end])

        # Generate all C(S, S/2) train/test partitions
        indices = list(range(S))
        combinations = list(itertools.combinations(indices, half))
        num_combos = len(combinations)

        logits = []
        oos_underperform_count = 0

        for train_idx in combinations:
            test_idx = tuple(i for i in indices if i not in train_idx)

            # Aggregate IS and OOS returns
            is_returns = np.concatenate([sub_returns[i] for i in train_idx], axis=0)
            oos_returns = np.concatenate([sub_returns[i] for i in test_idx], axis=0)

            # IS performance: mean return per strategy
            is_perf = is_returns.mean(axis=0)  # shape (N,)
            oos_perf = oos_returns.mean(axis=0)  # shape (N,)

            # IS-best strategy
            is_best = int(np.argmax(is_perf))

            # Rank of IS-best in OOS (0 = worst, N-1 = best)
            oos_ranks = np.argsort(np.argsort(oos_perf))
            oos_rank = oos_ranks[is_best]

            # Relative rank ∈ (0, 1)
            relative_rank = (oos_rank + 0.5) / N

            # Logit of relative rank
            logit = math.log(relative_rank / (1.0 - relative_rank + 1e-10))
            logits.append(logit)

            # Does IS-best underperform OOS median?
            if oos_rank < N / 2:
                oos_underperform_count += 1

        pbo = oos_underperform_count / num_combos if num_combos > 0 else 0.0

        return PBOResult(
            pbo=pbo,
            num_combinations=num_combos,
            logit_distribution=logits,
            mean_oos_rank=float(np.mean([l for l in logits])),
            degradation_ratio=pbo,
        )
