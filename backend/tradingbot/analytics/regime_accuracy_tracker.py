"""Regime Accuracy Tracker — measures regime detection quality over time.

Tracks predicted vs. actual regimes to surface detection accuracy,
per-regime precision, confusion matrices, and transition probabilities.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from backend.tradingbot.market_regime import MarketRegime


@dataclass
class RegimePrediction:
    """A single regime prediction record."""
    timestamp: datetime
    predicted_regime: MarketRegime
    confidence: float
    actual_regime: Optional[MarketRegime] = None


class ExPostRegimeLabeler:
    """Label regimes ex-post using realised returns and volatility.

    Provides a defensible "ground truth" that avoids circular validation
    by using a fixed-horizon lookforward with simple threshold rules.
    """

    def __init__(
        self,
        horizon: int = 20,
        bull_return_threshold: float = 0.02,
        bear_return_threshold: float = -0.02,
        high_vol_multiplier: float = 1.5,
    ):
        self.horizon = horizon
        self.bull_return_threshold = bull_return_threshold
        self.bear_return_threshold = bear_return_threshold
        self.high_vol_multiplier = high_vol_multiplier

    def label(self, prices: np.ndarray) -> List[Optional[MarketRegime]]:
        """Label each day with a regime based on future realised metrics.

        Parameters
        ----------
        prices : array-like, shape (T,)
            Daily closing prices.

        Returns
        -------
        list of MarketRegime (or None for the last ``horizon`` days)
        """
        prices = np.asarray(prices, dtype=np.float64)
        T = len(prices)
        labels: List[Optional[MarketRegime]] = []

        # Baseline volatility (rolling 60-day)
        daily_returns = np.diff(prices) / prices[:-1]
        baseline_vol = np.std(daily_returns[:min(60, len(daily_returns))]) if len(daily_returns) > 1 else 0.01

        for t in range(T):
            if t + self.horizon >= T:
                labels.append(None)
                continue

            future_ret = (prices[t + self.horizon] - prices[t]) / prices[t]
            future_vol = np.std(
                np.diff(prices[t:t + self.horizon + 1]) / prices[t:t + self.horizon]
            ) if self.horizon > 1 else 0.0

            is_high_vol = future_vol > baseline_vol * self.high_vol_multiplier

            if future_ret > self.bull_return_threshold and not is_high_vol:
                labels.append(MarketRegime.BULL)
            elif future_ret < self.bear_return_threshold:
                labels.append(MarketRegime.BEAR)
            else:
                labels.append(MarketRegime.SIDEWAYS)

            # Update baseline vol with expanding window
            if t + 1 < len(daily_returns):
                baseline_vol = np.std(daily_returns[:t + 1]) if t > 0 else baseline_vol

        return labels


class RegimeAccuracyTracker:
    """Track and evaluate regime detection accuracy.

    Records (predicted, actual) pairs over time and computes rolling
    accuracy, per-regime precision, confusion matrix, and empirical
    transition probabilities.
    """

    REGIMES = [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]

    def __init__(self):
        self._predictions: List[RegimePrediction] = []

    def record_prediction(
        self,
        predicted: MarketRegime,
        confidence: float,
        actual: Optional[MarketRegime] = None,
    ) -> None:
        """Record a new prediction."""
        self._predictions.append(
            RegimePrediction(
                timestamp=datetime.now(),
                predicted_regime=predicted,
                confidence=confidence,
                actual_regime=actual,
            )
        )

    def update_actual(self, timestamp: datetime, actual_regime: MarketRegime) -> None:
        """Back-fill the actual regime for a previously recorded prediction.

        Matches on the prediction whose timestamp is closest to *timestamp*.
        """
        if not self._predictions:
            return

        best_idx = 0
        best_diff = abs((self._predictions[0].timestamp - timestamp).total_seconds())
        for i, p in enumerate(self._predictions[1:], 1):
            diff = abs((p.timestamp - timestamp).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        self._predictions[best_idx].actual_regime = actual_regime

    def get_accuracy(self, window_days: int = 30) -> Dict:
        """Compute accuracy metrics over a rolling window.

        Returns
        -------
        dict with keys: accuracy, precision_per_regime, confusion_matrix, n_evaluated
        """
        cutoff = datetime.now() - timedelta(days=window_days)
        evaluated = [
            p for p in self._predictions
            if p.actual_regime is not None and p.timestamp >= cutoff
        ]

        if not evaluated:
            return {
                'accuracy': 0.0,
                'precision_per_regime': {},
                'confusion_matrix': {},
                'n_evaluated': 0,
            }

        correct = sum(1 for p in evaluated if p.predicted_regime == p.actual_regime)
        accuracy = correct / len(evaluated)

        # Per-regime precision: TP / (TP + FP)
        precision: Dict[str, float] = {}
        for regime in self.REGIMES:
            predicted_as = [p for p in evaluated if p.predicted_regime == regime]
            if predicted_as:
                tp = sum(1 for p in predicted_as if p.actual_regime == regime)
                precision[regime.value] = tp / len(predicted_as)
            else:
                precision[regime.value] = 0.0

        # Confusion matrix: {predicted: {actual: count}}
        cm: Dict[str, Dict[str, int]] = {}
        for pr in self.REGIMES:
            cm[pr.value] = {}
            for ar in self.REGIMES:
                cm[pr.value][ar.value] = sum(
                    1 for p in evaluated
                    if p.predicted_regime == pr and p.actual_regime == ar
                )

        return {
            'accuracy': accuracy,
            'precision_per_regime': precision,
            'confusion_matrix': cm,
            'n_evaluated': len(evaluated),
        }

    def get_regime_transition_matrix(self) -> np.ndarray:
        """Compute empirical transition probability matrix from actual regimes.

        Returns a 3×3 matrix where entry (i, j) is P(next = j | current = i)
        for regimes ordered as [BULL, BEAR, SIDEWAYS].
        """
        evaluated = [p for p in self._predictions if p.actual_regime is not None]
        n = len(self.REGIMES)
        counts = np.zeros((n, n), dtype=float)
        regime_idx = {r: i for i, r in enumerate(self.REGIMES)}

        for i in range(len(evaluated) - 1):
            cur = evaluated[i].actual_regime
            nxt = evaluated[i + 1].actual_regime
            if cur in regime_idx and nxt in regime_idx:
                counts[regime_idx[cur], regime_idx[nxt]] += 1

        # Normalise rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        return counts / row_sums

    def detect_regime_mismatch(
        self,
        current_regime: MarketRegime,
        strategy_assumptions: Dict,
    ) -> List[str]:
        """Warn when current regime conflicts with strategy assumptions.

        Parameters
        ----------
        current_regime : MarketRegime
        strategy_assumptions : dict
            Expected keys: ``expected_regime`` (MarketRegime or str),
            ``min_confidence`` (float, optional).

        Returns
        -------
        list of warning strings (empty = no mismatch)
        """
        warnings: List[str] = []

        expected = strategy_assumptions.get('expected_regime')
        if expected is None:
            return warnings

        # Normalise to MarketRegime
        if isinstance(expected, str):
            try:
                expected = MarketRegime(expected)
            except ValueError:
                return warnings

        if current_regime != expected:
            warnings.append(
                f"Regime mismatch: strategy expects {expected.value} "
                f"but current regime is {current_regime.value}"
            )

        # Check recent confidence
        min_conf = strategy_assumptions.get('min_confidence', 0.0)
        recent = [p for p in self._predictions if p.predicted_regime == current_regime]
        if recent:
            avg_conf = sum(p.confidence for p in recent[-10:]) / min(len(recent), 10)
            if avg_conf < min_conf:
                warnings.append(
                    f"Low regime confidence: avg {avg_conf:.2f} < threshold {min_conf:.2f}"
                )

        return warnings

    def brier_score(self, window_days: int = 30) -> float:
        """Compute Brier score for regime predictions.

        Brier score = mean( (p_predicted - I_actual)^2 ) across all regimes.
        Lower is better; 0 = perfect, 1 = worst.
        Uses confidence as the probability for the predicted regime.
        """
        cutoff = datetime.now() - timedelta(days=window_days)
        evaluated = [
            p for p in self._predictions
            if p.actual_regime is not None and p.timestamp >= cutoff
        ]
        if not evaluated:
            return 1.0

        total = 0.0
        for p in evaluated:
            for regime in self.REGIMES:
                # Probability assigned to this regime
                if regime == p.predicted_regime:
                    prob = p.confidence
                else:
                    prob = (1.0 - p.confidence) / max(len(self.REGIMES) - 1, 1)
                # Indicator: 1 if actual == this regime
                actual = 1.0 if regime == p.actual_regime else 0.0
                total += (prob - actual) ** 2

        return total / len(evaluated)

    def log_loss(self, window_days: int = 30, eps: float = 1e-10) -> float:
        """Compute log loss (cross-entropy) for regime predictions.

        Lower is better; penalises overconfident wrong predictions heavily.
        """
        cutoff = datetime.now() - timedelta(days=window_days)
        evaluated = [
            p for p in self._predictions
            if p.actual_regime is not None and p.timestamp >= cutoff
        ]
        if not evaluated:
            return float('inf')

        total = 0.0
        for p in evaluated:
            for regime in self.REGIMES:
                if regime == p.predicted_regime:
                    prob = max(p.confidence, eps)
                else:
                    prob = max((1.0 - p.confidence) / max(len(self.REGIMES) - 1, 1), eps)
                actual = 1.0 if regime == p.actual_regime else 0.0
                if actual > 0:
                    total -= math.log(prob)

        return total / len(evaluated)
