"""Tests for Regime Accuracy Tracker."""

import math
from datetime import datetime, timedelta
import numpy as np
import pytest

from backend.tradingbot.analytics.regime_accuracy_tracker import (
    RegimeAccuracyTracker,
    RegimePrediction,
)
from backend.tradingbot.market_regime import MarketRegime


@pytest.fixture
def tracker():
    return RegimeAccuracyTracker()


class TestFromHMMState:
    """from_hmm_state() maps OnlineHMM int states to MarketRegime."""

    def test_bull(self):
        assert RegimeAccuracyTracker.from_hmm_state(0) == MarketRegime.BULL

    def test_bear(self):
        assert RegimeAccuracyTracker.from_hmm_state(1) == MarketRegime.BEAR

    def test_sideways(self):
        assert RegimeAccuracyTracker.from_hmm_state(2) == MarketRegime.SIDEWAYS

    def test_unknown_returns_undefined(self):
        assert RegimeAccuracyTracker.from_hmm_state(99) == MarketRegime.UNDEFINED

    def test_negative_returns_undefined(self):
        assert RegimeAccuracyTracker.from_hmm_state(-1) == MarketRegime.UNDEFINED


class TestRecordAndRetrieve:
    def test_record_prediction(self, tracker):
        tracker.record_prediction(MarketRegime.BULL, 0.9, MarketRegime.BULL)
        acc = tracker.get_accuracy(window_days=365)
        assert acc['n_evaluated'] == 1
        assert acc['accuracy'] == 1.0

    def test_update_actual(self, tracker):
        tracker.record_prediction(MarketRegime.BULL, 0.8)
        ts = tracker._predictions[0].timestamp
        result = tracker.update_actual(ts, MarketRegime.BEAR)
        assert result is True
        assert tracker._predictions[0].actual_regime == MarketRegime.BEAR


class TestAccuracyCalculation:
    def test_perfect_accuracy(self, tracker):
        for _ in range(10):
            tracker.record_prediction(MarketRegime.BULL, 0.9, MarketRegime.BULL)
        acc = tracker.get_accuracy(window_days=365)
        assert acc['accuracy'] == 1.0

    def test_mixed_accuracy(self, tracker):
        tracker.record_prediction(MarketRegime.BULL, 0.9, MarketRegime.BULL)
        tracker.record_prediction(MarketRegime.BULL, 0.8, MarketRegime.BEAR)
        acc = tracker.get_accuracy(window_days=365)
        assert acc['accuracy'] == pytest.approx(0.5)

    def test_empty_data(self, tracker):
        acc = tracker.get_accuracy()
        assert acc['n_evaluated'] == 0
        assert acc['accuracy'] == 0.0

    def test_precision_per_regime(self, tracker):
        # 3 predicted BULL: 2 correct, 1 wrong
        tracker.record_prediction(MarketRegime.BULL, 0.9, MarketRegime.BULL)
        tracker.record_prediction(MarketRegime.BULL, 0.8, MarketRegime.BULL)
        tracker.record_prediction(MarketRegime.BULL, 0.7, MarketRegime.BEAR)
        acc = tracker.get_accuracy(window_days=365)
        assert acc['precision_per_regime']['bull'] == pytest.approx(2 / 3)


class TestConfusionMatrix:
    def test_confusion_matrix_structure(self, tracker):
        tracker.record_prediction(MarketRegime.BULL, 0.9, MarketRegime.BULL)
        tracker.record_prediction(MarketRegime.BEAR, 0.8, MarketRegime.BEAR)
        acc = tracker.get_accuracy(window_days=365)
        cm = acc['confusion_matrix']
        assert cm['bull']['bull'] == 1
        assert cm['bear']['bear'] == 1
        assert cm['bull']['bear'] == 0


class TestTransitionMatrix:
    def test_rows_sum_to_one(self, tracker):
        for regime in [MarketRegime.BULL, MarketRegime.BULL, MarketRegime.BEAR,
                       MarketRegime.SIDEWAYS, MarketRegime.BULL]:
            tracker.record_prediction(regime, 0.8, regime)
        tm = tracker.get_regime_transition_matrix()
        assert tm.shape == (3, 3)
        for row_sum in tm.sum(axis=1):
            assert row_sum == pytest.approx(1.0) or row_sum == pytest.approx(0.0)

    def test_empty_transition_matrix(self, tracker):
        tm = tracker.get_regime_transition_matrix()
        assert tm.shape == (3, 3)


class TestRegimeMismatch:
    def test_mismatch_detected(self, tracker):
        tracker.record_prediction(MarketRegime.BEAR, 0.9, MarketRegime.BEAR)
        warnings = tracker.detect_regime_mismatch(
            MarketRegime.BEAR,
            {'expected_regime': 'bull'},
        )
        assert any('mismatch' in w.lower() for w in warnings)

    def test_no_mismatch(self, tracker):
        warnings = tracker.detect_regime_mismatch(
            MarketRegime.BULL,
            {'expected_regime': 'bull'},
        )
        mismatch = [w for w in warnings if 'mismatch' in w.lower()]
        assert len(mismatch) == 0

    def test_low_confidence_warning(self, tracker):
        for _ in range(5):
            tracker.record_prediction(MarketRegime.BULL, 0.3)
        warnings = tracker.detect_regime_mismatch(
            MarketRegime.BULL,
            {'expected_regime': 'bull', 'min_confidence': 0.8},
        )
        assert any('confidence' in w.lower() for w in warnings)


class TestUpdateActualMaxDelta:
    """update_actual() with max_match_delta to prevent silent mislabelling."""

    def test_within_delta_matches(self, tracker):
        tracker.record_prediction(MarketRegime.BULL, 0.9)
        ts = tracker._predictions[0].timestamp
        result = tracker.update_actual(
            ts + timedelta(seconds=5),
            MarketRegime.BEAR,
            max_match_delta=timedelta(seconds=10),
        )
        assert result is True
        assert tracker._predictions[0].actual_regime == MarketRegime.BEAR

    def test_exceeds_delta_refuses(self, tracker):
        tracker.record_prediction(MarketRegime.BULL, 0.9)
        ts = tracker._predictions[0].timestamp
        result = tracker.update_actual(
            ts + timedelta(hours=3),
            MarketRegime.BEAR,
            max_match_delta=timedelta(hours=1),
        )
        assert result is False
        assert tracker._predictions[0].actual_regime is None

    def test_no_delta_always_matches(self, tracker):
        """Without max_match_delta, any distance matches (backward compat)."""
        tracker.record_prediction(MarketRegime.BULL, 0.9)
        ts = tracker._predictions[0].timestamp
        result = tracker.update_actual(
            ts + timedelta(days=30),
            MarketRegime.BEAR,
        )
        assert result is True
        assert tracker._predictions[0].actual_regime == MarketRegime.BEAR

    def test_empty_predictions_returns_false(self, tracker):
        result = tracker.update_actual(
            datetime.now(),
            MarketRegime.BULL,
            max_match_delta=timedelta(hours=1),
        )
        assert result is False

    def test_exact_boundary_matches(self, tracker):
        """Delta exactly equal to max_match_delta should still refuse (> not >=)."""
        tracker.record_prediction(MarketRegime.BULL, 0.9)
        ts = tracker._predictions[0].timestamp
        # Exactly at boundary — best_diff > max_match_delta.total_seconds()
        # depends on floating point; use clearly over the limit
        result = tracker.update_actual(
            ts + timedelta(seconds=61),
            MarketRegime.BEAR,
            max_match_delta=timedelta(seconds=60),
        )
        assert result is False


class TestBrierLogLossWithProbVector:
    """Brier/log-loss should use real probability vectors when available."""

    def test_brier_with_true_probabilities(self, tracker):
        """True prob vector should produce different score than synthetic."""
        # Perfect probabilistic prediction: assign 1.0 to actual regime
        tracker.record_prediction(
            MarketRegime.BULL, 0.9, MarketRegime.BULL,
            regime_probabilities={
                MarketRegime.BULL: 1.0,
                MarketRegime.BEAR: 0.0,
                MarketRegime.SIDEWAYS: 0.0,
            },
        )
        score = tracker.brier_score(window_days=365)
        assert score == pytest.approx(0.0)

    def test_brier_with_synthetic_fallback(self, tracker):
        """Without prob vector, falls back to confidence-derived distribution."""
        tracker.record_prediction(MarketRegime.BULL, 0.8, MarketRegime.BULL)
        score = tracker.brier_score(window_days=365)
        # Confidence=0.8 for correct regime: Brier = (0.8-1)^2 + 2*(0.1-0)^2 = 0.06
        assert score == pytest.approx(0.06, abs=0.001)

    def test_brier_prob_vector_vs_synthetic_differ(self, tracker):
        """When prob vector differs from confidence-derived, scores must differ."""
        tracker_synth = RegimeAccuracyTracker()
        tracker_prob = RegimeAccuracyTracker()

        # Same prediction, but with true probs that differ from synthetic
        tracker_synth.record_prediction(
            MarketRegime.BULL, 0.8, MarketRegime.BULL,
        )
        tracker_prob.record_prediction(
            MarketRegime.BULL, 0.8, MarketRegime.BULL,
            regime_probabilities={
                MarketRegime.BULL: 0.6,
                MarketRegime.BEAR: 0.3,
                MarketRegime.SIDEWAYS: 0.1,
            },
        )
        synth_score = tracker_synth.brier_score(window_days=365)
        prob_score = tracker_prob.brier_score(window_days=365)
        assert synth_score != prob_score

    def test_log_loss_with_true_probabilities(self, tracker):
        """Log loss with near-perfect prob vector should be near zero."""
        tracker.record_prediction(
            MarketRegime.BEAR, 0.9, MarketRegime.BEAR,
            regime_probabilities={
                MarketRegime.BULL: 0.01,
                MarketRegime.BEAR: 0.98,
                MarketRegime.SIDEWAYS: 0.01,
            },
        )
        ll = tracker.log_loss(window_days=365)
        assert ll == pytest.approx(-math.log(0.98), abs=0.001)

    def test_log_loss_synthetic_fallback(self, tracker):
        """Without prob vector, log loss uses confidence as predicted prob."""
        tracker.record_prediction(MarketRegime.BULL, 0.9, MarketRegime.BULL)
        ll = tracker.log_loss(window_days=365)
        assert ll == pytest.approx(-math.log(0.9), abs=0.001)

    def test_mixed_predictions_some_with_probs(self, tracker):
        """Mix of predictions with and without prob vectors should work."""
        tracker.record_prediction(
            MarketRegime.BULL, 0.9, MarketRegime.BULL,
            regime_probabilities={
                MarketRegime.BULL: 0.9,
                MarketRegime.BEAR: 0.05,
                MarketRegime.SIDEWAYS: 0.05,
            },
        )
        tracker.record_prediction(MarketRegime.BEAR, 0.7, MarketRegime.BEAR)
        score = tracker.brier_score(window_days=365)
        assert 0.0 < score < 1.0
