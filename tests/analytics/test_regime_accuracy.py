"""Tests for Regime Accuracy Tracker."""

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


class TestRecordAndRetrieve:
    def test_record_prediction(self, tracker):
        tracker.record_prediction(MarketRegime.BULL, 0.9, MarketRegime.BULL)
        acc = tracker.get_accuracy(window_days=365)
        assert acc['n_evaluated'] == 1
        assert acc['accuracy'] == 1.0

    def test_update_actual(self, tracker):
        tracker.record_prediction(MarketRegime.BULL, 0.8)
        ts = tracker._predictions[0].timestamp
        tracker.update_actual(ts, MarketRegime.BEAR)
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
