"""Tests for Deflated Sharpe Ratio — Bailey & López de Prado (2014)."""

import math
import pytest

from backend.validation.statistical_rigor.deflated_sharpe import (
    DeflatedSharpeRatio,
    DeflatedSharpeResult,
)


@pytest.fixture
def dsr():
    return DeflatedSharpeRatio(alpha=0.05)


class TestDeflatedSharpeResult:
    def test_dataclass_fields(self):
        r = DeflatedSharpeResult(
            observed_sharpe=1.5,
            deflated_sharpe=0.95,
            p_value=0.05,
            is_significant=True,
            num_trials=100,
            expected_max_sharpe=1.2,
        )
        assert r.observed_sharpe == 1.5
        assert r.num_trials == 100


class TestDeflatedSharpeRatio:
    def test_basic_calculation(self, dsr):
        result = dsr.calculate(
            observed_sharpe=2.0,
            num_trials=1,
            returns_skewness=0.0,
            returns_kurtosis=0.0,
            n_observations=252,
        )
        assert isinstance(result, DeflatedSharpeResult)
        assert result.observed_sharpe == 2.0
        # With 1 trial, expected max is 0 → high PSR → significant
        assert result.is_significant is True

    def test_high_trial_penalty(self, dsr):
        """Many trials should inflate expected max SR and reduce significance."""
        r_low = dsr.calculate(
            observed_sharpe=1.5,
            num_trials=5,
            returns_skewness=0.0,
            returns_kurtosis=0.0,
            n_observations=252,
        )
        r_high = dsr.calculate(
            observed_sharpe=1.5,
            num_trials=1000,
            returns_skewness=0.0,
            returns_kurtosis=0.0,
            n_observations=252,
        )
        # More trials → higher expected max SR → lower deflated Sharpe
        assert r_high.deflated_sharpe < r_low.deflated_sharpe
        assert r_high.expected_max_sharpe > r_low.expected_max_sharpe

    def test_skewness_correction(self, dsr):
        """Negative skewness should reduce significance."""
        r_norm = dsr.calculate(1.0, 10, 0.0, 0.0, 252)
        r_neg_skew = dsr.calculate(1.0, 10, -2.0, 0.0, 252)
        # Negative skew increases SR variance → PSR decreases
        assert r_neg_skew.deflated_sharpe <= r_norm.deflated_sharpe + 0.01

    def test_kurtosis_correction(self, dsr):
        """High excess kurtosis should reduce significance."""
        r_norm = dsr.calculate(1.0, 10, 0.0, 0.0, 252)
        r_fat_tail = dsr.calculate(1.0, 10, 0.0, 6.0, 252)
        # Fat tails widen SR distribution
        assert r_fat_tail.deflated_sharpe <= r_norm.deflated_sharpe + 0.01

    def test_significance_threshold(self, dsr):
        """Low Sharpe with many trials should not be significant."""
        result = dsr.calculate(
            observed_sharpe=0.5,
            num_trials=500,
            returns_skewness=0.0,
            returns_kurtosis=0.0,
            n_observations=252,
        )
        assert result.is_significant is False

    def test_one_trial_expected_max_zero(self, dsr):
        result = dsr.calculate(0.5, 1, 0.0, 0.0, 100)
        assert result.expected_max_sharpe == 0.0

    def test_zero_sharpe(self, dsr):
        result = dsr.calculate(0.0, 10, 0.0, 0.0, 252)
        # Zero Sharpe with trials → definitely not significant
        assert result.is_significant is False

    def test_p_value_range(self, dsr):
        result = dsr.calculate(1.5, 50, 0.0, 0.0, 252)
        assert 0.0 <= result.p_value <= 1.0

    def test_expected_max_increases_with_trials(self, dsr):
        e5 = DeflatedSharpeRatio._expected_max_sharpe(5, 1.0)
        e100 = DeflatedSharpeRatio._expected_max_sharpe(100, 1.0)
        e1000 = DeflatedSharpeRatio._expected_max_sharpe(1000, 1.0)
        assert e5 < e100 < e1000

    def test_psr_normal_case(self, dsr):
        """With normal returns, PSR should equal Φ((SR - SR*) * √(n-1))."""
        psr = DeflatedSharpeRatio._psr(2.0, 0.0, 252, 0.0, 0.0)
        # High SR (2.0) above 0 benchmark with 252 obs → extremely significant
        assert psr > 0.99

        # More moderate case: SR=0.5, benchmark=0.3
        psr2 = DeflatedSharpeRatio._psr(0.5, 0.3, 50, 0.0, 0.0)
        assert 0.0 < psr2 < 1.0
