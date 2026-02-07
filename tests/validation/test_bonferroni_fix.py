"""Tests for Bonferroni p-value fix and DSR integration in reality_check.py."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist

from backend.validation.statistical_rigor.reality_check import (
    MultipleTestingController,
)


@pytest.fixture
def controller():
    return MultipleTestingController()


@pytest.fixture
def sample_data():
    """Generate sample strategy and benchmark returns."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    benchmark = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
    strategies = {
        'strong_alpha': benchmark + 0.001 + pd.Series(np.random.normal(0, 0.003, 252), index=dates),
        'no_alpha': benchmark + pd.Series(np.random.normal(0, 0.008, 252), index=dates),
    }
    return strategies, benchmark


class TestBonferroniFix:
    def test_proper_p_values_match_scipy(self):
        """P-values should come from scipy t-distribution, not hardcoded."""
        np.random.seed(0)
        n = 100
        diff = np.random.normal(0.5, 1.0, n)
        t_stat = diff.mean() / (diff.std() / np.sqrt(n))
        df = n - 1
        expected_p = float(2 * (1 - t_dist.cdf(abs(t_stat), df)))

        # Run via controller
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        strat = pd.Series(diff + 1.0, index=dates)
        bench = pd.Series(np.ones(n), index=dates)

        ctrl = MultipleTestingController()
        result = ctrl._parametric_bonferroni({'test': strat}, bench)
        actual_p = result['individual_results']['test']['p_value']

        assert abs(actual_p - expected_p) < 1e-6

    def test_large_t_stat_small_p(self, controller):
        """Large t-stat → small p-value (not hardcoded 0.05)."""
        np.random.seed(1)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        strat = pd.Series(np.random.normal(5, 1, 200), index=dates)
        bench = pd.Series(np.zeros(200), index=dates)

        result = controller._parametric_bonferroni({'big_alpha': strat}, bench)
        p = result['individual_results']['big_alpha']['p_value']
        assert p < 0.001

    def test_small_t_stat_large_p(self, controller):
        """Near-zero excess returns → large p-value (not hardcoded 0.5)."""
        np.random.seed(2)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        noise = np.random.normal(0.0, 0.01, 200)
        strat = pd.Series(noise, index=dates)
        bench = pd.Series(np.zeros(200), index=dates)

        result = controller._parametric_bonferroni({'no_alpha': strat}, bench)
        p = result['individual_results']['no_alpha']['p_value']
        assert p > 0.05

    def test_bonferroni_correction_with_proper_p(self, controller):
        """Bonferroni alpha should shrink with more strategies."""
        np.random.seed(3)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        strategies = {}
        for i in range(20):
            strategies[f's{i}'] = pd.Series(
                np.random.normal(0.001, 0.01, 200), index=dates
            )
        bench = pd.Series(np.zeros(200), index=dates)

        result = controller._parametric_bonferroni(strategies, bench)
        assert result['bonferroni_alpha'] == pytest.approx(0.05 / 20)

    def test_result_dict_has_p_value_key(self, controller):
        """The fix changes key from 'p_value_approx' to 'p_value'."""
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        strat = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
        bench = pd.Series(np.zeros(100), index=dates)

        result = controller._parametric_bonferroni({'test': strat}, bench)
        entry = result['individual_results']['test']
        assert 'p_value' in entry
        assert 'p_value_approx' not in entry


class TestDSRIntegrationInController:
    def test_comprehensive_testing_includes_dsr(self, controller, sample_data):
        strategies, benchmark = sample_data
        results = controller.run_comprehensive_testing(strategies, benchmark)
        assert 'deflated_sharpe' in results
        assert 'individual_results' in results['deflated_sharpe']
        assert 'significant_strategies' in results['deflated_sharpe']

    def test_dsr_contributes_to_recommendation(self, controller, sample_data):
        strategies, benchmark = sample_data
        results = controller.run_comprehensive_testing(strategies, benchmark)
        rec = results['recommendation']
        # DSR should be among the tests used for consensus
        assert 'deflated_sharpe' in rec['significant_by_test']
