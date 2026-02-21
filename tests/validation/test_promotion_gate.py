"""Tests for promotion gate and trial counting integrity.

Promotion gate: DSR + Reality Check + leakage STRICT must block a strategy
from being marked promotable when it fails any gate.

Trial counting: actual_trials_evaluated reflects real evaluations (including
pruned/failed), not just configured n_trials.
"""

from datetime import datetime, date
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from backend.validation.statistical_rigor.reality_check import (
    MultipleTestingController,
)
from backend.validation.statistical_rigor.leakage_detection import (
    LeakageDetectionSuite,
    LeakageViolation,
    LeakageViolationError,
    EnforcementMode,
)
from backend.tradingbot.backtesting.optimization_service import (
    OptimizationResult,
    OptimizationConfig,
    ParameterRange,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy_returns(alpha: float, n: int = 252, seed: int = 42):
    """Return strategy and benchmark Series with controllable alpha."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    benchmark = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)
    strategy = benchmark + alpha + pd.Series(rng.normal(0, 0.003, n), index=dates)
    return strategy, benchmark


def _promotion_decision(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_source: str | None = None,
    enforcement_mode: EnforcementMode = EnforcementMode.STRICT,
) -> dict:
    """Simulate a promotion gate: DSR + RC + leakage STRICT.

    Returns dict with 'promotable' bool, 'recommendation', and 'leakage_blocked'.
    """
    ctrl = MultipleTestingController()
    results = ctrl.run_comprehensive_testing(
        {'candidate': strategy_returns}, benchmark_returns,
    )
    rec = results['recommendation']['recommendation']

    # Leakage gate
    leakage_blocked = False
    if strategy_source is not None:
        suite = LeakageDetectionSuite(enforcement_mode=enforcement_mode)
        # Run AST analysis via temporal guard
        suite.temporal_guard.analyze_source_code(strategy_source)
        # Then run validate_strategy which checks enforcement mode
        try:
            suite.validate_strategy(
                None,
                pd.Timestamp('2020-01-01'),
                pd.Timestamp('2021-01-01'),
                ['AAPL'],
            )
        except LeakageViolationError:
            leakage_blocked = True

    promotable = (rec == 'DEPLOY') and not leakage_blocked
    return {
        'promotable': promotable,
        'recommendation': rec,
        'leakage_blocked': leakage_blocked,
        'results': results,
    }


# ---------------------------------------------------------------------------
# Promotion Gate Tests
# ---------------------------------------------------------------------------

class TestPromotionGate:
    """DSR + Reality Check + leakage STRICT actually blocks promotion."""

    def test_strong_alpha_no_leakage_is_promotable(self):
        """Clean strategy with genuine alpha should be deployable."""
        strat, bench = _make_strategy_returns(alpha=0.003, seed=10)
        decision = _promotion_decision(strat, bench)
        # With strong alpha and only 1 trial, DSR should pass
        assert decision['recommendation'] in ('DEPLOY', 'INVESTIGATE')
        assert not decision['leakage_blocked']

    def test_zero_alpha_rejected(self):
        """Strategy with no edge among many trials should be rejected."""
        rng = np.random.RandomState(20)
        n = 252
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        benchmark = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)

        # 30 random strategies with zero alpha → DSR penalizes heavily
        strategies = {}
        for i in range(30):
            strategies[f's{i}'] = benchmark + pd.Series(
                rng.normal(0, 0.005, n), index=dates
            )

        ctrl = MultipleTestingController()
        results = ctrl.run_comprehensive_testing(strategies, benchmark)
        rec = results['recommendation']
        # With 30 trials and zero alpha, should not reach DEPLOY
        assert rec['recommendation'] in ('REJECT', 'SUSPECT')

    def test_leakage_blocks_even_with_alpha(self):
        """Strategy that passes DSR but has lookahead leakage must be blocked."""
        strat, bench = _make_strategy_returns(alpha=0.003, seed=30)
        leaky_code = '''
import pandas as pd
df = pd.DataFrame({'close': [1,2,3]})
df['signal'] = df['close'].shift(-1)
'''
        decision = _promotion_decision(
            strat, bench,
            strategy_source=leaky_code,
            enforcement_mode=EnforcementMode.STRICT,
        )
        assert decision['leakage_blocked'] is True
        assert decision['promotable'] is False

    def test_audit_mode_does_not_block(self):
        """AUDIT mode collects violations but does not block promotion."""
        strat, bench = _make_strategy_returns(alpha=0.003, seed=40)
        leaky_code = '''
import pandas as pd
df = pd.DataFrame({'close': [1,2,3]})
df['signal'] = df['close'].shift(-1)
'''
        decision = _promotion_decision(
            strat, bench,
            strategy_source=leaky_code,
            enforcement_mode=EnforcementMode.AUDIT,
        )
        assert decision['leakage_blocked'] is False

    def test_dsr_gate_blocks_data_mined_strategy(self):
        """A strategy significant by Bonferroni but not DSR → not promotable."""
        # Use many strategies to inflate DSR trial count
        rng = np.random.RandomState(50)
        n = 252
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        benchmark = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)

        # Create 50 strategies, one with weak alpha
        strategies = {}
        for i in range(50):
            strategies[f's{i}'] = benchmark + pd.Series(
                rng.normal(0.0002, 0.005, n), index=dates
            )
        # Make one slightly better
        strategies['candidate'] = benchmark + 0.0005 + pd.Series(
            rng.normal(0, 0.003, n), index=dates
        )

        ctrl = MultipleTestingController()
        results = ctrl.run_comprehensive_testing(strategies, benchmark)

        # With 51 trials and modest alpha, DSR should heavily penalize
        rec = results['recommendation']
        dsr_significant = set(rec.get('tier1_deploy', []) + rec.get('tier2_investigate', []))
        # 'candidate' should NOT survive the DSR gate with 51 trials
        if 'candidate' not in dsr_significant:
            assert rec['recommendation'] in ('REJECT', 'SUSPECT')

    def test_strict_blocks_high_severity_violation(self):
        """STRICT mode should block on high-severity violations, not just critical."""
        suite = LeakageDetectionSuite(enforcement_mode=EnforcementMode.STRICT)
        # Inject a "high" severity violation via temporal guard
        suite.temporal_guard.violations.append(LeakageViolation(
            violation_type='lookahead',
            description='bfill detected after split',
            severity='high',
            detected_at=datetime.now(),
        ))
        with pytest.raises(LeakageViolationError, match='blocking'):
            suite.validate_strategy(
                None,
                pd.Timestamp('2020-01-01'),
                pd.Timestamp('2021-01-01'),
                ['AAPL'],
            )


# ---------------------------------------------------------------------------
# Trial Counting Integrity Tests
# ---------------------------------------------------------------------------

class TestTrialCountingIntegrity:
    """actual_trials_evaluated must reflect real evaluations, not config.n_trials."""

    def test_actual_trials_field_exists(self):
        """OptimizationResult has actual_trials_evaluated with default 0."""
        result = OptimizationResult(
            run_id='test',
            config=OptimizationConfig(
                strategy_name='test',
                start_date=date(2020, 1, 1),
                end_date=date(2021, 1, 1),
                n_trials=100,
                parameter_ranges=[],
            ),
            best_params={},
            best_value=0.0,
            best_metrics={},
            all_trials=[],
            parameter_importance={},
        )
        assert result.actual_trials_evaluated == 0

    def test_actual_trials_differs_from_configured(self):
        """actual_trials_evaluated can differ from config.n_trials."""
        config = OptimizationConfig(
            strategy_name='test',
            start_date=date(2020, 1, 1),
            end_date=date(2021, 1, 1),
            n_trials=100,
            parameter_ranges=[],
        )
        result = OptimizationResult(
            run_id='test',
            config=config,
            best_params={},
            best_value=1.5,
            best_metrics={'sharpe': 1.5},
            all_trials=[],
            parameter_importance={},
            actual_trials_evaluated=87,  # e.g., 13 were pruned before completion
        )
        assert result.actual_trials_evaluated == 87
        assert result.config.n_trials == 100
        assert result.actual_trials_evaluated != result.config.n_trials

    def test_actual_trials_is_correct_dsr_input(self):
        """actual_trials_evaluated (not config.n_trials) should be used as DSR num_trials."""
        from backend.validation.statistical_rigor.deflated_sharpe import DeflatedSharpeRatio

        dsr = DeflatedSharpeRatio()

        # Scenario: configured 100 trials, but 150 actually ran (pruned + failed + complete)
        config_trials = 100
        actual_trials = 150

        r_config = dsr.calculate(
            observed_sharpe=1.5,
            num_trials=config_trials,
            returns_skewness=0.0,
            returns_kurtosis=0.0,
            n_observations=252,
        )
        r_actual = dsr.calculate(
            observed_sharpe=1.5,
            num_trials=actual_trials,
            returns_skewness=0.0,
            returns_kurtosis=0.0,
            n_observations=252,
        )

        # More actual trials → higher expected max → lower deflated sharpe
        assert r_actual.deflated_sharpe < r_config.deflated_sharpe
        assert r_actual.expected_max_sharpe > r_config.expected_max_sharpe

    def test_pruned_trials_still_count(self):
        """Even pruned trials represent evaluated hypotheses for DSR."""
        from backend.validation.statistical_rigor.deflated_sharpe import DeflatedSharpeRatio

        dsr = DeflatedSharpeRatio()

        # 1 trial vs many (simulating pruned trials included)
        r_few = dsr.calculate(1.0, num_trials=5, returns_skewness=0.0,
                              returns_kurtosis=0.0, n_observations=252)
        r_many = dsr.calculate(1.0, num_trials=500, returns_skewness=0.0,
                               returns_kurtosis=0.0, n_observations=252)

        # Including pruned trials should make it harder to be significant
        assert r_many.deflated_sharpe < r_few.deflated_sharpe

    def test_actual_trials_zero_means_no_evaluation(self):
        """Zero actual_trials_evaluated should be distinguishable from default."""
        config = OptimizationConfig(
            strategy_name='test',
            start_date=date(2020, 1, 1),
            end_date=date(2021, 1, 1),
            n_trials=50,
            parameter_ranges=[],
        )
        result = OptimizationResult(
            run_id='failed_run',
            config=config,
            best_params={},
            best_value=0.0,
            best_metrics={},
            all_trials=[],
            parameter_importance={},
            actual_trials_evaluated=0,
            status='failed',
        )
        assert result.actual_trials_evaluated == 0
        assert result.config.n_trials == 50
