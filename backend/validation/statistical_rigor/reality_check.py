"""
White's Reality Check and SPA Test Implementation
Prevents false discovery when testing multiple strategies/parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from numpy.random import default_rng
from scipy.stats import t as t_dist


@dataclass
class RealityCheckResult:
    """Results from Reality Check or SPA test."""
    test_statistics: Dict[str, float]
    p_values: Dict[str, float]
    critical_values: Dict[str, float]
    reject_null: Dict[str, bool]
    best_strategy: Optional[str]
    family_wise_error_rate: float
    bootstrap_iterations: int
    significant_strategies: List[str]


def stationary_bootstrap(series, p=0.1, B=1000, rng=None):
    """Stationary bootstrap for time series with dependence."""
    rng = rng or default_rng(0)
    n = len(series)
    idx = np.arange(n)
    out = np.empty((B, n), dtype=int)
    for b in range(B):
        pos = rng.integers(0, n)
        for t in range(n):
            out[b, t] = pos
            if rng.random() < p:
                pos = rng.integers(0, n)
            else:
                pos = (pos + 1) % n
    return series[out]


def spa_test(candidate_returns: Dict[str, np.ndarray], benchmark: np.ndarray, B=2000):
    """
    Hansen's SPA test (simplified).
    Returns p-values for each candidate vs benchmark.
    """
    X = {k: (v - benchmark) for k, v in candidate_returns.items()}
    means = {k: np.mean(v) for k, v in X.items()}
    pool = np.column_stack(list(X.values()))
    boot = stationary_bootstrap(pool, p=0.1, B=B)
    boot_means = boot.mean(axis=1, keepdims=True)
    pvals = {}
    for i, k in enumerate(X.keys()):
        t0 = means[k]
        tb = boot[:, i] - boot_means[:, 0]
        pvals[k] = float(np.mean(tb >= t0))
    return pvals


class WhitesRealityCheck:
    """White's Reality Check for data snooping bias."""

    def __init__(self, bootstrap_iterations: int = 2000, alpha: float = 0.05):
        self.B = bootstrap_iterations
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)

    def test(self, strategy_returns: Dict[str, pd.Series],
             benchmark_returns: pd.Series) -> RealityCheckResult:
        """Perform White's Reality Check."""
        try:
            # Convert to numpy arrays and align
            aligned_data = self._align_data(strategy_returns, benchmark_returns)
            if aligned_data is None:
                raise ValueError("Insufficient data for Reality Check")

            # Convert to dictionary of numpy arrays for SPA test
            strategy_arrays = {k: v.values for k, v in aligned_data.items() if k != 'benchmark'}
            benchmark_array = aligned_data['benchmark'].values

            # Run SPA test
            p_values = spa_test(strategy_arrays, benchmark_array, self.B)

            # Calculate test statistics and other metrics
            test_stats = {}
            critical_values = {}
            reject_null = {}

            for strategy, returns in strategy_arrays.items():
                excess_returns = returns - benchmark_array
                test_stats[strategy] = float(np.mean(excess_returns))
                critical_values[strategy] = np.quantile(excess_returns, 1 - self.alpha)
                reject_null[strategy] = p_values[strategy] <= self.alpha

            best_strategy = max(test_stats.keys(), key=lambda k: test_stats[k])
            significant_strategies = [k for k, reject in reject_null.items() if reject]

            return RealityCheckResult(
                test_statistics=test_stats,
                p_values=p_values,
                critical_values=critical_values,
                reject_null=reject_null,
                best_strategy=best_strategy,
                family_wise_error_rate=self.alpha,
                bootstrap_iterations=self.B,
                significant_strategies=significant_strategies
            )

        except Exception as e:
            self.logger.error(f"Reality Check failed: {e}")
            raise

    def _align_data(self, strategy_returns: Dict[str, pd.Series],
                   benchmark_returns: pd.Series) -> Optional[pd.DataFrame]:
        """Align all return series to common index."""
        try:
            all_series = {**strategy_returns, 'benchmark': benchmark_returns}
            df = pd.DataFrame(all_series).dropna()

            if len(df) < 50:
                self.logger.warning("Less than 50 observations after alignment")
                return None

            return df

        except Exception as e:
            self.logger.error(f"Data alignment failed: {e}")
            return None


class MultipleTestingController:
    """Controls family-wise error rate across multiple strategy tests."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run_comprehensive_testing(self, strategy_returns: Dict[str, pd.Series],
                                 benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Run multiple testing procedures to control false discoveries."""
        results = {}

        try:
            # White's Reality Check with SPA
            reality_check = WhitesRealityCheck()
            results['reality_check'] = reality_check.test(strategy_returns, benchmark_returns)

            # Bonferroni correction
            results['bonferroni'] = self._parametric_bonferroni(strategy_returns, benchmark_returns)

            # Deflated Sharpe Ratio
            results['deflated_sharpe'] = self._deflated_sharpe_test(strategy_returns, benchmark_returns)

            # Generate recommendation
            results['recommendation'] = self._generate_recommendation(results)

            return results

        except Exception as e:
            self.logger.error(f"Multiple testing failed: {e}")
            raise

    def _parametric_bonferroni(self, strategy_returns: Dict[str, pd.Series],
                              benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Parametric Bonferroni correction (t-distribution p-values, not bootstrap)."""
        try:
            k = len(strategy_returns)
            alpha_bonf = 0.05 / k if k > 0 else 0.05

            results = {}
            for strategy_name, returns in strategy_returns.items():
                aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()

                if len(aligned) < 30:
                    continue

                diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                if len(diff) > 1 and diff.std() > 1e-8:
                    t_stat = diff.mean() / (diff.std() / np.sqrt(len(diff)))
                    df = len(diff) - 1
                    p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df)))

                    results[strategy_name] = {
                        't_statistic': float(t_stat),
                        'p_value': p_value,
                        'significant': p_value < alpha_bonf
                    }

            return {
                'individual_results': results,
                'bonferroni_alpha': alpha_bonf,
                'significant_strategies': [k for k, v in results.items() if v['significant']]
            }

        except Exception as e:
            self.logger.error(f"Bonferroni correction failed: {e}")
            return {}

    def _deflated_sharpe_test(self, strategy_returns: Dict[str, pd.Series],
                             benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Deflated Sharpe Ratio test for each strategy."""
        from .deflated_sharpe import DeflatedSharpeRatio

        try:
            dsr = DeflatedSharpeRatio()
            num_trials = len(strategy_returns)
            results = {}
            significant_strategies = []

            for name, returns in strategy_returns.items():
                aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
                if len(aligned) < 30:
                    continue

                excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                n = len(excess)
                if excess.std() < 1e-8:
                    continue

                observed_sharpe = float(excess.mean() / excess.std()) * np.sqrt(252)
                skew = float(excess.skew())
                kurt = float(excess.kurtosis())  # pandas returns excess kurtosis

                dsr_result = dsr.calculate(
                    observed_sharpe=observed_sharpe,
                    num_trials=num_trials,
                    returns_skewness=skew,
                    returns_kurtosis=kurt,
                    n_observations=n,
                )
                results[name] = {
                    'observed_sharpe': dsr_result.observed_sharpe,
                    'deflated_sharpe': dsr_result.deflated_sharpe,
                    'p_value': dsr_result.p_value,
                    'is_significant': dsr_result.is_significant,
                    'expected_max_sharpe': dsr_result.expected_max_sharpe,
                }
                if dsr_result.is_significant:
                    significant_strategies.append(name)

            return {
                'individual_results': results,
                'num_trials': num_trials,
                'significant_strategies': significant_strategies,
            }

        except Exception as e:
            self.logger.error(f"Deflated Sharpe test failed: {e}")
            return {'individual_results': {}, 'num_trials': 0, 'significant_strategies': []}

    def _generate_recommendation(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendation with DSR as primary selection penalty.

        Hierarchy:
        - DSR: primary individual strategy selection penalty
        - Reality Check: family-wise error across the strategy set
        - Bonferroni: secondary parametric confirmation
        """
        try:
            significant_by_test = {}

            if 'reality_check' in test_results:
                significant_by_test['reality_check'] = test_results['reality_check'].significant_strategies

            if 'bonferroni' in test_results:
                significant_by_test['bonferroni'] = test_results['bonferroni']['significant_strategies']

            if 'deflated_sharpe' in test_results:
                significant_by_test['deflated_sharpe'] = test_results['deflated_sharpe']['significant_strategies']

            # DSR is the primary gate — strategy must pass DSR to be considered
            dsr_significant = set(significant_by_test.get('deflated_sharpe', []))
            rc_significant = set(significant_by_test.get('reality_check', []))
            bonf_significant = set(significant_by_test.get('bonferroni', []))

            # Tier 1: passes DSR + Reality Check (high confidence)
            tier1 = dsr_significant & rc_significant
            # Tier 2: passes DSR only (moderate confidence)
            tier2 = dsr_significant - tier1
            # Tier 3: passes other tests but not DSR (low confidence, likely data-mined)
            tier3 = (rc_significant | bonf_significant) - dsr_significant

            if tier1:
                recommendation = 'DEPLOY'
                confidence = 'HIGH'
            elif tier2:
                recommendation = 'INVESTIGATE'
                confidence = 'MEDIUM'
            elif tier3:
                recommendation = 'SUSPECT'
                confidence = 'LOW'
            else:
                recommendation = 'REJECT'
                confidence = 'NONE'

            return {
                'significant_by_test': significant_by_test,
                'primary_gate': 'deflated_sharpe',
                'tier1_deploy': sorted(tier1),
                'tier2_investigate': sorted(tier2),
                'tier3_suspect': sorted(tier3),
                'recommendation': recommendation,
                'confidence_level': confidence,
            }

        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return {'recommendation': 'ERROR', 'confidence_level': 'NONE'}


# Example usage
if __name__ == "__main__":
    def demo_reality_check():
        print("=== Reality Check & SPA Test Demo ===")

        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')

        # Benchmark returns
        benchmark = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)

        # Strategy returns
        strategies = {
            'real_alpha': benchmark + 0.0003 + np.random.normal(0, 0.005, len(dates)),
            'no_alpha': benchmark + np.random.normal(0, 0.008, len(dates)),
            'lucky': benchmark + np.random.normal(0.0001, 0.015, len(dates))
        }

        # Run testing
        controller = MultipleTestingController()
        results = controller.run_comprehensive_testing(strategies, benchmark)

        print("\nReality Check Results:")
        rc_result = results['reality_check']
        print(f"Best strategy: {rc_result.best_strategy}")
        print(f"Significant strategies: {rc_result.significant_strategies}")

        for strategy, p_val in rc_result.p_values.items():
            print(f"  {strategy}: p={p_val:.3f}")

        print(f"\nRecommendation: {results['recommendation']['recommendation']}")
        print(f"Confidence: {results['recommendation']['confidence_level']}")

    demo_reality_check()