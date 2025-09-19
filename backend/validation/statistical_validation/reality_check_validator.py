"""
White's Reality Check and SPA Test Implementation
===============================================

Implementation of White's Reality Check bootstrap methodology and
Hansen & Lunde's Superior Predictive Ability (SPA) test for multiple
strategy validation with proper statistical inference.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    from scipy import stats
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, using simplified implementations")


@dataclass
class RealityCheckResult:
    """Results from Reality Check validation."""
    strategy_name: str
    performance_metric: float  # Sharpe ratio or excess return
    bootstrap_p_value: float
    is_significant: bool
    rank_among_strategies: int
    total_strategies: int
    confidence_level: float
    bootstrap_samples: int
    test_statistic: float
    critical_value: float
    notes: List[str]


@dataclass
class SPATestResult:
    """Results from Superior Predictive Ability test."""
    best_strategy: str
    spa_statistic: float
    spa_p_value: float
    is_superior: bool
    consistent_p_value: float
    lower_p_value: float
    upper_p_value: float
    confidence_level: float
    bootstrap_samples: int
    num_strategies: int
    benchmark_name: str


class RealityCheckValidator:
    """
    Implementation of White's Reality Check and SPA test for strategy validation.

    White's Reality Check tests whether the best performing strategy from a universe
    of strategies is significantly better than a benchmark after accounting for
    data mining bias and multiple testing.
    """

    def __init__(self,
                 bootstrap_samples: int = 10000,
                 confidence_level: float = 0.95,
                 block_size: Optional[int] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize Reality Check validator.

        Args:
            bootstrap_samples: Number of bootstrap samples
            confidence_level: Confidence level for tests
            block_size: Block size for block bootstrap (None for iid bootstrap)
            random_seed: Random seed for reproducibility
        """
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.block_size = block_size
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        self.logger = logging.getLogger(__name__)

        # Performance cache
        self._performance_cache = {}

    def calculate_performance_metric(self,
                                   returns: np.ndarray,
                                   benchmark: Optional[np.ndarray] = None,
                                   metric: str = "sharpe") -> float:
        """
        Calculate performance metric for strategy evaluation.

        Args:
            returns: Strategy returns
            benchmark: Benchmark returns (optional)
            metric: Performance metric ("sharpe", "excess_return", "calmar")

        Returns:
            Performance metric value
        """
        if len(returns) == 0:
            return 0.0

        if metric == "sharpe":
            # Annualized Sharpe ratio
            if benchmark is not None:
                excess_returns = returns - benchmark[:len(returns)]
            else:
                excess_returns = returns

            if np.std(excess_returns) == 0:
                return 0.0

            sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
            return float(sharpe * np.sqrt(252))  # Annualize assuming daily returns

        elif metric == "excess_return":
            # Annualized excess return
            if benchmark is not None:
                excess_returns = returns - benchmark[:len(returns)]
            else:
                excess_returns = returns
            return float(np.mean(excess_returns) * 252)

        elif metric == "calmar":
            # Calmar ratio: return / max drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative / running_max) - 1
            max_drawdown = abs(drawdowns.min())

            if max_drawdown == 0:
                return 0.0

            annual_return = (cumulative[-1] ** (252 / len(returns))) - 1
            return float(annual_return / max_drawdown)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def block_bootstrap_sample(self, data: np.ndarray, block_size: int) -> np.ndarray:
        """
        Generate block bootstrap sample for dependent data.

        Args:
            data: Time series data
            block_size: Size of blocks

        Returns:
            Bootstrap sample
        """
        n = len(data)
        n_blocks = int(np.ceil(n / block_size))

        # Generate random starting indices
        start_indices = np.random.choice(n - block_size + 1, n_blocks, replace=True)

        bootstrap_sample = []
        for start_idx in start_indices:
            block = data[start_idx:start_idx + block_size]
            bootstrap_sample.extend(block)

        return np.array(bootstrap_sample[:n])  # Trim to original length

    def bootstrap_resample(self,
                          strategies_returns: Dict[str, np.ndarray],
                          benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Generate bootstrap resamples of all strategies.

        Args:
            strategies_returns: Dictionary of strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Dictionary of bootstrap resampled returns
        """
        # Determine sample length (use shortest series)
        min_length = min(len(returns) for returns in strategies_returns.values())

        if benchmark_returns is not None:
            min_length = min(min_length, len(benchmark_returns))

        # Choose bootstrap method
        if self.block_size is not None:
            # Block bootstrap for dependent data
            bootstrap_indices = None
            bootstrap_strategies = {}

            for name, returns in strategies_returns.items():
                returns_trimmed = returns[:min_length]
                bootstrap_strategies[name] = self.block_bootstrap_sample(
                    returns_trimmed, self.block_size
                )

            if benchmark_returns is not None:
                benchmark_trimmed = benchmark_returns[:min_length]
                bootstrap_benchmark = self.block_bootstrap_sample(
                    benchmark_trimmed, self.block_size
                )
            else:
                bootstrap_benchmark = None

        else:
            # IID bootstrap
            bootstrap_indices = np.random.choice(min_length, min_length, replace=True)

            bootstrap_strategies = {}
            for name, returns in strategies_returns.items():
                returns_trimmed = returns[:min_length]
                bootstrap_strategies[name] = returns_trimmed[bootstrap_indices]

            if benchmark_returns is not None:
                benchmark_trimmed = benchmark_returns[:min_length]
                bootstrap_benchmark = benchmark_trimmed[bootstrap_indices]
            else:
                bootstrap_benchmark = None

        return bootstrap_strategies, bootstrap_benchmark

    def run_reality_check(self,
                         strategies_returns: Dict[str, np.ndarray],
                         benchmark_returns: Optional[np.ndarray] = None,
                         metric: str = "sharpe") -> List[RealityCheckResult]:
        """
        Run White's Reality Check on multiple strategies.

        Args:
            strategies_returns: Dictionary mapping strategy names to return arrays
            benchmark_returns: Benchmark return array
            metric: Performance metric to use

        Returns:
            List of RealityCheckResult objects
        """

        self.logger.info(f"Running Reality Check on {len(strategies_returns)} strategies")
        self.logger.info(f"Bootstrap samples: {self.bootstrap_samples}, Metric: {metric}")

        # Calculate original performance metrics
        original_performance = {}
        for name, returns in strategies_returns.items():
            performance = self.calculate_performance_metric(returns, benchmark_returns, metric)
            original_performance[name] = performance

        # Sort strategies by performance
        sorted_strategies = sorted(original_performance.items(),
                                 key=lambda x: x[1], reverse=True)

        self.logger.info("Original performance ranking:")
        for i, (name, perf) in enumerate(sorted_strategies, 1):
            self.logger.info(f"  {i}. {name}: {perf:.4f}")

        # Run bootstrap
        self.logger.info("Running bootstrap resampling...")

        bootstrap_max_performance = []
        bootstrap_performance_matrix = {name: [] for name in strategies_returns.keys()}

        for i in range(self.bootstrap_samples):
            if i % (self.bootstrap_samples // 10) == 0:
                self.logger.info(f"  Bootstrap progress: {i}/{self.bootstrap_samples}")

            # Generate bootstrap sample
            bootstrap_strategies, bootstrap_benchmark = self.bootstrap_resample(
                strategies_returns, benchmark_returns
            )

            # Calculate performance for this bootstrap sample
            bootstrap_perfs = {}
            for name, bootstrap_returns in bootstrap_strategies.items():
                perf = self.calculate_performance_metric(
                    bootstrap_returns, bootstrap_benchmark, metric
                )
                bootstrap_perfs[name] = perf
                bootstrap_performance_matrix[name].append(perf)

            # Track maximum performance in this bootstrap sample
            max_perf = max(bootstrap_perfs.values())
            bootstrap_max_performance.append(max_perf)

        bootstrap_max_performance = np.array(bootstrap_max_performance)

        # Calculate p-values for each strategy
        results = []
        for i, (strategy_name, original_perf) in enumerate(sorted_strategies):

            # P-value: fraction of bootstrap maxima >= original performance
            p_value = np.mean(bootstrap_max_performance >= original_perf)

            # Test statistic and critical value
            test_statistic = original_perf
            critical_value = np.percentile(bootstrap_max_performance,
                                         self.confidence_level * 100)

            # Significance test
            is_significant = (p_value < self.alpha) and (original_perf > critical_value)

            # Notes
            notes = []
            if original_perf <= 0:
                notes.append("Negative or zero performance")
            if p_value < 0.001:
                notes.append("Highly significant (p < 0.001)")
            elif p_value < 0.01:
                notes.append("Very significant (p < 0.01)")
            elif p_value < 0.05:
                notes.append("Significant (p < 0.05)")
            else:
                notes.append("Not significant")

            result = RealityCheckResult(
                strategy_name=strategy_name,
                performance_metric=original_perf,
                bootstrap_p_value=p_value,
                is_significant=is_significant,
                rank_among_strategies=i + 1,
                total_strategies=len(strategies_returns),
                confidence_level=self.confidence_level,
                bootstrap_samples=self.bootstrap_samples,
                test_statistic=test_statistic,
                critical_value=critical_value,
                notes=notes
            )

            results.append(result)

            self.logger.info(f"Reality Check - {strategy_name}: "
                           f"p={p_value:.4f}, significant={is_significant}")

        self.logger.info("Reality Check completed")
        return results

    def run_spa_test(self,
                     strategies_returns: Dict[str, np.ndarray],
                     benchmark_returns: np.ndarray,
                     metric: str = "sharpe") -> SPATestResult:
        """
        Run Hansen & Lunde's Superior Predictive Ability (SPA) test.

        The SPA test improves on Reality Check by addressing issues with
        poor and irrelevant alternatives.

        Args:
            strategies_returns: Dictionary of strategy returns
            benchmark_returns: Benchmark returns
            metric: Performance metric

        Returns:
            SPATestResult object
        """

        self.logger.info(f"Running SPA test on {len(strategies_returns)} strategies")

        # Calculate excess performance relative to benchmark
        excess_performance = {}
        loss_series = {}  # L_k,t = benchmark_perf - strategy_perf

        min_length = min(len(benchmark_returns),
                        min(len(returns) for returns in strategies_returns.values()))

        benchmark_trimmed = benchmark_returns[:min_length]

        for name, returns in strategies_returns.items():
            returns_trimmed = returns[:min_length]

            # Calculate daily excess performance
            if metric == "sharpe":
                # Daily Sharpe differences (simplified)
                daily_excess = returns_trimmed - benchmark_trimmed
                excess_performance[name] = np.mean(daily_excess)
                loss_series[name] = benchmark_trimmed - returns_trimmed
            elif metric == "excess_return":
                daily_excess = returns_trimmed - benchmark_trimmed
                excess_performance[name] = np.mean(daily_excess)
                loss_series[name] = benchmark_trimmed - returns_trimmed

        # Find best performing strategy
        best_strategy = max(excess_performance.keys(),
                          key=lambda k: excess_performance[k])

        self.logger.info(f"Best strategy identified: {best_strategy}")

        # SPA test statistic
        # T_SPA = max(0, max_k(sqrt(n) * L_bar_k / omega_k))
        n = min_length

        spa_statistics = []
        for name, losses in loss_series.items():
            l_bar = np.mean(losses)  # Average loss
            omega = np.std(losses, ddof=1)  # Standard deviation of losses

            if omega > 0:
                t_stat = np.sqrt(n) * (-l_bar) / omega  # Note: negative because we want excess
                spa_statistics.append(max(0, t_stat))
            else:
                spa_statistics.append(0)

        spa_statistic = max(spa_statistics)

        # Bootstrap SPA test
        self.logger.info("Running SPA bootstrap...")

        bootstrap_spa_statistics = []

        for i in range(self.bootstrap_samples):
            if i % (self.bootstrap_samples // 10) == 0:
                self.logger.info(f"  SPA bootstrap progress: {i}/{self.bootstrap_samples}")

            # Bootstrap resample
            bootstrap_strategies, bootstrap_benchmark = self.bootstrap_resample(
                strategies_returns, benchmark_returns
            )

            # Calculate bootstrap SPA statistic
            bootstrap_spa_stats = []
            for name in strategies_returns.keys():
                boot_losses = bootstrap_benchmark - bootstrap_strategies[name]
                l_bar_boot = np.mean(boot_losses)
                omega_boot = np.std(boot_losses, ddof=1)

                if omega_boot > 0:
                    t_stat_boot = np.sqrt(n) * (-l_bar_boot) / omega_boot
                    bootstrap_spa_stats.append(max(0, t_stat_boot))
                else:
                    bootstrap_spa_stats.append(0)

            bootstrap_spa_statistics.append(max(bootstrap_spa_stats))

        bootstrap_spa_statistics = np.array(bootstrap_spa_statistics)

        # Calculate p-values
        # Consistent p-value
        consistent_p_value = np.mean(bootstrap_spa_statistics >= spa_statistic)

        # Lower and upper p-values (for robustness)
        lower_p_value = consistent_p_value
        upper_p_value = min(1.0, 2 * consistent_p_value)  # Conservative adjustment

        # Overall SPA p-value (use consistent)
        spa_p_value = consistent_p_value

        # Significance test
        is_superior = spa_p_value < self.alpha

        result = SPATestResult(
            best_strategy=best_strategy,
            spa_statistic=spa_statistic,
            spa_p_value=spa_p_value,
            is_superior=is_superior,
            consistent_p_value=consistent_p_value,
            lower_p_value=lower_p_value,
            upper_p_value=upper_p_value,
            confidence_level=self.confidence_level,
            bootstrap_samples=self.bootstrap_samples,
            num_strategies=len(strategies_returns),
            benchmark_name="Benchmark"
        )

        self.logger.info(f"SPA test completed - Best strategy: {best_strategy}, "
                        f"p-value: {spa_p_value:.4f}, superior: {is_superior}")

        return result

    def generate_reality_check_report(self,
                                    reality_check_results: List[RealityCheckResult],
                                    spa_result: Optional[SPATestResult] = None) -> str:
        """Generate comprehensive Reality Check validation report."""

        report = []
        report.append("=" * 80)
        report.append("WHITE'S REALITY CHECK & SPA TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Bootstrap samples: {self.bootstrap_samples}")
        report.append(f"Confidence level: {self.confidence_level}")
        report.append("")

        # Reality Check Results
        report.append("REALITY CHECK RESULTS:")
        report.append("-" * 50)

        significant_strategies = [r for r in reality_check_results if r.is_significant]

        report.append(f"Strategies tested: {len(reality_check_results)}")
        report.append(f"Significant strategies: {len(significant_strategies)}")
        report.append(f"Success rate: {len(significant_strategies)/len(reality_check_results)*100:.1f}%")
        report.append("")

        # Individual results
        for result in reality_check_results:
            status = "✅ SIGNIFICANT" if result.is_significant else "❌ NOT SIGNIFICANT"

            report.append(f"{result.rank_among_strategies}. {result.strategy_name}: {status}")
            report.append(f"   Performance: {result.performance_metric:.4f}")
            report.append(f"   Bootstrap p-value: {result.bootstrap_p_value:.4f}")
            report.append(f"   Critical value: {result.critical_value:.4f}")

            if result.notes:
                report.append(f"   Notes: {'; '.join(result.notes)}")
            report.append("")

        # SPA Test Results
        if spa_result:
            report.append("SUPERIOR PREDICTIVE ABILITY (SPA) TEST:")
            report.append("-" * 50)

            status = "✅ SUPERIOR" if spa_result.is_superior else "❌ NOT SUPERIOR"
            report.append(f"Best strategy: {spa_result.best_strategy} - {status}")
            report.append(f"SPA statistic: {spa_result.spa_statistic:.4f}")
            report.append(f"SPA p-value: {spa_result.spa_p_value:.4f}")
            report.append(f"Consistent p-value: {spa_result.consistent_p_value:.4f}")
            report.append(f"Lower p-value: {spa_result.lower_p_value:.4f}")
            report.append(f"Upper p-value: {spa_result.upper_p_value:.4f}")
            report.append("")

        # Interpretation
        report.append("INTERPRETATION:")
        report.append("-" * 50)

        if len(significant_strategies) == 0:
            report.append("❌ NO STRATEGIES SURVIVE REALITY CHECK")
            report.append("   • No strategy significantly outperforms after multiple testing")
            report.append("   • Results may be due to data mining/luck")
            report.append("   • Consider fundamental strategy improvements")
        elif len(significant_strategies) == 1:
            report.append("✅ ONE STRATEGY SURVIVES REALITY CHECK")
            report.append(f"   • {significant_strategies[0].strategy_name} shows genuine alpha")
            report.append("   • Consider focusing resources on this strategy")
        else:
            report.append(f"✅ {len(significant_strategies)} STRATEGIES SURVIVE REALITY CHECK")
            report.append("   • Multiple strategies show genuine outperformance")
            report.append("   • Consider portfolio allocation across significant strategies")

        if spa_result:
            if spa_result.is_superior:
                report.append(f"✅ SPA TEST CONFIRMS {spa_result.best_strategy} IS SUPERIOR")
                report.append("   • Best strategy significantly outperforms benchmark")
            else:
                report.append("❌ SPA TEST: NO SUPERIOR STRATEGY FOUND")
                report.append("   • Best strategy not significantly better than benchmark")

        report.append("")
        report.append("DEPLOYMENT RECOMMENDATIONS:")

        if len(significant_strategies) > 0:
            report.append("✅ PROCEED WITH DEPLOYMENT")
            for strategy in significant_strategies:
                report.append(f"   • Deploy {strategy.strategy_name} "
                            f"(rank #{strategy.rank_among_strategies})")
        else:
            report.append("❌ DO NOT DEPLOY")
            report.append("   • Return to strategy development")
            report.append("   • Address fundamental performance issues")

        report.append("=" * 80)

        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    def test_reality_check():
        """Test Reality Check and SPA validation."""

        print("Testing Reality Check & SPA Validation")
        print("=" * 50)

        # Create synthetic strategy data
        np.random.seed(42)

        # Strategy 1: Strong outperformer
        strategy1 = np.random.normal(0.0012, 0.015, 252)  # 12 bps daily mean

        # Strategy 2: Moderate outperformer
        strategy2 = np.random.normal(0.0008, 0.012, 252)  # 8 bps daily mean

        # Strategy 3: Underperformer
        strategy3 = np.random.normal(0.0003, 0.018, 252)  # 3 bps daily mean

        # Strategy 4: Random (no alpha)
        strategy4 = np.random.normal(0.0001, 0.016, 252)  # 1 bp daily mean

        # Benchmark
        benchmark = np.random.normal(0.0005, 0.012, 252)  # 5 bps daily mean

        strategies = {
            "Strong Strategy": strategy1,
            "Moderate Strategy": strategy2,
            "Weak Strategy": strategy3,
            "Random Strategy": strategy4
        }

        # Create validator
        validator = RealityCheckValidator(
            bootstrap_samples=1000,  # Reduced for testing speed
            confidence_level=0.95,
            random_seed=42
        )

        # Run Reality Check
        print("1. Running Reality Check...")
        reality_results = validator.run_reality_check(strategies, benchmark, "sharpe")

        print("   Results:")
        for result in reality_results:
            status = "✅" if result.is_significant else "❌"
            print(f"   {result.strategy_name}: {status} "
                  f"Sharpe={result.performance_metric:.3f}, p={result.bootstrap_p_value:.4f}")

        # Run SPA test
        print("\n2. Running SPA Test...")
        spa_result = validator.run_spa_test(strategies, benchmark, "sharpe")

        status = "✅ SUPERIOR" if spa_result.is_superior else "❌ NOT SUPERIOR"
        print(f"   Best Strategy: {spa_result.best_strategy} - {status}")
        print(f"   SPA p-value: {spa_result.spa_p_value:.4f}")

        # Generate report
        print("\n3. Validation Report:")
        report = validator.generate_reality_check_report(reality_results, spa_result)
        print(report[:800] + "..." if len(report) > 800 else report)

        return reality_results, spa_result

    # Run test
    test_reality_check()