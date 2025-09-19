"""
Phase 2: Statistical Validation Framework
========================================

Core signal validation system implementing multiple hypothesis testing,
bootstrap methods, and statistical significance testing for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import warnings

try:
    from scipy import stats
    from scipy.stats import jarque_bera, normaltest
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, using simplified statistical methods")

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available, using simplified implementations")


@dataclass
class SignalValidationResult:
    """Results from statistical signal validation."""
    signal_name: str
    sample_size: int
    test_statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    multiple_testing_adjusted_p: float
    validation_method: str
    notes: List[str]


@dataclass
class ValidationConfig:
    """Configuration for statistical validation."""
    significance_level: float = 0.05  # 5% significance level
    bootstrap_samples: int = 10000
    min_sample_size: int = 100
    confidence_level: float = 0.95
    multiple_testing_method: str = "bonferroni"  # bonferroni, benjamini_hochberg
    min_effect_size: float = 0.1  # Minimum meaningful effect size
    require_normality_tests: bool = True
    require_independence_tests: bool = True


class StatisticalTest(ABC):
    """Abstract base class for statistical tests."""

    @abstractmethod
    def run_test(self, returns: np.ndarray, benchmark: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Run the statistical test and return (test_statistic, p_value)."""
        pass

    @abstractmethod
    def get_test_name(self) -> str:
        """Return the name of the test."""
        pass


class TTestValidator(StatisticalTest):
    """T-test for mean return significance."""

    def run_test(self, returns: np.ndarray, benchmark: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Run one-sample or two-sample t-test."""
        if benchmark is not None:
            # Two-sample t-test
            if SCIPY_AVAILABLE:
                t_stat, p_value = stats.ttest_ind(returns, benchmark)
            else:
                # Simplified two-sample t-test
                n1, n2 = len(returns), len(benchmark)
                mean_diff = np.mean(returns) - np.mean(benchmark)
                s1, s2 = np.std(returns, ddof=1), np.std(benchmark, ddof=1)
                se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
                t_stat = mean_diff / se if se > 0 else 0
                # Approximate p-value for two-tailed test
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n2 - 2)) if SCIPY_AVAILABLE else 0.1
        else:
            # One-sample t-test against zero
            if SCIPY_AVAILABLE:
                t_stat, p_value = stats.ttest_1samp(returns, 0)
            else:
                # Simplified one-sample t-test
                n = len(returns)
                mean_ret = np.mean(returns)
                std_ret = np.std(returns, ddof=1)
                t_stat = mean_ret / (std_ret / np.sqrt(n)) if std_ret > 0 else 0
                # Approximate p-value
                p_value = 0.05 if abs(t_stat) > 2 else 0.5  # Rough approximation

        return float(t_stat), float(p_value)

    def get_test_name(self) -> str:
        return "T-Test"


class WilcoxonValidator(StatisticalTest):
    """Wilcoxon signed-rank test for non-parametric validation."""

    def run_test(self, returns: np.ndarray, benchmark: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Run Wilcoxon signed-rank test."""
        if benchmark is not None:
            # Wilcoxon rank-sum test
            if SCIPY_AVAILABLE:
                stat, p_value = stats.ranksums(returns, benchmark)
            else:
                # Simplified rank test approximation
                combined = np.concatenate([returns, benchmark])
                ranks = np.argsort(np.argsort(combined)) + 1
                rank_sum_1 = np.sum(ranks[:len(returns)])
                expected = len(returns) * (len(combined) + 1) / 2
                stat = abs(rank_sum_1 - expected)
                p_value = 0.05 if stat > len(returns) * 2 else 0.5
        else:
            # Wilcoxon signed-rank test against zero
            if SCIPY_AVAILABLE and len(returns) > 5:
                stat, p_value = stats.wilcoxon(returns)
            else:
                # Simplified: count positive vs negative returns
                positive_count = np.sum(returns > 0)
                total_count = len(returns)
                if total_count > 0:
                    ratio = positive_count / total_count
                    # Approximate significance
                    stat = abs(ratio - 0.5) * np.sqrt(total_count)
                    p_value = 0.05 if stat > 1.96 else 0.5
                else:
                    stat, p_value = 0, 1

        return float(stat), float(p_value)

    def get_test_name(self) -> str:
        return "Wilcoxon"


class BootstrapValidator(StatisticalTest):
    """Bootstrap-based significance testing."""

    def __init__(self, n_bootstrap: int = 10000):
        self.n_bootstrap = n_bootstrap

    def run_test(self, returns: np.ndarray, benchmark: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Run bootstrap test for mean return significance."""
        observed_mean = np.mean(returns)

        if benchmark is not None:
            # Bootstrap difference of means
            benchmark_mean = np.mean(benchmark)
            observed_diff = observed_mean - benchmark_mean

            # Bootstrap resampling
            bootstrap_diffs = []
            for _ in range(self.n_bootstrap):
                boot_returns = np.random.choice(returns, size=len(returns), replace=True)
                boot_benchmark = np.random.choice(benchmark, size=len(benchmark), replace=True)
                bootstrap_diffs.append(np.mean(boot_returns) - np.mean(boot_benchmark))

            bootstrap_diffs = np.array(bootstrap_diffs)
            # Center the bootstrap distribution under null hypothesis
            centered_diffs = bootstrap_diffs - np.mean(bootstrap_diffs)
            p_value = np.mean(np.abs(centered_diffs) >= abs(observed_diff))

        else:
            # Bootstrap mean against zero
            bootstrap_means = []
            for _ in range(self.n_bootstrap):
                boot_sample = np.random.choice(returns, size=len(returns), replace=True)
                bootstrap_means.append(np.mean(boot_sample))

            bootstrap_means = np.array(bootstrap_means)
            # Center under null hypothesis (mean = 0)
            centered_means = bootstrap_means - np.mean(bootstrap_means)
            p_value = np.mean(np.abs(centered_means) >= abs(observed_mean))

        return float(observed_mean), float(p_value)

    def get_test_name(self) -> str:
        return "Bootstrap"


class SignalValidator:
    """Main class for statistical validation of trading signals."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)

        # Available statistical tests
        self.tests = {
            'ttest': TTestValidator(),
            'wilcoxon': WilcoxonValidator(),
            'bootstrap': BootstrapValidator(self.config.bootstrap_samples)
        }

        # Track all p-values for multiple testing adjustment
        self.all_p_values: List[float] = []
        self.test_names: List[str] = []

    def check_assumptions(self, returns: np.ndarray) -> Dict[str, Any]:
        """Check statistical assumptions for the data."""
        assumptions = {
            'sample_size_adequate': len(returns) >= self.config.min_sample_size,
            'has_variance': np.std(returns) > 1e-8,
            'no_extreme_outliers': True,
            'normality_ok': True,
            'independence_ok': True,
            'notes': []
        }

        # Sample size check
        if not assumptions['sample_size_adequate']:
            assumptions['notes'].append(f"Sample size {len(returns)} < minimum {self.config.min_sample_size}")

        # Variance check
        if not assumptions['has_variance']:
            assumptions['notes'].append("Returns have zero or near-zero variance")

        # Outlier detection (simple IQR method)
        if len(returns) > 4:
            q1, q3 = np.percentile(returns, [25, 75])
            iqr = q3 - q1
            outlier_threshold = 3 * iqr
            outliers = np.abs(returns - np.median(returns)) > outlier_threshold
            if np.sum(outliers) > len(returns) * 0.1:  # More than 10% outliers
                assumptions['no_extreme_outliers'] = False
                assumptions['notes'].append(f"High outlier percentage: {np.sum(outliers)/len(returns):.1%}")

        # Normality tests
        if self.config.require_normality_tests and len(returns) > 8:
            try:
                if SCIPY_AVAILABLE:
                    # Jarque-Bera test
                    jb_stat, jb_p = jarque_bera(returns)
                    assumptions['normality_ok'] = jb_p > 0.05
                    if not assumptions['normality_ok']:
                        assumptions['notes'].append(f"Non-normal distribution (JB p={jb_p:.4f})")
                else:
                    # Simple normality check: skewness and kurtosis
                    skewness = stats.skew(returns) if SCIPY_AVAILABLE else 0
                    if abs(skewness) > 2:
                        assumptions['normality_ok'] = False
                        assumptions['notes'].append("High skewness detected")
            except Exception as e:
                assumptions['notes'].append(f"Normality test failed: {e}")

        # Independence tests (simple autocorrelation check)
        if self.config.require_independence_tests and len(returns) > 20:
            try:
                # Simple lag-1 autocorrelation
                returns_shifted = returns[1:]
                returns_lagged = returns[:-1]
                if len(returns_shifted) > 0 and np.std(returns_shifted) > 0 and np.std(returns_lagged) > 0:
                    autocorr = np.corrcoef(returns_shifted, returns_lagged)[0, 1]
                    if abs(autocorr) > 0.3:  # Significant autocorrelation
                        assumptions['independence_ok'] = False
                        assumptions['notes'].append(f"High autocorrelation: {autocorr:.3f}")
            except Exception as e:
                assumptions['notes'].append(f"Independence test failed: {e}")

        return assumptions

    def calculate_effect_size(self, returns: np.ndarray, benchmark: Optional[np.ndarray] = None) -> float:
        """Calculate effect size (Cohen's d or similar)."""
        if benchmark is not None:
            # Cohen's d for two samples
            mean_diff = np.mean(returns) - np.mean(benchmark)
            pooled_std = np.sqrt(((len(returns) - 1) * np.var(returns, ddof=1) +
                                 (len(benchmark) - 1) * np.var(benchmark, ddof=1)) /
                                (len(returns) + len(benchmark) - 2))
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
        else:
            # Effect size vs zero (standardized mean)
            effect_size = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        return float(effect_size)

    def bootstrap_confidence_interval(self, returns: np.ndarray,
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for the mean."""
        bootstrap_means = []

        for _ in range(self.config.bootstrap_samples):
            boot_sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_means.append(np.mean(boot_sample))

        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence_level

        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return float(ci_lower), float(ci_upper)

    def adjust_for_multiple_testing(self, p_values: List[float],
                                   method: str = "bonferroni") -> List[float]:
        """Adjust p-values for multiple testing."""
        p_values = np.array(p_values)
        n_tests = len(p_values)

        if method == "bonferroni":
            adjusted_p_values = p_values * n_tests
            adjusted_p_values = np.minimum(adjusted_p_values, 1.0)  # Cap at 1.0

        elif method == "benjamini_hochberg":
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            sorted_p_values = p_values[sorted_indices]

            adjusted_p_values = np.zeros_like(p_values)
            for i in range(n_tests - 1, -1, -1):
                if i == n_tests - 1:
                    adjusted_p_values[sorted_indices[i]] = sorted_p_values[i]
                else:
                    adjusted_p_values[sorted_indices[i]] = min(
                        sorted_p_values[i] * n_tests / (i + 1),
                        adjusted_p_values[sorted_indices[i + 1]]
                    )
        else:
            # No adjustment
            adjusted_p_values = p_values

        return adjusted_p_values.tolist()

    def validate_signal(self, signal_name: str, returns: np.ndarray,
                       benchmark: Optional[np.ndarray] = None,
                       test_method: str = "bootstrap") -> SignalValidationResult:
        """Validate a single trading signal for statistical significance."""

        self.logger.info(f"Validating signal: {signal_name} (n={len(returns)})")

        # Check assumptions
        assumptions = self.check_assumptions(returns)
        notes = assumptions['notes'].copy()

        if not assumptions['sample_size_adequate']:
            return SignalValidationResult(
                signal_name=signal_name,
                sample_size=len(returns),
                test_statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.config.confidence_level,
                effect_size=0.0,
                bootstrap_ci_lower=0.0,
                bootstrap_ci_upper=0.0,
                multiple_testing_adjusted_p=1.0,
                validation_method=test_method,
                notes=[*notes, "Insufficient sample size for validation"]
            )

        # Select appropriate test based on assumptions
        if test_method == "auto":
            if assumptions['normality_ok'] and assumptions['independence_ok']:
                test_method = "ttest"
            else:
                test_method = "wilcoxon" if not assumptions['normality_ok'] else "bootstrap"

        # Run statistical test
        try:
            test = self.tests[test_method]
            test_statistic, p_value = test.run_test(returns, benchmark)

            # Calculate additional metrics
            effect_size = self.calculate_effect_size(returns, benchmark)
            ci_lower, ci_upper = self.bootstrap_confidence_interval(returns, self.config.confidence_level)

            # Check significance
            is_significant = (p_value < self.config.significance_level and
                            abs(effect_size) >= self.config.min_effect_size)

            # Store for multiple testing adjustment
            self.all_p_values.append(p_value)
            self.test_names.append(signal_name)

            # Add validation notes
            if abs(effect_size) < self.config.min_effect_size:
                notes.append(f"Small effect size: {effect_size:.3f}")
            if not assumptions['normality_ok']:
                notes.append("Non-normal distribution detected")
            if not assumptions['independence_ok']:
                notes.append("Serial correlation detected")

            result = SignalValidationResult(
                signal_name=signal_name,
                sample_size=len(returns),
                test_statistic=test_statistic,
                p_value=p_value,
                is_significant=is_significant,
                confidence_level=self.config.confidence_level,
                effect_size=effect_size,
                bootstrap_ci_lower=ci_lower,
                bootstrap_ci_upper=ci_upper,
                multiple_testing_adjusted_p=p_value,  # Will be adjusted later
                validation_method=f"{test.get_test_name()}",
                notes=notes
            )

            self.logger.info(f"Validation complete: {signal_name} - "
                           f"p={p_value:.4f}, effect_size={effect_size:.3f}, "
                           f"significant={is_significant}")

            return result

        except Exception as e:
            self.logger.error(f"Validation failed for {signal_name}: {e}")
            return SignalValidationResult(
                signal_name=signal_name,
                sample_size=len(returns),
                test_statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.config.confidence_level,
                effect_size=0.0,
                bootstrap_ci_lower=0.0,
                bootstrap_ci_upper=0.0,
                multiple_testing_adjusted_p=1.0,
                validation_method=test_method,
                notes=[*notes, f"Validation error: {e!s}"]
            )

    def validate_multiple_signals(self, signals_data: Dict[str, np.ndarray],
                                 benchmark: Optional[np.ndarray] = None,
                                 test_method: str = "bootstrap") -> List[SignalValidationResult]:
        """Validate multiple signals with multiple testing correction."""

        self.logger.info(f"Validating {len(signals_data)} signals with multiple testing correction")

        # Reset p-value tracking
        self.all_p_values = []
        self.test_names = []

        # Validate each signal
        results = []
        for signal_name, returns in signals_data.items():
            result = self.validate_signal(signal_name, returns, benchmark, test_method)
            results.append(result)

        # Apply multiple testing correction
        if self.all_p_values:
            adjusted_p_values = self.adjust_for_multiple_testing(
                self.all_p_values,
                self.config.multiple_testing_method
            )

            # Update results with adjusted p-values
            for i, result in enumerate(results):
                result.multiple_testing_adjusted_p = adjusted_p_values[i]
                result.is_significant = (
                    adjusted_p_values[i] < self.config.significance_level and
                    abs(result.effect_size) >= self.config.min_effect_size
                )

                # Add multiple testing note
                if adjusted_p_values[i] != result.p_value:
                    result.notes.append(
                        f"Multiple testing adjusted: {result.p_value:.4f} → {adjusted_p_values[i]:.4f}"
                    )

        significant_count = sum(1 for r in results if r.is_significant)
        self.logger.info(f"Multiple testing validation complete: "
                        f"{significant_count}/{len(results)} signals significant")

        return results

    def generate_validation_report(self, results: List[SignalValidationResult]) -> str:
        """Generate a comprehensive validation report."""

        report = []
        report.append("=" * 80)
        report.append("STATISTICAL VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Configuration: α={self.config.significance_level}, "
                     f"min_effect_size={self.config.min_effect_size}")
        report.append(f"Multiple testing: {self.config.multiple_testing_method}")
        report.append("")

        # Summary statistics
        total_signals = len(results)
        significant_signals = sum(1 for r in results if r.is_significant)

        report.append("SUMMARY:")
        report.append(f"  Total signals tested: {total_signals}")
        report.append(f"  Significant signals: {significant_signals}")
        report.append(f"  Success rate: {significant_signals/total_signals*100:.1f}%")
        report.append("")

        # Individual results
        report.append("DETAILED RESULTS:")
        report.append("-" * 80)

        # Sort by significance and effect size
        sorted_results = sorted(results,
                              key=lambda r: (r.is_significant, abs(r.effect_size)),
                              reverse=True)

        for result in sorted_results:
            status = "✅ SIGNIFICANT" if result.is_significant else "❌ NOT SIGNIFICANT"

            report.append(f"\n{result.signal_name}: {status}")
            report.append(f"  Method: {result.validation_method}")
            report.append(f"  Sample size: {result.sample_size}")
            report.append(f"  Test statistic: {result.test_statistic:.4f}")
            report.append(f"  P-value: {result.p_value:.4f}")
            report.append(f"  Adjusted p-value: {result.multiple_testing_adjusted_p:.4f}")
            report.append(f"  Effect size: {result.effect_size:.4f}")
            report.append(f"  95% CI: [{result.bootstrap_ci_lower:.4f}, {result.bootstrap_ci_upper:.4f}]")

            if result.notes:
                report.append(f"  Notes: {'; '.join(result.notes)}")

        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS:")

        if significant_signals == 0:
            report.append("❌ No signals passed statistical validation")
            report.append("   - Review signal generation methodology")
            report.append("   - Consider increasing sample size")
            report.append("   - Check for data quality issues")
        elif significant_signals < total_signals * 0.3:
            report.append("⚠️  Low validation success rate")
            report.append("   - Focus resources on significant signals")
            report.append("   - Investigate failed signals for improvements")
        else:
            report.append("✅ Good validation results")
            report.append("   - Deploy significant signals for live testing")
            report.append("   - Continue monitoring performance")

        report.append("=" * 80)

        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    def test_signal_validator():
        """Test the statistical validation framework."""

        print("Testing Statistical Validation Framework")
        print("=" * 50)

        # Create synthetic signal data
        np.random.seed(42)

        # Signal 1: Strong positive signal
        strong_signal = np.random.normal(0.0015, 0.01, 200)  # 15 bps daily mean

        # Signal 2: Weak signal
        weak_signal = np.random.normal(0.0003, 0.012, 150)  # 3 bps daily mean

        # Signal 3: No signal (random)
        no_signal = np.random.normal(0.0001, 0.011, 180)  # 1 bp daily mean

        # Benchmark: Market returns
        benchmark = np.random.normal(0.0005, 0.01, 250)  # 5 bps daily mean

        # Create validator
        config = ValidationConfig(
            significance_level=0.05,
            bootstrap_samples=5000,  # Reduced for testing speed
            min_effect_size=0.1,
            multiple_testing_method="benjamini_hochberg"
        )

        validator = SignalValidator(config)

        # Test individual signal validation
        print("1. Individual Signal Validation:")
        result1 = validator.validate_signal("Strong Signal", strong_signal, benchmark)
        print(f"   Strong Signal: p={result1.p_value:.4f}, effect={result1.effect_size:.3f}")

        # Test multiple signals validation
        print("\n2. Multiple Signals Validation:")
        signals_data = {
            "Strong Signal": strong_signal,
            "Weak Signal": weak_signal,
            "Random Signal": no_signal
        }

        results = validator.validate_multiple_signals(signals_data, benchmark)

        for result in results:
            status = "✅" if result.is_significant else "❌"
            print(f"   {result.signal_name}: {status} p={result.multiple_testing_adjusted_p:.4f}")

        # Generate report
        print("\n3. Validation Report:")
        report = validator.generate_validation_report(results)
        print(report[:500] + "..." if len(report) > 500 else report)

        return results

    # Run test
    test_results = test_signal_validator()