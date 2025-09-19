"""
Factor Analysis with Correct OLS + Newey-West HAC Standard Errors
Regresses strategy returns on factors using statsmodels for proper statistical inference.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
import logging

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    import warnings
    warnings.warn("statsmodels not available, using fallback implementation")


@dataclass
class FactorResult:
    """Results from factor regression analysis."""
    daily_alpha: float
    annualized_alpha: float
    alpha_t_stat: float
    factor_exposures: Dict[str, float]
    r_squared: float
    alpha_significant: bool
    n_obs: int
    factor_t_stats: Dict[str, float] = None
    residuals: pd.Series = None
    model_summary: str = ""


class AlphaFactorAnalyzer:
    """
    Regress strategy daily returns on factors using OLS with HAC (Newey-West) std errors.
    Accepts either true Fama-French dataframe (recommended) or ETF proxies (fallback).
    Columns expected in factor_df: ['mkt', 'smb', 'hml', 'umd'] (or subset).
    Strategy returns should be daily arithmetic returns aligned to factor_df.index.
    """

    def __init__(self, min_obs: int = 126, hac_lags: int = 5):
        self.min_obs = min_obs
        self.hac_lags = hac_lags
        self.logger = logging.getLogger(__name__)

        if not STATSMODELS_AVAILABLE:
            self.logger.warning("statsmodels not available, using simplified implementation")

    def run_factor_regression(
        self,
        strategy_returns: pd.Series,
        factor_df: pd.DataFrame,
        factor_cols: Optional[List[str]] = None,
        alpha_sig_t: float = 2.0
    ) -> FactorResult:
        """
        Run factor regression with proper statistical inference.

        Args:
            strategy_returns: Daily strategy returns
            factor_df: Factor returns dataframe
            factor_cols: List of factor columns to use
            alpha_sig_t: T-statistic threshold for alpha significance

        Returns:
            FactorResult with alpha, exposures, and statistical tests
        """
        try:
            if factor_cols is None:
                # Try common Fama-French factors
                default_cols = ['mkt', 'smb', 'hml', 'umd']
                factor_cols = [c for c in default_cols if c in factor_df.columns]

                if not factor_cols:
                    # Fallback to any numeric columns
                    factor_cols = factor_df.select_dtypes(include=[np.number]).columns.tolist()

            if not factor_cols:
                raise ValueError("No factor columns found in factor_df")

            # Align and drop NaNs
            df = pd.concat([strategy_returns.rename('ret'), factor_df[factor_cols]], axis=1).dropna()

            if len(df) < self.min_obs:
                raise ValueError(f"Insufficient observations ({len(df)}), need ≥ {self.min_obs}")

            if STATSMODELS_AVAILABLE:
                return self._run_statsmodels_regression(df, factor_cols, alpha_sig_t)
            else:
                return self._run_fallback_regression(df, factor_cols, alpha_sig_t)

        except Exception as e:
            self.logger.error(f"Factor regression failed: {e}")
            raise

    def _run_statsmodels_regression(self, df: pd.DataFrame, factor_cols: List[str],
                                   alpha_sig_t: float) -> FactorResult:
        """Run regression using statsmodels with HAC standard errors."""
        y = df['ret'].astype(float)
        X = sm.add_constant(df[factor_cols].astype(float))

        # Fit OLS model
        model = sm.OLS(y, X, missing='drop')

        # Use HAC (Newey-West) standard errors
        try:
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': self.hac_lags})
        except Exception as e:
            self.logger.warning(f"HAC estimation failed ({e}), using robust standard errors")
            results = model.fit(cov_type='HC1')

        # Extract results
        alpha = results.params['const']
        t_alpha = results.tvalues['const']

        exposures = {k: float(results.params[k]) for k in factor_cols}
        factor_t_stats = {k: float(results.tvalues[k]) for k in factor_cols}

        r_squared = float(results.rsquared)
        alpha_significant = abs(t_alpha) >= alpha_sig_t

        return FactorResult(
            daily_alpha=float(alpha),
            annualized_alpha=float(alpha * 252.0),
            alpha_t_stat=float(t_alpha),
            factor_exposures=exposures,
            r_squared=r_squared,
            alpha_significant=alpha_significant,
            n_obs=len(df),
            factor_t_stats=factor_t_stats,
            residuals=pd.Series(results.resid, index=df.index),
            model_summary=str(results.summary())
        )

    def _run_fallback_regression(self, df: pd.DataFrame, factor_cols: List[str],
                                alpha_sig_t: float) -> FactorResult:
        """Fallback regression using numpy (without HAC errors)."""
        self.logger.warning("Using fallback regression - HAC standard errors not available")

        y = df['ret'].values
        X = np.column_stack([np.ones(len(df)), df[factor_cols].values])

        # Simple OLS
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            beta = np.zeros(len(factor_cols) + 1)

        # Calculate residuals and standard errors
        y_pred = X @ beta
        residuals = y - y_pred

        # Simple standard error calculation (not HAC)
        mse = np.mean(residuals**2)
        try:
            var_coef = mse * np.diag(np.linalg.inv(X.T @ X))
            std_errors = np.sqrt(var_coef)
        except np.linalg.LinAlgError:
            std_errors = np.ones(len(beta))

        # T-statistics
        t_stats = beta / (std_errors + 1e-8)

        alpha = beta[0]
        t_alpha = t_stats[0]

        exposures = {factor_cols[i]: float(beta[i+1]) for i in range(len(factor_cols))}
        factor_t_stats = {factor_cols[i]: float(t_stats[i+1]) for i in range(len(factor_cols))}

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))

        return FactorResult(
            daily_alpha=float(alpha),
            annualized_alpha=float(alpha * 252.0),
            alpha_t_stat=float(t_alpha),
            factor_exposures=exposures,
            r_squared=float(r_squared),
            alpha_significant=abs(t_alpha) >= alpha_sig_t,
            n_obs=len(df),
            factor_t_stats=factor_t_stats,
            residuals=pd.Series(residuals, index=df.index)
        )

    def isolate_components_alpha(
        self,
        component_returns: Dict[str, pd.Series],
        factor_df: pd.DataFrame
    ) -> Dict[str, FactorResult]:
        """
        Analyze alpha for different strategy components.

        Args:
            component_returns: Dict like {'entry_timing': pd.Series, 'exit_timing': ...}
            factor_df: Factor returns dataframe

        Returns:
            Dict of component name -> FactorResult
        """
        results = {}

        for name, series in component_returns.items():
            try:
                result = self.run_factor_regression(series, factor_df)
                results[name] = result
                self.logger.info(f"Component {name}: alpha={result.annualized_alpha:.2%}, "
                               f"t-stat={result.alpha_t_stat:.2f}")
            except Exception as e:
                self.logger.error(f"Factor analysis failed for component {name}: {e}")
                # Safe default result
                results[name] = FactorResult(
                    daily_alpha=0.0,
                    annualized_alpha=0.0,
                    alpha_t_stat=0.0,
                    factor_exposures={},
                    r_squared=0.0,
                    alpha_significant=False,
                    n_obs=0
                )

        return results

    def create_synthetic_factors(self, start_date: str = '2020-01-01',
                                end_date: str = '2024-12-31') -> pd.DataFrame:
        """
        Create synthetic factor returns for testing when real factors unavailable.

        Returns:
            DataFrame with synthetic mkt, smb, hml, umd factors
        """
        try:
            dates = pd.date_range(start_date, end_date, freq='B')
            np.random.seed(42)  # Reproducible for testing

            # Synthetic factor returns (roughly realistic)
            factors = pd.DataFrame({
                'mkt': np.random.normal(0.0005, 0.012, len(dates)),  # Market factor
                'smb': np.random.normal(0.0, 0.006, len(dates)),     # Size factor
                'hml': np.random.normal(0.0, 0.005, len(dates)),     # Value factor
                'umd': np.random.normal(0.0, 0.008, len(dates))      # Momentum factor
            }, index=dates)

            self.logger.info(f"Created synthetic factors for {len(dates)} business days")
            return factors

        except Exception as e:
            self.logger.error(f"Failed to create synthetic factors: {e}")
            raise

    def analyze_factor_loadings(self, result: FactorResult) -> Dict[str, str]:
        """
        Interpret factor loadings and provide insights.

        Args:
            result: FactorResult from regression

        Returns:
            Dict with interpretations of factor exposures
        """
        interpretations = {}

        try:
            exposures = result.factor_exposures
            t_stats = result.factor_t_stats or {}

            for factor, loading in exposures.items():
                significance = ""
                if factor in t_stats:
                    t_stat = abs(t_stats[factor])
                    if t_stat >= 2.58:
                        significance = " (highly significant)"
                    elif t_stat >= 1.96:
                        significance = " (significant)"
                    else:
                        significance = " (not significant)"

                if factor == 'mkt':
                    if loading > 1.1:
                        interpretations[factor] = f"High market beta ({loading:.2f}){significance}"
                    elif loading < 0.9:
                        interpretations[factor] = f"Low market beta ({loading:.2f}){significance}"
                    else:
                        interpretations[factor] = f"Market-neutral beta ({loading:.2f}){significance}"

                elif factor == 'smb':
                    if loading > 0.2:
                        interpretations[factor] = f"Small-cap tilt ({loading:.2f}){significance}"
                    elif loading < -0.2:
                        interpretations[factor] = f"Large-cap tilt ({loading:.2f}){significance}"
                    else:
                        interpretations[factor] = f"Size-neutral ({loading:.2f}){significance}"

                elif factor == 'hml':
                    if loading > 0.2:
                        interpretations[factor] = f"Value tilt ({loading:.2f}){significance}"
                    elif loading < -0.2:
                        interpretations[factor] = f"Growth tilt ({loading:.2f}){significance}"
                    else:
                        interpretations[factor] = f"Style-neutral ({loading:.2f}){significance}"

                elif factor == 'umd':
                    if loading > 0.2:
                        interpretations[factor] = f"Momentum exposure ({loading:.2f}){significance}"
                    elif loading < -0.2:
                        interpretations[factor] = f"Contrarian exposure ({loading:.2f}){significance}"
                    else:
                        interpretations[factor] = f"Momentum-neutral ({loading:.2f}){significance}"

                else:
                    interpretations[factor] = f"Loading: {loading:.2f}{significance}"

            return interpretations

        except Exception as e:
            self.logger.error(f"Factor interpretation failed: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    def test_factor_analysis():
        print("=== Factor Analysis with HAC Standard Errors Demo ===")

        analyzer = AlphaFactorAnalyzer()

        # Create synthetic factor data
        factors = analyzer.create_synthetic_factors('2020-01-01', '2023-12-31')
        print(f"Created factors: {list(factors.columns)}")

        # Create synthetic strategy returns with known alpha
        np.random.seed(42)
        true_alpha = 0.0002  # 20 bps daily = ~5% annual

        strategy_returns = (
            true_alpha +
            0.8 * factors['mkt'] +      # Market beta of 0.8
            0.3 * factors['smb'] +      # Small cap tilt
            -0.1 * factors['hml'] +     # Slight growth tilt
            np.random.normal(0, 0.008, len(factors))  # Idiosyncratic risk
        )
        strategy_returns.name = 'strategy'

        # Run factor regression
        result = analyzer.run_factor_regression(strategy_returns, factors)

        print("\n=== Regression Results ===")
        print(f"Daily Alpha: {result.daily_alpha:.6f} ({result.daily_alpha*10000:.1f} bps)")
        print(f"Annualized Alpha: {result.annualized_alpha:.2%}")
        print(f"Alpha T-Stat: {result.alpha_t_stat:.2f}")
        print(f"Alpha Significant: {result.alpha_significant}")
        print(f"R-Squared: {result.r_squared:.3f}")
        print(f"Observations: {result.n_obs}")

        print("\n=== Factor Exposures ===")
        for factor, exposure in result.factor_exposures.items():
            t_stat = result.factor_t_stats.get(factor, 0) if result.factor_t_stats else 0
            print(f"{factor.upper()}: {exposure:.3f} (t={t_stat:.2f})")

        # Interpret factor loadings
        interpretations = analyzer.analyze_factor_loadings(result)
        print("\n=== Factor Interpretations ===")
        for factor, interpretation in interpretations.items():
            print(f"{factor.upper()}: {interpretation}")

        # Test component analysis
        print("\n=== Component Analysis ===")
        components = {
            'entry_timing': strategy_returns * 0.6 + np.random.normal(0, 0.005, len(strategy_returns)),
            'exit_timing': strategy_returns * 0.4 + np.random.normal(0, 0.003, len(strategy_returns)),
            'position_sizing': np.random.normal(0.0001, 0.004, len(strategy_returns))
        }

        component_results = analyzer.isolate_components_alpha(components, factors)

        for comp_name, comp_result in component_results.items():
            print(f"{comp_name}: alpha={comp_result.annualized_alpha:.2%}, "
                  f"t-stat={comp_result.alpha_t_stat:.2f}")

    test_factor_analysis()