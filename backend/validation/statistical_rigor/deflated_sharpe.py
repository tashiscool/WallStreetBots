"""Deflated Sharpe Ratio — Bailey & López de Prado (2014).

Adjusts Sharpe Ratio for multiple testing by estimating the expected
maximum Sharpe under the null hypothesis and computing the probability
that the observed Sharpe exceeds it.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


EULER_MASCHERONI = 0.5772156649015329


@dataclass
class DeflatedSharpeResult:
    """Result from Deflated Sharpe Ratio calculation."""
    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    is_significant: bool
    num_trials: int
    expected_max_sharpe: float


class DeflatedSharpeRatio:
    """Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014).

    The DSR deflates an observed Sharpe ratio by the expected maximum
    Sharpe ratio that would arise from *num_trials* independent tests,
    given the return distribution's non-normality (skewness/kurtosis).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def calculate(
        self,
        observed_sharpe: float,
        num_trials: int,
        returns_skewness: float,
        returns_kurtosis: float,
        n_observations: int,
        sharpe_std: float | None = None,
    ) -> DeflatedSharpeResult:
        """Compute the Deflated Sharpe Ratio.

        Parameters
        ----------
        observed_sharpe : float
            The Sharpe ratio of the strategy being tested.
        num_trials : int
            Number of strategies / parameter combos tried.
        returns_skewness : float
            Skewness of strategy returns.
        returns_kurtosis : float
            *Excess* kurtosis of strategy returns (normal = 0).
        n_observations : int
            Number of return observations.
        sharpe_std : float, optional
            Standard deviation of Sharpe across trials.  When *None*,
            defaults to 1.0 (i.e. Sharpe ratios are roughly unit-scale).

        Returns
        -------
        DeflatedSharpeResult
        """
        if sharpe_std is None:
            sharpe_std = 1.0

        e_max_sr = self._expected_max_sharpe(num_trials, sharpe_std)
        psr = self._psr(observed_sharpe, e_max_sr, n_observations,
                        returns_skewness, returns_kurtosis)

        return DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=psr,
            p_value=1.0 - psr,
            is_significant=(1.0 - psr) < self.alpha,
            num_trials=num_trials,
            expected_max_sharpe=e_max_sr,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_max_sharpe(
        num_trials: int,
        sharpe_std: float,
        euler_mascheroni: float = EULER_MASCHERONI,
    ) -> float:
        r"""E[\max(SR)] under the null (all SRs ~ N(0, sharpe_std^2)).

        Uses the approximation for the expected maximum of *num_trials*
        i.i.d. standard normal draws:

            E[max] ≈ σ * { (1 − γ) * Φ⁻¹(1 − 1/N) + γ * Φ⁻¹(1 − 1/(N*e)) }

        where γ is the Euler-Mascheroni constant and Φ⁻¹ is the normal
        quantile function.
        """
        if num_trials <= 1:
            return 0.0

        # Use inverse survival function to avoid 1-1/N precision loss
        # and asymptotic sqrt(2*log(N)) for extremely large N
        if num_trials > 1e15:
            z1 = math.sqrt(2.0 * math.log(num_trials))
            z2 = math.sqrt(2.0 * math.log(num_trials * math.e))
        else:
            z1 = float(norm.isf(1.0 / num_trials))
            z2 = float(norm.isf(1.0 / (num_trials * math.e)))

        return sharpe_std * (
            (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2
        )

    @staticmethod
    def _psr(
        observed: float,
        benchmark: float,
        n: int,
        skew: float,
        kurt: float,
    ) -> float:
        r"""Probabilistic Sharpe Ratio with non-normal correction.

        PSR = Φ( (SR − SR*) / σ(SR) )

        where σ(SR) adjusts for skewness and kurtosis:

            σ(SR) = sqrt( (1 − γ₃·SR + (γ₄−1)/4 · SR²) / (n−1) )

        γ₃ = skewness, γ₄ = excess kurtosis.
        """
        if n <= 1:
            return 0.5

        sr = observed
        sr_var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n - 1)

        if sr_var <= 0:
            # Degenerate — treat as perfectly significant if above benchmark
            return 1.0 if observed > benchmark else 0.0

        sr_std = math.sqrt(sr_var)
        z = (observed - benchmark) / sr_std
        return float(norm.cdf(z))
