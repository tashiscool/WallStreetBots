"""Online (Incremental) Hidden Markov Model for real-time regime detection.

Pure-numpy implementation that processes one observation at a time,
updating parameters via online EM with a forgetting factor.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class OnlineHMMConfig:
    """Configuration for Online HMM."""
    n_states: int = 3
    n_features: int = 1
    learning_rate: float = 0.01
    forgetting_factor: float = 0.99


class OnlineHMM:
    """Incremental Hidden Markov Model.

    Uses online EM (Cappé, 2011) to update parameters after every new
    observation, allowing real-time regime detection without re-fitting.
    """

    def __init__(self, config: Optional[OnlineHMMConfig] = None):
        self.config = config or OnlineHMMConfig()
        k = self.config.n_states
        d = self.config.n_features

        # Initial state distribution — uniform
        self.pi = np.ones(k) / k

        # Transition matrix — mildly sticky diagonals
        self.A = np.full((k, k), 0.1 / (k - 1) if k > 1 else 1.0)
        np.fill_diagonal(self.A, 0.9)
        # Normalise rows
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Emission parameters — Gaussian (mean, variance per state per feature)
        # Spread initial means so states start differentiated
        self.means = np.linspace(-1, 1, k).reshape(k, 1) * np.ones((1, d))
        self.variances = np.ones((k, d))

        # Running forward variable (alpha)
        self._alpha: Optional[np.ndarray] = None
        self._step = 0

    def update(self, observation: np.ndarray) -> np.ndarray:
        """Process one observation and return state probabilities.

        Parameters
        ----------
        observation : array-like, shape (n_features,)

        Returns
        -------
        gamma : np.ndarray, shape (n_states,)
            Posterior state probabilities (normalised to sum 1).
        """
        obs = np.atleast_1d(np.asarray(observation, dtype=np.float64))
        k = self.config.n_states

        # Emission likelihoods: P(obs | state_j)
        emission = self._emission_prob(obs)

        # Forward step
        if self._alpha is None:
            alpha = self.pi * emission
        else:
            alpha = self._forward_step(obs, self._alpha)

        # Normalise
        alpha_sum = alpha.sum()
        if alpha_sum > 0:
            alpha /= alpha_sum
        else:
            alpha = np.ones(k) / k

        self._alpha = alpha
        gamma = alpha.copy()

        # Online parameter update
        self._update_parameters(obs, gamma)
        self._step += 1

        return gamma

    def get_regime(self) -> int:
        """Return the most likely current state (0-indexed)."""
        if self._alpha is None:
            return 0
        return int(np.argmax(self._alpha))

    def get_transition_matrix(self) -> np.ndarray:
        """Return the current transition matrix."""
        return self.A.copy()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_emission_prob(self, obs: np.ndarray) -> np.ndarray:
        """Log Gaussian emission probability for each state (log-space)."""
        k = self.config.n_states
        log_probs = np.zeros(k)
        for j in range(k):
            diff = obs - self.means[j]
            var = self.variances[j]
            log_probs[j] = -0.5 * np.sum(diff ** 2 / var) - 0.5 * np.sum(np.log(2 * np.pi * var))
        return log_probs

    def _emission_prob(self, obs: np.ndarray) -> np.ndarray:
        """Gaussian emission probability for each state (with scaling)."""
        log_probs = self._log_emission_prob(obs)
        # Subtract max for numerical stability before exp
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs = np.maximum(probs, 1e-300)
        return probs

    def _forward_step(self, obs: np.ndarray, alpha_prev: np.ndarray) -> np.ndarray:
        """Single forward pass with log-space scaling to prevent underflow.

        a_t(j) = P(o_t|j) * sum_i a_{t-1}(i) * A(i,j)
        """
        predicted = alpha_prev @ self.A          # shape (k,)
        emission = self._emission_prob(obs)      # shape (k,)
        alpha = predicted * emission
        # Scale to prevent underflow accumulation
        alpha_sum = alpha.sum()
        if alpha_sum > 0:
            alpha /= alpha_sum
        else:
            alpha = np.ones(self.config.n_states) / self.config.n_states
        return alpha

    def _update_parameters(self, obs: np.ndarray, gamma: np.ndarray) -> None:
        """Online EM parameter update with forgetting factor."""
        lr = self.config.learning_rate
        ff = self.config.forgetting_factor
        k = self.config.n_states

        for j in range(k):
            w = gamma[j]
            if w < 1e-12:
                continue

            # Update means: μ_j ← ff * μ_j + lr * w * (obs - μ_j)
            self.means[j] += lr * w * (obs - self.means[j])

            # Update variances: σ²_j ← ff * σ²_j + lr * w * ((obs - μ_j)² - σ²_j)
            diff_sq = (obs - self.means[j]) ** 2
            self.variances[j] = ff * self.variances[j] + lr * w * (diff_sq - self.variances[j])
            # Floor variance to prevent collapse
            self.variances[j] = np.maximum(self.variances[j], 1e-6)

        # Update transition matrix from successive gammas
        if self._alpha is not None and self._step > 0:
            # Approximate: A(i,j) <- ff * A(i,j) + lr * alpha_prev(i) * gamma_curr(j)
            outer = np.outer(self._alpha, gamma)
            self.A = ff * self.A + lr * outer
            # Re-normalise rows
            row_sums = self.A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.A /= row_sums

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    VERSION = 2  # increment when format changes

    def _config_checksum(self) -> int:
        """Deterministic hash of config for compatibility checking."""
        import hashlib
        config_str = f"{self.config.n_states}:{self.config.n_features}:{self.config.learning_rate}:{self.config.forgetting_factor}"
        return int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16)  # noqa: S324

    def save(self, path: str) -> None:
        """Save model parameters to .npz file with version and checksum."""
        np.savez(
            path,
            pi=self.pi,
            A=self.A,
            means=self.means,
            variances=self.variances,
            step=np.array([self._step]),
            alpha=self._alpha if self._alpha is not None else np.array([]),
            # Config
            n_states=np.array([self.config.n_states]),
            n_features=np.array([self.config.n_features]),
            learning_rate=np.array([self.config.learning_rate]),
            forgetting_factor=np.array([self.config.forgetting_factor]),
            # Versioning
            version=np.array([self.VERSION]),
            config_checksum=np.array([self._config_checksum()]),
        )

    def load(self, path: str) -> None:
        """Load model parameters from .npz file with version check."""
        data = np.load(path)

        # Version check
        file_version = int(data['version'][0]) if 'version' in data else 1
        if file_version > self.VERSION:
            raise ValueError(
                f"Model file version {file_version} > supported {self.VERSION}"
            )

        self.config = OnlineHMMConfig(
            n_states=int(data['n_states'][0]),
            n_features=int(data['n_features'][0]),
            learning_rate=float(data['learning_rate'][0]),
            forgetting_factor=float(data['forgetting_factor'][0]),
        )

        # Config checksum validation (v2+)
        if 'config_checksum' in data:
            expected = self._config_checksum()
            actual = int(data['config_checksum'][0])
            if expected != actual:
                raise ValueError(
                    f"Config checksum mismatch: file={actual}, current={expected}. "
                    f"Model was saved with different config."
                )

        self.pi = data['pi']
        self.A = data['A']
        self.means = data['means']
        self.variances = data['variances']
        self._step = int(data['step'][0])
        alpha = data['alpha']
        self._alpha = alpha if alpha.size > 0 else None
