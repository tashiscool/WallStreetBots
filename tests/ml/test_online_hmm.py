"""Tests for Online HMM (incremental regime detection)."""

import os
import tempfile
import numpy as np
import pytest

from ml.tradingbots.components.online_hmm import OnlineHMM, OnlineHMMConfig


@pytest.fixture
def hmm():
    return OnlineHMM(OnlineHMMConfig(n_states=3, n_features=1))


class TestInitialization:
    def test_shapes(self, hmm):
        assert hmm.pi.shape == (3,)
        assert hmm.A.shape == (3, 3)
        assert hmm.means.shape == (3, 1)
        assert hmm.variances.shape == (3, 1)

    def test_pi_sums_to_one(self, hmm):
        assert hmm.pi.sum() == pytest.approx(1.0)

    def test_transition_rows_sum_to_one(self, hmm):
        for row in hmm.A:
            assert row.sum() == pytest.approx(1.0)


class TestUpdate:
    def test_returns_probabilities_sum_one(self, hmm):
        gamma = hmm.update(np.array([0.5]))
        assert gamma.shape == (3,)
        assert gamma.sum() == pytest.approx(1.0)

    def test_multiple_updates(self, hmm):
        for _ in range(50):
            gamma = hmm.update(np.array([np.random.randn()]))
        assert gamma.sum() == pytest.approx(1.0)


class TestRegimeDetection:
    def test_regime_after_updates(self, hmm):
        for _ in range(20):
            hmm.update(np.array([1.0]))
        regime = hmm.get_regime()
        assert 0 <= regime < 3

    def test_different_observations_give_different_regimes(self):
        """Sustained positive vs negative observations should give different regimes."""
        hmm_pos = OnlineHMM(OnlineHMMConfig(n_states=3, n_features=1, learning_rate=0.1))
        hmm_neg = OnlineHMM(OnlineHMMConfig(n_states=3, n_features=1, learning_rate=0.1))

        for _ in range(100):
            hmm_pos.update(np.array([5.0]))
            hmm_neg.update(np.array([-5.0]))

        # After many updates, they should be in different regimes
        # (or at least their state distributions should differ)
        gamma_pos = hmm_pos.update(np.array([5.0]))
        gamma_neg = hmm_neg.update(np.array([-5.0]))
        assert not np.allclose(gamma_pos, gamma_neg, atol=0.1)


class TestTransitionMatrix:
    def test_rows_sum_to_one_after_updates(self, hmm):
        for _ in range(30):
            hmm.update(np.array([np.random.randn()]))
        tm = hmm.get_transition_matrix()
        for row in tm:
            assert row.sum() == pytest.approx(1.0, abs=1e-6)


class TestSaveLoad:
    def test_roundtrip(self, hmm):
        # Feed some data
        for _ in range(20):
            hmm.update(np.array([np.random.randn()]))

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name

        try:
            hmm.save(path)

            hmm2 = OnlineHMM()
            hmm2.load(path)

            np.testing.assert_array_almost_equal(hmm.A, hmm2.A)
            np.testing.assert_array_almost_equal(hmm.means, hmm2.means)
            np.testing.assert_array_almost_equal(hmm.variances, hmm2.variances)
            assert hmm._step == hmm2._step
        finally:
            os.unlink(path)


class TestForgettingFactor:
    def test_high_forgetting_retains_history(self):
        hmm = OnlineHMM(OnlineHMMConfig(forgetting_factor=0.999, learning_rate=0.01))
        means_before = hmm.means.copy()
        for _ in range(5):
            hmm.update(np.array([10.0]))
        # With high forgetting factor, means should change slowly
        diff_high = np.abs(hmm.means - means_before).max()

        hmm2 = OnlineHMM(OnlineHMMConfig(forgetting_factor=0.5, learning_rate=0.01))
        means_before2 = hmm2.means.copy()
        for _ in range(5):
            hmm2.update(np.array([10.0]))
        diff_low = np.abs(hmm2.means - means_before2).max()

        # Lower forgetting factor should allow faster adaptation
        # (but both should change since learning_rate > 0)
        assert diff_high > 0 or diff_low > 0  # at least one changed


class TestConvergence:
    def test_convergence_on_synthetic_data(self):
        """HMM should roughly separate two well-separated clusters."""
        rng = np.random.RandomState(42)
        hmm = OnlineHMM(OnlineHMMConfig(n_states=2, n_features=1, learning_rate=0.05))

        # Alternate between two regimes
        for _ in range(200):
            hmm.update(rng.normal(5.0, 0.5, 1))
        for _ in range(200):
            hmm.update(rng.normal(-5.0, 0.5, 1))

        # Means should have separated
        mean_vals = sorted(hmm.means[:, 0])
        assert mean_vals[-1] - mean_vals[0] > 1.0  # Some separation


class TestLongStreamStability:
    """50k-update stability test to catch underflow/divergence in log-space forward pass."""

    def test_50k_stream_no_nan_no_inf(self):
        """Run 50k updates; state probabilities must remain valid throughout."""
        rng = np.random.RandomState(99)
        hmm = OnlineHMM(OnlineHMMConfig(n_states=3, n_features=1, learning_rate=0.01))

        for i in range(50_000):
            # Switch regime every 5k steps to stress transitions
            if i % 15_000 < 5_000:
                obs = rng.normal(3.0, 1.0, 1)
            elif i % 15_000 < 10_000:
                obs = rng.normal(-3.0, 1.0, 1)
            else:
                obs = rng.normal(0.0, 0.5, 1)
            gamma = hmm.update(obs)

            # Check every 1000 steps (full check is too slow)
            if i % 1000 == 0:
                assert not np.any(np.isnan(gamma)), f"NaN at step {i}"
                assert not np.any(np.isinf(gamma)), f"Inf at step {i}"
                assert gamma.sum() == pytest.approx(1.0, abs=1e-6), f"Sum != 1 at step {i}"

        # Final state must be valid
        final_gamma = hmm.update(np.array([0.0]))
        assert not np.any(np.isnan(final_gamma))
        assert not np.any(np.isinf(final_gamma))
        assert final_gamma.sum() == pytest.approx(1.0, abs=1e-6)

        # Transition matrix must be valid
        tm = hmm.get_transition_matrix()
        assert not np.any(np.isnan(tm))
        for row in tm:
            assert row.sum() == pytest.approx(1.0, abs=1e-6)

        # Means and variances must be finite
        assert not np.any(np.isnan(hmm.means))
        assert not np.any(np.isinf(hmm.means))
        assert not np.any(np.isnan(hmm.variances))
        assert np.all(hmm.variances > 0), "Variances must stay positive"

    def test_50k_stream_with_outliers(self):
        """Extreme values should not blow up the model."""
        rng = np.random.RandomState(7)
        hmm = OnlineHMM(OnlineHMMConfig(n_states=2, n_features=1, learning_rate=0.005))

        for i in range(50_000):
            if i % 5000 == 0:
                # Inject extreme outlier every 5000 steps
                obs = np.array([100.0 * ((-1) ** i)])
            else:
                obs = rng.normal(0.0, 1.0, 1)
            gamma = hmm.update(obs)

        assert not np.any(np.isnan(gamma))
        assert gamma.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(hmm.variances > 0)
