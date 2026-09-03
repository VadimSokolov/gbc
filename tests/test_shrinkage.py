"""Tests for gbc.shrinkage — horseshoe and hierarchical shrinkage priors."""

import numpy as np
import pytest
from gbc.shrinkage import (
    horseshoe_draw,
    horseshoe_batch,
    shrinkage_factor,
    horseshoe_posterior,
    simulate_hierarchical_training_data,
)


class TestHorseshoeDraw:
    def test_shape(self):
        draw = horseshoe_draw(5, global_tau=0.1, rng=np.random.default_rng(0))
        assert draw["theta"].shape == (5,)
        assert draw["lambda_"].shape == (5,)
        assert draw["kappa"].shape == (5,)
        assert isinstance(draw["tau"], float)

    def test_kappa_range(self):
        draw = horseshoe_draw(100, global_tau=0.1, rng=np.random.default_rng(0))
        assert np.all(draw["kappa"] >= 0)
        assert np.all(draw["kappa"] <= 1)

    def test_reproducible(self):
        d1 = horseshoe_draw(5, rng=np.random.default_rng(42))
        d2 = horseshoe_draw(5, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(d1["theta"], d2["theta"])


class TestHorseshoeBatch:
    def test_shapes(self):
        batch = horseshoe_batch(n_draws=100, n_groups=5, seed=0)
        assert batch["theta"].shape == (100, 5)
        assert batch["kappa"].shape == (100, 5)

    def test_sparsity(self):
        """Most kappa values should be near 0 or 1 (horseshoe property)."""
        batch = horseshoe_batch(n_draws=1000, n_groups=10, global_tau=0.05, seed=0)
        kappa = batch["kappa"].ravel()
        near_zero = np.mean(kappa < 0.3)
        near_one = np.mean(kappa > 0.7)
        # At least 60% should be near the extremes
        assert near_zero + near_one > 0.5


class TestShrinkageFactor:
    def test_shape(self):
        mle = np.array([0.1, 0.5, 0.05, 0.8])
        se = np.array([0.1, 0.1, 0.1, 0.1])
        k = shrinkage_factor(mle, se)
        assert k.shape == (4,)

    def test_range(self):
        mle = np.array([0.0, 1.0, 0.5])
        se = np.array([0.1, 0.1, 0.1])
        k = shrinkage_factor(mle, se)
        assert np.all(k >= 0) and np.all(k <= 1)

    def test_large_signal_little_shrinkage(self):
        """Points far from the mean should have kappa near 1."""
        mle = np.array([0.0, 0.0, 0.0, 5.0])
        se = np.array([0.1, 0.1, 0.1, 0.1])
        k = shrinkage_factor(mle, se)
        assert k[3] > 0.9


class TestHorseshoePosterior:
    def test_shapes(self):
        mle = np.array([0.1, 0.05, 0.3, 0.02, 0.15])
        se = np.array([0.08, 0.09, 0.10, 0.08, 0.11])
        result = horseshoe_posterior(mle, se, global_tau=0.1, n_iter=200, seed=0)
        assert result["theta_mean"].shape == (5,)
        assert result["theta_lower"].shape == (5,)
        assert result["theta_upper"].shape == (5,)
        assert result["kappa_mean"].shape == (5,)
        assert result["theta_samples"].shape[1] == 5

    def test_shrinkage_toward_mean(self):
        """Posterior means should be between MLE and global mean."""
        mle = np.array([0.0, 0.0, 0.0, 0.0, 2.0])
        se = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = horseshoe_posterior(mle, se, global_tau=0.1, n_iter=500, seed=42)
        global_mean = np.mean(mle)
        # The outlier (2.0) should be shrunk toward the mean
        assert result["theta_mean"][4] < mle[4]

    def test_ci_contains_mean(self):
        mle = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        se = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        result = horseshoe_posterior(mle, se, n_iter=500, seed=0)
        # Credible intervals should be reasonable
        assert np.all(result["theta_lower"] < result["theta_upper"])


class TestHierarchicalSimulation:
    def test_shapes(self):
        data = simulate_hierarchical_training_data(
            n_sim=20, n_groups=3, n_obs_per_group=10, seed=0
        )
        assert data["X"].shape[0] == 20
        assert data["y"].shape == (20, 3)
        assert data["theta_true"].shape == (20, 3)
        assert data["kappa_true"].shape == (20, 3)

    def test_custom_simulator(self):
        def sim(theta, n, rng):
            scale = np.exp(np.clip(theta, -5, 5))
            return rng.exponential(scale=scale, size=n)

        data = simulate_hierarchical_training_data(
            n_sim=10, n_groups=2, group_simulator=sim, seed=0
        )
        assert data["X"].shape[0] == 10
        assert np.all(np.isfinite(data["X"]))
