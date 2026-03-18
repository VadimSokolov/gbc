"""Multivariate use case: bivariate posterior via autoregressive IQN.

DGP: (θ₁, θ₂) ~ N(μ, Σ)  with μ = [1, 2], Σ = [[1, 0.6], [0.6, 1]]
     Conditioning feature: x ~ U(0, 1) (dummy, not used in DGP).
"""

import numpy as np
import pytest

from gbc.multivariate import (
    train_multivariate_iqn, sample_multivariate_iqn,
    cholesky_precondition, cholesky_inverse,
)


@pytest.fixture(scope="module")
def mv_data():
    rng = np.random.default_rng(42)
    n = 500
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[1.0, 0.6], [0.6, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = rng.standard_normal((n, 2)) @ L.T + mu
    X = rng.uniform(0, 1, (n, 1))  # dummy feature
    return X, Y, mu, Sigma


# ── Cholesky preconditioning ────────────────────────────────────

class TestCholesky:
    def test_precondition_shape(self, mv_data):
        _, Y, _, _ = mv_data
        Z, L, mu = cholesky_precondition(Y)
        assert Z.shape == Y.shape
        assert L.shape == (2, 2)
        assert mu.shape == (2,)

    def test_whitened_near_standard(self, mv_data):
        _, Y, _, _ = mv_data
        Z, L, mu = cholesky_precondition(Y)
        # Z should have ~zero mean and ~unit variance
        assert np.abs(Z.mean(axis=0)).max() < 0.15
        assert np.abs(Z.std(axis=0) - 1.0).max() < 0.15

    def test_round_trip(self, mv_data):
        _, Y, _, _ = mv_data
        Z, L, mu = cholesky_precondition(Y)
        Y_rec = cholesky_inverse(Z, L, mu)
        assert np.allclose(Y, Y_rec, atol=1e-10)

    def test_inverse_batch_shape(self, mv_data):
        _, Y, _, _ = mv_data
        Z, L, mu = cholesky_precondition(Y)
        # Test with (B, n, d) shape
        Z_batch = np.stack([Z[:10], Z[:10]], axis=0)  # (2, 10, 2)
        Y_batch = cholesky_inverse(Z_batch, L, mu)
        assert Y_batch.shape == (2, 10, 2)


# ── Autoregressive IQN chain ───────────────────────────────────

class TestMultivariateIQN:
    @pytest.fixture(scope="class")
    def chain(self, mv_data):
        X, Y, _, _ = mv_data
        return train_multivariate_iqn(
            X, Y, epochs=200, hdim=32, nh=8, seed=0,
        )

    def test_chain_length(self, chain, mv_data):
        _, Y, _, _ = mv_data
        assert len(chain) == Y.shape[1]  # d=2 models

    def test_chain_input_dims_grow(self, chain, mv_data):
        X, _, _, _ = mv_data
        p = X.shape[1]
        for j, (model_j, xm_j, xs_j, ym_j, ys_j) in enumerate(chain):
            expected_dim = p + j
            assert xm_j.shape == (expected_dim,)

    def test_sample_shape(self, chain, mv_data):
        X, _, _, _ = mv_data
        samples = sample_multivariate_iqn(chain, X[:20], B=30, seed=0)
        assert samples.shape == (30, 20, 2)

    def test_samples_finite(self, chain, mv_data):
        X, _, _, _ = mv_data
        samples = sample_multivariate_iqn(chain, X[:20], B=30, seed=0)
        assert np.all(np.isfinite(samples))


# ── Cholesky + IQN pipeline ────────────────────────────────────

class TestCholeskyPipeline:
    def test_preconditioned_training(self, mv_data):
        """Train on whitened targets, invert samples back."""
        X, Y, _, _ = mv_data
        Z, L, mu = cholesky_precondition(Y)

        chain = train_multivariate_iqn(
            X, Z, epochs=200, hdim=32, nh=8, seed=0,
        )
        Z_samples = sample_multivariate_iqn(chain, X[:10], B=20, seed=0)
        Y_samples = cholesky_inverse(Z_samples, L, mu)

        assert Y_samples.shape == (20, 10, 2)
        assert np.all(np.isfinite(Y_samples))
