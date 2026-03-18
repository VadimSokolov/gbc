"""Financial application: multivariate returns, MGBC, and portfolio allocation.

DGP: 3-asset daily log-returns with constant correlation structure.
     r_t ~ N(μ, Σ)
     μ = [0.0002, 0.0001, 0.00015]  (daily expected returns)
     Σ encodes realistic cross-asset correlations (ρ ≈ 0.5–0.7)

This tests the full MGBC pipeline from Section 7 of
Lopes, Polson & Sokolov (2026):
  1. Simulate multivariate returns
  2. Cholesky preconditioning
  3. Variable ordering by variance
  4. Train autoregressive IQN chain
  5. Draw joint posterior samples
  6. Evaluate with energy score
  7. Bayesian portfolio allocation via MEU
"""

import numpy as np
import torch
import pytest

from gbc.multivariate import (
    train_multivariate_iqn, sample_multivariate_iqn,
    predict_multivariate_iqn, order_by_variance,
    cholesky_precondition, cholesky_inverse,
)
from gbc.metrics import energy_score, crps_samples, rmse
from gbc.welfare import portfolio_meu


# ── Synthetic financial data ────────────────────────────────────

@pytest.fixture(scope="module")
def returns_data():
    """Simulate 3-asset daily log-returns with known correlation."""
    rng = np.random.default_rng(42)
    d = 3
    n = 600

    # Daily expected returns
    mu = np.array([0.0002, 0.0001, 0.00015])

    # Correlation matrix → covariance (daily vol ~1-2%)
    vols = np.array([0.015, 0.012, 0.018])
    corr = np.array([
        [1.0, 0.6, 0.5],
        [0.6, 1.0, 0.7],
        [0.5, 0.7, 1.0],
    ])
    Sigma = np.outer(vols, vols) * corr

    # Generate returns
    L = np.linalg.cholesky(Sigma)
    R = rng.standard_normal((n, d)) @ L.T + mu

    # Conditioning features: lagged returns + constant
    X = np.column_stack([
        np.roll(R, 1, axis=0),  # 1-day lag
        np.ones(n),
    ])
    # Remove first row (no valid lag)
    X, R = X[1:], R[1:]

    return X, R, mu, Sigma


# ── Energy Score ────────────────────────────────────────────────

class TestEnergyScore:
    def test_shape_and_positive(self, returns_data):
        X, R, _, _ = returns_data
        rng = np.random.default_rng(0)
        # Fake samples: (B, n, d)
        B, n, d = 50, len(R), R.shape[1]
        samples = rng.standard_normal((B, n, d)) * 0.02
        es = energy_score(R, samples)
        assert np.isfinite(es)
        assert es > 0

    def test_perfect_samples_lower_score(self, returns_data):
        _, R, _, _ = returns_data
        n, d = R.shape
        rng = np.random.default_rng(0)

        # Good samples: actual data with small noise
        good = R[np.newaxis, :, :] + rng.normal(0, 0.001, (30, n, d))
        # Bad samples: random noise
        bad = rng.standard_normal((30, n, d)) * 0.1

        es_good = energy_score(R, good)
        es_bad = energy_score(R, bad)
        assert es_good < es_bad

    def test_single_observation(self):
        """Energy score should work for a single multivariate observation."""
        y = np.array([[1.0, 2.0, 3.0]])  # (1, 3)
        samples = np.random.randn(50, 1, 3) * 0.5 + y
        es = energy_score(y, samples)
        assert np.isfinite(es) and es > 0


# ── Variable Ordering ──────────────────────────────────────────

class TestOrderByVariance:
    def test_decreasing_variance(self, returns_data):
        _, R, _, _ = returns_data
        order = order_by_variance(R)
        variances = np.var(R, axis=0)
        ordered_vars = variances[order]
        # Should be non-increasing
        assert np.all(np.diff(ordered_vars) <= 1e-15)

    def test_highest_vol_first(self, returns_data):
        _, R, _, Sigma = returns_data
        order = order_by_variance(R)
        # Asset 2 (vol=0.018) should be first
        assert order[0] == 2

    def test_shape(self, returns_data):
        _, R, _, _ = returns_data
        order = order_by_variance(R)
        assert order.shape == (R.shape[1],)
        assert set(order) == {0, 1, 2}


# ── MGBC Pipeline ──────────────────────────────────────────────

class TestMGBCPipeline:
    @pytest.fixture(scope="class")
    def mgbc_chain(self, returns_data):
        X, R, _, _ = returns_data
        # Use variance ordering
        order = order_by_variance(R)
        R_ordered = R[:, order]

        chain = train_multivariate_iqn(
            X, R_ordered, epochs=300, hdim=64, nh=16, seed=0,
        )
        return chain, order

    def test_chain_length(self, mgbc_chain, returns_data):
        chain, _ = mgbc_chain
        _, R, _, _ = returns_data
        assert len(chain) == R.shape[1]

    def test_sample_shape(self, mgbc_chain, returns_data):
        chain, _ = mgbc_chain
        X, _, _, _ = returns_data
        samples = sample_multivariate_iqn(chain, X[:20], B=30, seed=0)
        assert samples.shape == (30, 20, 3)

    def test_samples_finite(self, mgbc_chain, returns_data):
        chain, _ = mgbc_chain
        X, _, _, _ = returns_data
        samples = sample_multivariate_iqn(chain, X[:20], B=30, seed=0)
        assert np.all(np.isfinite(samples))

    def test_predict_shape(self, mgbc_chain, returns_data):
        chain, _ = mgbc_chain
        X, _, _, _ = returns_data
        taus = [0.1, 0.5, 0.9]
        preds = predict_multivariate_iqn(chain, X[:10], taus=taus)
        assert preds.shape == (10, 3, 3)  # (n, d, n_q)

    def test_predict_monotonicity(self, mgbc_chain, returns_data):
        """Lower quantiles should generally be below upper."""
        chain, _ = mgbc_chain
        X, _, _, _ = returns_data
        preds = predict_multivariate_iqn(chain, X[:50], taus=[0.1, 0.9])
        # Check each component
        for j in range(3):
            frac = np.mean(preds[:, j, 0] <= preds[:, j, 1])
            assert frac > 0.7

    def test_energy_score_on_samples(self, mgbc_chain, returns_data):
        chain, order = mgbc_chain
        X, R, _, _ = returns_data
        R_ordered = R[:, order]
        samples = sample_multivariate_iqn(chain, X[:50], B=50, seed=0)
        es = energy_score(R_ordered[:50], samples)
        assert np.isfinite(es)


# ── Cholesky + MGBC Pipeline ───────────────────────────────────

class TestCholeskyMGBC:
    def test_whitened_pipeline(self, returns_data):
        """Cholesky preconditioning → train → sample → inverse."""
        X, R, _, _ = returns_data
        Z, L, mu = cholesky_precondition(R)

        chain = train_multivariate_iqn(
            X, Z, epochs=200, hdim=32, nh=8, seed=0,
        )
        Z_samples = sample_multivariate_iqn(chain, X[:10], B=20, seed=0)
        R_samples = cholesky_inverse(Z_samples, L, mu)

        assert R_samples.shape == (20, 10, 3)
        assert np.all(np.isfinite(R_samples))

    def test_ordered_whitened_pipeline(self, returns_data):
        """Ordering + Cholesky preconditioning → train → sample → inverse → reorder."""
        X, R, _, _ = returns_data
        order = order_by_variance(R)
        R_ordered = R[:, order]

        Z, L, mu = cholesky_precondition(R_ordered)
        chain = train_multivariate_iqn(
            X, Z, epochs=200, hdim=32, nh=8, seed=0,
        )
        Z_samples = sample_multivariate_iqn(chain, X[:10], B=20, seed=0)
        R_samples_ordered = cholesky_inverse(Z_samples, L, mu)

        # Undo ordering
        inv_order = np.argsort(order)
        R_samples = R_samples_ordered[:, :, inv_order]

        assert R_samples.shape == (20, 10, 3)
        assert np.all(np.isfinite(R_samples))


# ── Portfolio Allocation ────────────────────────────────────────

class TestPortfolioMEU:
    @pytest.fixture(scope="class")
    def return_samples(self, returns_data):
        """Gross returns (1 + r) for portfolio tests — always positive."""
        _, R, mu, Sigma = returns_data
        rng = np.random.default_rng(42)
        L = np.linalg.cholesky(Sigma)
        # 1000 MC draws of gross returns (1 + r), always positive
        samples = 1.0 + rng.standard_normal((1000, 3)) @ L.T + mu
        return samples

    def test_weights_on_simplex(self, return_samples):
        result = portfolio_meu(return_samples)
        w = result["weights"]
        assert w.shape == (3,)
        assert np.all(w >= -1e-8)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)

    def test_expected_utility_finite(self, return_samples):
        result = portfolio_meu(return_samples)
        assert np.isfinite(result["expected_utility"])

    def test_portfolio_returns_shape(self, return_samples):
        result = portfolio_meu(return_samples)
        assert result["portfolio_returns"].shape == (len(return_samples),)

    def test_custom_utility(self, return_samples):
        """Power utility: U(x) = x^(1-γ) / (1-γ) for γ=2."""
        gamma = 2.0
        power_util = lambda x: x ** (1 - gamma) / (1 - gamma)
        result = portfolio_meu(return_samples, utility=power_util)
        w = result["weights"]
        assert np.all(w >= -1e-8)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-6)

    def test_higher_return_gets_more_weight(self):
        """Asset with highest expected return should get nontrivial weight."""
        rng = np.random.default_rng(0)
        # Asset 0: high return, Asset 1-2: low return
        samples = np.column_stack([
            rng.normal(0.02, 0.01, 500),
            rng.normal(0.005, 0.01, 500),
            rng.normal(0.005, 0.01, 500),
        ])
        result = portfolio_meu(samples)
        # Asset 0 should get the most weight
        assert result["weights"][0] > result["weights"][1]
        assert result["weights"][0] > result["weights"][2]


# ── End-to-end Financial Test ───────────────────────────────────

class TestEndToEnd:
    def test_full_financial_pipeline(self, returns_data):
        """Complete MGBC financial pipeline:
        order → precondition → train → sample → de-whiten → reorder → portfolio.
        """
        X, R, _, _ = returns_data

        # 1. Order by variance
        order = order_by_variance(R)
        R_ord = R[:, order]

        # 2. Cholesky precondition
        Z, L, mu = cholesky_precondition(R_ord)

        # 3. Train MGBC chain
        chain = train_multivariate_iqn(
            X, Z, epochs=200, hdim=32, nh=8, seed=0,
        )

        # 4. Sample
        Z_samples = sample_multivariate_iqn(chain, X[:50], B=100, seed=0)

        # 5. Inverse Cholesky + reorder
        R_samples_ord = cholesky_inverse(Z_samples, L, mu)
        inv_order = np.argsort(order)
        R_samples = R_samples_ord[:, :, inv_order]
        assert R_samples.shape == (100, 50, 3)

        # 6. Energy score
        es = energy_score(R[:50], R_samples)
        assert np.isfinite(es)

        # 7. Portfolio allocation (using mean across test observations)
        mean_returns = R_samples.mean(axis=1)  # (100, 3) — MC draws of mean return
        # Shift to positive for log utility
        mean_returns_pos = mean_returns - mean_returns.min() + 0.001
        result = portfolio_meu(mean_returns_pos)
        assert np.sum(result["weights"]) == pytest.approx(1.0, abs=1e-6)
        assert np.isfinite(result["expected_utility"])
