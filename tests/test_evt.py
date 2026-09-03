"""Tests for gbc.evt — EVT distributions and simulation."""

import numpy as np
import pytest
from gbc.evt import (
    gev_quantile,
    gev_cdf,
    gev_survivor,
    gev_density,
    gev_return_level,
    gpd_quantile,
    gpd_survivor,
    gpd_density,
    poisson_loglik,
    simulate_gev_training_data,
    simulate_gpd_training_data,
    simulate_nonstationary_gev_training_data,
)


# ── GEV tests ──────────────────────────────────────────────────────

class TestGEV:
    def test_quantile_cdf_inverse(self):
        """Q(G(x)) should equal x."""
        for xi in [-0.2, 0.0, 0.1, 0.5]:
            x = np.linspace(-1, 5, 50)
            tau = gev_cdf(x, mu=1.0, sigma=2.0, xi=xi)
            x_back = gev_quantile(tau, mu=1.0, sigma=2.0, xi=xi)
            np.testing.assert_allclose(x_back, x, atol=1e-6)

    def test_survivor_plus_cdf(self):
        """S(t) + G(t) = 1."""
        t = np.linspace(-2, 10, 30)
        for xi in [-0.3, 0.0, 0.2]:
            S = gev_survivor(t, 1.0, 2.0, xi)
            G = gev_cdf(t, 1.0, 2.0, xi)
            np.testing.assert_allclose(S + G, 1.0, atol=1e-12)

    def test_cdf_range(self):
        t = np.linspace(-5, 15, 100)
        G = gev_cdf(t, 0.0, 1.0, 0.1)
        assert np.all(G >= 0) and np.all(G <= 1)

    def test_density_integrates_to_one(self):
        """Numerical integration of the density should be close to 1."""
        t = np.linspace(-10, 20, 5000)
        for xi in [-0.2, 0.0, 0.3]:
            g = gev_density(t, 0.0, 1.0, xi)
            integral = np.trapezoid(g, t)
            assert abs(integral - 1.0) < 0.02

    def test_density_nonnegative(self):
        t = np.linspace(-5, 15, 200)
        g = gev_density(t, 0.0, 1.0, 0.2)
        assert np.all(g >= 0)

    def test_gumbel_special_case(self):
        """xi=0 should give the Gumbel distribution."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        G_gev = gev_cdf(x, 0.0, 1.0, 0.0)
        G_gumbel = np.exp(-np.exp(-x))
        np.testing.assert_allclose(G_gev, G_gumbel, atol=1e-8)

    def test_return_level_ordering(self):
        """Longer return periods give larger return levels."""
        rl50 = gev_return_level(50, 0.0, 1.0, 0.1)
        rl100 = gev_return_level(100, 0.0, 1.0, 0.1)
        rl500 = gev_return_level(500, 0.0, 1.0, 0.1)
        assert rl50 < rl100 < rl500

    def test_quantile_monotone(self):
        tau = np.linspace(0.01, 0.99, 100)
        q = gev_quantile(tau, 0.0, 1.0, 0.2)
        assert np.all(np.diff(q) > 0)


# ── GPD tests ──────────────────────────────────────────────────────

class TestGPD:
    def test_quantile_survivor_inverse(self):
        """S(Q(tau)) = 1 - tau."""
        for xi in [-0.2, 0.0, 0.3]:
            tau = np.linspace(0.01, 0.99, 50)
            q = gpd_quantile(tau, sigma=2.0, xi=xi)
            S = gpd_survivor(q, sigma=2.0, xi=xi)
            np.testing.assert_allclose(S, 1.0 - tau, atol=1e-6)

    def test_survivor_range(self):
        y = np.linspace(0.01, 20, 100)
        S = gpd_survivor(y, sigma=1.0, xi=0.2)
        assert np.all(S >= 0) and np.all(S <= 1)

    def test_density_nonnegative(self):
        y = np.linspace(0.01, 10, 200)
        g = gpd_density(y, sigma=1.0, xi=0.2)
        assert np.all(g >= 0)

    def test_exponential_special_case(self):
        """xi=0 should give the exponential distribution."""
        y = np.array([0.5, 1.0, 2.0])
        S_gpd = gpd_survivor(y, sigma=1.0, xi=0.0)
        S_exp = np.exp(-y)
        np.testing.assert_allclose(S_gpd, S_exp, atol=1e-8)

    def test_quantile_monotone(self):
        tau = np.linspace(0.01, 0.99, 100)
        q = gpd_quantile(tau, sigma=1.0, xi=0.2)
        assert np.all(np.diff(q) > 0)


# ── Poisson process likelihood ─────────────────────────────────────

class TestPoissonLoglik:
    def test_returns_finite(self):
        y = np.array([30, 32, 35, 37, 38, 40, 41, 33, 34, 36])
        ll = poisson_loglik(y, mu=35.0, sigma=3.0, xi=0.1, u=37.0)
        assert np.isfinite(ll)

    def test_invalid_sigma(self):
        y = np.array([30, 32, 35])
        assert poisson_loglik(y, 35.0, -1.0, 0.1, 32.0) == -np.inf

    def test_higher_at_true_params(self):
        """Likelihood should be higher at true parameters."""
        rng = np.random.default_rng(123)
        n = 100
        u_vals = rng.uniform(0, 1, size=n)
        y = gev_quantile(u_vals, mu=35.0, sigma=3.0, xi=0.1)
        u = np.quantile(y, 0.85)
        ll_true = poisson_loglik(y, 35.0, 3.0, 0.1, u)
        ll_wrong = poisson_loglik(y, 40.0, 5.0, 0.5, u)
        assert ll_true > ll_wrong


# ── Simulation tests ───────────────────────────────────────────────

class TestSimulation:
    def test_gev_training_data_shapes(self):
        data = simulate_gev_training_data(n_sim=50, n_obs=30, seed=0)
        assert data["X"].shape[0] == 50
        assert data["y"].shape == (50,)
        assert data["tau"].shape == (50,)
        assert data["params"].shape == (50, 3)

    def test_gev_training_data_ranges(self):
        data = simulate_gev_training_data(n_sim=100, seed=1)
        assert np.all(data["tau"] > 0) and np.all(data["tau"] < 1)
        assert np.all(np.isfinite(data["y"]))
        assert np.all(np.isfinite(data["X"]))

    def test_gpd_training_data_shapes(self):
        data = simulate_gpd_training_data(n_sim=50, n_exc=20, seed=0)
        assert data["X"].shape[0] == 50
        assert data["y"].shape == (50,)
        assert data["params"].shape == (50, 2)

    def test_nonstationary_training_data_shapes(self):
        data = simulate_nonstationary_gev_training_data(
            n_sim=50, n_obs=30, seed=0
        )
        assert data["X"].shape[0] == 50
        assert data["y"].shape == (50,)
        assert data["params"].shape == (50, 4)

    def test_nonstationary_has_covariate_features(self):
        data = simulate_nonstationary_gev_training_data(
            n_sim=10, n_obs=20, seed=0
        )
        # X should have block maxima stats + covariate stats + T_star
        assert data["X"].shape[1] == 11 + 4 + 1  # 16 features
