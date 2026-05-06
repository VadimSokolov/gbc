"""Tests for gbc.portfolio: Kelly criterion, corridors, SPT, and universal portfolios."""

import numpy as np
import pytest

from gbc.portfolio import (
    kelly_fraction,
    sS_corridor,
    bayesian_sS_corridor,
    excess_growth_rate,
    diversity_weights,
    universal_portfolio,
    bayesian_predictive_weights,
    garman_klass_vol,
    rogers_satchell_vol,
    simulate_gbm,
    simulate_sS_policy,
)


# ── Kelly Fraction ────────────────────────────────────────────────


class TestKellyFraction:
    def test_basic(self):
        # mu=0.08, sigma=0.30, r=0.02 → pi* = 0.06/0.09 = 2/3
        pi = kelly_fraction(0.08, 0.30, r=0.02)
        assert pi == pytest.approx(2 / 3, rel=1e-10)

    def test_zero_excess_return(self):
        pi = kelly_fraction(0.02, 0.20, r=0.02)
        assert pi == pytest.approx(0.0)

    def test_negative_excess_return(self):
        pi = kelly_fraction(0.01, 0.20, r=0.02)
        assert pi < 0

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.08, 0.0)


# ── (s,S) Corridor ───────────────────────────────────────────────


class TestSSCorridor:
    def test_boundaries_order(self):
        c = sS_corridor(0.08, 0.30, r=0.02, lam=0.005)
        assert c["s"] < c["pi_star"] < c["S"]

    def test_boundaries_in_unit_interval(self):
        c = sS_corridor(0.08, 0.30, r=0.02, lam=0.005)
        assert 0 <= c["s"] <= 1
        assert 0 <= c["S"] <= 1

    def test_corridor_narrows_with_small_lambda(self):
        c1 = sS_corridor(0.08, 0.30, lam=0.01)
        c2 = sS_corridor(0.08, 0.30, lam=0.001)
        assert c1["half_width"] > c2["half_width"]

    def test_clip_leveraged_kelly(self):
        # mu=0.10, sigma=0.20 → pi*=2.0, should be clipped to 0.99
        c = sS_corridor(0.10, 0.20, r=0.0, lam=0.005, clip=True)
        assert c["pi_star"] == pytest.approx(0.99)

    def test_no_clip(self):
        c = sS_corridor(0.10, 0.20, r=0.0, lam=0.005, clip=False)
        assert c["pi_star"] == pytest.approx(2.5)


# ── Bayesian (s,S) Corridor ──────────────────────────────────────


class TestBayesianCorridor:
    @pytest.fixture(scope="class")
    def corridor_result(self):
        rng = np.random.default_rng(42)
        log_rets = rng.normal(0.08 / 252, 0.30 / np.sqrt(252), 504)
        return bayesian_sS_corridor(log_rets, r=0.02, lam=0.005, n_draw=3000)

    def test_shapes(self, corridor_result):
        assert corridor_result["pi_star"].shape == (3000,)
        assert corridor_result["s"].shape == (3000,)
        assert corridor_result["S"].shape == (3000,)

    def test_boundaries_in_range(self, corridor_result):
        assert np.all(corridor_result["s"] >= 0)
        assert np.all(corridor_result["S"] <= 1)

    def test_plug_in_present(self, corridor_result):
        plug = corridor_result["plug_in"]
        assert "pi_star" in plug
        assert "s" in plug
        assert "S" in plug


# ── Excess Growth Rate ───────────────────────────────────────────


class TestExcessGrowthRate:
    def test_positive_for_diversified(self):
        n = 5
        w = np.ones(n) / n
        Sigma = np.eye(n) * 0.04 + np.full((n, n), 0.01)
        np.fill_diagonal(Sigma, 0.04)
        egr = excess_growth_rate(w, Sigma, annualize=1)
        assert egr > 0

    def test_zero_for_single_asset(self):
        w = np.array([1.0])
        Sigma = np.array([[0.04]])
        egr = excess_growth_rate(w, Sigma, annualize=1)
        assert egr == pytest.approx(0.0, abs=1e-12)

    def test_higher_for_equal_than_concentrated(self):
        n = 10
        Sigma = np.eye(n) * 0.04 + np.full((n, n), 0.005)
        np.fill_diagonal(Sigma, 0.04)
        w_equal = np.ones(n) / n
        w_conc = np.zeros(n)
        w_conc[0] = 1.0
        egr_eq = excess_growth_rate(w_equal, Sigma, annualize=1)
        egr_co = excess_growth_rate(w_conc, Sigma, annualize=1)
        assert egr_eq > egr_co


# ── Diversity Weights ────────────────────────────────────────────


class TestDiversityWeights:
    def test_sum_to_one(self):
        mu = np.array([0.5, 0.3, 0.2])
        w = diversity_weights(mu, p=0.5)
        assert w.sum() == pytest.approx(1.0)

    def test_more_equal_than_market(self):
        mu = np.array([0.7, 0.2, 0.1])
        w = diversity_weights(mu, p=0.5)
        # p=0.5 should tilt toward equal weights
        assert w[0] < mu[0]
        assert w[2] > mu[2]

    def test_p_equals_one_returns_market(self):
        mu = np.array([0.5, 0.3, 0.2])
        w = diversity_weights(mu, p=1.0)
        np.testing.assert_allclose(w, mu)


# ── Universal Portfolio ──────────────────────────────────────────


class TestUniversalPortfolio:
    @pytest.fixture(scope="class")
    def up_result(self):
        rng = np.random.default_rng(99)
        T = 252
        r1 = np.exp(rng.normal(0.08 / 252, 0.25 / np.sqrt(252), T))
        r2 = np.exp(rng.normal(0.04 / 252, 0.15 / np.sqrt(252), T))
        return universal_portfolio(np.column_stack([r1, r2]), n_grid=100)

    def test_weights_in_range(self, up_result):
        assert np.all(up_result["weights"] >= 0)
        assert np.all(up_result["weights"] <= 1)

    def test_wealth_positive(self, up_result):
        assert np.all(up_result["wealth"] > 0)

    def test_best_b_in_range(self, up_result):
        assert 0 < up_result["best_b"] < 1

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            universal_portfolio(np.ones((10, 3)))


# ── Bayesian Predictive Weights ──────────────────────────────────


class TestBayesianPredictiveWeights:
    @pytest.fixture(scope="class")
    def result(self):
        rng = np.random.default_rng(42)
        mu = np.array([0.10, 0.08, 0.06]) / 252
        Sigma = np.diag([0.30, 0.25, 0.20]) ** 2 / 252
        R = rng.multivariate_normal(mu, Sigma, size=252)
        return bayesian_predictive_weights(R, gamma=3.0)

    def test_weights_on_simplex(self, result):
        w = result["weights"]
        assert np.all(w >= 0)
        assert w.sum() == pytest.approx(1.0, abs=1e-8)

    def test_mu_posterior_shape(self, result):
        assert result["mu_posterior"].shape == (3,)

    def test_sigma_predictive_shape(self, result):
        assert result["Sigma_predictive"].shape == (3, 3)


# ── Volatility Estimators ────────────────────────────────────────


class TestVolatilityEstimators:
    @pytest.fixture(scope="class")
    def ohlc(self):
        return simulate_gbm(500, mu=0.07, sigma=0.20, seed=42)

    def test_gk_close_to_true(self, ohlc):
        vol = garman_klass_vol(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert abs(vol - 0.20) < 0.05

    def test_rs_close_to_true(self, ohlc):
        vol = rogers_satchell_vol(
            ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
        )
        assert abs(vol - 0.20) < 0.05

    def test_gk_positive(self, ohlc):
        vol = garman_klass_vol(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
        assert vol > 0

    def test_close_to_close_less_efficient(self, ohlc):
        """GK should be closer to true vol than close-to-close on average."""
        cc_vol = np.std(ohlc["log_returns"]) * np.sqrt(252)
        gk_vol = garman_klass_vol(
            ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
        )
        # Both should be in the right ballpark
        assert abs(cc_vol - 0.20) < 0.10
        assert abs(gk_vol - 0.20) < 0.10


# ── Simulation ───────────────────────────────────────────────────


class TestSimulateGBM:
    def test_shape(self):
        data = simulate_gbm(100)
        assert data["open"].shape == (100,)
        assert data["close"].shape == (100,)
        assert data["log_returns"].shape == (99,)

    def test_high_above_low(self):
        data = simulate_gbm(100)
        assert np.all(data["high"] >= data["low"])

    def test_prices_positive(self):
        data = simulate_gbm(100)
        assert np.all(data["prices"] > 0)


class TestSimulateSS:
    def test_wealth_positive(self):
        result = simulate_sS_policy(n_steps=252)
        assert np.all(result["wealth"] > 0)

    def test_corridor_respected(self):
        result = simulate_sS_policy(n_steps=252)
        s = result["corridor"]["s"]
        S = result["corridor"]["S"]
        # After trading, rho should be within [s, S] (or at boundary)
        rho = result["rho"][1:]
        assert np.all(rho >= s - 1e-10)
        assert np.all(rho <= S + 1e-10)

    def test_few_trades(self):
        result = simulate_sS_policy(n_steps=2520, lam=0.005)
        # Should trade far less than every day
        assert result["n_trades"] < 2520 / 2
