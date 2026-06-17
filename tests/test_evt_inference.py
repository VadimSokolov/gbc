"""Tests for the EVT inference layer (classical baselines + GBC-QNN estimators)."""

import numpy as np

from gbc.evt import gev_quantile, gev_return_level
from gbc.evt_inference import (fit_stationary_gev, fit_ns_gev, ns_params_at, hill,
                               gev_mcmc, return_level_ci_delta,
                               train_functional_iqn, gbc_return_level_posterior,
                               gbc_crps_coverage_loyo)


def _gev_sample(n, mu, sigma, xi, seed):
    rng = np.random.default_rng(seed)
    return gev_quantile(rng.uniform(size=n), mu, sigma, xi)


def test_fit_stationary_gev_recovers_params():
    x = _gev_sample(4000, 35.0, 3.0, 0.1, 0)
    f = fit_stationary_gev(x)
    assert abs(f["mu"] - 35.0) < 0.5
    assert abs(f["sigma"] - 3.0) < 0.5
    assert abs(f["xi"] - 0.1) < 0.1


def test_hill_recovers_tail_index():
    rng = np.random.default_rng(1)
    alpha = 2.5
    x = (1 - rng.uniform(size=20000)) ** (-1.0 / alpha)  # Pareto(alpha)
    assert abs(hill(x, 500) - 1.0 / alpha) < 0.07


def test_ns_gev_recovers_trend():
    rng = np.random.default_rng(2)
    T = np.linspace(-1, 1, 2000)
    mu = 35.0 + 2.0 * T
    x = np.array([gev_quantile(rng.uniform(), mu[i], 3.0, 0.05) for i in range(len(T))])
    f = fit_ns_gev(x, T)
    assert abs(f["mu1"] - 2.0) < 0.6
    m, s, xi = ns_params_at(f, 1.0)
    assert abs(m - (f["mu0"] + f["mu1"])) < 1e-8


def test_return_level_ci_delta_brackets_point():
    x = _gev_sample(500, 35.0, 3.0, 0.1, 3)
    r = return_level_ci_delta(x, N=100)
    assert r["lo"] < r["zN"] < r["hi"]
    assert r["zN"] > np.quantile(x, 0.9)  # 100-yr level exceeds the sample 90th pct


def test_gev_mcmc_posterior_near_truth():
    x = _gev_sample(800, 35.0, 3.0, 0.1, 4)
    s = gev_mcmc(x, n_iter=8000, burn=3000, thin=5, seed=0)
    assert s.shape[1] == 3 and len(s) > 500
    assert abs(s[:, 0].mean() - 35.0) < 1.0      # mu
    assert abs(s[:, 2].mean() - 0.1) < 0.15      # xi


def test_gbc_return_level_posterior_runs():
    x = _gev_sample(60, 35.0, 3.0, 0.1, 5)
    trained = train_functional_iqn(x, N=100, n_sim=3000, epochs=300, seed=1)
    r = gbc_return_level_posterior(trained, x)
    assert r["lo"] < r["zN"] < r["hi"]
    assert 35.0 < r["zN"] < 70.0


def test_gbc_crps_coverage_loyo_runs():
    x = _gev_sample(40, 35.0, 3.0, 0.1, 6)
    out = gbc_crps_coverage_loyo(x, n_sim=2500, epochs=250, seed=0, B=400)
    assert out["crps"] > 0
    assert 0.0 <= out["coverage"] <= 1.0
