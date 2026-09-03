"""Tests for gbc.survivor — survivor function estimation and tail functionals."""

import numpy as np
import pytest
from gbc.iqn import train_iqn, IQN
from gbc.survivor import (
    survivor_curve,
    posterior_mean,
    return_level,
    exceedance_prob,
    expected_shortfall,
    tail_weight_crps,
    meu_optimal_action,
)


@pytest.fixture(scope="module")
def trained_model():
    """Train a small IQN on synthetic data for testing."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.uniform(-1, 1, size=(n, 2))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.3, n)
    model, xm, xs, ym, ys = train_iqn(X, y, epochs=300, hdim=64, nh=16, seed=42)
    return model, xm, xs, ym, ys, X, y


def test_survivor_curve_shape(trained_model):
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:5]
    t_grid = np.linspace(-2, 3, 20)
    S = survivor_curve(model, X_te, xm, xs, ym, ys, t_grid, B=100)
    assert S.shape == (20, 5)


def test_survivor_curve_monotone(trained_model):
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:3]
    t_grid = np.linspace(-2, 3, 50)
    S = survivor_curve(model, X_te, xm, xs, ym, ys, t_grid, B=200)
    # S should be non-increasing in t for each test point
    for i in range(3):
        diffs = np.diff(S[:, i])
        assert np.all(diffs <= 0.05)  # allow small noise


def test_survivor_curve_bounds(trained_model):
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:3]
    t_grid = np.linspace(-5, 5, 30)
    S = survivor_curve(model, X_te, xm, xs, ym, ys, t_grid, B=100)
    assert np.all(S >= 0)
    assert np.all(S <= 1)


def test_posterior_mean_shape(trained_model):
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:5]
    pm = posterior_mean(model, X_te, xm, xs, ym, ys, B=100)
    assert pm.shape == (5,)
    assert np.all(np.isfinite(pm))


def test_return_level_shape(trained_model):
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:5]
    rl = return_level(model, X_te, xm, xs, ym, ys, N=100)
    assert rl.shape == (5,)
    assert np.all(np.isfinite(rl))


def test_return_level_ordering(trained_model):
    """Longer return periods should give larger return levels."""
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:3]
    rl50 = return_level(model, X_te, xm, xs, ym, ys, N=50)
    rl100 = return_level(model, X_te, xm, xs, ym, ys, N=100)
    # Not strictly monotone due to noise, but on average should hold
    assert np.mean(rl100) >= np.mean(rl50) - 0.5


def test_exceedance_prob_shape(trained_model):
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:5]
    ep = exceedance_prob(model, X_te, xm, xs, ym, ys, threshold=0.5, B=100)
    assert ep.shape == (5,)
    assert np.all(ep >= 0) and np.all(ep <= 1)


def test_exceedance_prob_monotone_in_threshold(trained_model):
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:3]
    ep_low = exceedance_prob(model, X_te, xm, xs, ym, ys, threshold=0.0, B=200)
    ep_high = exceedance_prob(model, X_te, xm, xs, ym, ys, threshold=1.0, B=200)
    assert np.all(ep_low >= ep_high - 0.05)


def test_expected_shortfall_shape(trained_model):
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:5]
    es = expected_shortfall(model, X_te, xm, xs, ym, ys, p=0.90, B=100)
    assert es.shape == (5,)
    assert np.all(np.isfinite(es))


def test_expected_shortfall_exceeds_quantile(trained_model):
    """ES should be >= the corresponding quantile."""
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:5]
    from gbc.iqn import predict_iqn
    q90 = predict_iqn(model, X_te, xm, xs, ym, ys, taus=[0.90])[0]
    es = expected_shortfall(model, X_te, xm, xs, ym, ys, p=0.90, B=300)
    assert np.all(es >= q90 - 0.3)  # allow small estimation noise


def test_tail_weight_crps():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 100)
    samples = rng.normal(0, 1, (200, 100))
    tw = tail_weight_crps(y, samples, quantile=0.9)
    assert isinstance(tw, float)
    assert tw >= 0


def test_meu_optimal_action(trained_model):
    model, xm, xs, ym, ys, X, _ = trained_model
    X_te = X[:3]
    actions = np.linspace(0, 2, 10)
    def utility(a, x):
        return -(np.maximum(x - a, 0)) ** 2
    opt_a, eu = meu_optimal_action(
        model, X_te, xm, xs, ym, ys, actions, utility, B=100
    )
    assert opt_a.shape == (3,)
    assert eu.shape == (3,)
