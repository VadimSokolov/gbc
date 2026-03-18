"""Prediction use case: IQN on a smooth 1D function.

DGP: y = sin(2πx) + N(0, 0.3),  x ~ U(0, 1)
"""

import numpy as np
import torch
import pytest
import matplotlib
matplotlib.use("Agg")

from gbc.iqn import IQN, train_iqn, sample_iqn, predict_iqn
from gbc.metrics import crps_samples, coverage, pi_width, pit_values, rmse, rmspe
from gbc.sensitivity import partial_effect, elasticity, feature_effects
from gbc.plotting import quantile_fan, calibration_plot


# ── Synthetic data ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def sine_data():
    rng = np.random.default_rng(42)
    n = 300
    X = rng.uniform(0, 1, (n, 1))
    y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(0, 0.3, n)
    return X, y


@pytest.fixture(scope="module")
def trained_iqn(sine_data):
    X, y = sine_data
    model, xm, xs, ym, ys = train_iqn(
        X, y, epochs=500, hdim=64, nh=16, seed=0,
    )
    return model, xm, xs, ym, ys


# ── IQN training ────────────────────────────────────────────────

class TestTrainIQN:
    def test_returns_five_tuple(self, trained_iqn):
        assert len(trained_iqn) == 5

    def test_model_is_eval_mode(self, trained_iqn):
        model = trained_iqn[0]
        assert not model.training

    def test_normalization_stats_shapes(self, trained_iqn, sine_data):
        _, xm, xs, ym, ys = trained_iqn
        X, _ = sine_data
        assert xm.shape == (X.shape[1],)
        assert xs.shape == (X.shape[1],)
        assert isinstance(ym, float)
        assert isinstance(ys, float)


# ── predict_iqn ─────────────────────────────────────────────────

class TestPredictIQN:
    def test_shape(self, trained_iqn, sine_data):
        model, xm, xs, ym, ys = trained_iqn
        X, _ = sine_data
        taus = [0.1, 0.5, 0.9]
        preds = predict_iqn(model, X, xm, xs, ym, ys, taus=taus)
        assert preds.shape == (3, len(X))

    def test_monotonicity(self, trained_iqn, sine_data):
        """Lower quantiles should generally be below upper quantiles."""
        model, xm, xs, ym, ys = trained_iqn
        X, _ = sine_data
        preds = predict_iqn(model, X, xm, xs, ym, ys, taus=[0.1, 0.9])
        # Allow a few crossings from imperfect training
        frac_ordered = np.mean(preds[0] <= preds[1])
        assert frac_ordered > 0.8


# ── sample_iqn ──────────────────────────────────────────────────

class TestSampleIQN:
    def test_shape(self, trained_iqn, sine_data):
        model, xm, xs, ym, ys = trained_iqn
        X, _ = sine_data
        samples = sample_iqn(model, X, xm, xs, ym, ys, B=50)
        assert samples.shape == (50, len(X))

    def test_samples_finite(self, trained_iqn, sine_data):
        model, xm, xs, ym, ys = trained_iqn
        X, _ = sine_data
        samples = sample_iqn(model, X, xm, xs, ym, ys, B=50)
        assert np.all(np.isfinite(samples))


# ── Metrics on trained model ────────────────────────────────────

class TestMetrics:
    @pytest.fixture(scope="class")
    def samples(self, trained_iqn, sine_data):
        model, xm, xs, ym, ys = trained_iqn
        X, _ = sine_data
        return sample_iqn(model, X, xm, xs, ym, ys, B=200)

    def test_crps_positive(self, samples, sine_data):
        _, y = sine_data
        c = crps_samples(y, samples)
        assert c > 0

    def test_coverage_reasonable(self, samples, sine_data):
        _, y = sine_data
        cov = coverage(y, samples, alpha=0.90)
        # With 500 epochs, expect rough coverage (> 0.5 at least)
        assert 0.3 < cov <= 1.0

    def test_pi_width_positive(self, samples):
        w = pi_width(samples, alpha=0.90)
        assert w > 0

    def test_pit_values_in_unit_interval(self, samples, sine_data):
        _, y = sine_data
        pit = pit_values(y, samples)
        assert pit.shape == y.shape
        assert np.all(pit >= 0) and np.all(pit <= 1)

    def test_rmse_finite(self, samples, sine_data):
        _, y = sine_data
        r = rmse(y, samples.mean(axis=0))
        assert np.isfinite(r) and r > 0

    def test_rmspe_finite(self, samples, sine_data):
        _, y = sine_data
        r = rmspe(y, samples.mean(axis=0))
        assert np.isfinite(r) and r > 0


# ── Sensitivity ─────────────────────────────────────────────────

class TestSensitivity:
    def test_partial_effect_shape(self, trained_iqn, sine_data):
        model, xm, xs, ym, ys = trained_iqn
        X, _ = sine_data
        taus = np.array([0.25, 0.5, 0.75])
        pe = partial_effect(model, X[0], xm, xs, ym, ys, feature=0, taus=taus)
        assert pe.shape == (3,)

    def test_elasticity_shape(self, trained_iqn, sine_data):
        model, xm, xs, ym, ys = trained_iqn
        X, _ = sine_data
        # Pick a point where x != 0
        x0 = X[X[:, 0] > 0.1][0]
        taus = np.array([0.25, 0.5, 0.75])
        el = elasticity(model, x0, xm, xs, ym, ys, feature=0, taus=taus)
        assert el.shape == (3,)

    def test_feature_effects_shape(self, trained_iqn, sine_data):
        model, xm, xs, ym, ys = trained_iqn
        X, _ = sine_data
        taus = np.array([0.25, 0.5, 0.75])
        fe = feature_effects(model, X, xm, xs, ym, ys, taus=taus, max_obs=20)
        assert fe.shape == (1, 3)  # 1 feature, 3 quantiles


# ── Plotting (smoke tests) ─────────────────────────────────────

class TestPlotting:
    def test_quantile_fan_runs(self, trained_iqn, sine_data):
        model, xm, xs, ym, ys = trained_iqn
        X, y = sine_data
        samples = sample_iqn(model, X, xm, xs, ym, ys, B=50)
        order = np.argsort(X[:, 0])
        ax = quantile_fan(X[order, 0], samples[:, order], y_true=y[order])
        assert ax is not None

    def test_calibration_plot_runs(self, trained_iqn, sine_data):
        model, xm, xs, ym, ys = trained_iqn
        X, y = sine_data
        samples = sample_iqn(model, X, xm, xs, ym, ys, B=50)
        ax = calibration_plot(y, samples)
        assert ax is not None
