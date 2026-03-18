"""Ensemble + conformal use case: heteroskedastic prediction.

DGP: y = x + (0.1 + |x|)·ε,  x ~ U(-2, 2),  ε ~ N(0, 1)
Variance grows with |x| — HetMLP should capture this.
"""

import numpy as np
import torch
import pytest

from gbc.ensemble import HetMLP, train_het_mlp, ensemble_predict
from gbc.conformal import temporal_cv_quantiles, conformal_pi
from gbc.metrics import crps_gaussian


@pytest.fixture(scope="module")
def het_data():
    rng = np.random.default_rng(42)
    n = 500
    x = rng.uniform(-2, 2, n)
    sigma = 0.1 + np.abs(x)
    y = x + sigma * rng.normal(0, 1, n)
    X = x.reshape(-1, 1).astype(np.float32)
    y = y.astype(np.float32)
    # Normalize
    xm, xs = X.mean(0), X.std(0) + 1e-8
    X_n = (X - xm) / xs
    return X_n, y, xm, xs


@pytest.fixture(scope="module")
def het_tensors(het_data):
    X_n, y, _, _ = het_data
    return (
        torch.tensor(X_n, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )


# ── HetMLP ──────────────────────────────────────────────────────

class TestHetMLP:
    def test_forward_shapes(self):
        model = HetMLP(xdim=3, hdim=32, nlayers=2)
        x = torch.randn(10, 3)
        mu, logvar = model(x)
        assert mu.shape == (10, 1)
        assert logvar.shape == (10, 1)


# ── train_het_mlp ───────────────────────────────────────────────

class TestTrainHetMLP:
    def test_returns_eval_mode(self, het_tensors):
        X_t, y_t = het_tensors
        model = train_het_mlp(X_t, y_t, epochs=100, hdim=32, nlayers=2, seed=0)
        assert not model.training

    def test_predictions_finite(self, het_tensors):
        X_t, y_t = het_tensors
        model = train_het_mlp(X_t, y_t, epochs=100, hdim=32, nlayers=2, seed=0)
        with torch.no_grad():
            mu, logvar = model(X_t)
        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(logvar))


# ── ensemble_predict ────────────────────────────────────────────

class TestEnsemblePredict:
    @pytest.fixture(scope="class")
    def ensemble(self, het_tensors):
        X_t, y_t = het_tensors
        models = []
        for k in range(3):
            m = train_het_mlp(X_t, y_t, epochs=200, hdim=32, nlayers=2, seed=k)
            models.append(m)
        return models

    def test_shapes(self, ensemble, het_tensors):
        X_t, _ = het_tensors
        ens_mean, ens_std = ensemble_predict(ensemble, X_t)
        assert ens_mean.shape == (len(X_t),)
        assert ens_std.shape == (len(X_t),)

    def test_std_positive(self, ensemble, het_tensors):
        X_t, _ = het_tensors
        _, ens_std = ensemble_predict(ensemble, X_t)
        assert np.all(ens_std > 0)

    def test_crps_gaussian_finite(self, ensemble, het_tensors, het_data):
        X_t, _ = het_tensors
        _, y, _, _ = het_data
        ens_mean, ens_std = ensemble_predict(ensemble, X_t)
        c = crps_gaussian(y, ens_mean, ens_std)
        assert np.isfinite(c) and c > 0


# ── Conformal calibration ──────────────────────────────────────

class TestConformal:
    def test_temporal_cv_quantiles(self):
        rng = np.random.default_rng(0)
        residuals = np.abs(rng.normal(0, 1, 200))
        strata = np.repeat([1, 2, 3, 4], 50)
        qmap = temporal_cv_quantiles(residuals, strata, alpha=0.90)
        assert len(qmap) == 4
        for s in [1, 2, 3, 4]:
            assert s in qmap
            assert qmap[s] > 0

    def test_small_stratum_gets_global(self):
        rng = np.random.default_rng(0)
        residuals = np.abs(rng.normal(0, 1, 100))
        strata = np.concatenate([np.full(95, 1), np.full(5, 2)])
        qmap = temporal_cv_quantiles(residuals, strata, alpha=0.90, min_count=20)
        global_q = float(np.percentile(residuals, 90))
        # Stratum 2 has only 5 obs → should get global quantile
        assert qmap[2] == pytest.approx(global_q)

    def test_conformal_pi_shapes(self):
        y_hat = np.array([1.0, 2.0, 3.0])
        strata = np.array([1, 2, 1])
        qmap = {1: 0.5, 2: 1.0}
        lower, upper = conformal_pi(y_hat, strata, qmap)
        assert lower.shape == (3,)
        assert upper.shape == (3,)

    def test_conformal_pi_symmetric(self):
        y_hat = np.array([5.0])
        strata = np.array([1])
        qmap = {1: 2.0}
        lower, upper = conformal_pi(y_hat, strata, qmap)
        assert lower[0] == pytest.approx(3.0)
        assert upper[0] == pytest.approx(7.0)

    def test_conformal_pi_coverage_on_synthetic(self):
        """End-to-end: conformal intervals should achieve ~target coverage."""
        rng = np.random.default_rng(42)
        n = 1000
        y_true = rng.normal(0, 1, n)
        y_hat = y_true + rng.normal(0, 0.3, n)  # noisy predictions
        residuals = np.abs(y_true - y_hat)
        strata = np.ones(n)

        # Use first half as calibration, second half as test
        cal_res = residuals[:500]
        cal_strata = strata[:500]
        qmap = temporal_cv_quantiles(cal_res, cal_strata, alpha=0.90)

        lower, upper = conformal_pi(y_hat[500:], strata[500:], qmap)
        cov = np.mean((y_true[500:] >= lower) & (y_true[500:] <= upper))
        assert cov > 0.85  # should be close to 0.90
