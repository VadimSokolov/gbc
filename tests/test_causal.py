"""Causal inference use case: treatment effect estimation + welfare.

DGP: Y = X + 2·Z + N(0, 0.5)
     X ~ N(0, 1)
     Z ~ Bernoulli(sigmoid(X))   (confounded treatment)
     True ATE = 2, True CATE = 2 (constant)
"""

import numpy as np
import torch
import pytest

from gbc.causal import CausalIQN, CausalIQNv2, CausalEnsemble
from gbc.welfare import (
    meu, welfare_change, yaari_weighted, individual_welfare,
    distortion_identity, distortion_cvar, distortion_power, distortion_wang,
)


@pytest.fixture(scope="module")
def causal_data():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, (n, 1))
    prob = 1 / (1 + np.exp(-X[:, 0]))
    Z = rng.binomial(1, prob).astype(float)
    Y = X[:, 0] + 2.0 * Z + rng.normal(0, 0.5, n)
    return X, Y, Z


@pytest.fixture(scope="module")
def causal_tensors(causal_data):
    X, Y, Z = causal_data
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        torch.tensor(Z, dtype=torch.float32),
    )


# ── CausalIQN ──────────────────────────────────────────────────

class TestCausalIQN:
    def test_forward_shapes(self, causal_tensors):
        x, _, z = causal_tensors
        model = CausalIQN(xdim=1)
        f, pi, te, mu = model(x, z, 0.5)
        assert f.shape == (len(x), 2)
        assert pi.shape == (len(x), 1)
        assert te.shape == (len(x), 2)
        assert mu.shape == (len(x), 1)

    def test_loss_fn_scalar(self, causal_tensors):
        x, y, z = causal_tensors
        model = CausalIQN(xdim=1)
        loss = model.loss_fn(x, y, z)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_fit_reduces_loss(self, causal_tensors):
        x, y, z = causal_tensors
        torch.manual_seed(0)
        model = CausalIQN(xdim=1)
        loss_before = model.loss_fn(x, y, z).item()
        model.fit(x, y, z, epochs=200, lr=1e-3)
        loss_after = model.loss_fn(x, y, z).item()
        assert loss_after < loss_before

    def test_estimate_cate_shapes(self, causal_tensors):
        x, y, z = causal_tensors
        torch.manual_seed(0)
        model = CausalIQN(xdim=1)
        model.fit(x, y, z, epochs=200)
        cate_point, cate_samples = model.estimate_cate(x, z, n_mc=50)
        assert cate_point.shape == (len(x),)
        assert cate_samples.shape == (len(x), 50)


# ── CausalIQNv2 ────────────────────────────────────────────────

class TestCausalIQNv2:
    def test_forward_shapes(self, causal_tensors):
        x, _, z = causal_tensors
        model = CausalIQNv2(xdim=1, hdim=32, nh=8)
        f, pi, te, mu = model(x, z, 0.5)
        assert f.shape == (len(x), 2)
        assert pi.shape == (len(x), 1)
        assert te.shape == (len(x), 2)
        assert mu.shape == (len(x), 1)

    def test_loss_fn_scalar(self, causal_tensors):
        x, y, z = causal_tensors
        model = CausalIQNv2(xdim=1, hdim=32, nh=8)
        loss = model.loss_fn(x, y, z)
        assert loss.dim() == 0


# ── CausalEnsemble ──────────────────────────────────────────────

class TestCausalEnsemble:
    @pytest.fixture(scope="class")
    def fitted_ensemble(self, causal_data):
        X, Y, Z = causal_data
        ens = CausalEnsemble(
            model_cls=CausalIQNv2,
            model_kwargs={"xdim": 1, "hdim": 32, "nh": 8},
            n_models=2,
            device="cpu",
        )
        ens.fit(X, Y, Z, epochs=300, verbose=False)
        return ens

    def test_estimate_cate_keys(self, fitted_ensemble, causal_data):
        X, _, _ = causal_data
        result = fitted_ensemble.estimate_cate(X, n_mc=50)
        assert "cate" in result
        assert "ate" in result
        assert "ate_se" in result
        assert "ate_ci" in result
        assert result["cate"].shape == (len(X),)

    def test_estimate_qte_shape(self, fitted_ensemble, causal_data):
        X, _, _ = causal_data
        qs = np.linspace(0.1, 0.9, 9)
        qte = fitted_ensemble.estimate_qte(X, quantiles=qs)
        assert qte.shape == (len(X), 9)

    def test_estimate_qte_separate_shape(self, fitted_ensemble, causal_data):
        X, Y, Z = causal_data
        qs = np.linspace(0.1, 0.9, 5)
        qte = fitted_ensemble.estimate_qte_separate(
            X, Y, Z, quantiles=qs, epochs=100, hdim=32, nh=8,
        )
        assert qte.shape == (len(X), 5)

    def test_predict_propensity_range(self, fitted_ensemble, causal_data):
        X, _, _ = causal_data
        pi = fitted_ensemble.predict_propensity(X)
        assert pi.shape == (len(X),)
        assert np.all(pi >= 0) and np.all(pi <= 1)


# ── Welfare ─────────────────────────────────────────────────────

class TestWelfare:
    @pytest.fixture(scope="class")
    def quantile_functions(self):
        taus = np.linspace(0.05, 0.95, 19)
        # Treated: N(5, 1) quantile function
        from scipy.stats import norm
        qf_treated = norm.ppf(taus, loc=5, scale=1)
        qf_control = norm.ppf(taus, loc=3, scale=1)
        return qf_treated, qf_control, taus

    def test_meu_approximates_mean(self, quantile_functions):
        qf_treated, _, taus = quantile_functions
        # E[X] for N(5,1) should be ≈ 5
        e = meu(qf_treated, taus)
        assert abs(e - 5.0) < 0.6

    def test_welfare_change_positive(self, quantile_functions):
        qf_treated, qf_control, taus = quantile_functions
        result = welfare_change(qf_treated, qf_control, taus)
        assert result["welfare_change"] > 0
        assert result["eu_treated"] > result["eu_control"]
        assert result["qte"].shape == taus.shape

    def test_yaari_identity_equals_meu(self, quantile_functions):
        qf_treated, _, taus = quantile_functions
        e_meu = meu(qf_treated, taus)
        e_yaari = yaari_weighted(qf_treated, taus, distortion_identity)
        assert abs(e_meu - e_yaari) < 0.5

    def test_cvar_less_than_mean(self, quantile_functions):
        qf_treated, _, taus = quantile_functions
        e_mean = meu(qf_treated, taus)
        e_cvar = yaari_weighted(qf_treated, taus, distortion_cvar(0.1))
        # CVaR focuses on lower tail → should be less than mean
        assert e_cvar < e_mean

    def test_individual_welfare_shape(self, quantile_functions):
        _, _, taus = quantile_functions
        qte_matrix = np.random.randn(10, len(taus))
        iw = individual_welfare(qte_matrix, taus)
        assert iw.shape == (10,)

    def test_distortion_power(self):
        h = distortion_power(2.0)
        assert h(0.0) == 0.0
        assert h(1.0) == 1.0
        assert h(0.5) == pytest.approx(0.25)

    def test_distortion_wang(self):
        h = distortion_wang(0.0)
        # lambda=0 → identity: h(z) = Φ(Φ⁻¹(z)) = z
        assert h(0.5) == pytest.approx(0.5, abs=1e-5)
