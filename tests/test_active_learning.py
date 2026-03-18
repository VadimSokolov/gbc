"""Active learning use case: ensemble disagreement acquisition.

DGP: y = sin(3x) + N(0, 0.1),  x ~ U(0, 2π)
Sparse initial training set; candidate pool for acquisition.
"""

import numpy as np
import torch
import pytest

from gbc.active_learning import RandomPriorNet, ensemble_disagreement, select_next


@pytest.fixture(scope="module")
def al_data():
    rng = np.random.default_rng(42)
    # Sparse training set
    n_train = 20
    X_train = rng.uniform(0, 2 * np.pi, (n_train, 1))
    y_train = np.sin(3 * X_train[:, 0]) + rng.normal(0, 0.1, n_train)
    # Dense candidate pool
    n_cand = 200
    X_cand = rng.uniform(0, 2 * np.pi, (n_cand, 1))
    return X_train, y_train, X_cand


@pytest.fixture(scope="module")
def trained_ensemble(al_data):
    X_train, y_train, _ = al_data
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    K = 3
    models = []
    for k in range(K):
        torch.manual_seed(k)
        m = RandomPriorNet(xdim=1, hdim=32, nlayers=2, prior_scale=1.0)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        m.train()
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(m(X_t), y_t)
            loss.backward()
            opt.step()
        m.eval()
        models.append(m)
    return models


# ── RandomPriorNet ──────────────────────────────────────────────

class TestRandomPriorNet:
    def test_output_shape(self):
        m = RandomPriorNet(xdim=2, hdim=16, nlayers=2)
        x = torch.randn(10, 2)
        out = m(x)
        assert out.shape == (10, 1)

    def test_prior_frozen(self):
        m = RandomPriorNet(xdim=1, hdim=16, nlayers=2)
        for p in m.prior_net.parameters():
            assert not p.requires_grad

    def test_different_seeds_give_different_priors(self):
        torch.manual_seed(0)
        m1 = RandomPriorNet(xdim=1, hdim=16, nlayers=2)
        torch.manual_seed(1)
        m2 = RandomPriorNet(xdim=1, hdim=16, nlayers=2)
        x = torch.randn(5, 1)
        with torch.no_grad():
            out1 = m1.prior_net(x)
            out2 = m2.prior_net(x)
        assert not torch.allclose(out1, out2)


# ── Ensemble disagreement ──────────────────────────────────────

class TestEnsembleDisagreement:
    def test_shape(self, trained_ensemble, al_data):
        _, _, X_cand = al_data
        X_t = torch.tensor(X_cand, dtype=torch.float32)
        scores = ensemble_disagreement(trained_ensemble, X_t)
        assert scores.shape == (len(X_cand),)

    def test_non_negative(self, trained_ensemble, al_data):
        _, _, X_cand = al_data
        X_t = torch.tensor(X_cand, dtype=torch.float32)
        scores = ensemble_disagreement(trained_ensemble, X_t)
        assert np.all(scores >= 0)


# ── select_next ─────────────────────────────────────────────────

class TestSelectNext:
    def test_returns_correct_count(self, trained_ensemble, al_data):
        _, _, X_cand = al_data
        X_t = torch.tensor(X_cand, dtype=torch.float32)
        idx = select_next(trained_ensemble, X_t, batch_size=5)
        assert len(idx) == 5

    def test_indices_in_range(self, trained_ensemble, al_data):
        _, _, X_cand = al_data
        X_t = torch.tensor(X_cand, dtype=torch.float32)
        idx = select_next(trained_ensemble, X_t, batch_size=3)
        assert np.all(idx >= 0) and np.all(idx < len(X_cand))

    def test_selects_highest_disagreement(self, trained_ensemble, al_data):
        _, _, X_cand = al_data
        X_t = torch.tensor(X_cand, dtype=torch.float32)
        scores = ensemble_disagreement(trained_ensemble, X_t)
        idx = select_next(trained_ensemble, X_t, batch_size=1)
        # The selected index should be the one with max disagreement
        assert idx[0] == np.argmax(scores)
