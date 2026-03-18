"""Unit tests for loss functions, cosine embedding, utilities, and data loaders."""

import numpy as np
import torch
import pytest

from gbc.loss import pinball_loss, composite_loss, gaussian_nll
from gbc.iqn import cosine_embed, IQN
from gbc.utils import set_seed, get_device, cosine_schedule
from gbc.data import load_motorcycle, friedman1, jump_fn, make_bimodal, lhs_1d


# ── cosine_embed ────────────────────────────────────────────────

class TestCosineEmbed:
    def test_shape(self):
        e = cosine_embed(0.5, 32)
        assert e.shape == (32,)

    def test_boundary_values(self):
        e0 = cosine_embed(0.0, 4)
        # cos(i * pi * 0) = cos(0) = 1 for all i
        assert torch.allclose(e0, torch.ones(4))

    def test_device_propagation(self):
        e = cosine_embed(0.5, 16, device="cpu", dtype=torch.float64)
        assert e.device == torch.device("cpu")
        assert e.dtype == torch.float64

    def test_symmetry(self):
        # cos(i*pi*0.5) should be 0, -1, 0, 1, 0, -1, ...
        e = cosine_embed(0.5, 4)
        expected = torch.cos(torch.arange(1, 5).float() * torch.pi * 0.5)
        assert torch.allclose(e, expected)


# ── pinball_loss ────────────────────────────────────────────────

class TestPinballLoss:
    def test_zero_residual(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        loss = pinball_loss(y, y, 0.5)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_median_symmetry(self):
        y = torch.tensor([0.0])
        # tau=0.5: pinball should be symmetric around 0
        loss_pos = pinball_loss(y, torch.tensor([-1.0]), 0.5)
        loss_neg = pinball_loss(y, torch.tensor([1.0]), 0.5)
        assert loss_pos.item() == pytest.approx(loss_neg.item(), abs=1e-7)

    def test_asymmetry(self):
        y = torch.tensor([0.0])
        y_hat = torch.tensor([-1.0])  # underprediction
        # tau=0.9: heavy penalty for underprediction
        loss_hi = pinball_loss(y, y_hat, 0.9)
        loss_lo = pinball_loss(y, y_hat, 0.1)
        assert loss_hi.item() > loss_lo.item()


# ── composite_loss ──────────────────────────────────────────────

class TestCompositeLoss:
    def test_shape_and_scalar(self):
        y = torch.randn(10)
        f = torch.randn(10, 2)
        loss = composite_loss(y, f, 0.5)
        assert loss.dim() == 0  # scalar

    def test_perfect_prediction(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        f = torch.stack([y, y], dim=1)  # both cols = y
        loss = composite_loss(y, f, 0.5)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ── gaussian_nll ────────────────────────────────────────────────

class TestGaussianNLL:
    def test_perfect_mean_unit_var(self):
        mu = torch.tensor([[1.0], [2.0]])
        logvar = torch.tensor([[0.0], [0.0]])  # var = 1
        y = mu.clone()
        loss = gaussian_nll(mu, logvar, y)
        # Only log(1) = 0 term remains
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_higher_var_lower_loss_for_outlier(self):
        mu = torch.tensor([[0.0]])
        y = torch.tensor([[5.0]])
        loss_small_var = gaussian_nll(mu, torch.tensor([[0.0]]), y)
        loss_large_var = gaussian_nll(mu, torch.tensor([[4.0]]), y)
        assert loss_large_var.item() < loss_small_var.item()


# ── IQN save/load ───────────────────────────────────────────────

class TestIQNSaveLoad:
    def test_round_trip(self, tmp_path):
        torch.manual_seed(0)
        model = IQN(xdim=3, hdim=32, nh=8)
        x = torch.randn(5, 3)
        with torch.no_grad():
            out_before = model(x, 0.5)

        path = str(tmp_path / "model.pt")
        model.save(path)
        loaded = IQN.load(path, xdim=3, hdim=32, nh=8)
        with torch.no_grad():
            out_after = loaded(x, 0.5)

        assert torch.allclose(out_before, out_after)


# ── utils ───────────────────────────────────────────────────────

class TestUtils:
    def test_set_seed_reproducibility(self):
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_get_device(self):
        d = get_device()
        assert isinstance(d, torch.device)

    def test_cosine_schedule(self):
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        sched = cosine_schedule(opt, T_max=100)
        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(100):
            sched.step()
        final_lr = opt.param_groups[0]["lr"]
        assert final_lr < initial_lr
        assert final_lr == pytest.approx(0.01 * 0.01, abs=1e-6)


# ── data loaders ────────────────────────────────────────────────

class TestDataLoaders:
    def test_load_motorcycle(self):
        X, y = load_motorcycle()
        assert X.shape == (133, 1)
        assert y.shape == (133,)

    def test_friedman1(self):
        X = np.random.rand(100, 10)
        y = friedman1(X)
        assert y.shape == (100,)
        # last 5 features should be inert
        X2 = X.copy()
        X2[:, 5:] = 0.0
        y2 = friedman1(X2)
        assert np.allclose(y, y2)

    def test_jump_fn(self):
        x = np.array([0.0, 55.0, 80.0])
        y = jump_fn(x)
        # x=55 is in [50,70], so jump is +10
        assert y[1] - np.sin(55.0) == pytest.approx(10.0, abs=1e-10)
        # x=80 is outside, no jump
        assert y[2] == pytest.approx(np.sin(80.0), abs=1e-10)

    def test_make_bimodal(self):
        S, theta, y_obs = make_bimodal(n=1000, n_obs=10, theta_true=2.0)
        assert S.shape == (1000, 1)
        assert theta.shape == (1000,)
        assert y_obs.shape == (10,)

    def test_lhs_1d(self):
        samples = lhs_1d(50, lo=0.0, hi=1.0)
        assert samples.shape == (50,)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0
