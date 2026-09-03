"""Regressions for three defects that passed every mechanical check.

Each of these shipped in v0.5.1 and was found only by asking what the code
computes rather than what it is named.  The tests assert the *property*, not the
presence of a token, which is exactly what the earlier checks failed to do.
"""
import math

import numpy as np
import pytest
import torch

from gbc.evt import _truncated_normal, simulate_gev_training_data
from gbc.loss import composite_loss
from gbc.shrinkage import horseshoe_posterior


class TestMonotonicityPenalty:
    """The penalty must compare two quantile levels, and must not move the target."""

    def _f(self, q, n=64):
        return torch.stack([torch.zeros(n), torch.full((n,), float(q))], dim=1)

    def test_zero_when_correctly_ordered(self):
        # Q(0.2) < Q(0.8): no crossing, so the penalty contributes nothing.
        y = torch.zeros(64)
        w = (0.0, 1.0, 0.0)                      # isolate the penalty
        loss = composite_loss(y, self._f(-1.0), 0.2, w,
                              f_other=self._f(1.0), tau_other=0.8)
        assert float(loss) == pytest.approx(0.0, abs=1e-12)

    def test_positive_when_crossing(self):
        # Q(0.2) > Q(0.8) by 2.0: a genuine crossing must be penalised.
        y = torch.zeros(64)
        w = (0.0, 1.0, 0.0)
        loss = composite_loss(y, self._f(1.0), 0.2, w,
                              f_other=self._f(-1.0), tau_other=0.8)
        assert float(loss) == pytest.approx(2.0, abs=1e-6)

    def test_symmetric_in_argument_order(self):
        y = torch.zeros(64)
        w = (0.0, 1.0, 0.0)
        a = composite_loss(y, self._f(1.0), 0.8, w,
                           f_other=self._f(-1.0), tau_other=0.2)
        b = composite_loss(y, self._f(-1.0), 0.2, w,
                           f_other=self._f(1.0), tau_other=0.8)
        assert float(a) == pytest.approx(float(b), abs=1e-12)

    @pytest.mark.parametrize("tau", [0.05, 0.25, 0.75, 0.95])
    def test_minimiser_is_the_tau_quantile(self, tau):
        """The whole point: the composite loss must still target tau.

        The predecessor penalised the sign of the residual, which is active at
        the correct quantile and pulled the minimiser outward (tau=0.95 landed
        near 0.963), so nominal 90% intervals were really 92.5%.
        """
        rng = np.random.default_rng(0)
        y = torch.tensor(rng.standard_normal(200000), dtype=torch.float64)
        q = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([q], lr=0.3, max_iter=200, tolerance_grad=1e-12)

        def closure():
            opt.zero_grad()
            f = torch.stack([torch.zeros_like(y), q.expand_as(y)], dim=1)
            # A non-crossing partner: the penalty is inactive, as it should be.
            other = torch.stack([torch.zeros_like(y),
                                 torch.full_like(y, 20.0)], dim=1)
            L = composite_loss(y, f, tau, (0.3, 0.3, 0.4),
                               f_other=other, tau_other=0.999)
            L.backward()
            return L

        opt.step(closure)
        implied = 0.5 * math.erfc(-q.detach().item() / math.sqrt(2.0))
        assert implied == pytest.approx(tau, abs=0.005)


class TestShapePriorIsTruncated:
    """xi is truncated, not clipped: clipping left a 4.8% atom at the bound."""

    def test_no_atom_at_bounds(self):
        rng = np.random.default_rng(1)
        lo, hi = -0.8, 0.5
        x = np.array([_truncated_normal(rng, 0.0, 0.3, lo, hi) for _ in range(20000)])
        assert x.min() > lo and x.max() < hi
        assert np.mean(np.isclose(x, hi, atol=1e-9)) == 0.0
        assert np.mean(np.isclose(x, lo, atol=1e-9)) == 0.0

    def test_upper_tail_mass_is_not_piled_up(self):
        """Under clipping ~4.8% of draws sat exactly at 0.5; under truncation
        the mass just below the bound is the truncated-normal density."""
        rng = np.random.default_rng(2)
        d = simulate_gev_training_data(4000, n_obs=60, xi_prior=(0.0, 0.3),
                                       xi_bounds=(-0.8, 0.5), seed=3)
        xi = d["params"][:, 2]
        assert np.mean(xi >= 0.5 - 1e-9) < 0.005      # clipping gave ~0.048
        assert xi.max() < 0.5 and xi.min() > -0.8


class TestHorseshoeGlobalScale:
    """A small global scale must actually shrink."""

    def test_small_global_tau_shrinks_towards_the_mean(self):
        """With tau0 far below the spread of the MLEs the horseshoe should pool
        hard.  The mis-scaled auxiliary inflated tau by ~30x and returned the
        MLEs almost unchanged.
        """
        mle = np.array([-0.233, -0.240, -0.208, -0.309, -0.130])
        se = np.array([0.0436, 0.0326, 0.0453, 0.0433, 0.0694])
        out = horseshoe_posterior(mle, se, global_tau=0.01, n_iter=4000, seed=0)
        spread_in = mle.max() - mle.min()
        spread_out = out["theta_mean"].max() - out["theta_mean"].min()
        assert spread_out < 0.5 * spread_in
        assert out["kappa_mean"].mean() > 0.7


class TestGPDTailHead:
    """A body-only IQN saturates just past the levels it was trained on.

    The controlled pair behind the tail head: same data, same capacity, ``tail``
    the only difference. Both fit a network, so both are marked slow.

    Each asserts the body fits *below* the splice before claiming anything about
    what happens above it. Without that gate an underpowered network passes
    test_body_only_saturates for the wrong reason, by learning nothing and
    returning a flat line. Measured at hdim=64, n=1200: the body predicted 0.55
    at tau=0.9 against a true 1.72, its tail head ran to 7.35 against a true
    endpoint of 4.0, and both tests still went green.
    """

    MU, SIGMA, XI = 0.0, 1.0, -0.25              # short tail: finite endpoint
    ENDPOINT = MU - SIGMA / XI                   # 4.0
    TAUS = np.array([0.5, 0.9, 0.99, 0.999, 0.9999])

    def _fit(self, tail):
        from gbc.evt import gev_quantile
        from gbc.iqn import train_iqn, predict_iqn
        rng = np.random.default_rng(4)
        n = 6000
        X = rng.standard_normal((n, 3))
        y = np.array([gev_quantile(u, self.MU, self.SIGMA, self.XI)
                      for u in rng.uniform(1e-4, 1 - 1e-4, n)])
        # Mini-batched: 12 steps an epoch over n=6000 reaches the same body
        # accuracy as 900 full-batch epochs (0.12 abs err at tau=0.9) in a fifth
        # of the time. Both n and hidden width have to stay put; at n=3000 the
        # tail head overshoots the finite endpoint.
        m, xm, xs, ym, ys = train_iqn(X, y, epochs=120, batch_size=512,
                                      seed=0, tail=tail, tau0=0.9)
        q = predict_iqn(m, X[:1], xm, xs, ym, ys, taus=self.TAUS).ravel()
        true = np.array([gev_quantile(t, self.MU, self.SIGMA, self.XI)
                         for t in self.TAUS])
        return q, true

    def _assert_body_fits(self, q, true):
        """Below the splice the fit must be real, or the tail claim is vacuous."""
        err = float(np.abs(q[:2] - true[:2]).max())
        assert err < 0.25, (
            f"body does not fit below the splice: max abs err {err:.3f} at "
            f"tau=0.5,0.9 (fitted {q[:2].round(3)}, true {true[:2].round(3)}). "
            f"Anything this test then says about the tail is vacuous.")

    @pytest.mark.slow
    def test_body_only_saturates(self):
        q, true = self._fit(tail=False)
        self._assert_body_fits(q, true)
        # Having fitted the body, the last levels collapse onto one another.
        assert abs(q[-1] - q[-2]) < 0.05 * max(abs(q[-2]), 1e-6)

    @pytest.mark.slow
    def test_tail_head_keeps_climbing(self):
        q, true = self._fit(tail=True)
        self._assert_body_fits(q, true)
        assert q[-1] > q[-2] > q[-3], f"tail did not extrapolate: {q}"
        # xi < 0 gives a finite endpoint; running past it is not extrapolation,
        # it is a tail that has stopped meaning anything.
        assert q[-1] < self.ENDPOINT, (
            f"tail ran past the finite endpoint {self.ENDPOINT}: {q.round(3)}")

    def test_splice_is_continuous_at_tau0(self):
        """Q_GPD(tau0) = u exactly, so the two pieces must agree at the splice."""
        torch.manual_seed(0)
        from gbc.iqn import IQN
        m = IQN(3, hdim=32, nh=8, tail=True, tau0=0.9)
        x = torch.randn(16, 3)
        with torch.no_grad():
            below = float(m(x, 0.9 - 1e-9)[:, 1].mean())
            at = float(m(x, 0.9)[:, 1].mean())
            above = float(m(x, 0.9 + 1e-9)[:, 1].mean())
        assert at == pytest.approx(below, abs=1e-5)
        assert above == pytest.approx(at, abs=1e-5)

    def test_tail_params_do_not_depend_on_tau(self):
        """(sigma_u, xi) are properties of the conditional law, not of tau."""
        torch.manual_seed(0)
        from gbc.iqn import IQN
        m = IQN(3, hdim=32, nh=8, tail=True)
        x = torch.randn(8, 3)
        with torch.no_grad():
            a = m.tail_params(x)
            b = m.tail_params(x)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


class TestMCMCPriorJacobian:
    """The prior is stated on log sigma; the chain walks on sigma."""

    M_REF, S_REF, SD = 30.0, 2.0, 0.5

    def _logprior_in_log_sigma(self, u):
        """gev_logprior re-expressed as a density in u = log sigma."""
        from gbc.evt_inference import gev_logprior
        sigma = math.exp(u)
        # p(u) = p(sigma) * |dsigma/du| = p(sigma) * sigma
        return gev_logprior((self.M_REF, sigma, 0.0), self.M_REF, self.S_REF) + u

    def test_prior_on_log_sigma_is_centred_where_stated(self):
        # Without the Jacobian this density is N(log S_REF + 0.25, 0.5^2) and
        # the mode lands a quarter of a log unit high.  A grid maximum is enough
        # to separate the two: they differ by half a prior standard deviation.
        grid = np.linspace(math.log(self.S_REF) - 2, math.log(self.S_REF) + 2, 20001)
        vals = np.array([self._logprior_in_log_sigma(u) for u in grid])
        mode = grid[int(np.argmax(vals))]
        assert mode == pytest.approx(math.log(self.S_REF), abs=5e-3), (
            f"prior on log sigma peaks at {mode:.4f}, stated "
            f"{math.log(self.S_REF):.4f}; a peak near "
            f"{math.log(self.S_REF) + self.SD ** 2:.4f} means the Jacobian is missing")

    def test_prior_on_log_sigma_is_normal_with_the_stated_scale(self):
        # Normalise on a wide grid and compare the first two moments.
        # Plain Riemann sums: np.trapz is gone in numpy 2 and np.trapezoid is
        # absent in numpy 1.24, and this package has to run under both.
        grid = np.linspace(math.log(self.S_REF) - 6, math.log(self.S_REF) + 6, 40001)
        w = np.exp(np.array([self._logprior_in_log_sigma(u) for u in grid]))
        w /= w.sum()
        mean = float(np.sum(grid * w))
        var = float(np.sum((grid - mean) ** 2 * w))
        assert mean == pytest.approx(math.log(self.S_REF), abs=1e-3)
        assert math.sqrt(var) == pytest.approx(self.SD, rel=1e-3)

    def test_shape_prior_is_truncated_not_reweighted(self):
        from gbc.evt_inference import gev_logprior
        assert gev_logprior((30.0, 2.0, -1.5), 30.0, 2.0) == -np.inf
        assert gev_logprior((30.0, 2.0, 2.5), 30.0, 2.0) == -np.inf
        assert np.isfinite(gev_logprior((30.0, 2.0, 0.4), 30.0, 2.0))


class TestPoissonExponentMeasure:
    """The Poisson mean must be the exponent measure, not the GEV survivor."""

    MU, SIGMA = 30.0, 2.0

    def _lam(self, u, xi):
        if abs(xi) < 1e-8:
            return math.exp(-(u - self.MU) / self.SIGMA)
        return (1.0 + xi * (u - self.MU) / self.SIGMA) ** (-1.0 / xi)

    @pytest.mark.parametrize("xi", [-0.2, 0.0, 0.3])
    def test_count_term_uses_lambda_not_one_minus_exp(self, xi):
        from gbc.evt import poisson_loglik, gpd_density, gev_survivor
        rng = np.random.default_rng(0)
        u = self.MU + 1.5 * self.SIGMA
        # A record with a known exceedance set, so the mark term is common to
        # both candidate count terms and differences out exactly.
        y = np.concatenate([np.full(80, u - 1.0), np.array([u + 0.5, u + 1.5, u + 3.0])])
        n, exc = len(y), y[y > u]
        sigma_u = self.SIGMA + xi * (u - self.MU)
        marks = float(np.sum(np.log(gpd_density(exc - u, sigma_u, xi))))
        lam = self._lam(u, xi)
        want = -n * lam + len(exc) * math.log(lam) + marks
        got = poisson_loglik(y, self.MU, self.SIGMA, xi, u)
        assert got == pytest.approx(want, rel=1e-9), "count term is not -n*Lambda + N*log(Lambda)"
        # and it must differ from the survivor version, or the test proves nothing
        surv = float(gev_survivor(u, self.MU, self.SIGMA, xi))
        wrong = -n * surv + len(exc) * math.log(surv) + marks
        assert abs(got - wrong) > 1e-6

    def test_lambda_exceeds_the_survivor(self):
        # 1 - exp(-L) < L for L > 0, so the old code always understated the mean.
        from gbc.evt import gev_survivor
        for xi in (-0.2, 0.0, 0.3):
            u = self.MU + 1.5 * self.SIGMA
            assert self._lam(u, xi) > float(gev_survivor(u, self.MU, self.SIGMA, xi))
