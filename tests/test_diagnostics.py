"""Tests for MCMC convergence diagnostics against cases with known answers."""
import numpy as np
import pytest

from gbc.diagnostics import (ess_bulk, ess_per_second, rank_normalize, rhat,
                             split_chains, summarize_chains)


def ar1(rho, m=4, n=4000, seed=0):
    """m chains of a stationary AR(1) with autocorrelation rho."""
    rng = np.random.default_rng(seed)
    x = np.zeros((m, n))
    for k in range(m):
        e = rng.standard_normal(n)
        for t in range(1, n):
            x[k, t] = rho * x[k, t - 1] + e[t]
    return x


def test_split_chains_doubles_and_halves():
    d = np.arange(40.0).reshape(4, 10)
    s = split_chains(d)
    assert s.shape == (8, 5)
    assert np.allclose(s[0], d[0, :5])
    assert np.allclose(s[4], d[0, 5:])


def test_rank_normalize_is_standard_normal_scale():
    rng = np.random.default_rng(1)
    d = np.exp(rng.standard_normal((4, 2000)))      # heavily skewed
    z = rank_normalize(d)
    assert z.shape == d.shape
    assert abs(z.mean()) < 0.05
    assert 0.9 < z.std() < 1.1


def test_rhat_near_one_for_converged_chains():
    rng = np.random.default_rng(2)
    assert rhat(rng.standard_normal((4, 4000))) < 1.01


def test_rhat_flags_chains_that_disagree():
    rng = np.random.default_rng(3)
    d = rng.standard_normal((4, 4000)) + np.arange(4)[:, None] * 3.0
    assert rhat(d) > 1.5


def test_ess_matches_iid_draw_count():
    rng = np.random.default_rng(4)
    d = rng.standard_normal((4, 4000))
    # split-chains keeps 8 x 2000 = 16000 draws; iid ESS should be close to that
    assert 12000 < ess_bulk(d) < 20000


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_ess_tracks_the_ar1_integrated_autocorrelation(rho):
    d = ar1(rho, seed=5)
    expected = 16000 * (1 - rho) / (1 + rho)        # tau = (1+rho)/(1-rho)
    assert 0.6 * expected < ess_bulk(d) < 1.5 * expected


def test_ess_collapses_for_stuck_chains():
    d = np.repeat(np.arange(4.0)[:, None], 4000, axis=1)   # each chain constant
    assert ess_bulk(d) < 50


def test_ess_per_second_divides_by_wall_clock():
    rng = np.random.default_rng(6)
    d = rng.standard_normal((4, 4000))
    assert ess_per_second(d, 2.0) == pytest.approx(ess_bulk(d) / 2.0)


def test_summarize_chains_returns_one_row_per_parameter():
    rng = np.random.default_rng(7)
    d = rng.standard_normal((4, 2000, 3))
    rows = summarize_chains(d, names=["mu", "sigma", "xi"])
    assert [r["name"] for r in rows] == ["mu", "sigma", "xi"]
    assert all(r["rhat"] < 1.01 for r in rows)
    assert all(r["ess_bulk"] > 4000 for r in rows)
