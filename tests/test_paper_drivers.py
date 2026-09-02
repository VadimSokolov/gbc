"""Regressions for the WSC 2026 drivers in paper-wsc2026/.

The drivers are not part of the importable package, but they are shipped in this
repository and they are what produced every number in the paper, so a defect in
one of them is a defect in a published result.

This file exists because of a specific miss.  Three estimator defects were found
and fixed in the package, and `gbc.evt._truncated_normal` replaced a
`np.clip` on the shape prior.  `run_scale.py`, however, carries its own inline
prior-predictive simulator and never calls `simulate_gev_training_data`, so it
kept the clip and kept teaching the amortised networks a 4.7% point mass at
xi = 0.5.  Fixing a library function is not the same as fixing the study; these
tests assert the property at the driver's own call site.
"""
import importlib.util
import os

import numpy as np
import pytest

_DRIVERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "paper-wsc2026", "experiments")


def _load(name):
    """Import a driver by path; they are scripts, not an installed package."""
    path = os.path.join(_DRIVERS, name + ".py")
    if not os.path.exists(path):
        pytest.skip(f"{name}.py not present in this checkout")
    spec = importlib.util.spec_from_file_location("_driver_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shapes_drawn(n_sim, seed=0):
    """Every xi the standardised simulator draws, captured at the call site.

    `_sim_standardised` does not return its parameters, so we intercept the one
    function it hands them to rather than reimplementing the draw, which would
    test a copy instead of the code that runs.
    """
    run_scale = _load("run_scale")
    seen, real = [], run_scale.gev_quantile

    def spy(u, mu, sigma, xi):
        seen.append(xi)
        return real(u, mu, sigma, xi)

    run_scale.gev_quantile = spy
    try:
        run_scale._sim_standardised(n_sim, "rl", np.random.default_rng(seed))
    finally:
        run_scale.gev_quantile = real
    # One call per simulated record, plus one per 'draw' target; take the record
    # draws, which are the first of each pair.
    return np.array(seen, dtype=float)


class TestAmortisedShapePrior:
    """xi ~ N(0, 0.3^2) truncated to [-0.8, 0.5], as the paper's Method claims."""

    LO, HI, SD = -0.8, 0.5, 0.3

    @pytest.fixture(scope="class")
    def xi(self):
        return _shapes_drawn(6000)

    def test_within_bounds(self, xi):
        assert xi.min() >= self.LO and xi.max() <= self.HI

    def test_no_atom_at_either_bound(self, xi):
        # np.clip put 4.7% of the mass on the upper bound and 0.4% on the lower.
        # A truncated normal is continuous, so both must be empty.
        for bound in (self.LO, self.HI):
            atom = float(np.mean(np.isclose(xi, bound, atol=1e-9)))
            assert atom == 0.0, f"point mass {atom:.4f} at xi = {bound}"

    def test_matches_the_truncated_normal(self, xi):
        truncnorm = pytest.importorskip("scipy.stats").truncnorm
        ref = truncnorm(self.LO / self.SD, self.HI / self.SD, 0.0, self.SD)
        ks = pytest.importorskip("scipy.stats").kstest(xi, ref.cdf)
        assert ks.pvalue > 0.01, f"KS p = {ks.pvalue:.4f}, D = {ks.statistic:.4f}"
