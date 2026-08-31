"""GBC-QNN estimators for EVT, built on the gbc package's IQN engine.

Two IQNs per analysis:
  * predictive IQN  : target = a fresh GEV draw  -> predictive quantiles (CRPS, coverage).
  * functional IQN  : target = z_N(theta)        -> posterior quantiles of the return level (CI).
Both are trained once on prior-predictive simulations (amortized) and applied by a forward pass.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The gbc package sits at ROOT/gbc/gbc in the paper tree and one level higher
# when these drivers ship inside the gbc repository itself; add whichever parent
# actually holds it, so a driver runs unchanged from either checkout.
for _cand in (os.path.join(ROOT, "gbc"), ROOT, os.path.dirname(ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break
from gbc.evt import (gev_return_level, simulate_gev_training_data,  # noqa: E402
                     _block_maxima_summary)
from gbc.iqn import train_iqn, predict_iqn, sample_iqn  # noqa: E402
from gbc.metrics import crps_samples, coverage  # noqa: E402

TAUS = np.linspace(0.01, 0.99, 99)


def priors_from_data(x):
    """Empirical-Bayes reference values for the weakly-informative priors (Coles 1996)."""
    return {"mu_prior": (float(np.mean(x)), 5.0),
            "log_sigma_prior": (float(np.log(np.std(x, ddof=1))), 0.5),
            "xi_prior": (0.0, 0.3), "xi_bounds": (-1.0, 2.0)}


def _sim(n_sim, n_obs, pr, seed):
    return simulate_gev_training_data(n_sim=n_sim, n_obs=n_obs, mu_prior=pr["mu_prior"],
                                      log_sigma_prior=pr["log_sigma_prior"],
                                      xi_prior=pr["xi_prior"], xi_bounds=pr["xi_bounds"], seed=seed)


def train_predictive(x, n_sim=40000, epochs=1500, seed=0):
    pr = priors_from_data(x)
    d = _sim(n_sim, len(x), pr, seed)
    model, xm, xs, ym, ys = train_iqn(d["X"], d["y"], epochs=epochs, seed=seed)
    return ("pred", model, xm, xs, ym, ys)


def train_functional(x, N=100, n_sim=40000, epochs=1500, seed=1):
    """IQN whose target is the true z_N(theta); its predictive quantiles ARE the z_N posterior."""
    pr = priors_from_data(x)
    d = _sim(n_sim, len(x), pr, seed)
    zN = np.array([gev_return_level(N, *p) for p in d["params"]])
    model, xm, xs, ym, ys = train_iqn(d["X"], zN, epochs=epochs, seed=seed)
    return ("func", model, xm, xs, ym, ys)


def _feat(x):
    return _block_maxima_summary(np.asarray(x, float)).reshape(1, -1)


def return_level_posterior(fitted, x, alpha=0.10):
    """Posterior median + (1-alpha) CI for z_N from a functional IQN."""
    _, model, xm, xs, ym, ys = fitted
    q = predict_iqn(model, _feat(x), xm, xs, ym, ys,
                    np.array([alpha / 2, 0.5, 1 - alpha / 2])).ravel()
    return {"zN": float(q[1]), "lo": float(q[0]), "hi": float(q[2])}


def predictive_samples(fitted, x, B=2000):
    _, model, xm, xs, ym, ys = fitted
    return sample_iqn(model, _feat(x), xm, xs, ym, ys, B=B).ravel()


def crps_coverage_loyo(x, n_sim=20000, epochs=900, seed=0, alpha=0.10):
    """Leave-one-year-out predictive CRPS and (1-alpha) coverage (amortized: train once)."""
    x = np.asarray(x, float)
    fitted = train_predictive(x, n_sim=n_sim, epochs=epochs, seed=seed)
    _, model, xm, xs, ym, ys = fitted
    crps_list, hits = [], 0
    lo_t, hi_t = alpha / 2, 1 - alpha / 2
    for i in range(len(x)):
        xho = np.delete(x, i)
        samp = sample_iqn(model, _feat(xho), xm, xs, ym, ys, B=1000).ravel()
        crps_list.append(crps_samples(np.array([x[i]]), samp.reshape(-1, 1)))
        lo, hi = np.quantile(samp, [lo_t, hi_t])
        hits += int(lo <= x[i] <= hi)
    return {"crps": float(np.mean(crps_list)), "coverage": hits / len(x)}


if __name__ == "__main__":  # smoke test on real SEA data
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "data", "pnw_jja_maxima.csv"), index_col="year")
    x = df["SEA"].dropna().values
    print(f"SEA n={len(x)}  training GBC-QNN (functional + predictive)...")
    fF = train_functional(x, N=100, n_sim=15000, epochs=700, seed=1)
    rl = return_level_posterior(fF, x)
    print(f"GBC-QNN z100 = {rl['zN']:.1f}C  90% CI [{rl['lo']:.1f}, {rl['hi']:.1f}]")
    cc = crps_coverage_loyo(x, n_sim=8000, epochs=500)
    print(f"GBC-QNN LOYO: CRPS={cc['crps']:.3f}  coverage={cc['coverage']:.2f}")
