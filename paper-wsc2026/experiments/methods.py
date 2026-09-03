"""Classical baselines for the GBC-EVT paper, re-exported from the package.

All return-level math uses the Coles (2001) GEV parameterisation.  The GBC-QNN
estimators live in gbc_qnn.py; this file is the classical comparison set.

This file used to reimplement every estimator below, line for line, next to the
package.  The copies then diverged: the local `gev_mcmc` kept a log-posterior
without the `-log(sigma)` Jacobian, so it targeted a prior centring sigma 28%
high while its own docstring stated the correct one.  No driver imported it, so
no published number was affected, but the smoke test at the foot of this file
did run it, and it shipped in the reproduction bundle as a second, wrong answer
to a question the package already answers.  Re-exporting is what makes that
impossible rather than merely unlikely: there is now one implementation.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _cand in (os.path.join(_ROOT, "gbc"), _ROOT, os.path.dirname(_ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break

from gbc.evt import gev_return_level  # noqa: E402,F401
from gbc.evt_inference import (  # noqa: E402,F401
    gev_nll,
    _init,
    fit_stationary_gev,
    ns_gev_nll,
    fit_ns_gev,
    ns_params_at,
    hill,
    gev_logprior,
    gev_mcmc,
    return_level_ci_delta,
)

__all__ = [
    "gev_return_level", "gev_nll", "fit_stationary_gev", "ns_gev_nll",
    "fit_ns_gev", "ns_params_at", "hill", "gev_logprior", "gev_mcmc",
    "return_level_ci_delta",
]


if __name__ == "__main__":  # smoke test on real Seattle data
    import pandas as pd
    df = pd.read_csv(os.path.join(_ROOT, "data", "pnw_jja_maxima.csv"), index_col="year")
    x = df["SEA"].dropna()
    print(f"SEA n={len(x)} years, max={x.max():.1f}C")
    s = fit_stationary_gev(x.values)
    print(f"stationary GEV: mu={s['mu']:.2f} sigma={s['sigma']:.2f} xi={s['xi']:.3f}")
    ci = return_level_ci_delta(x.values, 100)
    print(f"z100 = {ci['zN']:.1f}C  90% CI [{ci['lo']:.1f}, {ci['hi']:.1f}]  width={ci['hi']-ci['lo']:.1f}")
    print(f"Hill (k=20) gamma = {hill(x.values, 20):.3f}")
    sm = gev_mcmc(x.values, seed=1)
    z = np.array([gev_return_level(100, *p) for p in sm])
    print(f"MCMC: {len(sm)} samples, xi post mean={sm[:,2].mean():.3f}, "
          f"z100 post median={np.median(z):.1f} 90% CI [{np.quantile(z,.05):.1f}, {np.quantile(z,.95):.1f}]")
