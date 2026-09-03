"""GBC-QNN estimators for EVT, re-exported from the package.

Two IQNs per analysis:
  * predictive IQN  : target = a fresh GEV draw  -> predictive quantiles.
  * functional IQN  : target = z_N(theta)        -> posterior quantiles of z_N.
Both are trained once on prior-predictive simulations (amortized) and applied
by a forward pass.

This file used to reimplement both training paths next to the package, and the
copies had already diverged.  The local ``train_predictive`` called ``train_iqn``
without ``tail=True, tau0=0.9``, so it built a predictive network with no GPD
tail head, the very splice the paper states in eq:splice and the first defect
round added.  Its ``crps_coverage_loyo`` then scored that tail-less network on
``sample_iqn``'s fixed 0.005 to 0.995 grid and returned the result as a plain
predictive coverage, with none of the truncation disclosure the package
docstrings now carry.  No driver imported it, so no published number was
affected, but the smoke test at the foot of this file did run it, and it shipped
in the reproduction bundle as a second and wrong answer to a question the
package already answers.

Re-exporting is what makes that impossible rather than merely unlikely: there is
now one implementation.  The ``train_*`` helpers return the package's 5-tuple
``(model, xm, xs, ym, ys)`` rather than the 6-tuple with a leading tag string
that the local copies used, which is what the consumers below expect.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The gbc package sits at ROOT/gbc/gbc in the paper tree and one level higher
# when these drivers ship inside the gbc repository itself; add whichever parent
# actually holds it, so a driver runs unchanged from either checkout.
for _cand in (os.path.join(_ROOT, "gbc"), _ROOT, os.path.dirname(_ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break

from gbc.evt_inference import (  # noqa: E402,F401
    _feat,
    _sim,
    gbc_priors_from_data as priors_from_data,
    train_predictive_iqn as train_predictive,
    train_functional_iqn as train_functional,
    gbc_return_level_posterior as return_level_posterior,
    gbc_predictive_samples as predictive_samples,
    gbc_crps_coverage_loyo as crps_coverage_loyo,
)

if __name__ == "__main__":  # smoke test on real SEA data
    import pandas as pd
    df = pd.read_csv(os.path.join(_ROOT, "data", "pnw_jja_maxima.csv"), index_col="year")
    x = df["SEA"].dropna().values
    print(f"SEA n={len(x)}  training GBC-QNN (functional + predictive)...")
    fF = train_functional(x, N=100, n_sim=15000, epochs=700, seed=1)
    rl = return_level_posterior(fF, x)
    print(f"GBC-QNN z100 = {rl['zN']:.1f}C  90% CI [{rl['lo']:.1f}, {rl['hi']:.1f}]")
    cc = crps_coverage_loyo(x, n_sim=8000, epochs=500)
    # Both diagnostics come off sample_iqn's fixed interior tau grid, so the
    # interval carries slightly less than 0.90 model mass.  See sample_iqn.
    print(f"GBC-QNN LOYO (grid): CRPS={cc['crps']:.3f}  grid coverage={cc['coverage']:.2f}")
