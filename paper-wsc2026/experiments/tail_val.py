"""Does the GPD tail head recover the far tail a body-only network cannot?

Reference is the stationary GEV fitted to the same detrended Seattle record: it
puts P(X >= 42.2) = 8.4e-3 with an upper endpoint at 45.13 C, so the far tail is
genuinely there to be found and a saturating network is simply wrong.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _c in (os.path.join(ROOT, "gbc"), ROOT, os.path.dirname(ROOT)):
    if os.path.isfile(os.path.join(_c, "gbc", "__init__.py")):
        sys.path.insert(0, _c); break
import numpy as np, pandas as pd
from gbc.evt import gev_quantile, gev_survivor
from gbc.evt_inference import (fit_stationary_gev, fit_ns_gev,
                               train_predictive_iqn, _feat)
from gbc.iqn import predict_iqn

HD = 42.2
d = pd.read_csv(os.path.join(ROOT, "data", "pnw_jja_maxima.csv"), index_col="year")
h = pd.read_csv(os.path.join(ROOT, "data", "hadcrut5_annual.csv"), index_col="year")["anomaly"]
x = d["SEA"].dropna(); T = h.reindex(x.index).values; x = x.values.astype(float)
nsf = fit_ns_gev(x, T)
x23 = x + nsf["mu1"] * (float(h.loc[2023]) - T)
f = fit_stationary_gev(x23); mu, sig, xi = f["mu"], f["sigma"], f["xi"]
print(f"reference GEV: mu={mu:.3f} sigma={sig:.3f} xi={xi:+.4f} "
      f"endpoint={mu - sig/xi:.2f} C, P(>=42.2)={gev_survivor(HD,mu,sig,xi):.3e}")

TAUS = np.array([0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999])
ref = np.array([gev_quantile(t, mu, sig, xi) for t in TAUS])
rows = {}
for label, kw in (("body only", dict(tail=False)), ("GPD tail head", dict(tail=True))):
    m, xm, xs, ym, ys = train_predictive_iqn(x23, n_sim=30000, epochs=1500, seed=0, **kw)
    q = predict_iqn(m, _feat(x23), xm, xs, ym, ys, taus=TAUS).ravel()
    rows[label] = q
    fine = np.linspace(0.5, 1 - 1e-6, 4000)
    qf = predict_iqn(m, _feat(x23), xm, xs, ym, ys, taus=fine).ravel()
    mono = bool(np.all(np.diff(qf) >= -1e-6))
    reach = qf.max()
    p = float(1 - np.interp(HD, qf, fine)) if reach >= HD else 0.0
    rows[label + "_meta"] = (mono, reach, p)

print(f"\n{'tau':>9} {'GEV ref':>9} {'body only':>11} {'GPD tail':>11}")
for i, t in enumerate(TAUS):
    print(f"{t:9.5f} {ref[i]:9.2f} {rows['body only'][i]:11.2f} {rows['GPD tail head'][i]:11.2f}")
for label in ("body only", "GPD tail head"):
    mono, reach, p = rows[label + "_meta"]
    rp = f"{1/p:,.0f} yr" if p > 0 else "unreachable"
    print(f"\n{label:14s} monotone={mono}  max Q={reach:.2f} C  "
          f"P(>=42.2)={p:.3e}  return period {rp}")
