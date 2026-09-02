"""Can the corrected body-only network reach the 2021 heat dome at all?

run_attr estimates P(X >= 42.2) by counting an evenly spaced tau grid of size
B=4000, which floors the answer at 2.5e-4 and returned exactly zero once the
quantile target was corrected.  The question is whether that is a resolution
limit (curable with larger B) or a saturating tail (not curable at all), so
evaluate Q on a fine ASCENDING grid and look at the largest value it attains.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _c in (os.path.join(ROOT, "gbc"), ROOT, os.path.dirname(ROOT)):
    if os.path.isfile(os.path.join(_c, "gbc", "__init__.py")):
        sys.path.insert(0, _c); break
import numpy as np, pandas as pd
from gbc.evt_inference import fit_ns_gev, train_predictive_iqn, _feat
from gbc.iqn import predict_iqn

HD = 42.2
d = pd.read_csv(os.path.join(ROOT, "data", "pnw_jja_maxima.csv"), index_col="year")
h = pd.read_csv(os.path.join(ROOT, "data", "hadcrut5_annual.csv"), index_col="year")["anomaly"]
x = d["SEA"].dropna(); T = h.reindex(x.index).values; x = x.values.astype(float)
nsf = fit_ns_gev(x, T)
x23 = x + nsf["mu1"] * (float(h.loc[2023]) - T)
print(f"SEA n={len(x)}  observed max {x.max():.1f}  detrended-to-2023 max {x23.max():.1f}")

model, xm, xs, ym, ys = train_predictive_iqn(x23, n_sim=30000, epochs=1500, seed=0)
feat = _feat(x23)

taus = np.sort(np.concatenate([
    np.linspace(0.50, 0.99, 200),
    1 - np.logspace(-6, -2, 200),          # 0.99 .. 0.999999
]))
q = predict_iqn(model, feat, xm, xs, ym, ys, taus=taus).ravel()

for t in (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999):
    print(f"  Q({t:<8}) = {q[np.searchsorted(taus, t)]:.2f} C")
dq = np.diff(q)
print(f"max Q attained over tau<=1-1e-6 : {q.max():.2f} C   (heat dome {HD} C)")
print(f"monotone non-decreasing in tau  : {bool(np.all(dq >= -1e-9))}"
      f"   (largest decrease {min(dq.min(), 0.0):+.4f} C)")
print("VERDICT:", "saturating tail, larger B cannot help"
      if q.max() < HD else f"resolution limit: crossing exists at Q={q.max():.2f}")
