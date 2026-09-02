"""Conditional risk of the 2021 Seattle heat dome (tab:attr): exceedance
probability and return period of a 42.2 degC JJA maximum under the 1950 versus
2023 fitted climates, with the warming risk ratio.
"""
import os
import sys

from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The gbc package sits at ROOT/gbc/gbc in the paper tree and one level higher
# when these drivers ship inside the gbc repository itself; add whichever parent
# actually holds it, so a driver runs unchanged from either checkout.
for _cand in (os.path.join(ROOT, "gbc"), ROOT, os.path.dirname(ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break
from gbc.evt import gev_survivor, gev_quantile
from gbc.evt_inference import fit_stationary_gev, fit_ns_gev, ns_params_at

STAMP = date.today().isoformat()
HD = 42.2                      # 2021 Seattle-Tacoma JJA maximum (degC)
T1950, T2023 = -0.227, 1.100
_LEDGER = os.path.join(ROOT, "results", "numbers.txt")


def reset_ledger():
    if not os.path.exists(_LEDGER):
        return
    with open(_LEDGER) as fh:
        keep = [ln for ln in fh if "\trun_attr.py\t" not in ln]
    with open(_LEDGER, "w") as fh:
        fh.writelines(keep)


def fmt_p(v):
    """Uniform scientific formatting so the P column does not mix 6.9e-05 with 0.003."""
    if v <= 0:
        return "0"
    from math import floor, log10
    e = int(floor(log10(v)))
    return f"${v/10**e:.1f}\\times 10^{{{e}}}$"


def ledger(value, ident, fmt="{:.4g}"):
    with open(_LEDGER, "a") as fh:
        fh.write(f"{fmt.format(value)}\trun_attr.py\t{ident}\t{STAMP}\n")
    return value


def rp(p):
    return np.inf if p <= 0 else 1.0 / p


def fmt_rp(r):
    if not np.isfinite(r) or r > 1e5:
        return r"$>10^5$"
    if r >= 1000:
        return f"{r/1000:.1f}k"
    return f"{r:.0f}"


def main():
    reset_ledger()
    d = pd.read_csv(os.path.join(ROOT, "data", "pnw_jja_maxima.csv"), index_col="year")
    h = pd.read_csv(os.path.join(ROOT, "data", "hadcrut5_annual.csv"), index_col="year")["anomaly"]
    x = d["SEA"].dropna()
    T = h.reindex(x.index).values
    x = x.values.astype(float)
    print(f"SEA n={len(x)}, heat dome HD={HD} degC")

    # point estimates
    sf = fit_stationary_gev(x)
    p_stat = float(gev_survivor(HD, sf["mu"], sf["sigma"], sf["xi"]))
    nsf = fit_ns_gev(x, T)
    p50 = float(gev_survivor(HD, *ns_params_at(nsf, T1950)))
    p23 = float(gev_survivor(HD, *ns_params_at(nsf, T2023)))
    rr = (p23 / p50) if p50 > 0 else np.inf

    # parametric bootstrap CI for the NS quantities
    rng = np.random.default_rng(7)
    b50, b23, brr = [], [], []
    for _ in range(400):
        xb = np.array([gev_quantile(rng.uniform(), *ns_params_at(nsf, T[k])) for k in range(len(x))])
        fb = fit_ns_gev(xb, T)
        q50 = float(gev_survivor(HD, *ns_params_at(fb, T1950)))
        q23 = float(gev_survivor(HD, *ns_params_at(fb, T2023)))
        b50.append(q50); b23.append(q23); brr.append((q23 / q50) if q50 > 0 else np.inf)
    def ci_rp(arr):
        r = np.array([rp(v) for v in arr])
        r = np.clip(r, 0, 1e6)
        return np.quantile(r, 0.05), np.quantile(r, 0.95)

    rows = [
        ("Stationary GEV (full record)", p_stat, rp(p_stat), None),
        ("NS-GEV, 1950 climate", p50, rp(p50), ci_rp(b50)),
        ("NS-GEV, 2023 climate", p23, rp(p23), ci_rp(b23)),
    ]
    brr_arr = np.array(brr, dtype=float)
    frac_inc = float(np.mean(np.array(b23) > np.array(b50)))
    rr_med = float(np.median(brr_arr[np.isfinite(brr_arr)]))

    print("\n=== tab:attr (2021 Seattle heat dome, 42.2 degC) ===")
    for name, p, r, ci in rows:
        cis = "" if ci is None else f"  CI[{fmt_rp(ci[0])},{fmt_rp(ci[1])}]"
        print(f"{name:32s} P={p:.3g}  RP={fmt_rp(r)} yr{cis}")
    print(f"Risk ratio (2023 vs 1950): point={rr:.1f}x, bootstrap median={rr_med:.1f}x; "
          f"warming increased risk in {frac_inc:.0%} of bootstrap fits")

    for name, p, r, ci in rows:
        # include the climate year: "NS-GEV, 1950 climate" and "NS-GEV, 2023 climate"
        # both reduced to "nsgev" before, overwriting each other in the ledger.
        tag = (name.replace(",", " ").replace("(", " ").replace(")", " ")
               .strip().lower().replace("-", "").split())
        tag = "_".join(t for t in tag if t not in ("climate", "full", "record"))
        ledger(p, f"attr:{tag}:p")
        ledger(min(r, 1e6), f"attr:{tag}:rp")
        # Interval endpoints belong in the ledger too: they are printed in the
        # table and quoted in the prose, so a reader must be able to trace them.
        if ci is not None:
            ledger(min(ci[0], 1e6), f"attr:{tag}:rp_lo")
            ledger(min(ci[1], 1e6), f"attr:{tag}:rp_hi")
    ledger(rr, "attr:risk_ratio", "{:.2f}")
    ledger(rr_med, "attr:risk_ratio_bootmedian", "{:.2f}")
    ledger(frac_inc, "attr:frac_increase", "{:.3f}")

    with open(os.path.join(ROOT, "tab", "attr.tex"), "w") as fh:
        fh.write("% generated by experiments/run_attr.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{lccc}\n\\toprule\n")
        fh.write("Model & $P(\\text{JJA max}\\geq 42.2^\\circ\\text{C})$ & Return period (yr) & 90\\% CI \\\\\n\\midrule\n")
        for name, p, r, ci in rows:
            cis = "n/a" if ci is None else f"[{fmt_rp(ci[0])}, {fmt_rp(ci[1])}]"
            fh.write(f"{name} & {fmt_p(p)} & {fmt_rp(r)} & {cis} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"\\multicolumn{{4}}{{p{{0.92\\linewidth}}}}{{\\small Warming risk ratio "
                 f"(2023 vs.\\ 1950): ${rr:.0f}\\times$ (point), with the direction of the change "
                 f"reproduced in {frac_inc:.0%}".replace("%", "\\%") + " of bootstrap refits.  "
                 f"Neither return-period interval is informative: both are unbounded above because "
                 f"42.2$^\\circ$C lies near the estimated upper endpoint.}} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("\nwrote tab/attr.tex and appended results/numbers.txt")


if __name__ == "__main__":
    main()
