"""Heavy-tailed hazard demo (tab:heavytail, fig:heavytail).

CONUS annual-maximum daily PRECIPITATION is genuinely heavy-tailed (xi>0), the
regime where small-sample maximum likelihood is unstable and the simulation
study (Section 8) predicts GBC's largest gains. We show, on real precipitation:
  1. the tail-index distribution is positive (vs negative for temperature);
  2. recovering the full-record 100-yr level from a SHORT (n=25) record, GBC's
     amortised, prior-regularised estimate has lower RMSE and better coverage
     than maximum likelihood.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbc.evt_inference import fit_stationary_gev, return_level_ci_delta
from gbc.evt import gev_return_level
from run_scale import train_amortised, gbc_rl

STAMP = date.today().isoformat()
N_RET, MIN_YEARS, N_SUB, R = 100, 40, 25, 20
_LEDGER = os.path.join(ROOT, "results", "numbers.txt")


def _rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def evaluate(ref, rl_net, rng):
    """Paired short-record recovery, grouped by station.

    A subsample is kept only when BOTH estimators return a finite return level,
    so the two RMSEs are computed over an identical set of subsamples. Errors are
    grouped by station because the 20 subsamples drawn from one record are not
    independent, which is what the clustered bootstrap below needs.
    """
    per_station, n_dropped, n_eval = {}, 0, 0
    for c, x in ref.items():
        ztrue = gev_return_level(N_RET, **{k: fit_stationary_gev(x)[k] for k in ("mu", "sigma", "xi")})
        if not np.isfinite(ztrue) or ztrue <= 0 or ztrue > 5 * np.max(x):
            continue
        em, eg, cm, cg = [], [], 0, 0
        for _ in range(R):
            xs = rng.choice(x, size=N_SUB, replace=False)
            m = return_level_ci_delta(xs, N_RET)
            g = gbc_rl(rl_net, xs)
            if not (np.isfinite(m["zN"]) and np.isfinite(g["zN"])):
                n_dropped += 1
                continue
            em.append(m["zN"] - ztrue); cm += int(m["lo"] <= ztrue <= m["hi"])
            eg.append(g["zN"] - ztrue); cg += int(g["lo"] <= ztrue <= g["hi"])
            n_eval += 1
        if em:
            per_station[c] = (np.array(em), np.array(eg), cm, cg)
    return per_station, n_dropped, n_eval


def clustered_reduction_ci(per_station, n_boot=2000, seed=0, alpha=0.10):
    """Station-clustered bootstrap CI for the percentage RMSE reduction.

    Resamples STATIONS with replacement, not subsamples: the 20 draws from one
    record share that record, so an unclustered interval would be too narrow.
    """
    keys = list(per_station)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(keys), size=len(keys))
        em = np.concatenate([per_station[keys[i]][0] for i in idx])
        eg = np.concatenate([per_station[keys[i]][1] for i in idx])
        out[b] = 100 * (1 - _rmse(eg) / _rmse(em))
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def reset_ledger():
    if not os.path.exists(_LEDGER):
        return
    with open(_LEDGER) as fh:
        keep = [ln for ln in fh if "\trun_heavytail.py\t" not in ln]
    with open(_LEDGER, "w") as fh:
        fh.writelines(keep)


def ledger(value, ident, fmt="{:.4g}"):
    with open(_LEDGER, "a") as fh:
        fh.write(f"{fmt.format(value)}\trun_heavytail.py\t{ident}\t{STAMP}\n")
    return value


def xi_dist(csv, n_min):
    df = pd.read_csv(os.path.join(ROOT, "data", csv), index_col="year")
    xis = {}
    for c in df.columns:
        x = df[c].dropna().values.astype(float)
        if len(x) >= n_min:
            xis[c] = fit_stationary_gev(x)["xi"]
    return df, xis


def main():
    reset_ledger()
    pdf, pxi = xi_dist("conus_prcp_maxima.csv", MIN_YEARS)
    _, txi = xi_dist("conus_tmax_maxima.csv", MIN_YEARS)
    pvals, tvals = np.array(list(pxi.values())), np.array(list(txi.values()))
    print(f"PRCP: {len(pvals)} stations, median xi={np.median(pvals):+.3f}, "
          f"frac xi>0 = {np.mean(pvals > 0):.2f}")
    print(f"TMAX: {len(tvals)} stations, median xi={np.median(tvals):+.3f}, "
          f"frac xi>0 = {np.mean(tvals > 0):.2f}")

    # one amortised GBC network (prior covers heavy tails, xi up to 0.5)
    print("training amortised GBC network ...")
    rl_net = train_amortised("rl", n_sim=60000, epochs=2000, seed=7)

    # short-record recovery: full-record MLE z100 is the reference; estimate from n=25.
    # Two station sets: the screened band used in the paper, and an UNSCREENED set
    # (every station with >=60 yr) that does not condition on the reference statistic.
    ref = {c: pdf[c].dropna().values.astype(float) for c in pxi
           if pdf[c].notna().sum() >= 60 and -0.1 < pxi[c] < 0.45}
    ref_all = {c: pdf[c].dropna().values.astype(float) for c in pxi
               if pdf[c].notna().sum() >= 60}
    print(f"reference (screened band, >=60 yr) stations: {len(ref)}")
    print(f"reference (unscreened,      >=60 yr) stations: {len(ref_all)}")

    per_station, n_dropped, n_eval = evaluate(ref, rl_net, np.random.default_rng(3))
    e_mle = np.concatenate([v[0] for v in per_station.values()])
    e_gbc = np.concatenate([v[1] for v in per_station.values()])
    cov_mle = sum(v[2] for v in per_station.values()) / len(e_mle)
    cov_gbc = sum(v[3] for v in per_station.values()) / len(e_gbc)
    rmse_mle, rmse_gbc = _rmse(e_mle), _rmse(e_gbc)
    red = 100 * (1 - rmse_gbc / rmse_mle)
    red_lo, red_hi = clustered_reduction_ci(per_station)

    print(f"short-record recovery over {n_eval} PAIRED subsamples "
          f"({n_dropped} pairs dropped for non-finite MLE):")
    print(f"  RMSE  MLE={rmse_mle:.1f}mm  GBC={rmse_gbc:.1f}mm  "
          f"reduction={red:.0f}% [{red_lo:.0f}, {red_hi:.0f}] (station-clustered, 90%)")
    print(f"  90% coverage of full-record z100:  MLE={cov_mle:.2f}  GBC={cov_gbc:.2f}")

    # sensitivity: same evaluation without conditioning on the reference shape
    ps_all, n_drop_all, n_eval_all = evaluate(ref_all, rl_net, np.random.default_rng(3))
    em_all = np.concatenate([v[0] for v in ps_all.values()])
    eg_all = np.concatenate([v[1] for v in ps_all.values()])
    red_all = 100 * (1 - _rmse(eg_all) / _rmse(em_all))
    red_all_lo, red_all_hi = clustered_reduction_ci(ps_all)
    print(f"  UNSCREENED ({len(ps_all)} stations, {n_eval_all} pairs): "
          f"reduction={red_all:.0f}% [{red_all_lo:.0f}, {red_all_hi:.0f}]")

    for v, k, f in [(len(pvals), "ht:n_prcp", "{:.0f}"), (float(np.median(pvals)), "ht:median_xi_prcp", "{:.3f}"),
                    (float(np.mean(pvals > 0)), "ht:frac_xi_pos", "{:.3f}"),
                    (float(np.median(tvals)), "ht:median_xi_tmax", "{:.3f}"),
                    (float(np.mean(tvals > 0)), "ht:frac_xi_pos_tmax", "{:.3f}"),
                    (len(tvals), "ht:n_tmax", "{:.0f}"),
                    (int(sum(1 for c in ref if pxi[c] <= 0)), "ht:n_ref_nonpositive", "{:.0f}"),
                    (len(ref), "ht:n_ref", "{:.0f}"), (n_eval, "ht:n_eval", "{:.0f}"),
                    (rmse_mle, "ht:rmse_mle", "{:.2f}"), (rmse_gbc, "ht:rmse_gbc", "{:.2f}"),
                    (red, "ht:rmse_reduction", "{:.1f}"),
                    (red_lo, "ht:rmse_reduction_lo", "{:.1f}"),
                    (red_hi, "ht:rmse_reduction_hi", "{:.1f}"),
                    (n_dropped, "ht:n_dropped", "{:.0f}"),
                    (len(ps_all), "ht:n_ref_unscreened", "{:.0f}"),
                    (red_all, "ht:rmse_reduction_unscreened", "{:.1f}"),
                    (red_all_lo, "ht:rmse_reduction_unscreened_lo", "{:.1f}"),
                    (red_all_hi, "ht:rmse_reduction_unscreened_hi", "{:.1f}"),
                    (cov_mle, "ht:cov_mle", "{:.3f}"), (cov_gbc, "ht:cov_gbc", "{:.3f}")]:
        ledger(v, k, f)

    np.save(os.path.join(ROOT, "results", "ht_xi_prcp.npy"), pvals)
    np.save(os.path.join(ROOT, "results", "ht_xi_tmax.npy"), tvals)
    with open(os.path.join(ROOT, "tab", "heavytail.tex"), "w") as fh:
        fh.write("% generated by experiments/run_heavytail.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{lcc}\n\\toprule\n")
        fh.write("Recover full-record $z_{100}$ from $n{=}25$ yr & MLE & GBC-QNN \\\\\n\\midrule\n")
        fh.write(f"RMSE (mm, lower better) & {rmse_mle:.1f} & \\textbf{{{rmse_gbc:.1f}}} \\\\\n")
        fh.write(f"RMSE reduction (\\%, station-clustered 90\\% CI) & \\multicolumn{{2}}{{c}}{{"
                 f"${red:.0f}$ $[{red_lo:.0f},\\,{red_hi:.0f}]$}} \\\\\n")
        fh.write(f"\\quad without the shape screen ({len(ps_all)} stations) & \\multicolumn{{2}}{{c}}{{"
                 f"${red_all:.0f}$ $[{red_all_lo:.0f},\\,{red_all_hi:.0f}]$}} \\\\\n")
        fh.write(f"Coverage of $z_{{100}}$ (nominal 0.90) & {cov_mle:.2f} & {cov_gbc:.2f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote tab/heavytail.tex, results/ht_xi_*.npy")


if __name__ == "__main__":
    main()
