"""Real Pacific Northwest analysis: Table tab:rl (Seattle return levels) and
tab:spatial (horseshoe shrinkage of xi). Every number is computed from real data
and appended to results/numbers.txt; LaTeX fragments go to tab/.
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
from gbc.evt import (gev_return_level, gev_quantile, simulate_nonstationary_gev_training_data,
                     _block_maxima_summary, _covariate_summary)
from gbc.evt_inference import (fit_stationary_gev, fit_ns_gev, ns_params_at, hill,
                               gev_mcmc, return_level_ci_delta, gev_nll,
                               train_functional_iqn, gbc_return_level_posterior,
                               gbc_crps_coverage_loyo)
from gbc.iqn import train_iqn, predict_iqn, sample_iqn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_scale import train_amortised, gbc_predict_samples
from gbc.metrics import crps_samples
from gbc.shrinkage import horseshoe_posterior, shrinkage_factor

STAMP = date.today().isoformat()
T1950, T2023 = -0.227, 1.100
_LEDGER = os.path.join(ROOT, "results", "numbers.txt")


def reset_ledger():
    """Drop this script's previous rows so a re-run replaces rather than duplicates."""
    if not os.path.exists(_LEDGER):
        return
    with open(_LEDGER) as fh:
        keep = [ln for ln in fh if "\trun_pnw.py\t" not in ln]
    with open(_LEDGER, "w") as fh:
        fh.writelines(keep)


def ledger(value, ident, fmt="{:.3f}"):
    os.makedirs(os.path.dirname(_LEDGER), exist_ok=True)
    with open(_LEDGER, "a") as fh:
        fh.write(f"{fmt.format(value)}\trun_pnw.py\t{ident}\t{STAMP}\n")
    return value


def load():
    d = pd.read_csv(os.path.join(ROOT, "data", "pnw_jja_maxima.csv"), index_col="year")
    h = pd.read_csv(os.path.join(ROOT, "data", "hadcrut5_annual.csv"), index_col="year")["anomaly"]
    return d, h


def aligned(d, h, station):
    x = d[station].dropna()
    T = h.reindex(x.index).values
    return x.values.astype(float), T, x.index.values


# ── plug-in predictive LOYO (CRPS, coverage) for classical methods ──────────

def loyo_plugin(x, T, params_fn, B=1500, alpha=0.10, seed=0):
    """params_fn(x_train, T_train, T_heldout) -> (mu,sigma,xi) for the held-out predictive."""
    rng = np.random.default_rng(seed)
    crps, hits = [], 0
    for i in range(len(x)):
        xtr = np.delete(x, i); Ttr = np.delete(T, i)
        mu, sigma, xi = params_fn(xtr, Ttr, T[i])
        samp = gev_quantile(rng.uniform(size=B), mu, sigma, xi)
        crps.append(crps_samples(np.array([x[i]]), samp.reshape(-1, 1)))
        lo, hi = np.quantile(samp, [alpha / 2, 1 - alpha / 2])
        hits += int(lo <= x[i] <= hi)
    return float(np.mean(crps)), hits / len(x)


def loyo_amortised(x, trained, alpha=0.10, B=1000):
    """Leave-one-year-out predictive CRPS and coverage, with NO leakage.

    `trained` is an amortised network fitted to standardised prior-predictive
    draws only, so no observation from this series ever entered training. For
    each fold the held-out year is removed BEFORE the series is standardised and
    summarised, so it touches neither the network nor the preprocessing. This is
    the like-for-like counterpart of the per-fold refits used by the classical
    baselines below.
    """
    x = np.asarray(x, float)
    crps_list, hits = [], 0
    lo_t, hi_t = alpha / 2, 1 - alpha / 2
    for i in range(len(x)):
        xho = np.delete(x, i)
        samp = gbc_predict_samples(trained, xho, B=B)
        crps_list.append(crps_samples(np.array([x[i]]), samp.reshape(-1, 1)))
        lo, hi = np.quantile(samp, [lo_t, hi_t])
        hits += int(lo <= x[i] <= hi)
    n = len(x)
    cov = hits / n
    return {"crps": float(np.mean(crps_list)), "coverage": cov,
            "coverage_se": float(np.sqrt(cov * (1 - cov) / n)), "n_folds": n}


def loyo_mcmc(x, B=1500, alpha=0.10, seed=0):
    rng = np.random.default_rng(seed)
    crps, hits = [], 0
    for i in range(len(x)):
        xtr = np.delete(x, i)
        sm = gev_mcmc(xtr, n_iter=4000, burn=1500, thin=3, seed=i)
        idx = rng.integers(0, len(sm), size=B)
        samp = np.array([gev_quantile(rng.uniform(), *sm[j]) for j in idx])
        crps.append(crps_samples(np.array([x[i]]), samp.reshape(-1, 1)))
        lo, hi = np.quantile(samp, [alpha / 2, 1 - alpha / 2])
        hits += int(lo <= x[i] <= hi)
    return float(np.mean(crps)), hits / len(x)


# ── non-stationary GBC functional IQN for z_N at a covariate state ───────────

def gbc_rl_detrended(x, T, T_star, mu1, n_sim=20000, epochs=1500, seed=7):
    """GBC return level at climate state T_star: shift the maxima to that state
    via the estimated trend, then apply the validated stationary GBC functional."""
    x_shift = x + mu1 * (T_star - T)
    trained = train_functional_iqn(x_shift, N=100, n_sim=n_sim, epochs=epochs, seed=seed)
    r = gbc_return_level_posterior(trained, x_shift)
    return r["zN"], r["lo"], r["hi"]


def fit_gev_fixed_xi(x, xi):
    from scipy.optimize import minimize
    nll = lambda ms: gev_nll(np.array([ms[0], np.exp(ms[1]), xi]), x)
    r = minimize(nll, [float(np.mean(x)), float(np.log(np.std(x, ddof=1)))], method="Nelder-Mead")
    return r.x[0], float(np.exp(r.x[1]))


def gbc_hs_rl(x, T, T_star, mu1, xi_samples, alpha=0.10, B=400, seed=3):
    """GBC + Horseshoe return level: bootstrap (mu,sigma) from the climate-shifted
    maxima while drawing the tail shape xi from the spatially-pooled horseshoe
    posterior (tighter than a single-station fit). Both sources of uncertainty
    propagate into the interval."""
    rng = np.random.default_rng(seed)
    x_shift = x + mu1 * (T_star - T)
    z = []
    for _ in range(B):
        xb = rng.choice(x_shift, size=len(x_shift), replace=True)
        xi = float(xi_samples[rng.integers(0, len(xi_samples))])
        mu, sigma = fit_gev_fixed_xi(xb, xi)
        z.append(gev_return_level(100, mu, sigma, xi))
    z = np.array(z)
    return float(np.median(z)), float(np.quantile(z, alpha / 2)), float(np.quantile(z, 1 - alpha / 2))


def xi_se(x):
    """xi MLE and its standard error from the numerical GEV Hessian."""
    f = fit_stationary_gev(x)
    p = np.array([f["mu"], f["sigma"], f["xi"]])
    h = 1e-4 * (np.abs(p) + 1)
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            pp = p.copy(); pp[i] += h[i]; pp[j] += h[j]
            pm = p.copy(); pm[i] += h[i]; pm[j] -= h[j]
            mp = p.copy(); mp[i] -= h[i]; mp[j] += h[j]
            mm = p.copy(); mm[i] -= h[i]; mm[j] -= h[j]
            H[i, j] = (gev_nll(pp, x) - gev_nll(pm, x) - gev_nll(mp, x) + gev_nll(mm, x)) / (4 * h[i] * h[j])
    cov = np.linalg.pinv(H)
    return f["xi"], float(np.sqrt(max(cov[2, 2], 1e-8)))


def main():
    reset_ledger()
    d, h = load()
    x, T, years = aligned(d, h, "SEA")
    print(f"SEA n={len(x)} ({years[0]}-{years[-1]}), max={x.max():.1f}")

    rows = {}

    # 1. Stationary GEV MLE
    sci = return_level_ci_delta(x, 100)
    c, cov = loyo_plugin(x, T, lambda xt, Tt, Th: tuple(fit_stationary_gev(xt)[k] for k in ("mu", "sigma", "xi")))
    rows["Stationary GEV MLE"] = [sci["zN"], sci["zN"], sci["hi"] - sci["lo"], cov, c]

    # 2. NS GEV MLE  (RL at 1950 vs 2023 climate)
    nsf = fit_ns_gev(x, T)
    rl50 = gev_return_level(100, *ns_params_at(nsf, T1950))
    rl23 = gev_return_level(100, *ns_params_at(nsf, T2023))
    # parametric-bootstrap CI width for z100 at T2023
    rng = np.random.default_rng(1)
    boot = []
    m23, s23, xi23 = ns_params_at(nsf, T2023)
    for _ in range(300):
        xb = np.array([gev_quantile(rng.uniform(), *ns_params_at(nsf, T[k])) for k in range(len(x))])
        fb = fit_ns_gev(xb, T)
        boot.append(gev_return_level(100, *ns_params_at(fb, T2023)))
    ciw_ns = float(np.quantile(boot, 0.95) - np.quantile(boot, 0.05))
    c, cov = loyo_plugin(x, T, lambda xt, Tt, Th: ns_params_at(fit_ns_gev(xt, Tt), Th))
    rows["NS GEV MLE"] = [rl50, rl23, ciw_ns, cov, c]

    # 3. Bayes GEV MCMC (stationary)
    sm = gev_mcmc(x, seed=2)
    z = np.array([gev_return_level(100, *p) for p in sm])
    c, cov = loyo_mcmc(x)
    rows["Bayes GEV MCMC"] = [float(np.median(z)), float(np.median(z)),
                              float(np.quantile(z, .95) - np.quantile(z, .05)), cov, c]

    # 4. Hill (Weissman high-quantile RL); no CI/coverage/CRPS
    k = max(8, len(x) // 4)
    g = hill(x, k)
    xs = np.sort(x)
    rl_hill = xs[len(x) - k] * (k / (len(x) / 100.0)) ** g
    rows["Hill Estimator"] = [rl_hill, rl_hill, None, None, None]

    # spatial horseshoe (needed for tab:spatial and the GBC+HS prior) -------
    stations = ["SEA", "PDX", "GEG", "EUG", "PDT"]
    xis, ses, trends = {}, {}, {}
    for s in stations:
        xs_, Ts_, _ = aligned(d, h, s)
        xi_hat, se = xi_se(xs_)
        xis[s], ses[s] = xi_hat, se
        fns = fit_ns_gev(xs_, Ts_)
        # trend in degC/decade: mu1 * (HadCRUT5 change per decade over the record)
        dT_decade = (Ts_[-1] - Ts_[0]) / ((len(xs_) - 1) / 10.0)
        trends[s] = fns["mu1"] * dT_decade
    mle = np.array([xis[s] for s in stations])
    se = np.array([ses[s] for s in stations])
    hs = horseshoe_posterior(mle, se, global_tau=float(np.std(mle)), n_iter=4000, seed=0)
    hs_mean, hs_lo, hs_hi = hs["theta_mean"], hs["theta_lower"], hs["theta_upper"]
    kappa = hs["kappa_mean"]
    hs_samples = hs["theta_samples"]

    # 5. GBC-QNN (non-stationary via detrend-then-stationary-GBC)
    mu1_sea = nsf["mu1"]
    g50 = gbc_rl_detrended(x, T, T1950, mu1_sea, seed=7)
    g23 = gbc_rl_detrended(x, T, T2023, mu1_sea, seed=17)
    pred_net = train_amortised("draw", n_sim=60000, epochs=2000, seed=11)
    cc = loyo_amortised(x, pred_net)
    ledger(cc["coverage_se"], "rl:gbc:cov_se")
    rows["GBC-QNN (ours)"] = [g50[0], g23[0], g23[2] - g23[1], cc["coverage"], cc["crps"]]

    # 6. GBC + Horseshoe: propagate the spatially-pooled xi posterior for SEA
    sea_i = stations.index("SEA")
    h50 = gbc_hs_rl(x, T, T1950, mu1_sea, hs_samples[:, sea_i])
    h23 = gbc_hs_rl(x, T, T2023, mu1_sea, hs_samples[:, sea_i])
    # gbc_hs_rl is a nonparametric bootstrap of the climate-shifted maxima with xi
    # drawn from the pooled horseshoe posterior. It never evaluates the predictive
    # network and yields an interval rather than a predictive distribution, so LOYO
    # coverage/CRPS are undefined for it. Leave them blank.
    rows["GBC + Horseshoe"] = [h50[0], h23[0], h23[2] - h23[1], None, None]

    # ---- emit tab:rl ----
    print("\n=== tab:rl (Seattle 100-yr return levels) ===")
    order = ["Stationary GEV MLE", "NS GEV MLE", "Bayes GEV MCMC", "Hill Estimator",
             "GBC-QNN (ours)", "GBC + Horseshoe"]
    lines = []
    for m in order:
        rl50, rl23, ciw, cvg, crps = rows[m]
        def f(v, p="{:.1f}"):
            return "n/a" if v is None else p.format(v)
        print(f"{m:20s} RL50={f(rl50)} RL23={f(rl23)} CIw={f(ciw)} cov={f(cvg,'{:.2f}')} crps={f(crps,'{:.2f}')}")
        lines.append(f"{m} & {f(rl50)} & {f(rl23)} & {f(ciw)} & {f(cvg,'{:.2f}')} & {f(crps,'{:.2f}')} \\\\")
        tag = m.split()[0].lower()
        for v, nm in [(rl50, "rl1950"), (rl23, "rl2023"), (ciw, "ciw"), (cvg, "cov"), (crps, "crps")]:
            if v is not None:
                ledger(v, f"rl:{tag}:{nm}")
    with open(os.path.join(ROOT, "tab", "rl.tex"), "w") as fh:
        fh.write("% generated by experiments/run_pnw.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{lccccc}\n\\toprule\nMethod & RL 1950 & RL 2023 & CI Width & Coverage & CRPS \\\\\n\\midrule\n")
        fh.write("\n".join(lines))
        fh.write("\n\\bottomrule\n\\end{tabular}\n")

    # ---- emit tab:spatial ----
    print("\n=== tab:spatial (horseshoe shrinkage of xi) ===")
    slines = []
    for i, s in enumerate(stations):
        print(f"{s}: xi_mle={mle[i]:+.3f} xi_hs={hs_mean[i]:+.3f} CI[{hs_lo[i]:+.2f},{hs_hi[i]:+.2f}] kappa={kappa[i]:.2f} trend={trends[s]:+.2f}")
        slines.append(f"{s} & {mle[i]:+.2f} & {hs_mean[i]:+.2f} & $[{hs_lo[i]:+.2f},{hs_hi[i]:+.2f}]$ & {kappa[i]:.2f} & {trends[s]:+.2f} \\\\")
        for v, nm in [(mle[i], "ximle"), (hs_mean[i], "xihs"), (kappa[i], "kappa"), (trends[s], "trend")]:
            ledger(v, f"sp:{s}:{nm}", "{:+.3f}")
    with open(os.path.join(ROOT, "tab", "spatial.tex"), "w") as fh:
        fh.write("% generated by experiments/run_pnw.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{lccccc}\n\\toprule\nStation & $\\xi$ (MLE) & $\\xi$ (HS) & 90\\% CI & $\\kappa_s$ & Trend \\\\\n\\midrule\n")
        fh.write("\n".join(slines))
        fh.write("\n\\bottomrule\n\\end{tabular}\n")
    print("\nwrote tab/rl.tex, tab/spatial.tex, results/numbers.txt")


if __name__ == "__main__":
    main()
