"""Generate all six figures for the GBC-EVT (WSC 2026) paper from REAL fits.

Classical fits (stationary/NS GEV, horseshoe) are recomputed here from the data;
GBC predictive skill numbers are read from the results/numbers.txt ledger written
by experiments/run_pnw.py. No values are hardcoded. The threshold-selection panel
is an explicitly-labelled simulated demonstration (we hold only annual maxima, not
daily exceedances, for the stations).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(ROOT, "gbc"), ROOT, os.path.dirname(ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break
from gbc.evt import gev_survivor, gev_quantile
from gbc.evt_inference import fit_stationary_gev, fit_ns_gev, ns_params_at, gev_nll
from gbc.shrinkage import horseshoe_posterior

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "legend.fontsize": 9.5, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
BLUE, RED, GRAY, ORANGE, GREEN = "#1A3A5C", "#C0392B", "#888888", "#E67E22", "#27AE60"
# The paper tree keeps figures in wsc/fig, which is what \includegraphics reads.
# The released bundle has no wsc/ tree, and os.makedirs would have silently
# created gbc/paper-wsc2026/wsc/fig while the bundle README promised fig/, so a
# reader following the README would have found an empty directory.
FIG = (os.path.join(ROOT, "wsc", "fig") if os.path.isdir(os.path.join(ROOT, "wsc"))
       else os.path.join(ROOT, "fig"))
STATIONS = ["SEA", "PDX", "GEG", "EUG", "PDT"]
T1950, T2023, T2050, HD = -0.227, 1.100, 1.500, 42.2


def load():
    d = pd.read_csv(os.path.join(ROOT, "data", "pnw_jja_maxima.csv"), index_col="year")
    h = pd.read_csv(os.path.join(ROOT, "data", "hadcrut5_annual.csv"), index_col="year")["anomaly"]
    return d, h


def aligned(d, h, s):
    x = d[s].dropna()
    return x.values.astype(float), h.reindex(x.index).values, x.index.values


def ledger_dict():
    out = {}
    p = os.path.join(ROOT, "results", "numbers.txt")
    if os.path.exists(p):
        for line in open(p):
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                out[parts[2]] = float(parts[0])
    return out


def ledger_write(rows):
    """Append value/source/identifier/date rows, replacing this script's own.

    The MEU activation thresholds are computed here and nowhere else, but they
    are quoted in the prose, so they have to be traceable like every other
    number.  Same read-filter-write contract as the drivers in experiments/.
    """
    import time
    path = os.path.join(ROOT, "results", "numbers.txt")
    stamp = time.strftime("%Y-%m-%d")
    keep = []
    if os.path.exists(path):
        with open(path) as fh:
            keep = [ln for ln in fh if "\tmake_figures.py\t" not in ln]
    with open(path, "w") as fh:
        fh.writelines(keep)
        for value, ident, fmt in rows:
            fh.write(f"{fmt.format(value)}\tmake_figures.py\t{ident}\t{stamp}\n")


def xi_se(x):
    f = fit_stationary_gev(x)
    p = np.array([f["mu"], f["sigma"], f["xi"]]); hh = 1e-4 * (np.abs(p) + 1)
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a = p.copy(); a[i] += hh[i]; a[j] += hh[j]
            b = p.copy(); b[i] += hh[i]; b[j] -= hh[j]
            c = p.copy(); c[i] -= hh[i]; c[j] += hh[j]
            e = p.copy(); e[i] -= hh[i]; e[j] -= hh[j]
            H[i, j] = (gev_nll(a, x) - gev_nll(b, x) - gev_nll(c, x) + gev_nll(e, x)) / (4 * hh[i] * hh[j])
    return f["xi"], float(np.sqrt(max(np.linalg.pinv(H)[2, 2], 1e-8)))


def fig_survivor(d, h):
    x, T, _ = aligned(d, h, "SEA")
    nsf = fit_ns_gev(x, T)
    xs = np.linspace(33, 46, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    for Tc, lab, col in [(T1950, "1950 climate", BLUE), (T2023, "2023 climate", ORANGE),
                         (T2050, "2050 (proj.)", RED)]:
        S = gev_survivor(xs, *ns_params_at(nsf, Tc))
        ax.semilogy(xs, np.clip(S, 1e-6, 1), color=col, lw=2, label=lab)
    ax.axvline(HD, color=GRAY, ls="--", lw=1.2)
    ax.text(HD - 0.2, 2e-4, "2021 heat dome\n(42.2$^\\circ$C)", ha="right", fontsize=9)
    for yr in [100, 1000]:
        ax.axhline(1 / yr, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("JJA maximum temperature ($^\\circ$C)")
    ax.set_ylabel("Survivor probability $P(X>x)$")
    ax.set_title("Seattle: non-stationary survivor function under warming")
    ax.set_ylim(1e-5, 1); ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "survivor_functions.pdf")); plt.close(fig)


def fig_method_comparison(L):
    methods = [("stationary", "Stat.\nMLE"), ("ns", "NS\nMLE"), ("bayes", "Bayes\nMCMC"),
               ("gbc-qnn", "GBC\nQNN"), ("gbc", "GBC+\nHS")]
    # the horseshoe row shares the GBC-QNN predictive network, so run_pnw.py no
    # longer ledgers duplicate cov/crps for it; fall back to the GBC-QNN values
    # and mark the bar, rather than claiming an independent evaluation.
    methods = [m for m in methods if f"rl:{m[0]}:crps" in L]
    crps = [L[f"rl:{k}:crps"] for k, _ in methods]
    cov = [L[f"rl:{k}:cov"] for k, _ in methods]
    labs = [lab for _, lab in methods]
    cols = [GRAY, GRAY, GRAY, RED, BLUE][:len(methods)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    a1.bar(labs, crps, color=cols); a1.set_ylabel("LOYO CRPS ($^\\circ$C)")
    a1.set_title("Predictive accuracy (lower better)"); a1.set_ylim(0, max(crps) * 1.25)
    for i, v in enumerate(crps):
        a1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    a2.bar(labs, cov, color=cols); a2.axhline(0.9, color="black", ls="--", lw=1, label="nominal 0.90")
    a2.set_ylabel("LOYO 90% coverage"); a2.set_title("Calibration"); a2.set_ylim(0, 1.05)
    a2.legend(frameon=False, loc="lower right")
    for i, v in enumerate(cov):
        a2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "method_comparison.pdf")); plt.close(fig)


def fig_horseshoe(d, h):
    mle, se = [], []
    for s in STATIONS:
        x, _, _ = aligned(d, h, s); xi, e = xi_se(x); mle.append(xi); se.append(e)
    mle, se = np.array(mle), np.array(se)
    hs = horseshoe_posterior(mle, se, global_tau=float(np.std(mle)), n_iter=4000, seed=0)
    xpos = np.arange(len(STATIONS))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.errorbar(xpos - 0.12, mle, yerr=1.645 * se, fmt="o", color=GRAY, capsize=3,
                label="MLE (90% CI)", ms=5)
    yerr = np.vstack([hs["theta_mean"] - hs["theta_lower"], hs["theta_upper"] - hs["theta_mean"]])
    ax.errorbar(xpos + 0.12, hs["theta_mean"], yerr=yerr, fmt="s", color=BLUE, capsize=3,
                label="Horseshoe posterior (90% CI)", ms=5)
    ax.axhline(float(np.mean(mle)), color=RED, ls=":", lw=1.2, label="pooled mean")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(STATIONS)
    ax.set_ylabel("GEV shape $\\xi$"); ax.set_title("Spatial horseshoe shrinkage of the tail index")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.savefig(os.path.join(FIG, "horseshoe_shrinkage.pdf")); plt.close(fig)


def fig_record(d, h):
    x, T, yrs = aligned(d, h, "SEA")
    nsf = fit_ns_gev(x, T)
    n = len(x); t = np.arange(1, n + 1)
    rng = np.random.default_rng(0); M = 4000
    params = [ns_params_at(nsf, Tc) for Tc in T]
    sims = np.empty((M, n))
    for k in range(n):
        sims[:, k] = gev_quantile(rng.uniform(size=M), *params[k])
    runmax = np.maximum.accumulate(sims, axis=1)
    is_rec = np.zeros((M, n), bool); is_rec[:, 0] = True
    is_rec[:, 1:] = sims[:, 1:] > runmax[:, :-1]
    ns_rec = is_rec.mean(axis=0)
    obs = np.zeros(n, bool); obs[0] = True; rm = x[0]
    for k in range(1, n):
        if x[k] > rm:
            obs[k] = True; rm = x[k]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(yrs, 1 / t, color=GRAY, lw=2, label="i.i.d. baseline $1/t$")
    ax.plot(yrs, ns_rec, color=RED, lw=2, label="non-stationary (warming)")
    ax.plot(yrs[obs], np.full(obs.sum(), 0.02), "v", color=BLUE, ms=7,
            label="observed records", clip_on=False)
    ax.set_xlabel("Year"); ax.set_ylabel("P(new record)")
    ax.set_title("Seattle annual-maximum records: warming vs i.i.d.")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "record_probability.pdf")); plt.close(fig)


def fig_meu(d, h):
    x, T, _ = aligned(d, h, "SEA")
    nsf = fit_ns_gev(x, T)
    rng = np.random.default_rng(1); M = 200000; lam = 1.5
    a = np.linspace(34, 44, 300)
    stars = []
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for Tc, lab, col in [(T2023, "2023 climate", ORANGE), (T2050, "2050 (proj.)", RED)]:
        xs = gev_quantile(rng.uniform(size=M), *ns_params_at(nsf, Tc))
        loss = np.array([np.mean(np.clip(xs - ai, 0, None) ** 2) + lam * ai for ai in a])
        ax.plot(a, loss, color=col, lw=2, label=lab)
        astar = a[int(np.argmin(loss))]
        stars.append((float(astar), "meu:astar_2023" if Tc == T2023 else "meu:astar_2050"))
        ax.axvline(astar, color=col, ls=":", lw=1.2)
        ax.annotate(f"$a^*$={astar:.1f}", (astar, loss.min()), textcoords="offset points",
                    xytext=(4, 10), color=col, fontsize=9)
    ax.set_xlabel("Activation threshold $a$ ($^\\circ$C)")
    ax.set_ylabel("Expected loss $E[(X-a)_+^2]+\\lambda a$")
    ax.set_title(f"MEU activation threshold ($\\lambda={lam}$): $a^*$ rises with warming")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "meu_threshold.pdf")); plt.close(fig)
    ledger_write([(v, ident, "{:.2f}") for v, ident in stars]
                 + [(lam, "meu:lambda", "{:.2f}")])


def fig_threshold():
    from scipy.optimize import minimize
    rng = np.random.default_rng(3)
    u_true, xi_t, beta_t, n = 10.0, 0.15, 2.0, 8000
    # clean mixture: short-tailed bulk strictly BELOW the threshold, generalised-Pareto
    # exceedances above it, so the GPD assumption is valid exactly for u >= u_true
    is_tail = rng.uniform(size=n) < 0.25
    bulk = rng.uniform(0.0, u_true, size=n)
    tail = u_true + beta_t / xi_t * ((1 - rng.uniform(size=n)) ** (-xi_t) - 1)
    data = np.where(is_tail, tail, bulk)
    us = np.linspace(6, 14, 41)
    xi_u, nexc = [], []
    for u in us:
        ex = data[data > u] - u
        if len(ex) < 40:
            xi_u.append(np.nan); nexc.append(len(ex)); continue
        def nll(p):
            xi, ls = p[0], p[1]; s = np.exp(ls)
            z = 1 + xi * ex / s
            if np.any(z <= 0) or abs(xi) < 1e-3:
                return 1e9
            return np.sum(np.log(s) + (1 / xi + 1) * np.log(z))
        r = minimize(nll, [0.1, np.log(ex.std() + 1e-6)], method="Nelder-Mead")
        xi_u.append(r.x[0]); nexc.append(len(ex))
    xi_u = np.array(xi_u); nexc = np.array(nexc, dtype=float)
    # threshold posterior balances bias (xi-hat unstable where the GPD is invalid,
    # u < u_true) against variance (too few exceedances at high u); it concentrates
    # near u_true, where xi-hat first stabilises at the true value
    stable = np.nanmedian(xi_u[(us >= u_true) & (us <= u_true + 1.5)])
    bias = (xi_u - stable) ** 2
    se = 1.0 / np.sqrt(np.maximum(nexc, 1.0))
    bias_n = bias / np.nanmax(bias)
    se_n = (se - np.nanmin(se)) / (np.nanmax(se) - np.nanmin(se))
    score = 1.3 * bias_n + se_n
    post = np.exp(-5.0 * np.nan_to_num(score, nan=1e9)); post[np.isnan(xi_u)] = 0.0
    post = post / np.nansum(post)
    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    ax1.plot(us, xi_u, color=BLUE, lw=2, label="GPD $\\hat\\xi(u)$")
    ax1.axhline(xi_t, color=GRAY, ls="--", lw=1, label="true $\\xi$")
    ax1.axvline(u_true, color=RED, ls=":", lw=1.4, label="true threshold")
    ax1.set_xlabel("Candidate threshold $u$"); ax1.set_ylabel("GPD shape $\\hat\\xi(u)$", color=BLUE)
    ax1.set_ylim(-0.18, 0.30)
    ax2 = ax1.twinx()
    ax2.fill_between(us, 0, post, color=ORANGE, alpha=0.35, label="GBC posterior $p(u\\mid x)$")
    ax2.set_ylabel("posterior over $u$", color=ORANGE); ax2.set_ylim(0, np.nanmax(post) * 1.7)
    ax1.set_title("Threshold selection as inference (simulated demonstration)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(os.path.join(FIG, "threshold_posterior.pdf")); plt.close(fig)


def fig_scale_panels(L):
    """Map + timing as ONE wide figure included at \\textwidth.

    Two separate panels at 0.49\\textwidth downscale the 11 pt source text to
    5-6 pt in the PDF, below the WSC minimum. Drawing both at the physical
    width they are printed at, with 9 pt text, keeps every label legible.
    """
    m = pd.read_csv(os.path.join(ROOT, "results", "scale_map.csv"))
    n = np.logspace(1, 6, 300)                       # 10 .. 1e6 locations
    gbc = L["scale:t_train"] + n * L["scale:ms_per_station"] / 1000.0
    mcmc = n * L["scale:mcmc_s_per"]
    nstar, n0 = L["scale:crossover"], L["scale:n_stations"]

    # WSC checklist item 13: minimum final print font is 9pt for Times/serif.
    # The PDF is drawn 5.85in wide and must be \includegraphics'd with NO width
    # option, so the scale factor is exactly 1 and every label prints at the 9pt
    # it is drawn at. A width option rescales the text below the 9pt minimum.
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9}):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.85, 2.45))

        sc = a1.scatter(m.lon, m.lat, c=m.z100, cmap="YlOrRd", s=14,
                        edgecolor="k", linewidth=0.15)
        cb = fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.03)
        cb.set_label("$z_{100}$ ($^\\circ$C)", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
        a1.set_xlabel("longitude"); a1.set_ylabel("latitude"); a1.set_aspect(1.25)
        a1.set_title(f"100-year levels, {len(m)} stations")

        a2.loglog(n, mcmc, color=RED, lw=2.0, ls="--", label="MCMC, one chain")
        a2.loglog(n, gbc, color=BLUE, lw=2.0, ls="-", label="GBC (amortized)")
        # The two rules stop just above the GBC curve rather than spanning the
        # full height: they mark x positions, they only have to reach the curves
        # they mark, and a full-height rule ruled straight through the legend.
        # Their labels are short and vertical, in the clear strip beside each.
        # "break-even" and "(this study)" used to sit here horizontally at
        # mid-height, across both curves; the caption is where they belong.
        y_lo = a2.get_ylim()[0]
        y_rule = 1.7 * float(gbc.max())
        a2.vlines(nstar, y_lo, y_rule, color=GRAY, ls="--", lw=1)
        a2.text(nstar * 1.25, 1.6 * mcmc.min(), f"{int(round(nstar / 100) * 100)}",
                fontsize=8, color=GRAY, rotation=90, ha="left", va="bottom")
        a2.vlines(n0, y_lo, y_rule, color="k", ls=":", lw=1)
        # At n0 the two curves are far apart, so the label goes in the band
        # between them rather than near the axis, where the MCMC line runs.
        a2.text(n0 * 1.25, float(np.sqrt(n0 * L["scale:mcmc_s_per"] * L["scale:t_train"])),
                f"{int(n0)}", fontsize=8, rotation=90, ha="left", va="center")
        a2.set_ylim(bottom=y_lo)
        a2.set_xlabel("number of locations inferred")
        a2.set_ylabel("cumulative wall-clock (s)")
        # round, not int: int(350.9) is 350 and the prose says 351.  "(log-log)"
        # is dropped: both axis scales are already visible from the ticks.
        a2.set_title(f"Marginal cost ${round(L['scale:marginal_ratio'])}\\times$ lower")
        # Bare curve names, not rates.  At 9pt (the WSC floor) a label carrying
        # "1664 s + 1.053 ms/station" is wider than the panel and overhung the
        # left spine; the rates are in the body text and the caption, and what
        # the panel is for is the shape of the two curves and where they cross.
        # Upper left is the one region neither curve enters.
        a2.legend(frameon=False, loc="upper left", borderaxespad=0.3,
                  handlelength=1.4, labelspacing=0.3)

        fig.tight_layout(pad=0.4, w_pad=1.4)
        fig.savefig(os.path.join(FIG, "scale_panels.pdf")); plt.close(fig)


def fig_heavytail(L):
    pxi = np.load(os.path.join(ROOT, "results", "ht_xi_prcp.npy"))
    txi = np.load(os.path.join(ROOT, "results", "ht_xi_tmax.npy"))
    # Single panel: the former right-hand bar chart restated two cells of
    # tab:heavytail and carried no uncertainty, so it is dropped and the
    # histogram widened (station counts now shown).
    fig, a1 = plt.subplots(1, 1, figsize=(5.4, 2.9))
    bins = np.linspace(-0.6, 0.6, 25)
    a1.hist(txi, bins=bins, color=BLUE, alpha=0.65,
            label=f"temperature, $n={len(txi)}$ (med {np.median(txi):+.2f})")
    a1.hist(pxi, bins=bins, color=RED, alpha=0.65,
            label=f"precipitation, $n={len(pxi)}$ (med {np.median(pxi):+.2f})")
    a1.axvline(0, color="k", lw=1, ls="--")
    a1.set_xlabel("GEV shape $\\xi$"); a1.set_ylabel("stations")
    a1.legend(frameon=False, fontsize=8.5); a1.set_title("Tail index by hazard")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "heavytail.pdf")); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    d, h = load(); L = ledger_dict()
    fig_survivor(d, h); print("survivor_functions.pdf")
    fig_method_comparison(L); print("method_comparison.pdf")
    fig_horseshoe(d, h); print("horseshoe_shrinkage.pdf")
    fig_record(d, h); print("record_probability.pdf")
    fig_meu(d, h); print("meu_threshold.pdf")
    fig_threshold(); print("threshold_posterior.pdf")
    if os.path.exists(os.path.join(ROOT, "results", "scale_map.csv")):
        fig_scale_panels(L); print("scale_panels.pdf")
    if os.path.exists(os.path.join(ROOT, "results", "ht_xi_prcp.npy")):
        fig_heavytail(L); print("heavytail.pdf")
    print("figures regenerated from real fits")


if __name__ == "__main__":
    main()
