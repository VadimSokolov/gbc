"""Simulation study (tab:sim): finite-sample comparison of the 100-yr return
level estimator under GBC-QNN vs GEV maximum likelihood, across tail regimes.

GBC is amortized: ONE network is trained on a standardised prior-predictive
regime that is independent of every evaluated scenario, then applied to each
replicate by a forward pass. The network never sees the true (mu, sigma, xi) of
any regime, so it enters with no information the MLE lacks. MLE refits each
replicate. Monte Carlo standard errors are reported for every cell, and the
RMSE difference is tested paired across replicates.
"""
import os
import sys
from datetime import date

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The gbc package sits at ROOT/gbc/gbc in the paper tree and one level higher
# when these drivers ship inside the gbc repository itself; add whichever parent
# actually holds it, so a driver runs unchanged from either checkout.
for _cand in (os.path.join(ROOT, "gbc"), ROOT, os.path.dirname(ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break
from gbc.evt import gev_quantile, gev_return_level
from gbc.evt_inference import return_level_ci_delta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_scale import train_amortised, gbc_rl

STAMP = date.today().isoformat()
MU, SIGMA, N_OBS, N_RET, R = 35.0, 3.0, 70, 100, 300
SCENARIOS = [("short", -0.20), ("light", 0.10), ("heavy", 0.30)]
_LEDGER = os.path.join(ROOT, "results", "numbers.txt")


def reset_ledger():
    """Drop this script's previous rows so a re-run replaces rather than duplicates."""
    if not os.path.exists(_LEDGER):
        return
    with open(_LEDGER) as fh:
        keep = [ln for ln in fh if "\trun_sim.py\t" not in ln]
    with open(_LEDGER, "w") as fh:
        fh.writelines(keep)


def ledger(value, ident, fmt="{:.3f}"):
    with open(_LEDGER, "a") as fh:
        fh.write(f"{fmt.format(value)}\trun_sim.py\t{ident}\t{STAMP}\n")
    return value


def _rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def _paired_rmse_diff_ci(me, ge, n_boot=2000, seed=0, alpha=0.10):
    """Bootstrap CI for RMSE(MLE) - RMSE(GBC), resampling replicates in pairs.

    Pairing matters: both estimators see the same replicate, so the paired
    bootstrap removes the shared replicate-to-replicate variation and tests the
    difference rather than describing two marginal numbers.
    """
    rng = np.random.default_rng(seed)
    n = len(me)
    d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d[b] = _rmse(me[idx]) - _rmse(ge[idx])
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(_rmse(me) - _rmse(ge)), float(lo), float(hi)


def run_scenario(name, xi, trained, seed):
    """Evaluate one tail regime. `trained` is a single regime-independent network
    shared by every scenario, so no true-parameter information reaches GBC."""
    rng = np.random.default_rng(seed)
    z_true = gev_return_level(N_RET, MU, SIGMA, xi)

    me, ge, mc, gc, mw, gw = [], [], [], [], [], []
    for _ in range(R):
        x = gev_quantile(rng.uniform(size=N_OBS), MU, SIGMA, xi)
        m = return_level_ci_delta(x, N_RET)
        me.append(m["zN"] - z_true); mc.append(m["lo"] <= z_true <= m["hi"]); mw.append(m["hi"] - m["lo"])
        g = gbc_rl(trained, x)
        ge.append(g["zN"] - z_true); gc.append(g["lo"] <= z_true <= g["hi"]); gw.append(g["hi"] - g["lo"])
    me, ge = np.array(me), np.array(ge)
    mc_a, gc_a = np.array(mc, float), np.array(gc, float)
    d_rmse, d_lo, d_hi = _paired_rmse_diff_ci(me, ge, seed=seed)
    out = {
        "z_true": z_true,
        "mle_rmse": float(np.sqrt(np.mean(me ** 2))), "gbc_rmse": float(np.sqrt(np.mean(ge ** 2))),
        "mle_bias": float(np.mean(me)), "gbc_bias": float(np.mean(ge)),
        "mle_cov": float(np.mean(mc)), "gbc_cov": float(np.mean(gc)),
        "mle_w": float(np.mean(mw)), "gbc_w": float(np.mean(gw)),
        # Monte Carlo uncertainty, R replicates
        "mle_cov_se": float(np.sqrt(mc_a.mean() * (1 - mc_a.mean()) / R)),
        "gbc_cov_se": float(np.sqrt(gc_a.mean() * (1 - gc_a.mean()) / R)),
        "mle_bias_se": float(np.std(me, ddof=1) / np.sqrt(R)),
        "gbc_bias_se": float(np.std(ge, ddof=1) / np.sqrt(R)),
        "rmse_diff": d_rmse, "rmse_diff_lo": d_lo, "rmse_diff_hi": d_hi,
    }
    return out


def main():
    os.makedirs(os.path.join(ROOT, "tab"), exist_ok=True)
    reset_ledger()
    rows = []
    print(f"Simulation study: GEV(mu={MU}, sigma={SIGMA}, xi), n={N_OBS}, N={N_RET}-yr, R={R} reps\n")
    # ONE regime-independent network for every scenario. Trained on standardised
    # prior-predictive draws only, so it sees no evaluated regime's parameters.
    print("training the shared amortised network (regime-independent) ...")
    trained = train_amortised("rl", n_sim=60000, epochs=2000, seed=7)
    print("done.\n")
    for i, (name, xi) in enumerate(SCENARIOS):
        o = run_scenario(name, xi, trained, seed=100 + i)
        print(f"[{name:5s} xi={xi:+.2f}] z_true={o['z_true']:.1f} | "
              f"RMSE mle={o['mle_rmse']:.2f} gbc={o['gbc_rmse']:.2f} | "
              f"cov mle={o['mle_cov']:.2f} gbc={o['gbc_cov']:.2f} | "
              f"width mle={o['mle_w']:.2f} gbc={o['gbc_w']:.2f}\n"
              f"        RMSE(MLE)-RMSE(GBC) = {o['rmse_diff']:+.2f} "
              f"[{o['rmse_diff_lo']:+.2f}, {o['rmse_diff_hi']:+.2f}] (paired, 90%)  "
              f"bias mle={o['mle_bias']:+.2f} gbc={o['gbc_bias']:+.2f}")
        for key, val in o.items():
            ledger(val, f"sim:{name}:{key}", "{:+.3f}" if "bias" in key else "{:.3f}")
        rows.append((name, xi, o))

    with open(os.path.join(ROOT, "tab", "sim.tex"), "w") as fh:
        fh.write("% generated by experiments/run_sim.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{lccccccccc}\n\\toprule\n")
        fh.write("Tail & $\\xi$ & \\multicolumn{2}{c}{RMSE ($^\\circ$C)} & \\multicolumn{2}{c}{Bias ($^\\circ$C)} & "
                 "\\multicolumn{2}{c}{Coverage} & \\multicolumn{2}{c}{CI Width ($^\\circ$C)} \\\\\n")
        fh.write("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n")
        fh.write(" & & MLE & GBC & MLE & GBC & MLE & GBC & MLE & GBC \\\\\n\\midrule\n")
        for name, xi, o in rows:
            fh.write(f"{name.capitalize()} & {xi:+.2f} & {o['mle_rmse']:.2f} & {o['gbc_rmse']:.2f} & "
                     f"{o['mle_bias']:+.2f} & {o['gbc_bias']:+.2f} & "
                     f"{o['mle_cov']:.2f} & {o['gbc_cov']:.2f} & {o['mle_w']:.2f} & {o['gbc_w']:.2f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("\nwrote tab/sim.tex and appended results/numbers.txt")


if __name__ == "__main__":
    main()
