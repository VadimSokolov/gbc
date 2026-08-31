"""Training-budget selection inside the simulator, with no observed data involved.

The amortized networks were previously trained for a hand-set number of epochs,
and the notebook justified that choice by checking predictive quantiles against
the *observed* station record ("epochs>=1500 calibrates").  That check reads the
evaluation data, so the epoch count was, however weakly, tuned on it.

This driver replaces it.  Prior-predictive draws are split into a training and a
validation half; a network is trained at each candidate budget with its own
complete cosine schedule; and the budget is chosen by validation pinball loss on
held-out *simulated* pairs.  Because a GBC network is amortized and never sees a
station during training, a selection rule that touches only the simulator leaves
the observed-data evaluation untouched by construction: the outer loop over
stations is nested outside a selection that cannot see it.

Then the chosen budget is checked against the deployed one on the 112 CONUS
stations, so the effect on the published z_100 is measured rather than assumed.

    sbatch experiments/epochsel.slurm
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The gbc package sits at ROOT/gbc/gbc in the paper tree and one level higher
# when these drivers ship inside the gbc repository itself; add whichever parent
# actually holds it, so a driver runs unchanged from either checkout.
for _cand in (os.path.join(ROOT, "gbc"), ROOT, os.path.dirname(ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from gbc.iqn import IQN, predict_iqn                                # noqa: E402
from run_scale import _sim_standardised, gbc_rl                     # noqa: E402

STAMP = time.strftime("%Y-%m-%d")
BUDGETS = [250, 500, 1000, 2000, 3000, 4000]
DEPLOYED = 2000                     # what Section 5 currently trains for
SEEDS = [1, 2, 3]
N_SIM = 60000
VAL_FRAC = 0.25
TAU_GRID = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
MIN_YEARS = 50
N_RET = 100
_LEDGER = os.path.join(ROOT, "results", "numbers.txt")


def ledger(value, ident, fmt="{:.4g}"):
    with open(_LEDGER, "a") as fh:
        fh.write(f"{fmt.format(value)}\trun_epochsel.py\t{ident}\t{STAMP}\n")
    return value


def _pinball(y, q, tau):
    """Check-function loss rho_tau(y - q), the proper scoring rule for quantiles."""
    u = y - q
    return torch.mean(torch.maximum(tau * u, (tau - 1.0) * u))


def val_pinball(model, Xv, yv, xm, xs, ym, ys):
    """Mean pinball loss over TAU_GRID on held-out simulated pairs (original units)."""
    Xt = torch.tensor((Xv - xm) / xs, dtype=torch.float32)
    yt = torch.tensor((yv - ym) / ys, dtype=torch.float32)
    tot = 0.0
    with torch.no_grad():
        for tau in TAU_GRID:
            q = model(Xt, float(tau))[:, 1]
            tot += float(_pinball(yt, q, float(tau)))
    return ys * tot / len(TAU_GRID)          # rescale to the target's own units


def train_budget(Xtr, ytr, epochs, seed, hdim=256, nh=32, lr=1e-3, wd=1e-4,
                 w=(0.3, 0.3, 0.4)):
    """train_iqn with a complete cosine schedule for this budget (T_max=epochs)."""
    torch.manual_seed(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = float(ytr.mean()), float(ytr.std()) + 1e-8
    Xt = torch.tensor((Xtr - xm) / xs, dtype=torch.float32)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    model = IQN(Xtr.shape[1], hdim=hdim, nh=nh)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=lr * 0.01)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        model.loss_fn(Xt, yt, w).backward()
        opt.step()
        sched.step()
    model.eval()
    return model, xm, xs, ym, ys


def main():
    print(f"budget selection on simulated data only; seeds={SEEDS}, "
          f"budgets={BUDGETS}, n_sim={N_SIM}, val_frac={VAL_FRAC}")

    # ── simulated train/validation split; no station data anywhere here ──────
    X, y = _sim_standardised(N_SIM, "rl", np.random.default_rng(20260831))
    n_val = int(VAL_FRAC * len(y))
    Xv, yv, Xtr, ytr = X[:n_val], y[:n_val], X[n_val:], y[n_val:]
    print(f"train {len(ytr)}, validation {len(yv)}")

    rows = []
    for E in BUDGETS:
        for sd in SEEDS:
            t = time.perf_counter()
            model, xm, xs, ym, ys = train_budget(Xtr, ytr, E, sd)
            loss = val_pinball(model, Xv, yv, xm, xs, ym, ys)
            rows.append({"epochs": E, "seed": sd, "val_pinball": loss,
                         "train_s": time.perf_counter() - t})
            print(f"  epochs={E:5d} seed={sd}  val pinball={loss:.5f}  "
                  f"({rows[-1]['train_s']:.0f}s)")
    tab = pd.DataFrame(rows)
    agg = tab.groupby("epochs")["val_pinball"].agg(["mean", "std", "min"]).reset_index()
    print("\nvalidation pinball by budget (simulated hold-out):")
    print(agg.to_string(index=False))
    tab.to_csv(os.path.join(ROOT, "results", "epoch_selection.csv"), index=False)

    best = int(agg.loc[agg["mean"].idxmin(), "epochs"])
    best_loss = float(agg["mean"].min())
    dep_loss = float(agg.loc[agg["epochs"] == DEPLOYED, "mean"].iloc[0])
    rel = 100.0 * (dep_loss - best_loss) / best_loss
    print(f"\nselected budget {best} (val {best_loss:.5f}); "
          f"deployed {DEPLOYED} (val {dep_loss:.5f}), {rel:+.2f}% worse")

    # ── does the selection change the published numbers? ─────────────────────
    df = pd.read_csv(os.path.join(ROOT, "data", "conus_tmax_maxima.csv"),
                     index_col="year")
    cols = [c for c in df.columns if df[c].notna().sum() >= MIN_YEARS]
    series = {c: df[c].dropna().values.astype(float) for c in cols}

    net_best = train_budget(Xtr, ytr, best, SEEDS[0])
    net_dep = train_budget(Xtr, ytr, DEPLOYED, SEEDS[0])
    z_best = np.array([gbc_rl(net_best, x)["zN"] for x in series.values()])
    z_dep = np.array([gbc_rl(net_dep, x)["zN"] for x in series.values()])
    d = np.abs(z_best - z_dep)
    print(f"\nz100 over {len(series)} stations, budget {best} vs {DEPLOYED}: "
          f"median |diff| {np.median(d):.3f} C, max {d.max():.3f} C, "
          f"r={np.corrcoef(z_best, z_dep)[0,1]:.4f}")

    out = {"stamp": STAMP, "budgets": BUDGETS, "seeds": SEEDS, "n_sim": N_SIM,
           "val_frac": VAL_FRAC, "tau_grid": TAU_GRID.tolist(),
           "selected_epochs": best, "deployed_epochs": DEPLOYED,
           "val_pinball_selected": best_loss, "val_pinball_deployed": dep_loss,
           "deployed_excess_pct": rel,
           "z100_median_abs_diff": float(np.median(d)),
           "z100_max_abs_diff": float(d.max()),
           "z100_corr": float(np.corrcoef(z_best, z_dep)[0, 1]),
           "by_budget": agg.to_dict("records")}
    with open(os.path.join(ROOT, "results", "epoch_selection.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    ledger(best, "epochsel:selected_epochs", "{:.0f}")
    ledger(rel, "epochsel:deployed_excess_pct", "{:.2f}")
    ledger(float(np.median(d)), "epochsel:z100_median_abs_diff", "{:.3f}")
    ledger(float(d.max()), "epochsel:z100_max_abs_diff", "{:.2f}")
    print("\nwrote results/epoch_selection.{csv,json} and ledger lines")


if __name__ == "__main__":
    main()
