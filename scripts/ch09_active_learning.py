#!/usr/bin/env python3
"""Ch 9 — Sequential design and active learning.

Randomized-prior ensemble acquisition on the rocket (LGBB) benchmark.

Usage
-----
# Batch mode:
python scripts/ch09_active_learning.py --dataset rocket --n0 50 --budget 250 --reps 30

# SLURM array mode:
python scripts/ch09_active_learning.py --dataset rocket --rep $SLURM_ARRAY_TASK_ID

Reference numbers (Table 9.x in Ch 9):
  GBC RMSE ≈ 0.00723 vs DGP RMSE ≈ 0.02134
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from gbc.active_learning import RandomPriorNet, ensemble_disagreement, select_next
from gbc.iqn import train_iqn, sample_iqn
from gbc.metrics import rmse, crps_samples, coverage
from gbc.utils import set_seed


def _load_rocket():
    """Load LGBB rocket data (lgbb_fill.csv)."""
    path = "data/lgbb_fill.csv"
    if not os.path.exists(path):
        print(f"Error: {path} not found.", file=sys.stderr)
        print("\nDownload the LGBB dataset from:", file=sys.stderr)
        print("  https://bobby.gramacy.com/surrogates/", file=sys.stderr)
        print("\nPlace lgbb_fill.csv in the data/ directory.", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(path)
    # Expected columns: mach_s, alpha_s, beta_s, side_c (or similar)
    xcols = [c for c in df.columns if c != df.columns[-1]]
    ycol = df.columns[-1]
    X = df[xcols].values.astype(np.float64)
    y = df[ycol].values.astype(np.float64)
    return X, y


def _train_ensemble(X_tr, y_tr, K: int = 5, epochs: int = 1000, seed: int = 0):
    """Train K-member randomized-prior ensemble."""
    xdim = X_tr.shape[1]
    xm, xs = X_tr.mean(0), X_tr.std(0) + 1e-8
    ym, ys = y_tr.mean(), y_tr.std() + 1e-8
    X_t = torch.tensor((X_tr - xm) / xs, dtype=torch.float32)
    y_t = torch.tensor((y_tr - ym) / ys, dtype=torch.float32).unsqueeze(1)

    models = []
    for k in range(K):
        torch.manual_seed(seed * 100 + k)
        model = RandomPriorNet(xdim, hdim=128, nlayers=3, prior_scale=1.0)
        optimizer = torch.optim.Adam(model.train_net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = nn.functional.mse_loss(pred, y_t)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        models.append(model)

    return models, xm, xs, ym, ys


def run_rep(rep: int, n0: int = 50, budget: int = 250, batch_size: int = 25):
    """Run one active learning replicate."""
    set_seed(rep * 100)

    X_all, y_all = _load_rocket()
    n_total = len(y_all)

    # Hold out 20% for final evaluation
    idx = np.random.permutation(n_total)
    n_test = int(0.2 * n_total)
    test_idx = idx[:n_test]
    pool_idx = idx[n_test:]

    X_test, y_test = X_all[test_idx], y_all[test_idx]

    # Initial design: n0 random points from pool
    pool_perm = np.random.permutation(len(pool_idx))
    acquired = pool_idx[pool_perm[:n0]].tolist()
    candidate_mask = np.ones(len(pool_idx), dtype=bool)
    candidate_mask[pool_perm[:n0]] = False

    # Active learning loop
    n_rounds = (budget - n0) // batch_size
    for rnd in range(n_rounds):
        X_tr = X_all[acquired]
        y_tr = y_all[acquired]

        # Train ensemble
        models, xm, xs, _, _ = _train_ensemble(X_tr, y_tr, K=5, epochs=1000, seed=rep * 10 + rnd)

        # Candidates
        cand_idx = pool_idx[candidate_mask]
        X_cand = X_all[cand_idx]
        X_cand_t = torch.tensor((X_cand - xm) / xs, dtype=torch.float32)

        # Select next batch
        sel = select_next(models, X_cand_t, batch_size=batch_size)
        new_idx = cand_idx[sel]
        acquired.extend(new_idx.tolist())

        # Update candidate mask
        global_sel = np.where(candidate_mask)[0][sel]
        candidate_mask[global_sel] = False

        print(f"  Round {rnd+1}/{n_rounds}: acquired {len(acquired)} points")

    # Final evaluation: train IQN on full acquired set
    X_tr = X_all[acquired]
    y_tr = y_all[acquired]
    model, xm, xs, ym, ys = train_iqn(X_tr, y_tr, epochs=3000, hdim=256, nh=32)
    samples = sample_iqn(model, X_test, xm, xs, ym, ys, B=500)

    r = rmse(y_test, samples.mean(axis=0))
    c = crps_samples(y_test, samples)
    cov = coverage(y_test, samples)

    print(f"  Rep {rep} final: RMSE={r:.5f}  CRPS={c:.5f}  Coverage={cov:.3f}  "
          f"(n_acquired={len(acquired)})")

    return {"rmse": r, "crps": c, "coverage": cov, "n_acquired": len(acquired)}


def _aggregate(all_results):
    """Write mean ± SE summary."""
    os.makedirs("cache", exist_ok=True)
    outpath = "cache/al_numbers.txt"
    lines = []
    for metric in ["rmse", "crps", "coverage"]:
        arr = np.array([r[metric] for r in all_results])
        lines.append(f"{metric} = {arr.mean():.5f} +/- {arr.std() / np.sqrt(len(arr)):.5f}")
    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAggregated results written to {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Ch 9 — Sequential design and active learning")
    parser.add_argument("--dataset", required=True, choices=["rocket"],
                        help="Benchmark dataset")
    parser.add_argument("--n0", type=int, default=50,
                        help="Initial design size (default: 50)")
    parser.add_argument("--budget", type=int, default=250,
                        help="Total acquisition budget (default: 250)")
    parser.add_argument("--reps", type=int, default=None,
                        help="Number of replicates (batch mode)")
    parser.add_argument("--rep", type=int, default=None,
                        help="Single replicate ID (SLURM array mode)")
    args = parser.parse_args()

    if args.reps is None and args.rep is None:
        parser.error("Specify --reps N (batch) or --rep ID (SLURM array)")

    os.makedirs("cache", exist_ok=True)
    batch_size = 25

    if args.rep is not None:
        rep = args.rep
        print(f"Running rocket rep {rep}")
        result = run_rep(rep, n0=args.n0, budget=args.budget, batch_size=batch_size)
        torch.save(result, f"cache/ch09_rocket_rep{rep}.pt")
        print(f"Saved cache/ch09_rocket_rep{rep}.pt")

    else:
        all_results = []
        for rep in range(args.reps):
            print(f"\n=== Replicate {rep}/{args.reps} ===")
            result = run_rep(rep, n0=args.n0, budget=args.budget, batch_size=batch_size)
            torch.save(result, f"cache/ch09_rocket_rep{rep}.pt")
            all_results.append(result)

        _aggregate(all_results)


if __name__ == "__main__":
    main()
