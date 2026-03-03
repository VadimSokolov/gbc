#!/usr/bin/env python3
"""Ch 7 — GBC surrogates for computer experiments.

Train IQN surrogates on BGP and Friedman benchmarks; compare to GP baselines.

Usage
-----
# Run all 50 replicates sequentially:
python scripts/ch07_surrogates.py --dataset bgp --reps 50

# SLURM array mode (one replicate per task):
python scripts/ch07_surrogates.py --dataset bgp --rep $SLURM_ARRAY_TASK_ID

Reference numbers (Table 7.x in Ch 7):
  BGP d=2 : GBC RMSE=2.168, CRPS=1.224
  Friedman: GBC RMSE=0.520, CRPS=0.323
"""

import argparse
import os
import sys

import numpy as np
import torch

from gbc.iqn import train_iqn, sample_iqn
from gbc.data import friedman1
from gbc.metrics import rmse, crps_samples, coverage
from gbc.utils import set_seed


# ---------------------------------------------------------------------------
# BGP synthetic data generator
# ---------------------------------------------------------------------------

def _bgp_generate(d: int, n: int, seed: int):
    """Generate BGP data: random partition vector, two GP regimes."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))

    # Random partition vector a (unit vector)
    a = rng.standard_normal(d)
    a /= np.linalg.norm(a)

    # Projection onto a
    proj = X @ a  # (n,)

    # Two regimes: smooth sine vs linear, split at median
    boundary = np.median(proj)
    regime = (proj >= boundary).astype(float)

    y = np.where(
        regime == 0,
        3.0 * np.sin(2 * np.pi * proj) + rng.normal(0, 0.3, n),
        -2.0 * proj + rng.normal(0, 0.3, n),
    )
    return X, y


def _train_gp(X_train, y_train, X_test):
    """Fit a GP baseline (Matern-5/2 + WhiteKernel)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42)
    gp.fit(X_train, y_train)
    mu, sigma = gp.predict(X_test, return_std=True)
    return mu, sigma


# ---------------------------------------------------------------------------
# Per-replicate runner
# ---------------------------------------------------------------------------

def run_bgp_rep(rep: int, dims=(2, 3, 4), n: int = 2000):
    """Run one BGP replicate across dimensions."""
    results = {}
    for d in dims:
        set_seed(rep * 100 + d)
        X, y = _bgp_generate(d, n, seed=rep * 100 + d)

        # 80/20 split
        n_train = int(0.8 * n)
        idx = np.random.permutation(n)
        X_tr, y_tr = X[idx[:n_train]], y[idx[:n_train]]
        X_te, y_te = X[idx[n_train:]], y[idx[n_train:]]

        # GBC (IQN)
        model, xm, xs, ym, ys = train_iqn(X_tr, y_tr, epochs=3000, hdim=256, nh=32)
        samples = sample_iqn(model, X_te, xm, xs, ym, ys, B=500)
        gbc_rmse = rmse(y_te, samples.mean(axis=0))
        gbc_crps = crps_samples(y_te, samples)
        gbc_cov = coverage(y_te, samples)

        # GP baseline
        mu_gp, sigma_gp = _train_gp(X_tr, y_tr, X_te)
        gp_rmse = rmse(y_te, mu_gp)
        gp_samples = np.random.randn(500, len(y_te)) * sigma_gp[None, :] + mu_gp[None, :]
        gp_crps = crps_samples(y_te, gp_samples)
        gp_cov = coverage(y_te, gp_samples)

        results[f"d{d}"] = {
            "gbc_rmse": gbc_rmse, "gbc_crps": gbc_crps, "gbc_cov": gbc_cov,
            "gp_rmse": gp_rmse, "gp_crps": gp_crps, "gp_cov": gp_cov,
        }
        print(f"  BGP d={d}: GBC RMSE={gbc_rmse:.3f}  CRPS={gbc_crps:.3f} | "
              f"GP RMSE={gp_rmse:.3f}  CRPS={gp_crps:.3f}")

    return results


def run_friedman_rep(rep: int, d: int = 10, n: int = 2000):
    """Run one Friedman replicate."""
    set_seed(rep * 100)
    rng = np.random.default_rng(rep)
    X = rng.uniform(0, 1, (n, d))
    y = friedman1(X) + rng.normal(0, 1, n)

    n_train = int(0.8 * n)
    idx = np.random.permutation(n)
    X_tr, y_tr = X[idx[:n_train]], y[idx[:n_train]]
    X_te, y_te = X[idx[n_train:]], y[idx[n_train:]]

    model, xm, xs, ym, ys = train_iqn(X_tr, y_tr, epochs=3000, hdim=256, nh=32)
    samples = sample_iqn(model, X_te, xm, xs, ym, ys, B=500)
    gbc_rmse = rmse(y_te, samples.mean(axis=0))
    gbc_crps = crps_samples(y_te, samples)
    gbc_cov = coverage(y_te, samples)

    mu_gp, sigma_gp = _train_gp(X_tr, y_tr, X_te)
    gp_rmse = rmse(y_te, mu_gp)
    gp_samples = np.random.randn(500, len(y_te)) * sigma_gp[None, :] + mu_gp[None, :]
    gp_crps = crps_samples(y_te, gp_samples)

    print(f"  Friedman rep {rep}: GBC RMSE={gbc_rmse:.3f}  CRPS={gbc_crps:.3f} | "
          f"GP RMSE={gp_rmse:.3f}  CRPS={gp_crps:.3f}")

    return {
        "gbc_rmse": gbc_rmse, "gbc_crps": gbc_crps, "gbc_cov": gbc_cov,
        "gp_rmse": gp_rmse, "gp_crps": gp_crps,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(all_results, dataset: str):
    """Write mean ± SE summary to cache/<dataset>_numbers.txt."""
    os.makedirs("cache", exist_ok=True)
    outpath = f"cache/{dataset}_numbers.txt"

    if dataset == "bgp":
        lines = []
        for d_key in ["d2", "d3", "d4"]:
            vals = [r[d_key] for r in all_results if d_key in r]
            if not vals:
                continue
            for metric in ["gbc_rmse", "gbc_crps", "gbc_cov", "gp_rmse", "gp_crps", "gp_cov"]:
                arr = np.array([v[metric] for v in vals])
                lines.append(f"{d_key}_{metric} = {arr.mean():.4f} +/- {arr.std() / np.sqrt(len(arr)):.4f}")
        with open(outpath, "w") as f:
            f.write("\n".join(lines) + "\n")
    else:
        lines = []
        for metric in ["gbc_rmse", "gbc_crps", "gbc_cov", "gp_rmse", "gp_crps"]:
            arr = np.array([r[metric] for r in all_results])
            lines.append(f"{metric} = {arr.mean():.4f} +/- {arr.std() / np.sqrt(len(arr)):.4f}")
        with open(outpath, "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"\nAggregated results written to {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ch 7 — GBC surrogates for computer experiments")
    parser.add_argument("--dataset", required=True, choices=["bgp", "friedman"],
                        help="Benchmark dataset")
    parser.add_argument("--reps", type=int, default=None,
                        help="Number of replicates (batch mode)")
    parser.add_argument("--rep", type=int, default=None,
                        help="Single replicate ID (SLURM array mode)")
    args = parser.parse_args()

    if args.reps is None and args.rep is None:
        parser.error("Specify --reps N (batch) or --rep ID (SLURM array)")

    os.makedirs("cache", exist_ok=True)

    if args.rep is not None:
        # Single replicate
        rep = args.rep
        print(f"Running {args.dataset} rep {rep}")
        if args.dataset == "bgp":
            result = run_bgp_rep(rep)
        else:
            result = run_friedman_rep(rep)
        torch.save(result, f"cache/ch07_{args.dataset}_rep{rep}.pt")
        print(f"Saved cache/ch07_{args.dataset}_rep{rep}.pt")

    else:
        # Batch mode
        all_results = []
        for rep in range(args.reps):
            print(f"\n=== Replicate {rep}/{args.reps} ===")
            if args.dataset == "bgp":
                result = run_bgp_rep(rep)
            else:
                result = run_friedman_rep(rep)
            torch.save(result, f"cache/ch07_{args.dataset}_rep{rep}.pt")
            all_results.append(result)

        _aggregate(all_results, args.dataset)


if __name__ == "__main__":
    main()
