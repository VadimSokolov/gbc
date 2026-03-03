#!/usr/bin/env python3
"""Ch 8 — Aug-IQN for jump discontinuities.

Train boundary-augmented IQN on Phantom and Star datasets.

Usage
-----
# Batch mode:
python scripts/ch08_jumps.py --dataset phantom --reps 10

# SLURM array mode:
python scripts/ch08_jumps.py --dataset phantom --rep $SLURM_ARRAY_TASK_ID

Reference numbers (Table 8.x in Ch 8):
  Phantom: Aug-IQN RMSE=0.044, CRPS=0.009
  Star   : Aug-IQN RMSE=0.059, CRPS=0.013
"""

import argparse
import os
import sys

import numpy as np
import torch

from gbc.augment import em_labels, train_classifier, augment_features
from gbc.iqn import train_iqn, sample_iqn
from gbc.metrics import rmse, crps_samples, coverage
from gbc.utils import set_seed


def _load_data(dataset: str):
    """Load Phantom or Star CSV data."""
    path = f"data/{dataset}.csv"
    if not os.path.exists(path):
        print(f"Error: {path} not found.", file=sys.stderr)
        print(f"\nTo obtain the {dataset} dataset, export it from R:", file=sys.stderr)
        print(f"  library(jumpgp)", file=sys.stderr)
        print(f"  data({dataset})", file=sys.stderr)
        print(f'  write.csv({dataset}, "{dataset}.csv", row.names = FALSE)', file=sys.stderr)
        print(f"\nThen place the CSV in the data/ directory.", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(path)
    # jumpgp datasets have columns x1, x2, y (Phantom/Star are 2D)
    xcols = [c for c in df.columns if c.startswith("x")]
    ycol = [c for c in df.columns if c.startswith("y") or c == "Y"][0]
    X = df[xcols].values.astype(np.float64)
    y = df[ycol].values.astype(np.float64)
    return X, y


def run_rep(dataset: str, rep: int):
    """Run one Aug-IQN replicate on the given dataset."""
    set_seed(rep * 100)

    X, y = _load_data(dataset)
    n = len(y)

    # 90/10 train/test split
    idx = np.random.permutation(n)
    n_test = max(1, int(0.10 * n))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    # Step 1: EM labels
    labels = em_labels(y_tr, n_components=2, seed=rep)

    # Step 2: Train boundary classifier
    classifier = train_classifier(X_tr, labels, epochs=3000, hdim=256, nlayers=3, seed=rep)

    # Step 3: Augment features
    X_tr_aug = augment_features(X_tr, classifier)
    X_te_aug = augment_features(X_te, classifier)

    # Step 4: Train IQN on augmented features
    model, xm, xs, ym, ys = train_iqn(
        X_tr_aug, y_tr, epochs=8000, hdim=256, nh=32,
        w=(0.10, 0.20, 0.70), seed=rep,
    )

    # Step 5: Evaluate
    samples = sample_iqn(model, X_te_aug, xm, xs, ym, ys, B=500)
    r = rmse(y_te, samples.mean(axis=0))
    c = crps_samples(y_te, samples)
    cov = coverage(y_te, samples)

    print(f"  {dataset} rep {rep}: RMSE={r:.4f}  CRPS={c:.4f}  Coverage={cov:.3f}")

    return {"rmse": r, "crps": c, "coverage": cov}


def _aggregate(all_results, dataset: str):
    """Write mean ± SE summary."""
    os.makedirs("cache", exist_ok=True)
    outpath = f"cache/augiqn_{dataset}_numbers.txt"
    lines = []
    for metric in ["rmse", "crps", "coverage"]:
        arr = np.array([r[metric] for r in all_results])
        lines.append(f"{metric} = {arr.mean():.4f} +/- {arr.std() / np.sqrt(len(arr)):.4f}")
    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAggregated results written to {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Ch 8 — Aug-IQN for jump discontinuities")
    parser.add_argument("--dataset", required=True, choices=["phantom", "star"],
                        help="Jump-process dataset")
    parser.add_argument("--reps", type=int, default=None,
                        help="Number of replicates (batch mode)")
    parser.add_argument("--rep", type=int, default=None,
                        help="Single replicate ID (SLURM array mode)")
    args = parser.parse_args()

    if args.reps is None and args.rep is None:
        parser.error("Specify --reps N (batch) or --rep ID (SLURM array)")

    os.makedirs("cache", exist_ok=True)

    if args.rep is not None:
        rep = args.rep
        print(f"Running {args.dataset} rep {rep}")
        result = run_rep(args.dataset, rep)
        torch.save(result, f"cache/ch08_{args.dataset}_rep{rep}.pt")
        print(f"Saved cache/ch08_{args.dataset}_rep{rep}.pt")

    else:
        all_results = []
        for rep in range(args.reps):
            print(f"\n=== Replicate {rep}/{args.reps} ===")
            result = run_rep(args.dataset, rep)
            torch.save(result, f"cache/ch08_{args.dataset}_rep{rep}.pt")
            all_results.append(result)

        _aggregate(all_results, args.dataset)


if __name__ == "__main__":
    main()
