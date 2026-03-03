#!/usr/bin/env python3
"""Ch 14 — Het-MLP ensemble + conformal calibration on lake temperature data.

Usage
-----
python scripts/ch14_lake.py --epochs 5000 --cv-folds 5
python scripts/ch14_lake.py --epochs 5000 --cv-folds 5 --data data/lake_merged.csv

Reference numbers (Table 14.x in Ch 14):
  RMSE=1.219, Coverage=90.2%, PI width ≈ 3.8
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from gbc.ensemble import train_het_mlp, ensemble_predict
from gbc.conformal import temporal_cv_quantiles, conformal_pi
from gbc.metrics import rmse, coverage, pi_width
from gbc.utils import set_seed, get_device


def _load_lake(path: str):
    """Load lake temperature data."""
    if not os.path.exists(path):
        print(f"Error: {path} not found.", file=sys.stderr)
        print("\nThe lake temperature dataset is from:", file=sys.stderr)
        print("  Holthuijzen et al. (2025), 'Synthesizing data products....'", file=sys.stderr)
        print(f"\nPlace the merged CSV at {path}.", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(
        description="Ch 14 — Het-MLP ensemble + conformal calibration")
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Training epochs per ensemble member (default: 5000)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of temporal CV folds (default: 5)")
    parser.add_argument("--data", type=str, default="data/lake_merged.csv",
                        help="Path to lake data CSV")
    parser.add_argument("--K", type=int, default=5,
                        help="Ensemble size (default: 5)")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    df = _load_lake(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    # Features: DOY, Depth, Horizon, phi, GLM_mean, GLM_std -> temp_obs
    feature_cols = ["DOY", "Depth", "Horizon", "phi", "GLM_mean", "GLM_std"]
    target_col = "temp_obs"

    # Check columns exist (allow flexible naming)
    available = set(df.columns)
    missing = [c for c in feature_cols + [target_col] if c not in available]
    if missing:
        print(f"Warning: columns {missing} not found. Available: {sorted(available)}",
              file=sys.stderr)
        print("Attempting to use first 6 numeric columns as features, last as target.",
              file=sys.stderr)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric_cols[:6]
        target_col = numeric_cols[-1]
        print(f"Using features={feature_cols}, target={target_col}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Train/test split by date
    if "date" in df.columns:
        train_mask = pd.to_datetime(df["date"]) < "2022-06-11"
    elif "Date" in df.columns:
        train_mask = pd.to_datetime(df["Date"]) < "2022-06-11"
    else:
        # Fallback: 80/20 split
        print("No date column found; using 80/20 split.", file=sys.stderr)
        n_train = int(0.8 * len(df))
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[:n_train] = True

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[~train_mask], y[~train_mask]
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

    # Normalize features
    xm, xs = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr_n = (X_tr - xm) / xs
    X_te_n = (X_te - xm) / xs

    # Convert to tensors
    X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X_te_t = torch.tensor(X_te_n, dtype=torch.float32).to(device)

    # Train ensemble
    print(f"\nTraining {args.K}-member HetMLP ensemble ({args.epochs} epochs each)...")
    models = []
    for k in range(args.K):
        print(f"  Member {k+1}/{args.K}...")
        model = train_het_mlp(
            X_tr_t, y_tr_t,
            epochs=args.epochs, hdim=256, nlayers=3,
            batch_size=8192, seed=42 + k, device=device,
        )
        models.append(model)

    # Ensemble predictions on test
    ens_mean, ens_std = ensemble_predict(models, X_te_t)

    # Point-forecast RMSE
    r = rmse(y_te, ens_mean)
    print(f"\nTest RMSE: {r:.3f}")

    # Temporal CV for conformal calibration
    print(f"\n{args.cv_folds}-fold temporal CV for conformal quantiles...")
    horizon_col = "Horizon" if "Horizon" in df.columns else feature_cols[2]
    horizons_te = df.loc[~train_mask, horizon_col].values if horizon_col in df.columns else np.ones(len(X_te))

    # Out-of-fold residuals on training set
    n_tr = len(X_tr)
    fold_size = n_tr // args.cv_folds
    oof_residuals = np.zeros(n_tr)
    oof_horizons = df.loc[train_mask, horizon_col].values if horizon_col in df.columns else np.ones(n_tr)

    for fold in range(args.cv_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < args.cv_folds - 1 else n_tr
        val_mask = np.zeros(n_tr, dtype=bool)
        val_mask[val_start:val_end] = True

        X_fold_tr = X_tr_n[~val_mask]
        y_fold_tr = y_tr[~val_mask]
        X_fold_val = X_tr_n[val_mask]

        X_ft = torch.tensor(X_fold_tr, dtype=torch.float32).to(device)
        y_ft = torch.tensor(y_fold_tr, dtype=torch.float32).unsqueeze(1).to(device)
        X_vt = torch.tensor(X_fold_val, dtype=torch.float32).to(device)

        fold_models = []
        for k in range(args.K):
            m = train_het_mlp(X_ft, y_ft, epochs=args.epochs, hdim=256, nlayers=3,
                              batch_size=8192, seed=100 + fold * 10 + k, device=device)
            fold_models.append(m)

        fold_mean, _ = ensemble_predict(fold_models, X_vt)
        oof_residuals[val_mask] = np.abs(y_tr[val_mask] - fold_mean)

    # Per-horizon conformal quantiles
    quantile_map = temporal_cv_quantiles(oof_residuals, oof_horizons, alpha=0.90)

    # Conformal PI on test set
    lower, upper = conformal_pi(ens_mean, horizons_te, quantile_map)
    cov = float(np.mean((y_te >= lower) & (y_te <= upper)))
    width = float(np.mean(upper - lower))

    print(f"\nResults:")
    print(f"  RMSE     = {r:.3f}")
    print(f"  Coverage = {cov:.3f} (target: 0.900)")
    print(f"  PI width = {width:.2f}")

    # Per-horizon results
    unique_h = np.unique(horizons_te)
    if len(unique_h) > 1 and len(unique_h) <= 30:
        print(f"\nPer-horizon breakdown:")
        for h in sorted(unique_h):
            mask = horizons_te == h
            h_rmse = rmse(y_te[mask], ens_mean[mask])
            h_cov = float(np.mean((y_te[mask] >= lower[mask]) & (y_te[mask] <= upper[mask])))
            h_width = float(np.mean(upper[mask] - lower[mask]))
            print(f"  Horizon {h:>3.0f}: RMSE={h_rmse:.3f}  Cov={h_cov:.3f}  Width={h_width:.2f}")

    # Save results
    os.makedirs("cache", exist_ok=True)
    results_df = pd.DataFrame({
        "y_true": y_te, "y_hat": ens_mean, "ens_std": ens_std,
        "lower": lower, "upper": upper,
    })
    results_df.to_csv("cache/lake_results.csv", index=False)

    with open("cache/lake_numbers.txt", "w") as f:
        f.write(f"rmse = {r:.4f}\n")
        f.write(f"coverage = {cov:.4f}\n")
        f.write(f"pi_width = {width:.4f}\n")

    print(f"\nSaved cache/lake_results.csv and cache/lake_numbers.txt")


if __name__ == "__main__":
    main()
