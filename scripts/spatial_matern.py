"""GBC-Spatial: Amortized posterior inference for Matérn covariance parameters.

Generates training data from a stationary Matérn GP model, trains the
GBC-Spatial architecture (GNN encoder + IQN transport map), and evaluates
posterior quality on held-out test datasets.

Usage
-----
    python scripts/spatial_matern.py [--n_obs 500] [--n_train 200000] \
        [--n_test 500] [--epochs 500000] [--device cuda]

For Hopper (SLURM):
    sbatch scripts/spatial_matern.slurm

Results are saved to results/spatial_matern/.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, kv

from gbc.utils import set_seed, get_device
from gbc.metrics import crps_samples, coverage, pi_width

# Lazy import of GNN modules (requires torch_geometric)
from gbc.spatial_gnn import (
    GBCSpatial,
    build_batch,
    posterior_samples,
    huber_quantile_loss,
)


# ── Matérn covariance ────────────────────────────────────────────────

def matern_cov(dists: np.ndarray, sigma2: float, rho: float, nu: float) -> np.ndarray:
    """Compute Matérn covariance matrix from a distance matrix."""
    C = np.zeros_like(dists)
    nz = dists > 0
    h = dists[nz] / rho
    C[nz] = sigma2 * (2 ** (1 - nu) / gamma(nu)) * (h ** nu) * kv(nu, h)
    C[~nz] = sigma2
    return C


def sample_prior(rng: np.random.Generator) -> dict:
    """Sample Matérn parameters from the prior."""
    log_sigma2 = rng.normal(0, 1)
    log_rho = rng.normal(-1, 0.5)
    log_nu = rng.normal(0, 0.5)
    log_tau2 = rng.normal(-2, 0.5)
    return {
        "sigma2": np.exp(log_sigma2),
        "rho": np.exp(log_rho),
        "nu": np.exp(log_nu),
        "tau2": np.exp(log_tau2),
    }


def simulate_field(
    n: int, psi: dict, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a spatial field from the Matérn model.

    Returns (Y, S) where Y is (n,) observations and S is (n, 2) locations.
    """
    S = rng.uniform(size=(n, 2)).astype(np.float64)
    dists = squareform(pdist(S))
    C = matern_cov(dists, psi["sigma2"], psi["rho"], psi["nu"])
    Sigma = C + psi["tau2"] * np.eye(n)

    # Cholesky + draw
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n))
    z = rng.standard_normal(n)
    Y = L @ z

    return Y.astype(np.float32), S.astype(np.float32)


# ── Data generation ──────────────────────────────────────────────────

def _generate_one_field(seed: int, n_obs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single (Y, S, log_psi) triple.  Designed to run in a worker process."""
    rng = np.random.default_rng(seed)
    psi = sample_prior(rng)
    Y, S = simulate_field(n_obs, psi, rng)
    log_psi = np.array([
        np.log(psi["sigma2"]),
        np.log(psi["rho"]),
        np.log(psi["nu"]),
        np.log(psi["tau2"]),
    ], dtype=np.float32)
    return Y, S, log_psi


def generate_dataset(
    N: int, n_obs: int, seed: int = 42, max_workers: int | None = None,
) -> tuple[list, list, np.ndarray]:
    """Generate N simulated (Y, S, psi) triples using multiprocessing.

    Returns (Ys, Ss, psis) where psis is (N, 4) in log-space.
    """
    if max_workers is None:
        max_workers = min(N, os.cpu_count() or 8)

    seeds = [seed * 1_000_000 + i for i in range(N)]
    worker_fn = partial(_generate_one_field, n_obs=n_obs)

    Ys, Ss = [], []
    psis = np.zeros((N, 4), dtype=np.float32)

    # Process in chunks to show progress
    chunk = 5000
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            results = list(executor.map(worker_fn, seeds[start:end]))
            for k, (Y, S, log_psi) in enumerate(results):
                Ys.append(Y)
                Ss.append(S)
                psis[start + k] = log_psi
            print(f"  {end}/{N} fields generated...")

    return Ys, Ss, psis


# ── Training loop ────────────────────────────────────────────────────

def train(
    model: GBCSpatial,
    Ys_train: list,
    Ss_train: list,
    psis_train: np.ndarray,
    n_steps: int = 500_000,
    batch_size: int = 64,
    N_tau: int = 8,
    lr: float = 3e-4,
    device: str = "cpu",
):
    """Train the GBC-Spatial model on pre-generated data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01,
    )

    N = len(Ys_train)
    rng = np.random.default_rng(0)
    losses = []

    for step in range(n_steps):
        idx = rng.choice(N, size=batch_size, replace=False)
        batch_Ys = [Ys_train[i] for i in idx]
        batch_Ss = [Ss_train[i] for i in idx]
        batch_psi = torch.tensor(psis_train[idx], device=device)

        batch_data = build_batch(batch_Ys, batch_Ss).to(device)

        # Forward
        model.train()
        B = batch_size
        tau = torch.rand(B * N_tau, device=device)
        psi_pred = model(
            batch_data.x, batch_data.edge_index, batch_data.edge_attr,
            batch_data.batch, tau,
        )
        psi_rep = batch_psi.unsqueeze(1).expand(-1, N_tau, -1).reshape(-1, 4)
        loss = huber_quantile_loss(psi_rep, psi_pred, tau)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            avg_loss = np.mean(losses[-1000:])
            print(f"Step {step+1:>7d}/{n_steps}  loss={avg_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    return losses


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(
    model: GBCSpatial,
    Ys_test: list,
    Ss_test: list,
    psis_test: np.ndarray,
    B: int = 2000,
    device: str = "cpu",
) -> dict:
    """Evaluate posterior quality on test datasets."""
    N_test = len(Ys_test)
    crps_vals = np.zeros((N_test, 4))
    cov_vals = np.zeros((N_test, 4))
    mae_vals = np.zeros((N_test, 4))

    for i in range(N_test):
        samples = posterior_samples(
            model, Ys_test[i], Ss_test[i], B=B, device=device,
        )
        for j in range(4):
            crps_vals[i, j] = crps_samples(
                np.atleast_1d(psis_test[i, j]),
                samples[:, j:j+1],
            )
            lower = np.percentile(samples[:, j], 5)
            upper = np.percentile(samples[:, j], 95)
            cov_vals[i, j] = float(lower <= psis_test[i, j] <= upper)
            mae_vals[i, j] = abs(np.median(samples[:, j]) - psis_test[i, j])

    param_names = ["log_sigma2", "log_rho", "log_nu", "log_tau2"]
    results = {}
    for j, name in enumerate(param_names):
        results[name] = {
            "crps": float(crps_vals[:, j].mean()),
            "coverage_90": float(cov_vals[:, j].mean()),
            "mae": float(mae_vals[:, j].mean()),
        }

    results["overall"] = {
        "crps": float(crps_vals.mean()),
        "coverage_90": float(cov_vals.mean()),
        "mae": float(mae_vals.mean()),
    }
    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GBC-Spatial: Matérn experiment")
    parser.add_argument("--n_obs", type=int, default=500)
    parser.add_argument("--n_train", type=int, default=200_000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--n_steps", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--d_bottleneck", type=int, default=16)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Device: {device}")

    out_dir = "results/spatial_matern"
    os.makedirs(out_dir, exist_ok=True)

    # Respect SLURM allocation; fall back to os.cpu_count()
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    max_workers = int(slurm_cpus) if slurm_cpus else min(16, os.cpu_count() or 8)
    max_workers = min(max_workers, args.n_train)
    print(f"Using {max_workers} workers for parallel data generation")

    # Generate training data in parallel
    print(f"Generating {args.n_train} training fields (n={args.n_obs})...")
    t0 = time.time()
    Ys_train, Ss_train, psis_train = generate_dataset(
        args.n_train, args.n_obs, seed=args.seed, max_workers=max_workers,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # Generate test data in parallel
    print(f"Generating {args.n_test} test fields (n={args.n_obs})...")
    t0 = time.time()
    Ys_test, Ss_test, psis_test = generate_dataset(
        args.n_test, args.n_obs, seed=args.seed + 1, max_workers=max_workers,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # Build model
    model = GBCSpatial(
        encoder_kwargs=dict(
            d_bottleneck=args.d_bottleneck,
            H=args.H,
            H_prime=args.H * 2,
            n_layers=4,
        ),
        iqn_kwargs=dict(
            d_out=4,
            H=256,
            n_cosine=64,
            d_tau=128,
        ),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    print(f"Training for {args.n_steps} steps...")
    t0 = time.time()
    losses = train(
        model, Ys_train, Ss_train, psis_train,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=device,
    )
    train_time = time.time() - t0
    print(f"Training time: {train_time:.0f}s ({train_time/3600:.1f}h)")

    # Save model
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    np.save(os.path.join(out_dir, "losses.npy"), np.array(losses))

    # Evaluate
    print("Evaluating on test set...")
    t0 = time.time()
    results = evaluate(model, Ys_test, Ss_test, psis_test, B=2000, device=device)
    eval_time = time.time() - t0
    results["timing"] = {
        "train_seconds": train_time,
        "eval_seconds": eval_time,
        "eval_per_field": eval_time / args.n_test,
    }
    results["config"] = vars(args)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Results ===")
    for name in ["log_sigma2", "log_rho", "log_nu", "log_tau2"]:
        r = results[name]
        print(f"  {name:12s}: CRPS={r['crps']:.4f}  "
              f"Cov90={r['coverage_90']:.3f}  MAE={r['mae']:.4f}")
    r = results["overall"]
    print(f"  {'overall':12s}: CRPS={r['crps']:.4f}  "
          f"Cov90={r['coverage_90']:.3f}  MAE={r['mae']:.4f}")


if __name__ == "__main__":
    main()
