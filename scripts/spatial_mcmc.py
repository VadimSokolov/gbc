"""MCMC baseline for GBC-Spatial: exact-likelihood posterior inference.

Gold-standard Bayesian inference for stationary Matern covariance parameters
via adaptive Metropolis (Haario et al., 2001) on the *exact* Gaussian
log-likelihood. This is the reference posterior against which the amortized
GBC-Spatial posterior is compared.

The data-generating process (prior, Matern covariance, field simulation) is
reproduced verbatim from ``spatial_matern.py`` so that the test fields are
*identical*: with the same ``--seed`` and the same per-field seed formula,
field ``i`` here is the same field ``i`` the GBC model was evaluated on.
Posterior quality (CRPS, 90% coverage, MAE) is computed exactly as in
``spatial_matern.py:evaluate`` so the two methods are directly comparable.

Pure numpy + scipy; no torch / torch_geometric dependency.

Usage
-----
    python scripts/spatial_mcmc.py --n_obs 500 --n_fields 100 \
        --n_iter 20000 --burnin 5000 --seed 42
    python scripts/spatial_mcmc.py --smoke         # fast local correctness check

For Hopper (SLURM): run with --n_obs 500 --n_fields 500 under a CPU allocation.
Results are saved to results/spatial_mcmc/.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, kv

from gbc.metrics import crps_samples


# ── Matern model (verbatim from spatial_matern.py) ───────────────────

def matern_cov(dists: np.ndarray, sigma2: float, rho: float, nu: float) -> np.ndarray:
    """Compute Matern covariance matrix from a distance matrix."""
    C = np.zeros_like(dists)
    nz = dists > 0
    h = dists[nz] / rho
    C[nz] = sigma2 * (2 ** (1 - nu) / gamma(nu)) * (h ** nu) * kv(nu, h)
    C[~nz] = sigma2
    return C


def sample_prior(rng: np.random.Generator) -> dict:
    """Sample Matern parameters from the prior (log-normal in each param)."""
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


def simulate_field(n: int, psi: dict, rng: np.random.Generator):
    """Simulate a spatial field from the Matern model. Returns (Y, S)."""
    S = rng.uniform(size=(n, 2)).astype(np.float64)
    dists = squareform(pdist(S))
    C = matern_cov(dists, psi["sigma2"], psi["rho"], psi["nu"])
    Sigma = C + psi["tau2"] * np.eye(n)
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(n))
    z = rng.standard_normal(n)
    Y = L @ z
    return Y.astype(np.float32), S.astype(np.float32)


def _generate_one_field(seed: int, n_obs: int):
    """Generate one (Y, S, log_psi) triple. Matches spatial_matern.py exactly."""
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


# ── Exact Gaussian posterior in log-parameter space ──────────────────

# Prior on theta = [log_sigma2, log_rho, log_nu, log_tau2]
_PRIOR_MEAN = np.array([0.0, -1.0, 0.0, -2.0])
_PRIOR_SD = np.array([1.0, 0.5, 0.5, 0.5])


def log_prior(theta: np.ndarray) -> float:
    """Gaussian log-prior on the log-parameters (up to an additive constant)."""
    z = (theta - _PRIOR_MEAN) / _PRIOR_SD
    return float(-0.5 * np.sum(z ** 2))


def log_likelihood(theta: np.ndarray, Y: np.ndarray, dists: np.ndarray) -> float:
    """Exact Gaussian log-likelihood of Y given log-parameters theta.

    Sigma = C(sigma2, rho, nu) + tau2 I.  Uses a Cholesky solve; returns
    -inf on numerical failure or an out-of-support proposal.
    """
    log_sigma2, log_rho, log_nu, log_tau2 = theta
    # Guard against numerically extreme proposals (Bessel kv overflow etc.).
    if log_nu > 2.5 or log_nu < -3.0 or log_rho < -5.0 or log_sigma2 > 6.0:
        return -np.inf
    sigma2 = np.exp(log_sigma2)
    rho = np.exp(log_rho)
    nu = np.exp(log_nu)
    tau2 = np.exp(log_tau2)

    n = Y.shape[0]
    C = matern_cov(dists, sigma2, rho, nu)
    Sigma = C + tau2 * np.eye(n)

    jitter = 1e-8
    for _ in range(5):
        try:
            L = np.linalg.cholesky(Sigma + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        return -np.inf

    alpha = solve_triangular(L, Y, lower=True)
    quad = float(alpha @ alpha)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (quad + logdet + n * np.log(2.0 * np.pi))


# ── Adaptive Metropolis sampler (Haario et al., 2001) ────────────────

def adaptive_metropolis(
    Y: np.ndarray,
    dists: np.ndarray,
    n_iter: int,
    burnin: int,
    rng: np.random.Generator,
    init: np.ndarray | None = None,
):
    """Adaptive-Metropolis chain targeting the exact log-posterior.

    Returns (samples, accept_rate) where samples is (n_iter - burnin, 4) of
    post-burn-in log-parameter draws.
    """
    d = 4
    theta = _PRIOR_MEAN.copy() if init is None else init.copy()
    lp = log_likelihood(theta, Y, dists) + log_prior(theta)

    # Initial proposal covariance: modest, scaled to the prior.
    sd_scale = (2.38 ** 2) / d
    prop_cov = np.diag((0.3 * _PRIOR_SD) ** 2)
    chol = np.linalg.cholesky(prop_cov)

    chain = np.empty((n_iter, d))
    mean = theta.copy()
    cov = prop_cov.copy()
    n_accept = 0
    t0 = max(200, burnin // 4)  # start adapting after t0 samples

    for t in range(n_iter):
        prop = theta + chol @ rng.standard_normal(d)
        lp_prop = log_likelihood(prop, Y, dists) + log_prior(prop)
        if np.log(rng.uniform()) < lp_prop - lp:
            theta, lp = prop, lp_prop
            n_accept += 1
        chain[t] = theta

        # Recursive mean/covariance update + adaptation of the proposal.
        delta = theta - mean
        mean = mean + delta / (t + 2)
        cov = cov + (np.outer(delta, theta - mean) - cov) / (t + 2)
        if t > t0 and t % 50 == 0:
            prop_cov = sd_scale * (cov + 1e-8 * np.eye(d))
            try:
                chol = np.linalg.cholesky(prop_cov)
            except np.linalg.LinAlgError:
                pass

    return chain[burnin:], n_accept / n_iter


# ── Per-field worker + evaluation ────────────────────────────────────

def _mcmc_one_field(args_tuple) -> dict:
    """Run MCMC on a single test field and return per-parameter metrics."""
    field_seed, n_obs, n_iter, burnin, chain_seed = args_tuple
    Y, S, log_psi_true = _generate_one_field(field_seed, n_obs)
    Y = Y.astype(np.float64)
    dists = squareform(pdist(S.astype(np.float64)))

    rng = np.random.default_rng(chain_seed)
    samples, acc = adaptive_metropolis(Y, dists, n_iter, burnin, rng)

    crps = np.empty(4)
    cov = np.empty(4)
    mae = np.empty(4)
    for j in range(4):
        sj = samples[:, j:j + 1]
        crps[j] = crps_samples(np.atleast_1d(log_psi_true[j]), sj)
        lo, hi = np.percentile(samples[:, j], [5, 95])
        cov[j] = float(lo <= log_psi_true[j] <= hi)
        mae[j] = abs(np.median(samples[:, j]) - log_psi_true[j])
    return {"crps": crps, "cov": cov, "mae": mae, "acc": acc}


def main():
    parser = argparse.ArgumentParser(description="MCMC baseline for GBC-Spatial")
    parser.add_argument("--n_obs", type=int, default=500)
    parser.add_argument("--n_fields", type=int, default=100,
                        help="number of held-out test fields (<=500 to align with GBC eval)")
    parser.add_argument("--n_iter", type=int, default=20000)
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42,
                        help="must match the GBC run's --seed; test fields use seed+1")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="fast local correctness check (tiny problem)")
    args = parser.parse_args()

    if args.smoke:
        args.n_obs, args.n_fields, args.n_iter, args.burnin = 50, 3, 2000, 500

    # Reproduce the SAME test fields as spatial_matern.py: test seed = seed + 1,
    # per-field seed = (seed+1) * 1_000_000 + i.
    test_seed = args.seed + 1
    field_seeds = [test_seed * 1_000_000 + i for i in range(args.n_fields)]
    tasks = [
        (fs, args.n_obs, args.n_iter, args.burnin, 7_000_000 + i)
        for i, fs in enumerate(field_seeds)
    ]

    workers = args.workers or min(16, os.cpu_count() or 4)
    print(f"MCMC baseline: n_obs={args.n_obs}, n_fields={args.n_fields}, "
          f"n_iter={args.n_iter}, burnin={args.burnin}, workers={workers}")

    t0 = time.time()
    if workers > 1 and not args.smoke:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            per_field = list(ex.map(_mcmc_one_field, tasks))
    else:
        per_field = [_mcmc_one_field(t) for t in tasks]
    elapsed = time.time() - t0

    crps = np.array([r["crps"] for r in per_field])  # (n_fields, 4)
    cov = np.array([r["cov"] for r in per_field])
    mae = np.array([r["mae"] for r in per_field])
    acc = np.array([r["acc"] for r in per_field])

    names = ["log_sigma2", "log_rho", "log_nu", "log_tau2"]
    results = {}
    for j, name in enumerate(names):
        results[name] = {
            "crps": float(crps[:, j].mean()),
            "coverage_90": float(cov[:, j].mean()),
            "mae": float(mae[:, j].mean()),
        }
    results["overall"] = {
        "crps": float(crps.mean()),
        "coverage_90": float(cov.mean()),
        "mae": float(mae.mean()),
    }
    results["timing"] = {
        "total_seconds": elapsed,
        "seconds_per_field": elapsed / args.n_fields,
    }
    results["diagnostics"] = {"mean_accept_rate": float(acc.mean())}
    results["config"] = vars(args)

    out_dir = "results/spatial_mcmc"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"results_n{args.n_obs}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s ({elapsed/args.n_fields:.2f}s/field), "
          f"mean accept={acc.mean():.3f}")
    print("\n=== MCMC baseline results ===")
    for name in names:
        r = results[name]
        print(f"  {name:12s}: CRPS={r['crps']:.4f}  "
              f"Cov90={r['coverage_90']:.3f}  MAE={r['mae']:.4f}")
    r = results["overall"]
    print(f"  {'overall':12s}: CRPS={r['crps']:.4f}  "
          f"Cov90={r['coverage_90']:.3f}  MAE={r['mae']:.4f}")


if __name__ == "__main__":
    main()
