"""Multivariate GBC via autoregressive Cholesky decomposition (MGBC).

Implements the Rosenblatt-transform decomposition of a joint distribution
into a chain of d conditional univariate quantile regressions:

    θ_k = Q_k(τ_k | θ₁, …, θ_{k-1}, y),  τ_k ~ U[0,1],  k = 1, …, d

Each Q_k is estimated by an IQN.  The Cholesky preconditioning trick
exploits the fact that near-Gaussian targets can be whitened first, so
the IQN only learns the residual non-Gaussianity.

Book references
---------------
- Ch 4 §sec-noise-multi   : multivariate noise outsourcing (Theorem 4.2)
- Ch 10 §sec-bayes-multi  : autoregressive posterior sampling
- App. A §sec-proofs-multi : convergence guarantee for sequential decomposition
"""

import numpy as np
import torch

from gbc.iqn import IQN, train_iqn, sample_iqn


def train_multivariate_iqn(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 3000,
    hdim: int = 256,
    nh: int = 32,
    lr: float = 1e-3,
    wd: float = 1e-4,
    seed: int = 42,
    w: tuple[float, float, float] = (0.3, 0.3, 0.4),
    batch_size: int | None = None,
    verbose: bool = False,
) -> list[tuple]:
    """Train an autoregressive chain of IQNs for multivariate output.

    Fits *d* sequential IQNs where the *j*-th model conditions on the
    original features *X* plus the first *j-1* target components.

    Parameters
    ----------
    X : (n, p) input features.
    Y : (n, d) multivariate targets.
    epochs, hdim, nh, lr, wd, seed, w : passed to ``train_iqn``.
    batch_size : mini-batch size for each IQN. ``None`` uses full batch.
    verbose : if True, print progress for each component.

    Returns
    -------
    List of *d* tuples, each containing (model_j, xm_j, xs_j, ym_j, ys_j).
    The j-th model expects inputs of dimension p + j - 1.
    """
    n, d = Y.shape
    chain = []

    for j in range(d):
        if verbose:
            print(f"[MGBC] Training component {j+1}/{d}  "
                  f"(input dim = {X.shape[1] + j})")

        if j == 0:
            X_j = X
        else:
            X_j = np.column_stack([X, Y[:, :j]])

        model_j, xm_j, xs_j, ym_j, ys_j = train_iqn(
            X_j, Y[:, j], epochs=epochs, hdim=hdim, nh=nh,
            lr=lr, wd=wd, seed=seed + j, w=w,
            batch_size=batch_size, verbose=verbose,
        )
        chain.append((model_j, xm_j, xs_j, ym_j, ys_j))

    return chain


def sample_multivariate_iqn(
    chain: list[tuple],
    X_te: np.ndarray,
    B: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """Draw multivariate samples from an autoregressive IQN chain.

    For each draw *b*, samples τ₁, …, τ_d ~ U(0,1) independently and
    generates θ_j = Q_j(τ_j | x, θ₁, …, θ_{j-1}) sequentially.

    Parameters
    ----------
    chain : list of (model, xm, xs, ym, ys) from ``train_multivariate_iqn``.
    X_te : (n_test, p) test features (raw, unnormalized).
    B : number of Monte Carlo draws.
    seed : random seed for tau sampling.

    Returns
    -------
    (B, n_test, d) array of joint samples.
    """
    d = len(chain)
    n = X_te.shape[0]
    rng = np.random.default_rng(seed)
    taus = rng.uniform(0.01, 0.99, size=(B, d))

    samples = np.empty((B, n, d))

    for j in range(d):
        model_j, xm_j, xs_j, ym_j, ys_j = chain[j]

        # Build input: X_te + previously sampled components
        if j == 0:
            # (B, n, p) — replicate X_te for all B draws
            X_base = np.broadcast_to(X_te[np.newaxis, :, :], (B, n, X_te.shape[1]))
            X_j_all = X_base
        else:
            X_base = np.broadcast_to(X_te[np.newaxis, :, :], (B, n, X_te.shape[1]))
            X_j_all = np.concatenate([X_base, samples[:, :, :j]], axis=2)

        # Flatten (B, n) -> (B*n) for batch evaluation at per-draw taus
        X_flat = X_j_all.reshape(B * n, -1)
        X_norm = (X_flat - xm_j) / xs_j
        Xt = torch.tensor(X_norm, dtype=torch.float32)

        # Each of the B draws uses a different tau, but all n test points
        # in the same draw share the same tau
        tau_per_row = np.repeat(taus[:, j], n)  # (B*n,)

        # Evaluate in chunks of same-tau for efficiency
        theta_flat = np.empty(B * n)
        with torch.no_grad():
            for b in range(B):
                sl = slice(b * n, (b + 1) * n)
                f = model_j(Xt[sl], float(taus[b, j]))
                theta_flat[sl] = f[:, 1].numpy() * ys_j + ym_j

        samples[:, :, j] = theta_flat.reshape(B, n)

    return samples


def predict_multivariate_iqn(
    chain: list[tuple],
    X_te: np.ndarray,
    taus: np.ndarray | list[float] = (0.025, 0.25, 0.5, 0.75, 0.975),
) -> np.ndarray:
    """Predict quantiles at specific levels through the autoregressive chain.

    For each component k, predecessors θ₁, …, θ_{k-1} are fixed at their
    median predictions (τ=0.5), then the quantile function Q_k is evaluated
    at each requested τ level.

    Parameters
    ----------
    chain : list of (model, xm, xs, ym, ys) from ``train_multivariate_iqn``.
    X_te : (n_test, p) test features (raw, unnormalized).
    taus : quantile levels to evaluate.

    Returns
    -------
    (n_test, d, n_q) array of quantile predictions.
    """
    taus = np.asarray(taus)
    d = len(chain)
    n = X_te.shape[0]
    result = np.empty((n, d, len(taus)))

    theta_median = np.empty((n, 0))
    for j in range(d):
        model_j, xm_j, xs_j, ym_j, ys_j = chain[j]
        if j == 0:
            X_j = X_te
        else:
            X_j = np.column_stack([X_te, theta_median])

        Xt = torch.tensor((X_j - xm_j) / xs_j, dtype=torch.float32)
        with torch.no_grad():
            for q_idx, tau in enumerate(taus):
                f = model_j(Xt, float(tau))
                result[:, j, q_idx] = f[:, 1].numpy() * ys_j + ym_j
            f_med = model_j(Xt, 0.5)
            median_j = f_med[:, 1].numpy() * ys_j + ym_j

        theta_median = np.column_stack([theta_median, median_j])

    return result


def order_by_variance(Y: np.ndarray) -> np.ndarray:
    """Order components by decreasing marginal variance.

    Variables with larger marginal variance are placed first in the
    Cholesky ordering, as they tend to be more influential and benefit
    from being conditioned on early.

    Parameters
    ----------
    Y : (n, d) multivariate targets.

    Returns
    -------
    (d,) array of component indices, sorted by decreasing variance.
    """
    return np.argsort(-np.var(Y, axis=0))


def cholesky_precondition(
    Y: np.ndarray,
    regularize: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cholesky preconditioning for near-Gaussian multivariate targets.

    Transforms *Y* so that the marginal conditionals are approximately
    standard normal, making the IQN's job easier. The IQN then learns
    only the residual non-Gaussianity.

    Applies:  Z = L⁻¹ (Y − μ)  where  Σ = L Lᵀ  (Cholesky).

    To recover original samples:  Y = L Z + μ  (see ``cholesky_inverse``).

    Parameters
    ----------
    Y : (n, d) multivariate targets.
    regularize : small constant added to the diagonal of the covariance
        matrix for numerical stability. Set to 0 to disable.

    Returns
    -------
    Z : (n, d) whitened targets.
    L : (d, d) lower Cholesky factor.
    mu : (d,) mean vector.
    """
    mu = Y.mean(axis=0)
    Y_c = Y - mu
    cov = np.cov(Y_c, rowvar=False)
    if regularize > 0:
        cov += regularize * np.eye(cov.shape[0])
    L = np.linalg.cholesky(cov)
    L_inv = np.linalg.inv(L)
    Z = (L_inv @ Y_c.T).T
    return Z, L, mu


def cholesky_inverse(
    Z: np.ndarray,
    L: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Invert the Cholesky preconditioning to recover original scale.

    Parameters
    ----------
    Z : (..., d) whitened values (any batch shape).
    L : (d, d) lower Cholesky factor from ``cholesky_precondition``.
    mu : (d,) mean vector.

    Returns
    -------
    Y : (..., d) values in original scale.
    """
    return Z @ L.T + mu
