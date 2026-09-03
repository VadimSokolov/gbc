"""MCMC convergence diagnostics: rank-normalized split-Rhat and effective sample size.

Implements the estimators of Vehtari, Gelman, Simpson, Carpenter and Buerkner
(2021), "Rank-normalization, folding, and localization: an improved Rhat for
assessing convergence of MCMC", Bayesian Analysis 16(2):667-718.

Self-contained (numpy only) so that a benchmark run needs no extra dependency
beyond the package's own.  ``rhat`` and ``ess_bulk`` take an (M, N) array of M
chains of length N; ``ess_per_second`` converts an ESS into the cost-per-
effective-draw that a wall-clock comparison against an amortized network needs.
"""
import numpy as np
from scipy.stats import norm

__all__ = ["split_chains", "rank_normalize", "rhat", "ess_bulk",
           "ess_per_second", "summarize_chains"]


def split_chains(draws):
    """Split each of M chains in half, returning a (2M, N//2) array."""
    d = np.asarray(draws, float)
    if d.ndim == 1:
        d = d[None, :]
    m, n = d.shape
    h = n // 2
    return np.concatenate([d[:, :h], d[:, h:2 * h]], axis=0)


def rank_normalize(draws):
    """Rank-normalize pooled draws to standard normal scores (Blom transform)."""
    d = np.asarray(draws, float)
    flat = d.ravel()
    order = np.argsort(np.argsort(flat, kind="stable"), kind="stable") + 1.0
    s = flat.size
    return norm.ppf((order - 3.0 / 8.0) / (s - 1.0 / 4.0)).reshape(d.shape)


def _var_estimates(d):
    """Within-chain variance W and the marginal-posterior variance estimate."""
    m, n = d.shape
    w = float(np.mean(np.var(d, axis=1, ddof=1)))
    b = n * float(np.var(np.mean(d, axis=1), ddof=1)) if m > 1 else 0.0
    var_hat = ((n - 1.0) / n) * w + b / n
    return w, b, var_hat


def rhat(draws, rank_normalized=True):
    """Rank-normalized split-Rhat.  Values below 1.01 indicate convergence."""
    d = split_chains(draws)
    if rank_normalized:
        d = rank_normalize(d)
    w, _, var_hat = _var_estimates(d)
    if w <= 0:
        return np.inf
    return float(np.sqrt(var_hat / w))


def _autocov_fft(x):
    """Autocovariance of a single chain at all lags, via FFT."""
    n = x.size
    c = x - x.mean()
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(c, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n]
    return acov / n


def ess_bulk(draws, rank_normalized=True):
    """Bulk effective sample size (Geyer initial positive/monotone sequence)."""
    d = split_chains(draws)
    if rank_normalized:
        d = rank_normalize(d)
    m, n = d.shape
    if n < 4:
        return float("nan")
    w, _, var_hat = _var_estimates(d)
    if var_hat <= 0 or w <= 0:
        return float("nan")

    acov = np.array([_autocov_fft(row) for row in d])          # (m, n)
    mean_acov = acov.mean(axis=0)
    rho = 1.0 - (w - mean_acov) / var_hat                       # rho_hat_t
    rho[0] = 1.0

    # Paired autocorrelations P_t = rho_{2t} + rho_{2t+1}, starting at lag 0.
    npair = n // 2
    pair = rho[0:2 * npair:2] + rho[1:2 * npair:2]
    # Geyer initial positive sequence: truncate at the first non-positive pair.
    nonpos = np.nonzero(pair <= 0)[0]
    pair = pair[:nonpos[0]] if nonpos.size else pair
    if pair.size == 0:
        return float(m * n)
    # Geyer initial monotone sequence: non-increasing envelope.
    pair = np.minimum.accumulate(pair)
    tau = -1.0 + 2.0 * float(pair.sum())
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def ess_per_second(draws, seconds, rank_normalized=True):
    """Effective draws per second: the cost unit a wall-clock MCMC claim needs."""
    e = ess_bulk(draws, rank_normalized=rank_normalized)
    return float(e / seconds) if seconds > 0 else float("nan")


def summarize_chains(chains, names=None):
    """Per-parameter Rhat and bulk ESS for an (M, N, P) array of chains.

    Returns a list of dicts with keys ``name``, ``rhat``, ``ess_bulk``, ``mean``.
    """
    c = np.asarray(chains, float)
    if c.ndim == 2:
        c = c[:, :, None]
    p = c.shape[2]
    names = names or [f"theta[{j}]" for j in range(p)]
    return [{"name": names[j],
             "rhat": rhat(c[:, :, j]),
             "ess_bulk": ess_bulk(c[:, :, j]),
             "mean": float(c[:, :, j].mean())} for j in range(p)]
