"""Bayesian hypothesis testing via Tweedie's formula.

Implements the spike-and-slab testing framework from:

  Datta, Polson, Sokolov, Zantedeschi (2026).
  "Tweedie's Formula for Testing: Score Identities from
   Hypothesis Tests to Diffusion Models."
  ICML Workshop on Hypothesis Testing.

The central identity is Tweedie's formula

    E[theta | z] = z + ell'(z),

where ell(z) = log m(z) is the log marginal density.  For testing
H0: theta = 0 vs H1: theta ~ g1, the test statistic is

    T(z) = z + ell'(z) = Pr(H1|z) * E[theta | z, H1],

and the local false discovery rate is

    fdr(z) = pi0 * phi(z) / m(z) = pi0 * exp(ell0(z) - ell(z)).

The fdr threshold fdr(z) <= q is equivalent to a Bayes factor threshold
B10(z) >= K = (pi0/pi1) * (1-q)/q  (Corollary 1 of the paper).

For diffusion generative models with forward process x_t = alpha_t x0 + sigma_t eps,
Tweedie's formula yields

    E[x0 | x_t] = (x_t + sigma_t^2 * nabla log p_t(x_t)) / alpha_t,

so the Tweedie test statistic T(x_t) = alpha_t * E[x0 | x_t] is exactly
the denoised prediction computed by a trained score network.
"""

from __future__ import annotations

import numpy as np
from scipy import stats, integrate, optimize, interpolate


# ── slab marginal ─────────────────────────────────────────────────────────────

def build_slab_marginal(
    slab_pdf,
    z_lo: float = -9.0,
    z_hi: float = 9.0,
    n_grid: int = 1801,
    integration_limit: float = 50.0,
) -> interpolate.interp1d:
    """Pre-compute m1(z) = integral phi(z-theta) slab_pdf(theta) dtheta on a grid.

    Parameters
    ----------
    slab_pdf : callable
        Probability density of the slab prior; slab_pdf(theta) -> float.
    z_lo, z_hi : float
        Range for the pre-computation grid.
    n_grid : int
        Number of grid points (odd for symmetry).
    integration_limit : float
        Truncation limit for the theta integral.

    Returns
    -------
    interpolate.interp1d
        Interpolator for m1(z); extrapolates flat outside [z_lo, z_hi].

    Examples
    --------
    >>> from scipy.stats import cauchy
    >>> m1 = build_slab_marginal(cauchy.pdf)
    >>> float(m1(0.0)) > 0
    True
    """
    grid = np.linspace(z_lo, z_hi, n_grid)
    vals = np.empty(n_grid)
    for i, z in enumerate(grid):
        v, _ = integrate.quad(
            lambda t: stats.norm.pdf(z - t) * slab_pdf(t),
            -integration_limit, integration_limit, limit=300,
        )
        vals[i] = v
    return interpolate.interp1d(
        grid, vals, kind="cubic",
        bounds_error=False, fill_value=(vals[0], vals[-1]),
    )


def cauchy_slab_marginal(
    z_lo: float = -9.0, z_hi: float = 9.0, n_grid: int = 1801,
) -> interpolate.interp1d:
    """Pre-computed m1(z) for the standard Cauchy slab prior.

    Convenience wrapper around :func:`build_slab_marginal` for the most
    commonly used slab in sparse inference.

    Returns
    -------
    interpolate.interp1d
        Interpolator for the Cauchy-slab marginal.
    """
    return build_slab_marginal(stats.cauchy.pdf, z_lo=z_lo, z_hi=z_hi, n_grid=n_grid)


# ── empirical Bayes sparsity estimation ───────────────────────────────────────

def estimate_sparsity(z: np.ndarray, m1_interp) -> float:
    """Estimate pi1 by maximising the marginal log-likelihood.

    Solves  argmax_{w in (0,0.5)}  sum_i log[(1-w)*phi(z_i) + w*m1(z_i)].

    Parameters
    ----------
    z : (n,) array of z-scores.
    m1_interp : callable
        Interpolator for the slab marginal, e.g. from :func:`cauchy_slab_marginal`.

    Returns
    -------
    float
        Estimated non-null proportion pi1_hat in (0, 0.5).
    """
    phi0 = stats.norm.pdf(z)
    m1z = np.asarray(m1_interp(z), dtype=float)

    def neg_ll(w: float) -> float:
        marg = (1.0 - w) * phi0 + w * m1z
        return -np.sum(np.log(np.maximum(marg, 1e-300)))

    res = optimize.minimize_scalar(neg_ll, bounds=(1e-4, 0.499), method="bounded")
    return float(res.x)


# ── local fdr and Bayes factor ────────────────────────────────────────────────

def local_fdr(z: np.ndarray, pi1: float, m1_interp) -> np.ndarray:
    """Local false discovery rate  fdr(z) = pi0 phi(z) / m(z).

    Parameters
    ----------
    z : (n,) array of z-scores.
    pi1 : float
        Non-null proportion (use :func:`estimate_sparsity` for data-driven estimate).
    m1_interp : callable
        Interpolator for the slab marginal.

    Returns
    -------
    (n,) array of fdr values in [0, 1].
    """
    pi0  = 1.0 - pi1
    phi0 = stats.norm.pdf(z)
    m1z  = np.asarray(m1_interp(z), dtype=float)
    marg = pi0 * phi0 + pi1 * m1z
    marg = np.maximum(marg, 1e-300)
    return pi0 * phi0 / marg


def log_bayes_factor(z: np.ndarray, m1_interp) -> np.ndarray:
    """Log Bayes factor  log B10(z) = log m1(z) - log phi(z).

    Parameters
    ----------
    z : (n,) array of z-scores.
    m1_interp : callable
        Interpolator for the slab marginal.

    Returns
    -------
    (n,) array of log Bayes factors.
    """
    m1z  = np.maximum(np.asarray(m1_interp(z), dtype=float), 1e-300)
    phi0 = np.maximum(stats.norm.pdf(z), 1e-300)
    return np.log(m1z) - np.log(phi0)


# ── selection procedures ──────────────────────────────────────────────────────

def tweedie_select(
    z: np.ndarray,
    q: float,
    m1_interp,
    pi1: float | None = None,
) -> np.ndarray:
    """Select hypotheses by thresholding the local fdr at level q.

    Uses empirical Bayes to estimate pi1 when not supplied.

    Parameters
    ----------
    z : (n,) array of z-scores.
    q : float
        Target fdr level (e.g. 0.10).
    m1_interp : callable
        Interpolator for the slab marginal.
    pi1 : float or None
        Non-null proportion.  Estimated from data when None.

    Returns
    -------
    (n,) boolean array; True = rejected (signal declared).
    """
    if pi1 is None:
        pi1 = estimate_sparsity(z, m1_interp)
    fdr = local_fdr(z, pi1, m1_interp)
    return fdr <= q


def bh_reject(z: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg procedure at level q.

    Parameters
    ----------
    z : (n,) array of z-scores.
    q : float
        Target FDR level.

    Returns
    -------
    (n,) boolean array; True = rejected.
    """
    n      = len(z)
    pvals  = 2.0 * stats.norm.sf(np.abs(z))
    order  = np.argsort(pvals)
    psort  = pvals[order]
    thresh = q * np.arange(1, n + 1) / n
    below  = psort <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    cutoff = psort[np.where(below)[0].max()]
    return pvals <= cutoff


# ── screening metrics ─────────────────────────────────────────────────────────

def fdp_power(
    selected: np.ndarray,
    is_signal: np.ndarray,
) -> tuple[float, float]:
    """Compute false discovery proportion and power for a selection rule.

    Parameters
    ----------
    selected : (n,) boolean array.
    is_signal : (n,) boolean array; True = true non-null.

    Returns
    -------
    fdp : float
        False discovery proportion = FD / max(|selected|, 1).
    power : float
        Power = TD / |is_signal|.
    """
    sel = np.asarray(selected, dtype=bool)
    sig = np.asarray(is_signal, dtype=bool)
    n_sig = sig.sum()
    fd  = (sel & ~sig).sum()
    td  = (sel &  sig).sum()
    fdp = float(fd) / max(int(sel.sum()), 1)
    pwr = float(td) / max(int(n_sig), 1)
    return fdp, pwr


# ── diffusion model Tweedie test statistic ────────────────────────────────────

def mog_score(
    x_t: np.ndarray,
    means: list[float],
    weights: list[float],
    alpha_t: float,
    sigma_t: float,
    sigma0: float = 1.0,
) -> np.ndarray:
    """Analytical score nabla log p_t(x_t) for a Gaussian mixture prior.

    For  p0(x0) = sum_k w_k N(x0; mu_k, sigma0^2)  and forward process
    x_t = alpha_t x0 + sigma_t eps,  the diffused marginal is

        p_t(x_t) = sum_k w_k N(x_t; alpha_t mu_k,  alpha_t^2 sigma0^2 + sigma_t^2).

    Parameters
    ----------
    x_t : (n,) noisy observations.
    means : list of component means mu_k.
    weights : list of mixing weights w_k (must sum to 1).
    alpha_t : float
        Signal coefficient at noise level t.
    sigma_t : float
        Noise standard deviation at noise level t.
    sigma0 : float
        Component standard deviation of the prior.

    Returns
    -------
    (n,) array of score values nabla log p_t(x_t).
    """
    sigma_eff2 = alpha_t**2 * sigma0**2 + sigma_t**2
    scores_w   = np.zeros_like(x_t, dtype=float)
    norm_const = np.zeros_like(x_t, dtype=float)

    for mu_k, w_k in zip(means, weights):
        center  = alpha_t * mu_k
        prob_k  = stats.norm.pdf(x_t, loc=center, scale=np.sqrt(sigma_eff2)) * w_k
        score_k = -(x_t - center) / sigma_eff2
        scores_w   += prob_k * score_k
        norm_const += prob_k

    return scores_w / np.maximum(norm_const, 1e-300)


def tweedie_stat(
    x_t: np.ndarray,
    score: np.ndarray,
    sigma_t: float,
) -> np.ndarray:
    """Tweedie test statistic T(x_t) = x_t + sigma_t^2 * score.

    In a diffusion model this equals alpha_t * E[x0 | x_t], the scaled
    denoised prediction.  Under H0: x0 = 0, T(x_t) = 0 in expectation.
    Under H1: x0 ~ p0, T(x_t) tracks the expected signal amplitude.

    Parameters
    ----------
    x_t : (n,) noisy observations.
    score : (n,) score estimates nabla log p_t(x_t).
    sigma_t : float
        Noise standard deviation (matches the score network's noise level).

    Returns
    -------
    (n,) Tweedie test statistic values.
    """
    return x_t + sigma_t**2 * score
