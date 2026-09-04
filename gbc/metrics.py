"""Evaluation metrics: CRPS, coverage, PI width, RMSE, RMSPE.

Book references
---------------
- Ch 5 §sec-iqn-diagnostics : CRPS and coverage for IQN validation
- Ch 7 §sec-surrogates      : CRPS/RMSE for GP vs IQN benchmarks
- Ch 8 §sec-jumps-phantom   : RMSE/CRPS on jump-process benchmarks
- Ch 14 §sec-cal-lake       : coverage and PI width on lake temperature data

Notes
-----
``crps_samples`` computes empirical CRPS exactly by default using a sorted
sample identity. This takes O(B log B) time and O(B) working memory. A seeded
random-permutation approximation is available explicitly for unusually large B.
"""

import numpy as np
from scipy.stats import norm


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for Gaussian predictive distribution.

    Parameters
    ----------
    y : (n,) observed values.
    mu : (n,) predicted means.
    sigma : (n,) predicted standard deviations.
    """
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    return float(
        np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi)))
    )


def crps_samples(
    y: np.ndarray,
    samples: np.ndarray,
    *,
    method: str = "exact",
    seed: int | None = None,
) -> float:
    r"""CRPS for an empirical predictive distribution.

    .. math::
        \text{CRPS}(F, y) = E|Y - y| - \frac{1}{2}E|Y - Y'|

    Parameters
    ----------
    y : (n,) observed values.
    samples : (B, n) array of B draws from the predictive distribution.
    method : ``"exact"`` uses the deterministic sorted-sample identity;
        ``"mc"`` uses one seeded random permutation.
    seed : integer seed required when ``method="mc"``. The calculation uses
        an isolated generator and does not change NumPy's global random state.
    """
    y = np.asarray(y, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if samples.ndim != 2:
        raise ValueError("samples must be a two-dimensional (B, n) array")
    if samples.shape[0] == 0:
        raise ValueError("samples must contain at least one predictive value")
    if samples.shape[1] != y.shape[0]:
        raise ValueError("samples and y must have the same observation count")

    term1 = np.mean(np.abs(samples - y[np.newaxis, :]), axis=0)
    B = samples.shape[0]
    if method == "exact":
        ordered = np.sort(samples, axis=0)
        ranks = np.arange(1, B + 1, dtype=np.float64)
        weights = 2.0 * ranks - B - 1.0
        term2 = np.sum(ordered * weights[:, np.newaxis], axis=0) / float(B) ** 2
    elif method == "mc":
        if seed is None:
            raise ValueError("seed is required when method='mc'")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise ValueError("seed must be an integer")
        idx = np.random.default_rng(int(seed)).permutation(B)
        term2 = 0.5 * np.mean(np.abs(samples - samples[idx, :]), axis=0)
    else:
        raise ValueError("method must be 'exact' or 'mc'")
    return float(np.mean(term1 - term2))


def _quantile_interval(
    samples: np.ndarray, alpha: float = 0.90
) -> tuple[np.ndarray, np.ndarray]:
    """Extract lower and upper bounds of a prediction interval from samples."""
    lo = np.quantile(samples, (1 - alpha) / 2, axis=0)
    hi = np.quantile(samples, 1 - (1 - alpha) / 2, axis=0)
    return lo, hi


def coverage(y: np.ndarray, samples: np.ndarray, alpha: float = 0.90) -> float:
    """Empirical coverage of prediction intervals.

    Parameters
    ----------
    y : (n,) observed values.
    samples : (B, n) quantile samples.
    alpha : nominal coverage level.
    """
    lo, hi = _quantile_interval(samples, alpha)
    return float(np.mean((y >= lo) & (y <= hi)))


def pi_width(samples: np.ndarray, alpha: float = 0.90) -> float:
    """Mean prediction interval width.

    Parameters
    ----------
    samples : (B, n) quantile samples.
    alpha : nominal coverage level.
    """
    lo, hi = _quantile_interval(samples, alpha)
    return float(np.mean(hi - lo))


def pit_values(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Probability Integral Transform (PIT) values.

    For a well-calibrated model, PIT values should be approximately
    Uniform(0, 1). This function extracts the values; use
    ``calibration_plot`` to visualize them as a histogram.

    Parameters
    ----------
    y : (n,) observed values.
    samples : (B, n) quantile samples from the predictive distribution.

    Returns
    -------
    (n,) PIT values in [0, 1].
    """
    return np.mean(samples <= y[np.newaxis, :], axis=0)


def energy_score(y: np.ndarray, samples: np.ndarray) -> float:
    r"""Energy Score — multivariate proper scoring rule (Gneiting & Raftery, 2007).

    Multivariate generalization of CRPS:

    .. math::
        \text{ES}(F, y) = E_F\|\theta - y\| - \frac{1}{2}E_F\|\theta - \theta'\|

    Uses an O(B) randomised estimator via random permutation.

    Parameters
    ----------
    y : (n, d) observed multivariate values.
    samples : (B, n, d) draws from the predictive distribution.

    Returns
    -------
    Scalar mean energy score across observations.
    """
    B = samples.shape[0]
    term1 = np.mean(np.linalg.norm(samples - y[np.newaxis, :, :], axis=2), axis=0)
    idx = np.random.permutation(B)
    term2 = 0.5 * np.mean(np.linalg.norm(samples - samples[idx], axis=2), axis=0)
    return float(np.mean(term1 - term2))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def rmspe(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared percentage error."""
    return float(np.sqrt(np.mean(((y - y_hat) / np.maximum(np.abs(y), 1e-12)) ** 2)))
