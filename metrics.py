"""Evaluation metrics: CRPS, coverage, PI width, RMSE, RMSPE.

Book references
---------------
- Ch 5 §sec-iqn-diagnostics : CRPS and coverage for IQN validation
- Ch 7 §sec-surrogates      : CRPS/RMSE for GP vs IQN benchmarks
- Ch 8 §sec-jumps-phantom   : RMSE/CRPS on jump-process benchmarks
- Ch 14 §sec-cal-lake       : coverage and PI width on lake temperature data

Notes
-----
``crps_samples`` uses an O(B) randomised estimator (random permutation of
samples) rather than the O(B²) exhaustive estimator. For small B (<50) the
O(B²) estimator is more accurate; for B≥200 the randomised version is
essentially unbiased.
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


def crps_samples(y: np.ndarray, samples: np.ndarray) -> float:
    r"""Energy-score estimator of CRPS.

    .. math::
        \text{CRPS}(F, y) = E|Y - y| - \frac{1}{2}E|Y - Y'|

    Parameters
    ----------
    y : (n,) observed values.
    samples : (B, n) array of B draws from the predictive distribution.
    """
    term1 = np.mean(np.abs(samples - y[np.newaxis, :]), axis=0)
    idx = np.random.permutation(samples.shape[0])
    term2 = 0.5 * np.mean(np.abs(samples - samples[idx, :]), axis=0)
    return float(np.mean(term1 - term2))


def coverage(y: np.ndarray, samples: np.ndarray, alpha: float = 0.90) -> float:
    """Empirical coverage of prediction intervals.

    Parameters
    ----------
    y : (n,) observed values.
    samples : (B, n) quantile samples.
    alpha : nominal coverage level.
    """
    lo = np.quantile(samples, (1 - alpha) / 2, axis=0)
    hi = np.quantile(samples, 1 - (1 - alpha) / 2, axis=0)
    return float(np.mean((y >= lo) & (y <= hi)))


def pi_width(samples: np.ndarray, alpha: float = 0.90) -> float:
    """Mean prediction interval width.

    Parameters
    ----------
    samples : (B, n) quantile samples.
    alpha : nominal coverage level.
    """
    lo = np.quantile(samples, (1 - alpha) / 2, axis=0)
    hi = np.quantile(samples, 1 - (1 - alpha) / 2, axis=0)
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


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def rmspe(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root mean squared percentage error."""
    return float(np.sqrt(np.mean(((y - y_hat) / np.maximum(np.abs(y), 1e-12)) ** 2)))
