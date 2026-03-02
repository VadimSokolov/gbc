"""Conformal calibration for prediction intervals.

Temporal cross-validation conformal: split training dates into K folds,
collect out-of-fold residuals, compute per-stratum quantiles.

Book references
---------------
- Ch 14 §sec-cal-conformal     : conformal post-processing for coverage guarantees
- Ch 14 §sec-cal-per-horizon   : per-stratum (per-horizon) quantile adjustment
- Ch 14 §sec-cal-lake          : lake temperature benchmark (90.2% coverage)

Notes
-----
``conformal_pi`` constructs symmetric intervals (y_hat ± q). All stratum
keys in ``strata`` must be present in ``quantile_map``; use
``temporal_cv_quantiles`` to build the map, which falls back to the global
quantile for strata with fewer than ``min_count`` observations.
"""

import numpy as np


def temporal_cv_quantiles(
    residuals: np.ndarray,
    strata: np.ndarray,
    alpha: float = 0.90,
    min_count: int = 20,
) -> dict:
    """Compute per-stratum conformal quantiles from out-of-fold residuals.

    Parameters
    ----------
    residuals : (n,) absolute residuals from out-of-fold predictions.
    strata : (n,) stratum labels (e.g. forecast horizon).
    alpha : target coverage level.
    min_count : minimum observations per stratum; below this, use global quantile.

    Returns
    -------
    Dictionary mapping stratum -> quantile threshold.
    """
    global_q = float(np.percentile(residuals, 100 * alpha))
    quantiles = {}
    for s in np.unique(strata):
        mask = strata == s
        if mask.sum() >= min_count:
            quantiles[s] = float(np.percentile(residuals[mask], 100 * alpha))
        else:
            quantiles[s] = global_q
    return quantiles


def conformal_pi(
    y_hat: np.ndarray,
    strata: np.ndarray,
    quantile_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct conformal prediction intervals.

    Parameters
    ----------
    y_hat : (n,) point predictions.
    strata : (n,) stratum labels.
    quantile_map : dict from temporal_cv_quantiles.

    Returns
    -------
    (lower, upper) arrays of shape (n,).
    """
    q_vec = np.array([quantile_map[s] for s in strata])
    return y_hat - q_vec, y_hat + q_vec
