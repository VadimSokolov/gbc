"""Sensitivity analysis for trained IQN models.

Numerical partial derivatives, elasticities, and feature importance
across quantile levels. These tools apply to any trained IQN, regardless
of domain.

Book references
---------------
- Ch 7 §sec-surrogates : sensitivity of surrogates to input perturbations
- Ch 5 §sec-iqn-diagnostics : interpreting learned quantile functions

Notes
-----
All functions operate on raw (unnormalized) inputs and handle the
normalization internally using the (xm, xs, ym, ys) statistics returned
by ``train_iqn``.
"""

import numpy as np
import torch

from gbc.iqn import IQN


def _predict_at(
    model: IQN,
    x: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    taus: np.ndarray,
) -> np.ndarray:
    """Predict quantiles at specific tau values for a single input point.

    Parameters
    ----------
    model : trained IQN (eval mode).
    x : (1, d) or (d,) raw input point.
    xm, xs : input normalization stats from train_iqn.
    ym, ys : output normalization stats from train_iqn.
    taus : (T,) quantile levels.

    Returns
    -------
    (T,) predicted quantile values (denormalized).
    """
    x = np.atleast_2d(x)
    xt = torch.tensor((x - xm) / xs, dtype=torch.float32)
    qs = np.empty(len(taus))
    with torch.no_grad():
        for i, tau in enumerate(taus):
            f = model(xt, float(tau))
            qs[i] = f[0, 1].item() * ys + ym
    return qs


def partial_effect(
    model: IQN,
    x0: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    feature: int,
    delta: float = 0.01,
    taus: np.ndarray | None = None,
) -> np.ndarray:
    r"""Numerical partial derivative of the quantile function.

    Computes :math:`\partial \hat{Q}(\tau \mid x) / \partial x_j` via
    central differences at the point ``x0``.

    Parameters
    ----------
    model : trained IQN (eval mode).
    x0 : (d,) or (1, d) raw input point at which to evaluate.
    xm, xs : input normalization stats from train_iqn.
    ym, ys : output normalization stats from train_iqn.
    feature : index of the feature to perturb.
    delta : perturbation size (in raw input units).
    taus : (T,) quantile levels. Defaults to linspace(0.05, 0.95, 19).

    Returns
    -------
    (T,) array of partial derivatives, one per quantile level.
    """
    if taus is None:
        taus = np.linspace(0.05, 0.95, 19)
    x0 = np.atleast_1d(x0).astype(float)

    x_lo = x0.copy()
    x_hi = x0.copy()
    x_lo[feature] -= delta / 2
    x_hi[feature] += delta / 2

    q_lo = _predict_at(model, x_lo, xm, xs, ym, ys, taus)
    q_hi = _predict_at(model, x_hi, xm, xs, ym, ys, taus)
    return (q_hi - q_lo) / delta


def elasticity(
    model: IQN,
    x0: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    feature: int,
    pct: float = 0.01,
    taus: np.ndarray | None = None,
) -> np.ndarray:
    r"""Elasticity of the quantile function w.r.t. one feature.

    Computes the percentage change in :math:`\hat{Q}(\tau \mid x)` per
    percentage change in ``x[feature]``:

    .. math::
        \varepsilon_j(\tau) \approx
        \frac{\hat{Q}(\tau \mid x + \Delta) - \hat{Q}(\tau \mid x)}
             {\hat{Q}(\tau \mid x)} \Big/ \frac{\Delta}{x_j}

    where :math:`\Delta` perturbs only feature *j* by ``pct * x_j``.

    Parameters
    ----------
    model : trained IQN (eval mode).
    x0 : (d,) or (1, d) raw input point.
    xm, xs : input normalization stats.
    ym, ys : output normalization stats.
    feature : feature index.
    pct : fractional perturbation (default 0.01 = 1%).
    taus : (T,) quantile levels. Defaults to linspace(0.05, 0.95, 19).

    Returns
    -------
    (T,) array of elasticities, one per quantile level.
    """
    if taus is None:
        taus = np.linspace(0.05, 0.95, 19)
    x0 = np.atleast_1d(x0).astype(float)
    x_val = x0[feature]
    if abs(x_val) < 1e-12:
        raise ValueError(
            f"Feature {feature} is near zero ({x_val}); "
            "elasticity is undefined. Use partial_effect instead."
        )

    q_base = _predict_at(model, x0, xm, xs, ym, ys, taus)
    x_up = x0.copy()
    x_up[feature] *= 1 + pct
    q_up = _predict_at(model, x_up, xm, xs, ym, ys, taus)

    dq_pct = (q_up - q_base) / np.where(np.abs(q_base) > 1e-12, q_base, 1e-12)
    return dq_pct / pct


def feature_effects(
    model: IQN,
    X: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    delta: float = 0.01,
    taus: np.ndarray | None = None,
    max_obs: int = 200,
    seed: int = 42,
) -> np.ndarray:
    r"""Average partial effects for all features across quantile levels.

    Computes ``partial_effect`` at each observation in *X* (or a random
    subsample) and returns the mean, yielding a (d, T) matrix of average
    marginal effects.

    Parameters
    ----------
    model : trained IQN (eval mode).
    X : (n, d) raw input matrix.
    xm, xs : input normalization stats.
    ym, ys : output normalization stats.
    delta : perturbation size (passed to ``partial_effect``).
    taus : (T,) quantile levels.
    max_obs : subsample size if n > max_obs.
    seed : random seed for subsampling.

    Returns
    -------
    (d, T) array — average partial effect for each feature and quantile.
    """
    if taus is None:
        taus = np.linspace(0.05, 0.95, 19)
    n, d = X.shape
    if n > max_obs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_obs, replace=False)
        X = X[idx]
        n = max_obs

    effects = np.zeros((d, len(taus)))
    for j in range(d):
        acc = np.zeros(len(taus))
        for i in range(n):
            acc += partial_effect(model, X[i], xm, xs, ym, ys, j, delta, taus)
        effects[j] = acc / n
    return effects
