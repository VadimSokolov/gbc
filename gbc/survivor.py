"""Survivor function estimation and tail functionals from IQN quantile predictions.

The posterior mean of any non-negative random variable equals the integral
of its posterior survivor function:

    E[X | y] = integral_0^inf S(t | y) dt

where S(t | y) = P(X > t | y) = 1 - F(t | y).  This module converts IQN
quantile predictions into survivor function estimates and computes tail
functionals (return levels, exceedance probabilities, Expected Shortfall)
directly from the estimated survivor curve.

These tools work with *any* trained IQN — they are not specific to EVT.

Book references
---------------
- Polson & Sokolov (2026) §Pillar 1: survivor integral representation
- Polson, Ruggeri & Sokolov (2024): MEU via survivor integrals
"""

import numpy as np
from gbc.iqn import IQN, predict_iqn, sample_iqn


def survivor_curve(
    model: IQN,
    X: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    t_grid: np.ndarray,
    B: int = 500,
) -> np.ndarray:
    r"""Estimate the survivor function S(t | x) on a grid.

    For each test point x_i, the survivor function at level t is

        S(t | x_i) = 1 - F(t | x_i) = fraction of quantile samples > t.

    Parameters
    ----------
    model : trained IQN.
    X : (n, d) test inputs (raw, unnormalized).
    xm, xs : input normalization from train_iqn.
    ym, ys : output normalization from train_iqn.
    t_grid : (m,) sorted grid of values at which to evaluate S.
    B : number of quantile samples.

    Returns
    -------
    (m, n) array where entry [j, i] = S(t_grid[j] | X[i]).
    """
    samples = sample_iqn(model, X, xm, xs, ym, ys, B=B)  # (B, n)
    t_grid = np.asarray(t_grid)
    # S(t | x_i) = mean(samples[:, i] > t)
    return np.array([
        np.mean(samples > t, axis=0) for t in t_grid
    ])


def posterior_mean(
    model: IQN,
    X: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    B: int = 500,
) -> np.ndarray:
    r"""Posterior mean via the survivor integral identity.

    .. math::
        E[X | y] = \int_0^1 (1 - \tau)\, dQ(\tau | y)
                 \approx \sum_j (1 - \tau_j)(Q(\tau_{j+1}) - Q(\tau_j))

    Parameters
    ----------
    model : trained IQN.
    X : (n, d) test inputs.
    xm, xs, ym, ys : normalization stats from train_iqn.
    B : number of quantile levels for integration.

    Returns
    -------
    (n,) array of posterior means.
    """
    taus = np.linspace(0.005, 0.995, B)
    Q = predict_iqn(model, X, xm, xs, ym, ys, taus=taus)  # (B, n)
    # Trapezoidal integration: sum (1 - tau_j) * dQ_j
    dtau = np.diff(taus)
    tau_mid = 0.5 * (taus[:-1] + taus[1:])
    dQ = np.diff(Q, axis=0)  # (B-1, n)
    weights = (1.0 - tau_mid)[:, np.newaxis]  # (B-1, 1)
    return np.sum(weights * dQ, axis=0)


def return_level(
    model: IQN,
    X: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    N: float,
    B: int = 500,
) -> np.ndarray:
    r"""N-period return level: the (1 - 1/N) quantile.

    The return level z_N satisfies S(z_N | y) = 1/N.

    Parameters
    ----------
    model : trained IQN.
    X : (n, d) test inputs.
    xm, xs, ym, ys : normalization stats.
    N : return period (e.g. 100 for 100-year return level).
    B : number of quantile samples.

    Returns
    -------
    (n,) array of return levels.
    """
    tau = 1.0 - 1.0 / N
    Q = predict_iqn(model, X, xm, xs, ym, ys, taus=[tau])  # (1, n)
    return Q[0]


def exceedance_prob(
    model: IQN,
    X: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    threshold: float,
    B: int = 500,
) -> np.ndarray:
    r"""Exceedance probability P(Y > threshold | x).

    Parameters
    ----------
    model : trained IQN.
    X : (n, d) test inputs.
    xm, xs, ym, ys : normalization stats.
    threshold : scalar threshold.
    B : number of quantile samples.

    Returns
    -------
    (n,) array of exceedance probabilities.
    """
    samples = sample_iqn(model, X, xm, xs, ym, ys, B=B)  # (B, n)
    return np.mean(samples > threshold, axis=0)


def expected_shortfall(
    model: IQN,
    X: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    p: float = 0.95,
    B: int = 500,
) -> np.ndarray:
    r"""Expected Shortfall (Conditional Value at Risk) at level p.

    .. math::
        ES_p = E[X | X > Q(p)] = Q(p) + \frac{1}{S(Q(p))} \int_{Q(p)}^\infty S(t)\, dt

    In practice, computed as the mean of samples above the p-th quantile.

    Parameters
    ----------
    model : trained IQN.
    X : (n, d) test inputs.
    xm, xs, ym, ys : normalization stats.
    p : probability level in (0, 1).  ES_0.95 conditions on exceeding
        the 95th percentile.
    B : number of quantile samples.

    Returns
    -------
    (n,) array of Expected Shortfall values.
    """
    samples = sample_iqn(model, X, xm, xs, ym, ys, B=B)  # (B, n)
    q_p = np.quantile(samples, p, axis=0)  # (n,)
    n = X.shape[0]
    es = np.empty(n)
    for i in range(n):
        above = samples[:, i][samples[:, i] > q_p[i]]
        es[i] = np.mean(above) if len(above) > 0 else q_p[i]
    return es


def tail_weight_crps(
    y: np.ndarray,
    samples: np.ndarray,
    threshold: float | None = None,
    quantile: float = 0.9,
) -> float:
    r"""Tail-weighted CRPS that focuses evaluation on extreme quantiles.

    Computes CRPS restricted to observations exceeding a threshold,
    which better evaluates tail calibration than the standard CRPS.

    Parameters
    ----------
    y : (n,) observed values.
    samples : (B, n) quantile samples from the predictive distribution.
    threshold : if given, restrict to observations y > threshold.
        If None, uses the `quantile`-th quantile of y.
    quantile : quantile of y to use as threshold (default 0.9).

    Returns
    -------
    Scalar tail-weighted CRPS.
    """
    if threshold is None:
        threshold = np.quantile(y, quantile)
    mask = y > threshold
    if not np.any(mask):
        return 0.0
    y_tail = y[mask]
    s_tail = samples[:, mask]
    term1 = np.mean(np.abs(s_tail - y_tail[np.newaxis, :]), axis=0)
    idx = np.random.permutation(s_tail.shape[0])
    term2 = 0.5 * np.mean(np.abs(s_tail - s_tail[idx, :]), axis=0)
    return float(np.mean(term1 - term2))


def meu_optimal_action(
    model: IQN,
    X: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    action_grid: np.ndarray,
    utility_fn: callable,
    B: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Optimal action via Maximum Expected Utility and the survivor function.

    For each test point, solves

        a* = argmax_a  E[U(a, X) | y]

    where the expectation is computed from IQN samples.

    Parameters
    ----------
    model : trained IQN.
    X : (n, d) test inputs.
    xm, xs, ym, ys : normalization stats.
    action_grid : (m,) grid of candidate actions.
    utility_fn : callable(a, x) -> scalar utility.
        Must accept scalar a and array x of shape (B,).
    B : number of quantile samples.

    Returns
    -------
    (optimal_actions, expected_utilities) — both (n,) arrays.
    """
    samples = sample_iqn(model, X, xm, xs, ym, ys, B=B)  # (B, n)
    n = X.shape[0]
    m = len(action_grid)
    eu_matrix = np.empty((m, n))
    for j, a in enumerate(action_grid):
        for i in range(n):
            eu_matrix[j, i] = np.mean(utility_fn(a, samples[:, i]))
    best_idx = np.argmax(eu_matrix, axis=0)
    optimal_actions = action_grid[best_idx]
    expected_utilities = eu_matrix[best_idx, np.arange(n)]
    return optimal_actions, expected_utilities
