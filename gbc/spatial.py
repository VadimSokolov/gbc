"""Spatial utilities for GBC with areal or point-referenced data.

General-purpose tools for encoding spatial structure as IQN features.
These replace the need for explicit spatial priors (CAR, Matern) by
converting spatial dependence into covariates that a standard IQN can
learn from.

Book references
---------------
- Ch 10 §sec-bayes-iqn : GBC for spatially structured simulators
- Ch 7 §sec-surrogates : surrogate modeling with structured inputs

Notes
-----
These functions work with any spatial weight matrix — the user provides
the adjacency or distance structure, and the functions encode it as
features. No specific spatial library (e.g. PySAL, geopandas) is
required; inputs are plain numpy arrays.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def row_standardize(W: np.ndarray) -> np.ndarray:
    """Row-standardize a spatial weight matrix.

    Each row sums to 1 (or 0 for isolates). Converts a binary adjacency
    matrix into a row-stochastic matrix suitable for computing spatial lags.

    Parameters
    ----------
    W : (K, K) weight matrix (binary or general non-negative).

    Returns
    -------
    (K, K) row-standardized weight matrix.
    """
    W = np.asarray(W, dtype=float)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return W / row_sums


def spatial_lag(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute the spatial lag Wx.

    For a row-standardized *W*, this returns the neighbor-average of *x*.
    For a binary *W*, this returns the neighbor-sum.

    Parameters
    ----------
    W : (K, K) spatial weight matrix.
    x : (K,) or (K, p) feature vector(s).

    Returns
    -------
    Array of same shape as *x* containing spatially lagged values.
    """
    W = np.asarray(W, dtype=float)
    x = np.asarray(x, dtype=float)
    return W @ x


def spatial_features(
    X: np.ndarray,
    W: np.ndarray,
    lags: int = 1,
    standardize: bool = True,
) -> np.ndarray:
    """Augment a feature matrix with spatial lags.

    Appends first- through ``lags``-th order spatial lags of every column
    in *X*. For ``lags=1``, the output has 2*d columns (original + lag).
    For ``lags=2``, it has 3*d columns (original + W*X + W^2*X), etc.

    Parameters
    ----------
    X : (K, d) feature matrix (one row per spatial unit).
    W : (K, K) spatial weight matrix (will be row-standardized if requested).
    lags : number of spatial lag orders to include.
    standardize : if True, row-standardize W before computing lags.

    Returns
    -------
    (K, d * (1 + lags)) augmented feature matrix.
    """
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if standardize:
        W = row_standardize(W)

    parts = [X]
    lag_X = X.copy()
    for _ in range(lags):
        lag_X = W @ lag_X
        parts.append(lag_X)
    return np.column_stack(parts)


def moran_eigenvectors(
    W: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Extract Moran eigenvectors for spatial filtering.

    Moran eigenvectors are the eigenvectors of the doubly-centered
    spatial weight matrix ``MCM'`` where ``M = I - 11'/n`` and
    ``C = (W + W') / 2``. They form an orthogonal basis ordered by
    spatial autocorrelation (highest first) and can be used as IQN
    features to capture spatial structure non-parametrically.

    Parameters
    ----------
    W : (K, K) spatial weight matrix (binary or general).
    k : number of eigenvectors to return (largest eigenvalues).

    Returns
    -------
    (K, k) matrix of eigenvectors, columns ordered by descending
    eigenvalue (strongest spatial pattern first).

    References
    ----------
    Griffith, D. A. (2003). Spatial Autocorrelation and Spatial Filtering.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    C = (W + W.T) / 2
    M = np.eye(n) - np.ones((n, n)) / n
    MCM = M @ C @ M

    k = min(k, n - 1)
    if sparse.issparse(MCM):
        eigenvalues, eigenvectors = eigsh(MCM, k=k, which="LM")
    else:
        eigenvalues, eigenvectors = eigsh(
            sparse.csr_matrix(MCM), k=k, which="LM"
        )

    # Sort by descending eigenvalue
    order = np.argsort(-eigenvalues)
    return eigenvectors[:, order]


def spatial_panel_features(
    X: np.ndarray,
    W: np.ndarray,
    unit_ids: np.ndarray,
    time_ids: np.ndarray,
    lags: int = 1,
    time_normalize: bool = True,
) -> np.ndarray:
    """Augment a panel dataset with spatial lags computed per time period.

    For panel data where rows are (unit, time) pairs, computes spatial
    lags within each time period using the weight matrix *W*.

    Parameters
    ----------
    X : (n, d) feature matrix (n = K * T for balanced panel).
    W : (K, K) spatial weight matrix.
    unit_ids : (n,) integer unit identifiers (matching W's row order).
    time_ids : (n,) time period identifiers.
    lags : number of spatial lag orders.
    time_normalize : if True, append normalized time index as a feature.

    Returns
    -------
    (n, d_out) augmented feature matrix with spatial lags and optional time.
    """
    W_std = row_standardize(np.asarray(W, dtype=float))
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    unit_ids = np.asarray(unit_ids)
    time_ids = np.asarray(time_ids)

    unique_units = np.sort(np.unique(unit_ids))
    unique_times = np.sort(np.unique(time_ids))
    unit_to_idx = {u: i for i, u in enumerate(unique_units)}

    # Pre-allocate lag columns
    n, d = X.shape
    lag_cols = np.zeros((n, d * lags))

    for t in unique_times:
        mask = time_ids == t
        units_t = unit_ids[mask]
        idx_in_W = np.array([unit_to_idx[u] for u in units_t])

        W_sub = W_std[np.ix_(idx_in_W, idx_in_W)]
        X_t = X[mask]

        lag_X = X_t.copy()
        for lag_order in range(lags):
            lag_X = W_sub @ lag_X
            col_start = lag_order * d
            lag_cols[mask, col_start : col_start + d] = lag_X

    parts = [X, lag_cols]
    if time_normalize:
        t_min, t_max = time_ids.min(), time_ids.max()
        t_range = t_max - t_min if t_max > t_min else 1.0
        t_norm = ((time_ids - t_min) / t_range).reshape(-1, 1)
        parts.append(t_norm)

    return np.column_stack(parts)
