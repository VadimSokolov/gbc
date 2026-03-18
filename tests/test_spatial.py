"""Spatial use case: spatial feature engineering on a grid.

DGP: 5×5 grid (K=25 units), y = 0.5·Wy + x + N(0, 0.1)
     where W is queen contiguity and x ~ U(0, 1).
"""

import numpy as np
import pytest

from gbc.spatial import (
    row_standardize, spatial_lag, spatial_features,
    moran_eigenvectors, spatial_panel_features,
)


@pytest.fixture(scope="module")
def grid_data():
    """5×5 grid with queen contiguity weight matrix."""
    K = 25
    W = np.zeros((K, K))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 5 and 0 <= nj < 5:
                        W[idx, ni * 5 + nj] = 1.0

    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, K)
    # Spatial AR: y = (I - 0.5W)^{-1}(x + eps)
    W_std = row_standardize(W)
    eps = rng.normal(0, 0.1, K)
    y = np.linalg.solve(np.eye(K) - 0.5 * W_std, x + eps)
    return W, x, y


# ── row_standardize ─────────────────────────────────────────────

class TestRowStandardize:
    def test_rows_sum_to_one(self, grid_data):
        W, _, _ = grid_data
        W_std = row_standardize(W)
        row_sums = W_std.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_preserves_zeros(self, grid_data):
        W, _, _ = grid_data
        W_std = row_standardize(W)
        assert np.all(W_std[W == 0] == 0)

    def test_isolate_handling(self):
        W = np.array([[0, 0], [1, 0]], dtype=float)
        W_std = row_standardize(W)
        # Row 0 is an isolate → should stay all zeros
        assert W_std[0, 0] == 0.0
        assert W_std[0, 1] == 0.0


# ── spatial_lag ─────────────────────────────────────────────────

class TestSpatialLag:
    def test_shape_1d(self, grid_data):
        W, x, _ = grid_data
        lag = spatial_lag(row_standardize(W), x)
        assert lag.shape == x.shape

    def test_shape_2d(self, grid_data):
        W, x, _ = grid_data
        X = np.column_stack([x, x * 2])
        lag = spatial_lag(row_standardize(W), X)
        assert lag.shape == X.shape

    def test_corner_has_fewer_neighbors(self, grid_data):
        W, _, _ = grid_data
        # Corner (0,0) has 3 neighbors, center (2,2) has 8
        assert W[0].sum() == 3
        assert W[12].sum() == 8


# ── spatial_features ────────────────────────────────────────────

class TestSpatialFeatures:
    def test_shape_one_lag(self, grid_data):
        W, x, _ = grid_data
        X = x.reshape(-1, 1)
        X_aug = spatial_features(X, W, lags=1)
        assert X_aug.shape == (25, 2)  # original + 1 lag

    def test_shape_two_lags(self, grid_data):
        W, x, _ = grid_data
        X = x.reshape(-1, 1)
        X_aug = spatial_features(X, W, lags=2)
        assert X_aug.shape == (25, 3)  # original + 2 lags

    def test_original_preserved(self, grid_data):
        W, x, _ = grid_data
        X = x.reshape(-1, 1)
        X_aug = spatial_features(X, W, lags=1)
        assert np.allclose(X_aug[:, 0], x)


# ── moran_eigenvectors ─────────────────────────────────────────

class TestMoranEigenvectors:
    def test_shape(self, grid_data):
        W, _, _ = grid_data
        E = moran_eigenvectors(W, k=5)
        assert E.shape == (25, 5)

    def test_orthogonality(self, grid_data):
        W, _, _ = grid_data
        E = moran_eigenvectors(W, k=5)
        gram = E.T @ E
        assert np.allclose(gram, np.eye(5), atol=1e-6)

    def test_k_capped_at_n_minus_1(self):
        W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        E = moran_eigenvectors(W, k=10)  # k > n-1=2
        assert E.shape[1] == 2


# ── spatial_panel_features ──────────────────────────────────────

class TestSpatialPanelFeatures:
    def test_shape(self, grid_data):
        W, _, _ = grid_data
        K = 25
        T = 3
        rng = np.random.default_rng(0)
        X = rng.standard_normal((K * T, 2))
        unit_ids = np.tile(np.arange(K), T)
        time_ids = np.repeat(np.arange(T), K)

        X_aug = spatial_panel_features(X, W, unit_ids, time_ids, lags=1)
        # d=2, lags=1 → 2 original + 2 lag + 1 time = 5
        assert X_aug.shape == (K * T, 5)

    def test_time_column_normalized(self, grid_data):
        W, _, _ = grid_data
        K = 25
        T = 4
        X = np.ones((K * T, 1))
        unit_ids = np.tile(np.arange(K), T)
        time_ids = np.repeat(np.arange(T), K)

        X_aug = spatial_panel_features(X, W, unit_ids, time_ids, lags=1)
        time_col = X_aug[:, -1]
        assert time_col.min() == pytest.approx(0.0)
        assert time_col.max() == pytest.approx(1.0)
