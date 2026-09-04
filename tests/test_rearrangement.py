"""Tests for opt-in monotone rearrangement of IQN quantiles."""

import math

import numpy as np
import pytest
import torch

from gbc import rearrange_quantiles
from gbc.iqn import predict_iqn, sample_iqn
from gbc.metrics import crps_samples


class OscillatingIQN(torch.nn.Module):
    """Small deterministic model with a deliberately crossing quantile curve."""

    def forward(self, x, tau):
        value = tau + 0.2 * math.sin(4.0 * math.pi * tau)
        quantile = torch.full(
            (x.shape[0],), value, dtype=x.dtype, device=x.device
        ) + 0.1 * x[:, 0]
        return torch.stack([torch.zeros_like(quantile), quantile], dim=1)


@pytest.fixture
def prediction_args():
    model = OscillatingIQN()
    X = np.array([[0.0], [1.0], [-1.0]])
    return model, X, np.zeros(1), np.ones(1), 0.0, 1.0


class TestRearrangeQuantiles:
    def test_sorts_then_interpolates_in_target_order(self):
        quantiles = np.array([0.0, 3.0, 1.0, 2.0])
        taus = np.array([0.1, 0.3, 0.6, 0.9])
        targets = np.array([0.9, 0.1, 0.45, 0.45])
        result = rearrange_quantiles(quantiles, taus, targets)
        np.testing.assert_allclose(result, [3.0, 0.0, 1.5, 1.5])

    def test_matrix_is_rearranged_by_column(self):
        quantiles = np.array([[3.0, 0.0], [1.0, 4.0], [2.0, 2.0]])
        result = rearrange_quantiles(quantiles, [0.1, 0.5, 0.9])
        np.testing.assert_array_equal(
            result, np.array([[1.0, 0.0], [2.0, 2.0], [3.0, 4.0]])
        )

    @pytest.mark.parametrize(
        "quantiles,taus,targets",
        [
            ([], [], None),
            ([1.0, 2.0], [0.5, 0.5], None),
            ([1.0, np.nan], [0.1, 0.9], None),
            ([1.0], [0.1, 0.9], None),
            ([1.0, 2.0], [0.1, 0.9], [0.0]),
        ],
    )
    def test_invalid_inputs(self, quantiles, taus, targets):
        with pytest.raises(ValueError):
            rearrange_quantiles(quantiles, taus, targets)


class TestIQNRearrangementFlags:
    def test_predict_default_path_is_unchanged(self, prediction_args):
        taus = [0.1, 0.5, 0.9]
        default = predict_iqn(*prediction_args, taus=taus)
        explicit = predict_iqn(*prediction_args, taus=taus, rearrange=False)
        np.testing.assert_array_equal(default, explicit)

    def test_predict_rearranges_on_dense_grid(self, prediction_args):
        taus = np.linspace(0.01, 0.99, 1000)
        raw = predict_iqn(*prediction_args, taus=taus)
        ordered = predict_iqn(
            *prediction_args,
            taus=taus,
            rearrange=True,
            rearrange_grid_size=1000,
        )
        assert np.any(np.diff(raw, axis=0) < 0.0)
        assert np.all(np.diff(ordered, axis=0) >= -1e-12)

    def test_predict_preserves_unsorted_repeated_target_order(self, prediction_args):
        taus = np.array([0.9, 0.1, 0.5, 0.5, 0.3])
        ordered = predict_iqn(*prediction_args, taus=taus, rearrange=True)
        sorted_result = ordered[np.argsort(taus, kind="stable")]
        assert np.all(np.diff(sorted_result, axis=0) >= -1e-12)
        np.testing.assert_array_equal(ordered[2], ordered[3])

    def test_sample_preserves_multiset_and_crps(self, prediction_args):
        raw = sample_iqn(*prediction_args, B=201)
        ordered = sample_iqn(*prediction_args, B=201, rearrange=True)
        np.testing.assert_array_equal(np.sort(raw, axis=0), ordered)
        assert np.all(np.diff(ordered, axis=0) >= 0.0)

        y = np.array([0.0, 1.0, -1.0])
        assert crps_samples(y, raw) == pytest.approx(
            crps_samples(y, ordered), abs=1e-14
        )

    @pytest.mark.parametrize("grid_size", [0, 1])
    def test_predict_rejects_invalid_dense_grid_size(
        self, prediction_args, grid_size
    ):
        with pytest.raises(ValueError, match="grid"):
            predict_iqn(
                *prediction_args,
                taus=[0.2, 0.8],
                rearrange=True,
                rearrange_grid_size=grid_size,
            )
