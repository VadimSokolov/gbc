"""Focused tests for deterministic sample-based scoring rules."""

import numpy as np
import pytest

from gbc.metrics import crps_samples


def _brute_force_crps(y, samples):
    term1 = np.mean(np.abs(samples - y[None, :]), axis=0)
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :])
    term2 = 0.5 * np.mean(pairwise, axis=(0, 1))
    return float(np.mean(term1 - term2))


class TestCRPSSamples:
    def test_exact_matches_all_pairs_reference(self):
        rng = np.random.default_rng(9)
        y = rng.normal(size=7)
        samples = rng.normal(size=(31, 7))
        assert crps_samples(y, samples) == pytest.approx(
            _brute_force_crps(y, samples), abs=1e-14
        )

    def test_exact_is_deterministic(self):
        rng = np.random.default_rng(13)
        y = rng.normal(size=5)
        samples = rng.normal(size=(50, 5))
        values = [crps_samples(y, samples) for _ in range(10)]
        assert values == [values[0]] * len(values)

    def test_one_sample_reduces_to_mae(self):
        y = np.array([1.0, -2.0, 0.5])
        samples = np.array([[4.0, -1.0, 0.0]])
        assert crps_samples(y, samples) == pytest.approx(
            np.mean(np.abs(samples[0] - y))
        )

    def test_seeded_mc_repeats_and_does_not_touch_global_rng(self):
        y = np.array([0.0, 1.0])
        samples = np.arange(20.0).reshape(10, 2)

        np.random.seed(91)
        expected_next = np.random.random()
        np.random.seed(91)
        first = crps_samples(y, samples, method="mc", seed=42)
        actual_next = np.random.random()
        second = crps_samples(y, samples, method="mc", seed=42)

        assert first == second
        assert actual_next == expected_next

    def test_mc_requires_seed(self):
        with pytest.raises(ValueError, match="seed"):
            crps_samples(np.zeros(2), np.zeros((3, 2)), method="mc")

    @pytest.mark.parametrize(
        "y,samples",
        [
            (np.zeros((2, 1)), np.zeros((3, 2))),
            (np.zeros(2), np.zeros(2)),
            (np.zeros(2), np.zeros((0, 2))),
            (np.zeros(2), np.zeros((3, 4))),
        ],
    )
    def test_invalid_shapes(self, y, samples):
        with pytest.raises(ValueError):
            crps_samples(y, samples)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            crps_samples(np.zeros(2), np.zeros((3, 2)), method="fast")
