"""Jump discontinuity use case: Aug-IQN pipeline.

DGP: y = sin(2πx) + 3·1(x > 0.5) + N(0, 0.2),  x ~ U(0, 1)
Two regimes separated by a boundary at x=0.5.
"""

import numpy as np
import torch
import pytest

from gbc.augment import em_labels, BoundaryClassifier, train_classifier, augment_features
from gbc.iqn import train_iqn, sample_iqn
from gbc.metrics import rmse, crps_samples


@pytest.fixture(scope="module")
def jump_data():
    rng = np.random.default_rng(42)
    n = 400
    X = rng.uniform(0, 1, (n, 1))
    y = np.sin(2 * np.pi * X[:, 0]) + 3.0 * (X[:, 0] > 0.5) + rng.normal(0, 0.2, n)
    return X, y


# ── EM labels ───────────────────────────────────────────────────

class TestEMLabels:
    def test_binary_labels(self, jump_data):
        _, y = jump_data
        labels = em_labels(y, n_components=2, seed=0)
        assert set(np.unique(labels)).issubset({0.0, 1.0})
        assert labels.shape == y.shape

    def test_two_clusters_found(self, jump_data):
        _, y = jump_data
        labels = em_labels(y, n_components=2, seed=0)
        # Both labels should be present
        assert len(np.unique(labels)) == 2


# ── Boundary classifier ────────────────────────────────────────

class TestBoundaryClassifier:
    @pytest.fixture(scope="class")
    def classifier(self, jump_data):
        X, y = jump_data
        labels = em_labels(y, n_components=2, seed=0)
        return train_classifier(X, labels, epochs=300, hdim=32, nlayers=2, seed=0)

    def test_returns_eval_mode(self, classifier):
        assert not classifier.training

    def test_output_shape(self, classifier, jump_data):
        X, _ = jump_data
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = classifier(X_t)
        assert logits.shape == (len(X), 1)

    def test_probabilities_in_range(self, classifier, jump_data):
        X, _ = jump_data
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.sigmoid(classifier(X_t)).numpy()
        assert np.all(probs >= 0) and np.all(probs <= 1)


# ── Feature augmentation ───────────────────────────────────────

class TestAugmentFeatures:
    def test_shape(self, jump_data):
        X, y = jump_data
        labels = em_labels(y, n_components=2, seed=0)
        clf = train_classifier(X, labels, epochs=100, hdim=32, nlayers=2, seed=0)
        X_aug = augment_features(X, clf)
        assert X_aug.shape == (len(X), X.shape[1] + 1)

    def test_original_features_preserved(self, jump_data):
        X, y = jump_data
        labels = em_labels(y, n_components=2, seed=0)
        clf = train_classifier(X, labels, epochs=100, hdim=32, nlayers=2, seed=0)
        X_aug = augment_features(X, clf)
        assert np.allclose(X_aug[:, :X.shape[1]], X)


# ── Full Aug-IQN pipeline ──────────────────────────────────────

class TestAugIQNPipeline:
    def test_end_to_end(self, jump_data):
        X, y = jump_data
        n = len(y)
        # Split
        idx = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        tr, te = idx[:300], idx[300:]
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # Pipeline
        labels = em_labels(y_tr, n_components=2, seed=0)
        clf = train_classifier(X_tr, labels, epochs=300, hdim=32, nlayers=2, seed=0)
        X_tr_aug = augment_features(X_tr, clf)
        X_te_aug = augment_features(X_te, clf)

        model, xm, xs, ym, ys = train_iqn(
            X_tr_aug, y_tr, epochs=300, hdim=64, nh=16, seed=0,
        )
        samples = sample_iqn(model, X_te_aug, xm, xs, ym, ys, B=100)

        r = rmse(y_te, samples.mean(axis=0))
        c = crps_samples(y_te, samples)
        assert np.isfinite(r) and r > 0
        assert np.isfinite(c) and c > 0
