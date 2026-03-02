"""Aug-IQN pipeline for jump/boundary detection.

Pipeline:
  1. EM cluster Y into 2-component Gaussian mixture -> binary labels
  2. Train MLP classifier X -> P(regime=1 | X)
  3. Append classifier probability as extra feature (d+1 inputs)
  4. Train standard IQN on augmented inputs

Book references
---------------
- Ch 8 §sec-jumps-problem    : why plain IQN struggles with jump discontinuities
- Ch 8 §sec-jumps-em         : EM clustering to infer regime labels (eq. sec-gmm)
- Ch 8 §sec-jumps-classifier : MLP classifier for boundary detection
- Ch 8 §sec-jumps-augiqn     : feature augmentation map T(x) = [x, f̂(x)]

Benchmark results (Ch 8 Table 8.x):
  Phantom  — Aug-IQN RMSE=0.044, CRPS=0.009 vs MJGP RMSE=0.053, CRPS=0.010
  Star     — Aug-IQN RMSE=0.059, CRPS=0.013 vs MJGP RMSE=0.085, CRPS=0.024
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture


def em_labels(y: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    """Cluster responses into regimes via EM on a Gaussian mixture.

    Parameters
    ----------
    y : (n,) response values.
    n_components : number of mixture components.
    seed : random seed.

    Returns
    -------
    (n,) binary labels (0 or 1).
    """
    gm = GaussianMixture(n_components=n_components, random_state=seed)
    labels = gm.fit_predict(y.reshape(-1, 1))
    return labels.astype(np.float32)


class BoundaryClassifier(nn.Module):
    """MLP binary classifier for regime detection.

    Parameters
    ----------
    xdim : int
        Input dimension.
    hdim : int
        Hidden layer width.
    nlayers : int
        Number of hidden layers.
    """

    def __init__(self, xdim: int, hdim: int = 256, nlayers: int = 3):
        super().__init__()
        layers = [nn.Linear(xdim, hdim), nn.ReLU()]
        for _ in range(nlayers - 1):
            layers += [nn.Linear(hdim, hdim), nn.ReLU()]
        layers.append(nn.Linear(hdim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (pre-sigmoid)."""
        return self.net(x)


def train_classifier(
    X: np.ndarray,
    labels: np.ndarray,
    epochs: int = 3000,
    hdim: int = 256,
    nlayers: int = 3,
    lr: float = 1e-3,
    seed: int = 42,
) -> BoundaryClassifier:
    """Train the boundary classifier with BCE + cosine annealing.

    Parameters
    ----------
    X : (n, d) inputs.
    labels : (n,) binary labels from em_labels.

    Returns
    -------
    Trained classifier in eval mode.
    """
    torch.manual_seed(seed)
    xdim = X.shape[1]
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.float32)

    model = BoundaryClassifier(xdim, hdim, nlayers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_t).squeeze(1)
        loss = criterion(logits, y_t)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    return model


def augment_features(
    X: np.ndarray, classifier: BoundaryClassifier
) -> np.ndarray:
    """Append classifier probabilities as extra feature.

    Parameters
    ----------
    X : (n, d) original features.
    classifier : trained BoundaryClassifier.

    Returns
    -------
    (n, d+1) augmented feature array.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(classifier(X_t)).numpy()
    return np.hstack([X, probs])
