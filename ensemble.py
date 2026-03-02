"""Heteroskedastic MLP ensemble with Gaussian NLL loss.

Used for settings where IQN's multiplicative tau architecture
fails (high signal-to-noise ratio). Produces mean and variance
predictions via law of total variance.

Book references
---------------
- Ch 14 §sec-cal-failure   : IQN failure on high-SNR data (gradient collapse)
- Ch 14 §sec-cal-het-mlp   : HetMLP architecture — separate mean/logvar heads
- Ch 14 §sec-cal-nll       : Gaussian NLL training objective
- Ch 14 §sec-cal-lake      : lake temperature benchmark results
"""

import numpy as np
import torch
import torch.nn as nn

from gbc.loss import gaussian_nll


class HetMLP(nn.Module):
    """MLP with heteroskedastic output (mean and log-variance).

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
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hdim, 1)
        self.logvar_head = nn.Linear(hdim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, logvar) predictions."""
        h = self.backbone(x)
        return self.mean_head(h), self.logvar_head(h)


def train_het_mlp(
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 5000,
    hdim: int = 256,
    nlayers: int = 3,
    lr: float = 1e-3,
    batch_size: int = 8192,
    seed: int = 42,
    device: torch.device | None = None,
) -> HetMLP:
    """Train a single HetMLP model.

    Parameters
    ----------
    X : (n, d) input tensor on device.
    y : (n, 1) target tensor on device.
    epochs : training epochs.
    seed : random seed.
    device : torch device.

    Returns
    -------
    Trained HetMLP in eval mode.
    """
    if device is None:
        device = X.device
    xdim = X.shape[1]
    n = len(X)
    n_batches = (n + batch_size - 1) // batch_size

    torch.manual_seed(seed)
    model = HetMLP(xdim, hdim, nlayers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            mu, logvar = model(X[idx])
            loss = gaussian_nll(mu, logvar, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def ensemble_predict(
    models: list[HetMLP], X: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate predictions from an ensemble of HetMLPs (Ch 14 §sec-cal-het-mlp).

    Uses law of total variance (standard probability identity):
        Var[Y|X] = E[sigma^2(X)] + Var[mu(X)]

    Parameters
    ----------
    models : list of trained HetMLP models.
    X : (n, d) input tensor.

    Returns
    -------
    (ens_mean, ens_std) — numpy arrays of shape (n,).
    """
    K = len(models)
    n = X.shape[0]
    all_mu = np.zeros((K, n))
    all_var = np.zeros((K, n))

    for k, m in enumerate(models):
        m.eval()
        with torch.no_grad():
            mu, logvar = m(X)
            all_mu[k] = mu.squeeze(1).cpu().numpy()
            all_var[k] = torch.exp(logvar.squeeze(1)).cpu().numpy()

    ens_mean = all_mu.mean(axis=0)
    ens_var = (all_var + all_mu**2).mean(axis=0) - ens_mean**2
    ens_std = np.sqrt(np.maximum(ens_var, 1e-6))
    return ens_mean, ens_std
