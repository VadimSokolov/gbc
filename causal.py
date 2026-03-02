"""Causal inference via GBC: CATE/ATE estimation.

QuantNet architecture with propensity score embedding,
treatment effect module, and quantile-indexed output.

Book references
---------------
- Ch 13 §sec-causal           : potential outcomes framework and identification
- Ch 13 §sec-causal-cate      : CATE and ATE definitions (eq. sec-cate, sec-ate)
- Ch 13 §sec-causal-qte-motivation : quantile treatment effects beyond the mean
- Ch 13 §sec-causal-architecture   : CausalIQN three-part architecture

Architecture overview (Ch 13):
  pi module  : X -> propensity embedding -> P(Z=1|X)   [ignorability]
  mu module  : (X, pi_embed) -> baseline outcome E[Y(0)|X]
  te module  : X -> treatment effect (location + quantile components)
  Output     : mu(X) + te(X, tau) * Z

Notes
-----
``z`` must be a float tensor (0.0 / 1.0) for BCELoss compatibility.
"""

import numpy as np
import torch
import torch.nn as nn


class CausalIQN(nn.Module):
    """Quantile neural network for causal inference.

    Architecture:
        - pi module: X -> propensity embedding -> treatment probability
        - mu module: (X, pi_embed) -> baseline outcome
        - te module: X -> treatment effect (location + quantile)
        - tau module: cosine embedding for quantile level

    Output: mu(X) + te(X, tau) * Z

    Parameters
    ----------
    xdim : int
        Number of covariates.
    hsz : int
        Hidden size for treatment effect module.
    nh : int
        Number of cosine embedding frequencies.
    """

    def __init__(self, xdim: int = 1, hsz: int = 32, nh: int = 32):
        super().__init__()
        self.nh = nh
        pisz, lw = 8, 16
        self.pi = nn.Sequential(nn.Linear(xdim, 16), nn.ReLU(), nn.Linear(16, pisz))
        self.pi1 = nn.Sequential(nn.Linear(pisz, 16), nn.ReLU(), nn.Linear(16, 1))
        self.mu = nn.Sequential(
            nn.Linear(xdim + pisz, lw), nn.ReLU(),
            nn.Linear(lw, lw), nn.ReLU(),
            nn.Linear(lw, hsz),
        )
        self.mu1 = nn.Sequential(
            nn.Linear(hsz, lw), nn.ReLU(),
            nn.Linear(lw, lw), nn.ReLU(),
            nn.Linear(lw, 1),
        )
        self.te = nn.Sequential(
            nn.Linear(xdim, lw), nn.ReLU(),
            nn.Linear(lw, lw), nn.ReLU(),
            nn.Linear(lw, hsz),
        )
        self.te1 = nn.Sequential(nn.Linear(hsz, lw), nn.ReLU(), nn.Linear(lw, 2))
        self.tau_embed = nn.Sequential(nn.Linear(nh, hsz), nn.ReLU())

    def forward(
        self, x: torch.Tensor, z: torch.Tensor, tau: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        (y_pred, pi_logits, te_output, mu_output)
        """
        tau_e = self.tau_embed(
            torch.cos(torch.arange(1, self.nh + 1) * torch.pi * tau)
        )
        pi = self.pi(x)
        pi1 = self.pi1(pi)
        mu = self.mu1(self.mu(torch.cat((x, pi), 1)))
        te = self.te1(tau_e * self.te(x))
        return mu + te * z.view(-1, 1), pi1, te, mu

    def loss_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        w: tuple[float, float, float] = (0.3, 0.1, 0.6),
    ) -> torch.Tensor:
        """Composite loss with propensity BCE."""
        tau = torch.rand(1).item()
        tauind = float(tau < 0.5)
        f, pi, _, _ = self(x, z, tau)
        piloss = nn.BCELoss()(torch.sigmoid(pi.view(-1)), z)
        e = y.view(-1, 1) - f
        loss = w[0] * torch.mean(torch.abs(e[:, 0]))
        loss = loss + w[1] * abs(tau - 0.5) * (
            tauind * torch.mean(torch.relu(-e[:, 1]))
            + (1 - tauind) * torch.mean(torch.relu(e[:, 1]))
        )
        loss = loss + w[2] * torch.mean(
            torch.maximum(tau * e[:, 1], (tau - 1) * e[:, 1])
        )
        return loss + piloss

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        epochs: int = 3000,
        w: tuple[float, float, float] = (0.3, 0.1, 0.6),
        lr: float = 5e-4,
        wd: float = 3e-3,
    ):
        """Train the causal IQN."""
        opt = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            loss = self.loss_fn(x, y, z, w)
            loss.backward()
            opt.step()

    def estimate_cate(
        self, x: torch.Tensor, z: torch.Tensor, n_mc: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate CATE by Monte Carlo over quantile levels.

        Returns
        -------
        (cate_point, cate_samples) — point estimate (median) and full (n, n_mc) array.
        """
        n = x.shape[0]
        samples = torch.zeros((n, n_mc))
        with torch.no_grad():
            for i in range(n_mc):
                _, _, te, _ = self(x, z, torch.rand(1).item())
                samples[:, i] = te[:, 1]
        return samples.median(dim=1).values.numpy(), samples.numpy()
