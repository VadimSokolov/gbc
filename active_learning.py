"""Active learning with randomized-prior ensembles.

Implements the Osband et al. (2018) randomized prior approach:
each ensemble member has a frozen random prior network added to
a trainable network. Acquisition is by ensemble disagreement.

Book references
---------------
- Ch 9 §sec-al-problem      : sequential design problem statement (eq. sec-al-acquire)
- Ch 9 §sec-al-randprior    : randomized-prior ensemble construction
- Ch 9 §sec-al-acquisition  : ensemble disagreement as acquisition function
- Ch 9 §sec-al-rocket       : rocket aerodynamics benchmark (GBC vs DGP+ALC)

Key result (Ch 9): Rocket benchmark RMSE=0.00723 (GBC) vs 0.02134 (DGP+ALC).
"""

import numpy as np
import torch
import torch.nn as nn


class RandomPriorNet(nn.Module):
    """Trainable network + frozen random prior (Osband et al. 2018).

    Parameters
    ----------
    xdim : int
        Input dimension.
    hdim : int
        Hidden layer width.
    nlayers : int
        Number of hidden layers.
    prior_scale : float
        Scale of the frozen prior network output.
    """

    def __init__(
        self, xdim: int, hdim: int = 128, nlayers: int = 3, prior_scale: float = 1.0
    ):
        super().__init__()
        # Trainable network
        layers = [nn.Linear(xdim, hdim), nn.ReLU()]
        for _ in range(nlayers - 1):
            layers += [nn.Linear(hdim, hdim), nn.ReLU()]
        layers.append(nn.Linear(hdim, 1))
        self.train_net = nn.Sequential(*layers)

        # Frozen prior network (same architecture, random weights)
        prior_layers = [nn.Linear(xdim, hdim), nn.ReLU()]
        for _ in range(nlayers - 1):
            prior_layers += [nn.Linear(hdim, hdim), nn.ReLU()]
        prior_layers.append(nn.Linear(hdim, 1))
        self.prior_net = nn.Sequential(*prior_layers)
        for p in self.prior_net.parameters():
            p.requires_grad = False

        self.prior_scale = prior_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.train_net(x) + self.prior_scale * self.prior_net(x)


def ensemble_disagreement(
    models: list[nn.Module], X: torch.Tensor
) -> np.ndarray:
    """Compute ensemble disagreement (std of predictions) as acquisition function.

    Parameters
    ----------
    models : list of trained ensemble members.
    X : (n_candidates, d) candidate points.

    Returns
    -------
    (n_candidates,) disagreement scores.
    """
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(X).squeeze(1).cpu().numpy())
    preds = np.stack(preds)
    return preds.std(axis=0)


def select_next(
    models: list[nn.Module],
    X_candidates: torch.Tensor,
    batch_size: int = 1,
) -> np.ndarray:
    """Select next points to query by maximum disagreement.

    Parameters
    ----------
    models : list of ensemble members.
    X_candidates : (n_cand, d) candidate pool.
    batch_size : number of points to select.

    Returns
    -------
    Indices into X_candidates of selected points.
    """
    scores = ensemble_disagreement(models, X_candidates)
    return np.argsort(scores)[-batch_size:][::-1]
