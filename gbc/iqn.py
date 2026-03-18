"""Implicit Quantile Network (IQN).

Cosine quantile embedding following Dabney et al. (2018).
Three-term composite loss: L1 anchor + monotonicity + pinball.

Book references
---------------
- Ch 5 §sec-iqn              : IQN overview and motivation
- Ch 5 §sec-iqn-cosine       : cosine embedding of tau (Definition 5.1)
- Ch 5 §sec-iqn-architecture : multiplicative merge h_x ⊙ h_tau (Definition 5.2)
- Ch 5 §sec-iqn-loss         : three-term composite loss
- Ch 5 §sec-iqn-training     : Adam + cosine annealing schedule

Notes
-----
The default ``nh=32`` cosine frequencies is lower than the book's recommended
M=64 (§sec-iqn-cosine). For full fidelity to the book, pass ``nh=64``.
The smaller default trains faster for interactive chapter examples.
"""

import numpy as np
import torch
import torch.nn as nn

from gbc.loss import composite_loss


def cosine_embed(
    tau: float, nh: int, device=None, dtype=torch.float32
) -> torch.Tensor:
    """Cosine quantile embedding (Ch 5 §sec-iqn-cosine, Definition 5.1).

    Parameters
    ----------
    tau : scalar quantile level in (0, 1).
    nh : number of cosine frequencies.
    device : torch device.
    dtype : tensor dtype.

    Returns
    -------
    (nh,) tensor of cosine features.
    """
    i = torch.arange(1, nh + 1, device=device, dtype=dtype)
    return torch.cos(i * torch.pi * tau)


class IQN(nn.Module):
    r"""Implicit Quantile Network (Ch 5 §sec-iqn-architecture).

    Implements Definition 5.2:  Q̂(τ | x) = g_θ(ψ_θ(x) ⊙ φ_θ(τ))

    Architecture::

        tau -> cos(i * pi * tau), i=1..nh -> Linear(nh, hdim) -> ReLU -> h_tau
        x   -> Linear(xdim, hdim) -> ReLU -> h_x
        h = h_x * h_tau  (element-wise, §sec-iqn-architecture)
        h -> Linear(hdim, hdim) -> ReLU -> Linear(hdim, 64) -> Tanh
          -> Linear(64, 2)   # col 0 = mean anchor, col 1 = quantile

    The 2-output head supports the three-term composite loss (§sec-iqn-loss):
    col 0 is used by the L1 anchor term; col 1 by the pinball term.

    Parameters
    ----------
    xdim : int
        Input dimension.
    hdim : int
        Hidden layer width.
    nh : int
        Number of cosine embedding frequencies (book default M=64, §sec-iqn-cosine).
    """

    def __init__(self, xdim: int, hdim: int = 256, nh: int = 32):
        super().__init__()
        self.nh = nh
        self.fc_tau = nn.Sequential(nn.Linear(nh, hdim), nn.ReLU())
        self.fc_x = nn.Sequential(nn.Linear(xdim, hdim), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hdim, 64), nn.Tanh())
        self.fc_out = nn.Linear(64, 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (n, xdim) input features.
        tau : scalar quantile level in (0, 1).

        Returns
        -------
        (n, 2) tensor — column 0 is mean estimate, column 1 is quantile.
        """
        h_tau = self.fc_tau(cosine_embed(tau, self.nh, device=x.device, dtype=x.dtype))
        h_x = self.fc_x(x)
        h = self.fc1(h_x * h_tau.unsqueeze(0))
        h = self.fc2(h)
        return self.fc_out(h)

    def loss_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w: tuple[float, float, float] = (0.3, 0.3, 0.4),
    ) -> torch.Tensor:
        """Compute three-term loss with a randomly sampled tau."""
        tau = torch.rand(1).item()
        f = self(x, tau)
        return composite_loss(y, f, tau, w)

    def save(self, path: str):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, xdim: int, hdim: int = 256, nh: int = 32) -> "IQN":
        """Load a saved IQN model."""
        model = cls(xdim, hdim, nh)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model


def train_iqn(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 3000,
    hdim: int = 256,
    nh: int = 32,
    lr: float = 1e-3,
    wd: float = 1e-4,
    seed: int = 42,
    w: tuple[float, float, float] = (0.3, 0.3, 0.4),
) -> tuple:
    """Train an IQN with Adam + cosine annealing.

    Parameters
    ----------
    X : (n, d) input array.
    y : (n,) target array.
    epochs : number of training epochs.
    hdim : hidden dimension.
    nh : cosine embedding dimension.
    lr : learning rate.
    wd : weight decay.
    seed : random seed.
    w : composite loss weights (L1, monotonicity, pinball).

    Returns
    -------
    (model, xm, xs, ym, ys) — trained model and normalization stats.
    """
    torch.manual_seed(seed)
    xdim = X.shape[1]
    xm, xs = X.mean(0), X.std(0) + 1e-8
    ym, ys = float(y.mean()), float(y.std()) + 1e-8
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32)

    model = IQN(xdim, hdim=hdim, nh=nh)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01
    )
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = model.loss_fn(Xt, yt, w)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    return model, xm, xs, ym, ys


def predict_iqn(
    model: IQN,
    X_te: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    taus: np.ndarray | list[float] = (0.025, 0.25, 0.5, 0.75, 0.975),
) -> np.ndarray:
    """Predict at user-specified quantile levels.

    Unlike ``sample_iqn`` which uses evenly-spaced quantiles, this function
    evaluates the IQN at specific quantile levels chosen by the caller —
    useful for extracting credible intervals or comparing specific quantiles.

    Parameters
    ----------
    model : trained IQN.
    X_te : (n_test, d) test inputs (raw, unnormalized).
    xm, xs : input normalization (mean, std) from train_iqn.
    ym, ys : output normalization (mean, std) from train_iqn.
    taus : quantile levels to evaluate.

    Returns
    -------
    (len(taus), n_test) array of predicted values.
    """
    taus = np.asarray(taus)
    Xt = torch.tensor((X_te - xm) / xs, dtype=torch.float32)
    rows = []
    with torch.no_grad():
        for tau in taus:
            f = model(Xt, float(tau))
            rows.append(f[:, 1].numpy() * ys + ym)
    return np.array(rows)


def sample_iqn(
    model: IQN,
    X_te: np.ndarray,
    xm: np.ndarray,
    xs: np.ndarray,
    ym: float,
    ys: float,
    B: int = 500,
) -> np.ndarray:
    """Generate B quantile samples from a trained IQN.

    Parameters
    ----------
    model : trained IQN.
    X_te : (n_test, d) test inputs.
    xm, xs : input normalization (mean, std).
    ym, ys : output normalization (mean, std).
    B : number of quantile levels.

    Returns
    -------
    (B, n_test) array of predicted values at evenly-spaced quantiles.
    """
    Xt = torch.tensor((X_te - xm) / xs, dtype=torch.float32)
    taus = torch.linspace(0.005, 0.995, B)
    rows = []
    with torch.no_grad():
        for tau in taus:
            f = model(Xt, tau.item())
            rows.append(f[:, 1].numpy() * ys + ym)
    return np.array(rows)
