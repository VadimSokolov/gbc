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
          -> Linear(64, 2)   # col 0 = location anchor, col 1 = quantile

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

    def __init__(self, xdim: int, hdim: int = 256, nh: int = 32,
                 tail: bool = False, tau0: float = 0.9, xi_max: float = 0.9):
        super().__init__()
        self.nh = nh
        self.tail = tail
        self.tau0 = tau0
        self.xi_max = xi_max
        self.fc_tau = nn.Sequential(nn.Linear(nh, hdim), nn.ReLU())
        self.fc_x = nn.Sequential(nn.Linear(xdim, hdim), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hdim, 64), nn.Tanh())
        self.fc_out = nn.Linear(64, 2)
        # GPD tail head: (sigma_u, xi) as functions of the conditioning summary
        # only.  They are properties of the conditional law, not of tau, so they
        # are read off h_x and never see the quantile embedding.
        self.fc_tail = nn.Linear(hdim, 2) if tail else None
        self._init_weights()

    def tail_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GPD scale and shape at the splice, as (n,) tensors.

        sigma_u is softplus-positive; xi is bounded to (-xi_max, xi_max) so the
        implied endpoint u - sigma_u/xi cannot collapse onto the splice point or
        run away during early training.
        """
        raw = self.fc_tail(self.fc_x(x))
        sigma_u = nn.functional.softplus(raw[:, 0]) + 1e-4
        xi = self.xi_max * torch.tanh(raw[:, 1])
        return sigma_u, xi

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
        (n, 2) tensor; column 0 is mean estimate, column 1 is quantile.
        """
        h_x = self.fc_x(x)
        out = self._body(h_x, tau)
        if self.tail is False or self.fc_tail is None or tau <= self.tau0:
            return out
        # Above the splice the quantile is parametric, so it extrapolates instead
        # of saturating: a body-only network flattens beyond the tau it was
        # trained on and then cannot reach values its own record contains.
        u = self._body(h_x, self.tau0)[:, 1]
        sigma_u, xi = self.tail_params(x)
        z = (1.0 - tau) / (1.0 - self.tau0)           # in (0, 1]; z=1 at tau0
        logz = float(np.log(z))
        small = xi.abs() < 1e-6
        xi_safe = torch.where(small, torch.ones_like(xi), xi)
        q_gen = u + sigma_u / xi_safe * (torch.exp(-xi_safe * logz) - 1.0)
        q_exp = u + sigma_u * (-logz)                 # xi -> 0 limit
        q = torch.where(small, q_exp, q_gen)
        return torch.stack([out[:, 0], q], dim=1)

    def _body(self, h_x: torch.Tensor, tau: float) -> torch.Tensor:
        """The nonparametric body, given the already-computed conditioning map."""
        h_tau = self.fc_tau(
            cosine_embed(tau, self.nh, device=h_x.device, dtype=h_x.dtype))
        h = self.fc1(h_x * h_tau.unsqueeze(0))
        return self.fc_out(self.fc2(h))

    def loss_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w: tuple[float, float, float] = (0.3, 0.3, 0.4),
        w_tail: float = 0.4,
    ) -> torch.Tensor:
        """Three-term loss at a randomly sampled tau.

        A second, independently drawn level supplies the monotonicity term.  It
        is drawn separately rather than as the larger of a sorted pair so that
        the level scoring the pinball term stays Uniform(0,1); taking the
        smaller of two uniforms would be Beta(1,2) and would systematically
        under-train the upper tail, which is the part this library is for.
        """
        tau = torch.rand(1).item()
        tau_other = torch.rand(1).item()
        f = self(x, tau)
        f_other = self(x, tau_other)
        loss = composite_loss(y, f, tau, w, f_other=f_other, tau_other=tau_other)
        if self.tail and self.fc_tail is not None:
            # A uniform tau lands above the splice only (1 - tau0) of the time,
            # so the head would train on a tenth of the steps.  Add a dedicated
            # level drawn from Uniform(tau0, 1): a sum of pinball losses at
            # different levels is still minimised by the true quantile function.
            tt = self.tau0 + (1.0 - self.tau0) * torch.rand(1).item()
            e = y - self(x, tt)[:, 1]
            loss = loss + w_tail * torch.mean(
                torch.maximum(tt * e, (tt - 1.0) * e))
        return loss

    def save(self, path: str):
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, xdim: int, hdim: int = 256, nh: int = 32,
             tail: bool = False, tau0: float = 0.9) -> "IQN":
        """Load a saved IQN model.

        ``tail`` must match the saved model: the tail head adds parameters, so a
        mismatch fails on state_dict load rather than silently dropping it.
        """
        model = cls(xdim, hdim, nh, tail=tail, tau0=tau0)
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
    batch_size: int | None = None,
    verbose: bool = False,
    device: str | torch.device | None = None,
    tail: bool = False,
    tau0: float = 0.9,
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
    batch_size : mini-batch size. ``None`` uses full batch.
    verbose : if True, print loss every 500 epochs.
    device : torch device (``None`` selects automatically).
    tail : attach the parametric GPD tail head above ``tau0``.  Needed whenever
        the network must report probabilities far into the upper tail; a
        body-only network saturates just past the levels it was trained on.
    tau0 : splice level between the nonparametric body and the GPD tail.

    Returns
    -------
    (model, xm, xs, ym, ys): trained model and normalization stats.
    """
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    xdim = X.shape[1]
    n = X.shape[0]
    xm, xs = X.mean(0), X.std(0) + 1e-8
    ym, ys = float(y.mean()), float(y.std()) + 1e-8
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32, device=device)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32, device=device)

    model = IQN(xdim, hdim=hdim, nh=nh, tail=tail, tau0=tau0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01
    )
    use_batches = batch_size is not None and batch_size < n
    model.train()
    for ep in range(epochs):
        if use_batches:
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad()
                loss = model.loss_fn(Xt[idx], yt[idx], w)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
        else:
            opt.zero_grad()
            loss = model.loss_fn(Xt, yt, w)
            loss.backward()
            opt.step()
            epoch_loss = loss.item()
            n_batches = 1
        sched.step()
        if verbose and (ep + 1) % 500 == 0:
            print(f"  epoch {ep+1}/{epochs}  loss={epoch_loss/n_batches:.4f}")

    model.eval().cpu()
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
    evaluates the IQN at specific quantile levels chosen by the caller,
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
    """Evaluate a trained IQN on a deterministic grid of B quantile levels.

    These are NOT random draws, despite the name: tau runs on an evenly spaced
    grid from 0.005 to 0.995, so two calls with the same inputs return the same
    numbers.  Equal-weight summaries describe the quantile range on this grid,
    not the full predictive distribution: each omitted tail has mass 0.005, so
    the resulting truncation error does not vanish as B grows.  In particular,
    do not use a raw exceedance fraction for a full far-tail probability.

    That is also the trap.  **Do not bootstrap the returned values.**  Resampling
    them measures the sampling variability of B independent draws, and there are
    no draws here, so the resulting interval describes noise the estimate does
    not have.  For a genuine interval, resample the conditioning record in
    ``X_te`` and re-evaluate; the network is amortized, so that costs a forward
    pass rather than a refit.

    Parameters
    ----------
    model : trained IQN.
    X_te : (n_test, d) test inputs.
    xm, xs : input normalization (mean, std).
    ym, ys : output normalization (mean, std).
    B : number of quantile levels on the grid.

    Returns
    -------
    (B, n_test) array of predicted values at evenly-spaced quantile levels.
    """
    Xt = torch.tensor((X_te - xm) / xs, dtype=torch.float32)
    taus = torch.linspace(0.005, 0.995, B)
    rows = []
    with torch.no_grad():
        for tau in taus:
            f = model(Xt, tau.item())
            rows.append(f[:, 1].numpy() * ys + ym)
    return np.array(rows)
