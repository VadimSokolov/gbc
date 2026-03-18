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

Two architectures are provided:

``CausalIQN``  — original multiplicative design (tau_embed * te_encoder(x)).
    Compact and fast, best for low-dimensional settings (d < 10).

``CausalIQNv2`` — additive design (concat [x, tau_features] → TE head + skip).
    Preserves CATE heterogeneity by avoiding the multiplicative collapse that
    occurs when the quantile embedding dominates the covariate signal. Includes
    dropout regularisation and a skip connection from raw covariates to the
    final TE layer. Preferred when heterogeneous treatment effects are expected.

``CausalEnsemble`` — trains K models with different seeds, averages predictions.
    Uses Adam + cosine annealing. Provides ATE, CATE, quantile treatment effects,
    and propensity scores.

Notes
-----
``z`` must be a float tensor (0.0 / 1.0) for BCELoss compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from gbc.loss import composite_loss
from gbc.iqn import cosine_embed, train_iqn
from gbc.utils import get_device


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
        tau_e = self.tau_embed(cosine_embed(tau, self.nh, device=x.device, dtype=x.dtype))
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
        f, pi_logit, _, _ = self(x, z, tau)
        pi_loss = nn.functional.binary_cross_entropy_with_logits(
            pi_logit.view(-1), z.float()
        )
        return composite_loss(y, f, tau, w) + pi_loss

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


class CausalIQNv2(nn.Module):
    r"""Causal IQN with additive quantile--covariate interaction.

    Differs from :class:`CausalIQN` in the treatment effect head:
    instead of element-wise multiplication ``tau_embed * te_encoder(x)``,
    it concatenates ``[x, tau_features]`` and feeds through a deep network
    with a skip connection from ``x``.  This preserves CATE heterogeneity
    by preventing the quantile embedding from dominating the covariate signal.

    Architecture::

        pi  : x -> propensity embedding -> P(Z=1|X)
        mu  : (x, pi_embed) -> E[Y(0)|X]
        te  : concat(x, cos_features(tau)) -> MLP -> [+skip(x)] -> [loc, quant]
        out : mu + te * z

    Parameters
    ----------
    xdim : int
        Number of covariates.
    hdim : int
        Hidden layer width.
    nh : int
        Number of cosine embedding frequencies for the quantile level.
    pidim : int
        Dimension of propensity-score embedding.
    dropout : float
        Dropout probability in hidden layers.
    """

    def __init__(
        self,
        xdim: int,
        hdim: int = 64,
        nh: int = 32,
        pidim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nh = nh

        # Propensity score network
        self.pi_net = nn.Sequential(
            nn.Linear(xdim, hdim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hdim, hdim // 2), nn.ReLU(),
            nn.Linear(hdim // 2, pidim),
        )
        self.pi_head = nn.Linear(pidim, 1)

        # Outcome baseline network
        self.mu_net = nn.Sequential(
            nn.Linear(xdim + pidim, hdim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hdim, hdim), nn.ReLU(),
            nn.Linear(hdim, 1),
        )

        # Treatment effect network — additive design
        te_in = xdim + nh
        self.te_net = nn.Sequential(
            nn.Linear(te_in, hdim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hdim, hdim), nn.ReLU(),
        )
        self.te_head = nn.Sequential(
            nn.Linear(hdim + xdim, hdim // 2), nn.ReLU(),
            nn.Linear(hdim // 2, 2),
        )

    def forward(
        self, x: torch.Tensor, z: torch.Tensor, tau: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (n, xdim) covariates.
        z : (n,) treatment indicator (0/1 float).
        tau : scalar quantile level in (0, 1).

        Returns
        -------
        (y_pred, pi_logit, te, mu)
        """
        # Propensity
        pi_embed = self.pi_net(x)
        pi_logit = self.pi_head(pi_embed)

        # Outcome baseline
        mu = self.mu_net(torch.cat([x, pi_embed], dim=1))

        # Quantile features
        q_feat = cosine_embed(tau, self.nh, device=x.device, dtype=x.dtype)
        q_feat = q_feat.unsqueeze(0).expand(x.shape[0], -1)     # (n, nh)

        # TE: concat [x, q_feat] → hidden → concat [hidden, x] → head
        te_hidden = self.te_net(torch.cat([x, q_feat], dim=1))
        te = self.te_head(torch.cat([te_hidden, x], dim=1))     # skip

        f = mu + te * z.view(-1, 1)
        return f, pi_logit, te, mu

    def loss_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        w: tuple[float, float, float] = (0.3, 0.1, 0.6),
    ) -> torch.Tensor:
        """Composite loss with propensity BCE."""
        tau = torch.rand(1).item()
        f, pi_logit, _, _ = self(x, z, tau)
        pi_loss = nn.functional.binary_cross_entropy_with_logits(
            pi_logit.view(-1), z.float()
        )
        return composite_loss(y, f, tau, w) + pi_loss


class CausalEnsemble:
    """Ensemble of CausalIQN (v1 or v2) models.

    Trains ``n_models`` with different random seeds using Adam + cosine
    annealing.  Provides ATE, CATE, quantile treatment effects, and
    propensity scores by aggregating across ensemble members.

    Parameters
    ----------
    model_cls : {CausalIQN, CausalIQNv2}
        Model class to instantiate.
    model_kwargs : dict
        Keyword arguments forwarded to ``model_cls.__init__``.
    n_models : int
        Number of ensemble members.
    device : str
        ``"auto"`` selects CUDA if available.
    """

    def __init__(
        self,
        model_cls=CausalIQNv2,
        model_kwargs: dict | None = None,
        n_models: int = 3,
        device: str = "auto",
    ):
        self.n_models = n_models
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)

        if model_kwargs is None:
            model_kwargs = {}

        self.models: list[nn.Module] = []
        for i in range(n_models):
            torch.manual_seed(42 + i * 137)
            m = model_cls(**model_kwargs).to(self.device)
            self.models.append(m)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        epochs: int = 5000,
        lr: float = 5e-4,
        weight_decay: float = 3e-3,
        loss_weights: tuple[float, float, float] = (0.3, 0.1, 0.6),
        verbose: bool = True,
    ):
        """Train all ensemble members.

        Parameters
        ----------
        X : (n, d) covariates (numpy).
        Y : (n,) outcomes (numpy).
        Z : (n,) binary treatment (numpy, 0/1).
        epochs : training epochs per model.
        lr : initial learning rate.
        weight_decay : L2 regularisation.
        loss_weights : (L1, monotonicity, pinball) weights.
        verbose : print epoch-level loss.
        """
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        z_t = torch.tensor(Z, dtype=torch.float32, device=self.device)

        for m_idx, model in enumerate(self.models):
            if verbose:
                print(f"  Training model {m_idx + 1}/{self.n_models}...")
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
            model.train()
            for epoch in range(epochs):
                opt.zero_grad()
                loss = model.loss_fn(x_t, y_t, z_t, loss_weights)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                if verbose and (epoch + 1) % 1000 == 0:
                    print(f"    Epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")

    def estimate_cate(
        self, X: np.ndarray, n_mc: int = 500
    ) -> dict[str, np.ndarray | float]:
        """Estimate CATE by Monte Carlo over quantile levels.

        Parameters
        ----------
        X : (n, d) covariates.
        n_mc : Monte Carlo samples per ensemble member.

        Returns
        -------
        Dict with keys ``cate``, ``ci_lo``, ``ci_hi``, ``ate``, ``ate_se``,
        ``ate_ci``, ``cate_samples``.
        """
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        z_dum = torch.ones(X.shape[0], device=self.device)

        all_samples = []
        for model in self.models:
            model.eval()
            s = torch.zeros(X.shape[0], n_mc, device=self.device)
            with torch.no_grad():
                for i in range(n_mc):
                    _, _, te, _ = model(x_t, z_dum, torch.rand(1).item())
                    s[:, i] = te[:, 1]
            all_samples.append(s)

        samples = torch.cat(all_samples, dim=1).cpu().numpy()
        cate = np.median(samples, axis=1)
        ate_samples = samples.mean(axis=0)

        return {
            "cate": cate,
            "cate_samples": samples,
            "ci_lo": np.percentile(samples, 5, axis=1),
            "ci_hi": np.percentile(samples, 95, axis=1),
            "ate": float(cate.mean()),
            "ate_se": float(np.std(ate_samples)),
            "ate_ci": (
                float(np.percentile(ate_samples, 2.5)),
                float(np.percentile(ate_samples, 97.5)),
            ),
        }

    def estimate_qte(
        self, X: np.ndarray, quantiles: np.ndarray | None = None
    ) -> np.ndarray:
        """Estimate quantile treatment effects at specified levels.

        Parameters
        ----------
        X : (n, d) covariates.
        quantiles : (n_q,) quantile levels.  Default: 0.05, 0.10, …, 0.95.

        Returns
        -------
        (n, n_q) array of quantile treatment effects.
        """
        if quantiles is None:
            quantiles = np.linspace(0.05, 0.95, 19)
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        z_dum = torch.ones(X.shape[0], device=self.device)

        qte = np.zeros((X.shape[0], len(quantiles)))
        for model in self.models:
            model.eval()
            with torch.no_grad():
                for j, q in enumerate(quantiles):
                    _, _, te, _ = model(x_t, z_dum, q)
                    qte[:, j] += te[:, 1].cpu().numpy()
        qte /= self.n_models
        return qte

    def estimate_qte_separate(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
        quantiles: np.ndarray | None = None,
        epochs: int = 3000, hdim: int = 128, nh: int = 32,
    ) -> np.ndarray:
        """Estimate QTE by fitting separate IQN models for treated and control.

        This avoids the constant-QTE problem that arises when a single
        causal model factorizes the outcome as mu(x) + te(x,q)*z,
        because each group gets its own quantile function.

        QTE(q, x) = Q_{Y|X,Z=1}(q | x) - Q_{Y|X,Z=0}(q | x)

        Parameters
        ----------
        X : (n, d) covariates.
        Y : (n,) outcomes.
        Z : (n,) treatment (0/1).
        quantiles : (n_q,) quantile levels.
        epochs : training epochs for each IQN.
        hdim : hidden dimension for IQN.
        nh : cosine embedding dimension.

        Returns
        -------
        (n, n_q) quantile treatment effects.
        """
        if quantiles is None:
            quantiles = np.linspace(0.05, 0.95, 19)

        X1, Y1 = X[Z == 1], Y[Z == 1]
        X0, Y0 = X[Z == 0], Y[Z == 0]

        m1, xm1, xs1, ym1, ys1 = train_iqn(X1, Y1, epochs=epochs, hdim=hdim, nh=nh, seed=42)
        m0, xm0, xs0, ym0, ys0 = train_iqn(X0, Y0, epochs=epochs, hdim=hdim, nh=nh, seed=43)

        X1t = torch.tensor((X - xm1) / xs1, dtype=torch.float32)
        X0t = torch.tensor((X - xm0) / xs0, dtype=torch.float32)

        qte = np.zeros((X.shape[0], len(quantiles)))
        with torch.no_grad():
            for j, q in enumerate(quantiles):
                f1 = m1(X1t, q)[:, 1].numpy() * ys1 + ym1
                f0 = m0(X0t, q)[:, 1].numpy() * ys0 + ym0
                qte[:, j] = f1 - f0

        return qte

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        """Predict P(Z=1 | X) averaged over ensemble.

        Returns
        -------
        (n,) propensity scores.
        """
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        z_dum = torch.zeros(X.shape[0], device=self.device)

        pi_sum = np.zeros(X.shape[0])
        for model in self.models:
            model.eval()
            with torch.no_grad():
                _, pi_logit, _, _ = model(x_t, z_dum, 0.5)
                pi_sum += torch.sigmoid(pi_logit.view(-1)).cpu().numpy()
        return pi_sum / self.n_models
