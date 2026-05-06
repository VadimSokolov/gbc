"""GNN-based spatial encoder and IQN transport map for GBC-Spatial.

Provides a graph neural network that maps an irregularly observed spatial
field (Y, S) to a low-dimensional sufficient statistic T, and couples it
with an Implicit Quantile Network for amortized posterior sampling of
spatial model parameters.

Architecture
------------
(Y, S) --[GNN_phi]--> T in R^d --[IQN_gamma]--> psi_tilde(tau) in Theta

The GNN builds a spatial proximity graph, performs message passing with
multi-aggregation (sum, mean, max), and applies attention-weighted global
pooling to produce a permutation-invariant bottleneck statistic.

Book references
---------------
- Ch 10 §sec-bayes-iqn : GBC for spatially structured simulators
- Polson & Sokolov (2026), "GBC-Spatial" paper

Dependencies
------------
Requires ``torch`` and ``torch_geometric``.  Install via::

    pip install torch torch-geometric

Notes
-----
All modules work on CPU and CUDA.  The ``build_spatial_graph`` helper
converts raw numpy arrays into a ``torch_geometric.data.Data`` object
ready for batching.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import global_add_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ── Graph construction ───────────────────────────────────────────────

def build_spatial_graph(
    Y: np.ndarray,
    S: np.ndarray,
    radius: float | None = None,
    percentile: float = 15.0,
) -> "Data":
    """Convert a spatial field (Y, S) into a PyG Data object.

    Parameters
    ----------
    Y : (n,) observed values.
    S : (n, 2) observation locations.
    radius : connectivity radius.  If None, set at the ``percentile``-th
             percentile of pairwise distances.
    percentile : percentile of pairwise distances used when ``radius``
                 is None (default 15).

    Returns
    -------
    torch_geometric.data.Data with node features ``x``, edge features
    ``edge_attr``, and ``edge_index``.
    """
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for spatial GNN")

    Y = np.asarray(Y, dtype=np.float32).ravel()
    S = np.asarray(S, dtype=np.float32)
    n = len(Y)

    # Pairwise distances
    diff = S[:, None, :] - S[None, :, :]          # (n, n, 2)
    dist = np.linalg.norm(diff, axis=-1)           # (n, n)

    if radius is None:
        # Use upper triangle (no self-loops) for percentile
        triu_idx = np.triu_indices(n, k=1)
        radius = float(np.percentile(dist[triu_idx], percentile))

    # Build edges (exclude self-loops)
    mask = (dist <= radius) & (dist > 0)
    src, dst = np.where(mask)

    # Node features: [Y(s_i), s_i1, s_i2]
    x = np.column_stack([Y, S])                    # (n, 3)

    # Edge features: [distance, unit_dir_x, unit_dir_y, squared_diff]
    edge_dist = dist[src, dst]
    edge_dir = diff[src, dst] / np.maximum(edge_dist[:, None], 1e-8)
    edge_sq_diff = (Y[src] - Y[dst]) ** 2
    edge_attr = np.column_stack([edge_dist, edge_dir, edge_sq_diff])  # (E, 4)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([dst, src]), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        num_nodes=n,
    )


def build_batch(
    Ys: list[np.ndarray],
    Ss: list[np.ndarray],
    radius: float | None = None,
    percentile: float = 15.0,
) -> "Batch":
    """Build a batched graph from multiple spatial fields.

    Parameters
    ----------
    Ys : list of (n_i,) observation arrays.
    Ss : list of (n_i, 2) location arrays.
    radius, percentile : passed to ``build_spatial_graph``.

    Returns
    -------
    torch_geometric.data.Batch ready for GNN forward pass.
    """
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for spatial GNN")
    graphs = [
        build_spatial_graph(y, s, radius=radius, percentile=percentile)
        for y, s in zip(Ys, Ss)
    ]
    return Batch.from_data_list(graphs)


# ── GNN spatial encoder ─────────────────────────────────────────────

class MessageLayer(nn.Module):
    """One message-passing layer with multi-aggregation and residual."""

    def __init__(self, H: int = 128):
        super().__init__()
        self.mlp_m = nn.Sequential(
            nn.Linear(3 * H, H), nn.ReLU(),
            nn.Linear(H, H),
        )
        self.mlp_u = nn.Sequential(
            nn.Linear(4 * H, H), nn.ReLU(),
            nn.Linear(H, H),
        )
        self.norm = nn.LayerNorm(H)

    def forward(
        self, h: Tensor, edge_index: Tensor, f: Tensor,
    ) -> Tensor:
        dst, src = edge_index  # (E,) each

        # MESSAGE
        m = self.mlp_m(torch.cat([h[dst], h[src], f], dim=-1))

        # AGGREGATE: sum, mean, max
        n = h.size(0)
        H_dim = h.size(-1)

        agg_sum = torch.zeros(n, H_dim, device=h.device, dtype=h.dtype)
        agg_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(m), m)

        counts = torch.zeros(n, device=h.device, dtype=h.dtype)
        counts.scatter_add_(0, dst, torch.ones(dst.size(0), device=h.device))
        counts = counts.clamp(min=1).unsqueeze(-1)
        agg_mean = agg_sum / counts

        agg_max = torch.full((n, H_dim), -1e9, device=h.device, dtype=h.dtype)
        agg_max.scatter_reduce_(
            0, dst.unsqueeze(-1).expand_as(m), m, reduce="amax",
        )

        a = torch.cat([agg_sum, agg_mean, agg_max], dim=-1)  # (n, 3H)

        # UPDATE with residual + LayerNorm
        h_new = h + self.mlp_u(torch.cat([h, a], dim=-1))
        return self.norm(h_new)


class SpatialEncoder(nn.Module):
    """GNN that maps (Y, S) to a sufficient statistic T.

    Parameters
    ----------
    d_x : node input dimension (default 3: [Y, s1, s2]).
    d_e : edge input dimension (default 4: [dist, dir_x, dir_y, sq_diff]).
    H : hidden dimension in message passing layers.
    H_prime : pre-readout projection dimension.
    n_layers : number of message passing layers.
    d_bottleneck : output dimension of the sufficient statistic T.
    """

    def __init__(
        self,
        d_x: int = 3,
        d_e: int = 4,
        H: int = 128,
        H_prime: int = 256,
        n_layers: int = 4,
        d_bottleneck: int = 32,
    ):
        super().__init__()
        self.node_proj = nn.Sequential(nn.Linear(d_x, H), nn.ReLU())
        self.edge_proj = nn.Sequential(nn.Linear(d_e, H), nn.ReLU())
        self.layers = nn.ModuleList(
            [MessageLayer(H) for _ in range(n_layers)]
        )
        self.pre_readout = nn.Sequential(nn.Linear(H, H_prime), nn.ReLU())
        self.attn_w = nn.Linear(H_prime, 1)
        self.bottleneck = nn.Linear(H_prime, d_bottleneck)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (N_total, d_x) node features.
        edge_index : (2, E_total) edge connectivity.
        edge_attr : (E_total, d_e) edge features.
        batch : (N_total,) graph membership indices.

        Returns
        -------
        T : (B, d_bottleneck) sufficient statistics per graph.
        """
        h = self.node_proj(x)
        f = self.edge_proj(edge_attr)

        for layer in self.layers:
            h = layer(h, edge_index, f)

        h_tilde = self.pre_readout(h)  # (N_total, H')

        # Attention-weighted pooling (per graph)
        logits = self.attn_w(h_tilde)  # (N_total, 1)
        # Per-graph softmax
        from torch_geometric.utils import softmax as pyg_softmax
        alpha = pyg_softmax(logits, batch)  # (N_total, 1)
        if not HAS_PYG:
            raise ImportError("torch_geometric required")
        g = global_add_pool(alpha * h_tilde, batch)  # (B, H')

        T = self.bottleneck(g)  # (B, d_bottleneck)
        return T


# ── IQN transport map ────────────────────────────────────────────────

class SpatialIQN(nn.Module):
    """Implicit Quantile Network for spatial posterior sampling.

    Maps (T, tau) to a posterior draw psi_tilde(tau).

    Parameters
    ----------
    d_T : dimension of the sufficient statistic (bottleneck).
    n_cosine : number of cosine basis functions for tau embedding.
    d_tau : dimension of the quantile/statistic embedding.
    H : hidden dimension in the MLP decoder.
    d_out : number of output parameters (e.g. 4 for stationary Matern).
    out_activations : list of per-parameter output activations.
        'exp' for positive parameters, 'sigmoid' for (0,1), None for R.
    """

    def __init__(
        self,
        d_T: int = 32,
        n_cosine: int = 64,
        d_tau: int = 128,
        H: int = 256,
        d_out: int = 4,
        out_activations: list[str | None] | None = None,
    ):
        super().__init__()
        self.n_cosine = n_cosine
        self.W_tau = nn.Linear(n_cosine, d_tau)
        self.W_T = nn.Linear(d_T, d_tau)
        self.decoder = nn.Sequential(
            nn.Linear(d_tau, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, d_out),
        )
        self.out_activations = out_activations

    def embed_tau(self, tau: Tensor) -> Tensor:
        """Cosine embedding: tau in (B,) -> phi(tau) in (B, d_tau)."""
        i = torch.arange(
            1, self.n_cosine + 1, device=tau.device, dtype=tau.dtype,
        )
        phi = torch.cos(math.pi * i.unsqueeze(0) * tau.unsqueeze(1))
        return torch.relu(self.W_tau(phi))

    def forward(self, T: Tensor, tau: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        T : (B, d_T) sufficient statistics.
        tau : (B,) quantile levels in (0, 1).

        Returns
        -------
        psi : (B, d_out) posterior draws.
        """
        psi_T = torch.relu(self.W_T(T))       # (B, d_tau)
        phi_tau = self.embed_tau(tau)           # (B, d_tau)
        z = psi_T * phi_tau                    # Hadamard fusion
        out = self.decoder(z)                  # (B, d_out)

        if self.out_activations is not None:
            parts = []
            for k, act in enumerate(self.out_activations):
                col = out[:, k : k + 1]
                if act == "exp":
                    parts.append(torch.exp(col))
                elif act == "sigmoid":
                    parts.append(torch.sigmoid(col))
                else:
                    parts.append(col)
            out = torch.cat(parts, dim=-1)
        return out


# ── Joint model ──────────────────────────────────────────────────────

class GBCSpatial(nn.Module):
    """Joint GNN encoder + IQN transport map for spatial posterior sampling.

    Parameters
    ----------
    encoder_kwargs : keyword arguments for ``SpatialEncoder``.
    iqn_kwargs : keyword arguments for ``SpatialIQN``.  The ``d_T`` parameter
        is set automatically from the encoder's bottleneck dimension.
    """

    def __init__(
        self,
        encoder_kwargs: dict | None = None,
        iqn_kwargs: dict | None = None,
    ):
        super().__init__()
        enc_kw = encoder_kwargs or {}
        iqn_kw = iqn_kwargs or {}

        self.encoder = SpatialEncoder(**enc_kw)
        d_T = enc_kw.get("d_bottleneck", 32)
        iqn_kw.setdefault("d_T", d_T)
        self.iqn = SpatialIQN(**iqn_kw)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        tau: Tensor,
    ) -> Tensor:
        """End-to-end forward: spatial data -> posterior draw.

        Parameters
        ----------
        x, edge_index, edge_attr, batch : graph data (from PyG Batch).
        tau : (B * N_tau,) quantile levels.

        Returns
        -------
        psi : (B * N_tau, d_out) posterior draws.
        """
        T = self.encoder(x, edge_index, edge_attr, batch)  # (B, d_T)
        B = T.size(0)
        N_tau = tau.size(0) // B

        if N_tau > 1:
            T = T.unsqueeze(1).expand(-1, N_tau, -1).reshape(-1, T.size(-1))
        return self.iqn(T, tau)


# ── Training utilities ───────────────────────────────────────────────

def huber_quantile_loss(
    y_true: Tensor,
    y_pred: Tensor,
    tau: Tensor,
    kappa: float = 1.0,
) -> Tensor:
    """Huber quantile (pinball) loss.

    Parameters
    ----------
    y_true : (B, p) true parameter values.
    y_pred : (B, p) predicted quantiles.
    tau : (B,) quantile levels.
    kappa : Huber threshold (default 1.0).

    Returns
    -------
    Scalar loss.
    """
    u = y_true - y_pred
    huber = torch.where(
        u.abs() <= kappa,
        0.5 * u ** 2,
        kappa * (u.abs() - 0.5 * kappa),
    )
    weight = torch.abs(tau.unsqueeze(-1) - (u < 0).float())
    return (weight * huber / kappa).mean()


def train_step(
    model: GBCSpatial,
    optimizer: torch.optim.Optimizer,
    batch_data: "Batch",
    psi_true: Tensor,
    N_tau: int = 8,
    kappa: float = 1.0,
) -> float:
    """One training step: forward + backward + optimizer step.

    Parameters
    ----------
    model : GBCSpatial joint model.
    optimizer : torch optimizer.
    batch_data : PyG Batch of spatial graphs.
    psi_true : (B, p) true parameter values for this batch.
    N_tau : number of quantile levels sampled per instance.
    kappa : Huber threshold.

    Returns
    -------
    Scalar loss value.
    """
    model.train()
    B = psi_true.size(0)
    device = psi_true.device

    tau = torch.rand(B * N_tau, device=device)

    psi_pred = model(
        batch_data.x.to(device),
        batch_data.edge_index.to(device),
        batch_data.edge_attr.to(device),
        batch_data.batch.to(device),
        tau,
    )

    psi_rep = psi_true.unsqueeze(1).expand(-1, N_tau, -1).reshape(-1, psi_true.size(-1))

    loss = huber_quantile_loss(psi_rep, psi_pred, tau, kappa=kappa)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def posterior_samples(
    model: GBCSpatial,
    Y_obs: np.ndarray,
    S_obs: np.ndarray,
    B: int = 2000,
    radius: float | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Draw B posterior samples from p(psi | Y_obs, S_obs).

    Parameters
    ----------
    model : trained GBCSpatial model.
    Y_obs : (n,) observed values.
    S_obs : (n, 2) observation locations.
    B : number of posterior draws.
    radius : connectivity radius (None = auto).
    device : torch device.

    Returns
    -------
    (B, p) array of posterior samples.
    """
    model.eval()
    graph = build_spatial_graph(Y_obs, S_obs, radius=radius)
    batch = Batch.from_data_list([graph]).to(device)

    T = model.encoder(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch,
    )  # (1, d_T)
    T_rep = T.expand(B, -1)
    tau = torch.rand(B, device=device)
    psi = model.iqn(T_rep, tau)
    return psi.cpu().numpy()
