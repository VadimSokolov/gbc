"""Tests for the GNN-based spatial encoder and IQN transport map."""
import pytest
import numpy as np
import torch

try:
    from torch_geometric.data import Batch
    from gbc.spatial_gnn import (
        build_spatial_graph,
        build_batch,
        MessageLayer,
        SpatialEncoder,
        SpatialIQN,
        GBCSpatial,
        huber_quantile_loss,
        train_step,
        posterior_samples,
    )
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

pytestmark = pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def spatial_field(rng):
    """A small spatial field: n=50 locations on [0,1]^2."""
    n = 50
    S = rng.uniform(size=(n, 2)).astype(np.float32)
    Y = rng.standard_normal(n).astype(np.float32)
    return Y, S


@pytest.fixture
def spatial_batch(rng):
    """A batch of 4 spatial fields."""
    Ys, Ss = [], []
    for _ in range(4):
        n = rng.integers(30, 60)
        Ss.append(rng.uniform(size=(n, 2)).astype(np.float32))
        Ys.append(rng.standard_normal(n).astype(np.float32))
    return Ys, Ss


# ── Graph construction tests ────────────────────────────────────────

class TestBuildGraph:
    def test_basic_shape(self, spatial_field):
        Y, S = spatial_field
        g = build_spatial_graph(Y, S)
        assert g.x.shape == (50, 3)
        assert g.edge_attr.shape[1] == 4
        assert g.edge_index.shape[0] == 2

    def test_node_features(self, spatial_field):
        Y, S = spatial_field
        g = build_spatial_graph(Y, S)
        np.testing.assert_allclose(g.x[:, 0].numpy(), Y, atol=1e-6)
        np.testing.assert_allclose(g.x[:, 1:].numpy(), S, atol=1e-6)

    def test_edge_distance_positive(self, spatial_field):
        Y, S = spatial_field
        g = build_spatial_graph(Y, S)
        dists = g.edge_attr[:, 0]
        assert (dists > 0).all(), "No self-loops: distances must be positive"

    def test_custom_radius(self, spatial_field):
        Y, S = spatial_field
        g_small = build_spatial_graph(Y, S, radius=0.05)
        g_large = build_spatial_graph(Y, S, radius=0.5)
        assert g_small.edge_index.shape[1] <= g_large.edge_index.shape[1]

    def test_batch(self, spatial_batch):
        Ys, Ss = spatial_batch
        batch = build_batch(Ys, Ss)
        total_n = sum(len(y) for y in Ys)
        assert batch.x.shape[0] == total_n
        assert batch.batch.max().item() == 3  # 4 graphs: 0,1,2,3


# ── MessageLayer tests ──────────────────────────────────────────────

class TestMessageLayer:
    def test_output_shape(self, spatial_field):
        Y, S = spatial_field
        g = build_spatial_graph(Y, S)
        H = 64
        node_proj = torch.nn.Sequential(torch.nn.Linear(3, H), torch.nn.ReLU())
        edge_proj = torch.nn.Sequential(torch.nn.Linear(4, H), torch.nn.ReLU())
        layer = MessageLayer(H)

        h = node_proj(g.x)
        f = edge_proj(g.edge_attr)
        h_out = layer(h, g.edge_index, f)
        assert h_out.shape == (50, H)

    def test_deterministic(self, spatial_field):
        Y, S = spatial_field
        g = build_spatial_graph(Y, S)
        H = 32
        torch.manual_seed(0)
        layer = MessageLayer(H)
        h = torch.randn(50, H)
        f = torch.randn(g.edge_index.shape[1], H)

        out1 = layer(h, g.edge_index, f)
        out2 = layer(h, g.edge_index, f)
        torch.testing.assert_close(out1, out2)


# ── SpatialEncoder tests ────────────────────────────────────────────

class TestSpatialEncoder:
    def test_output_shape(self, spatial_batch):
        Ys, Ss = spatial_batch
        batch = build_batch(Ys, Ss)
        enc = SpatialEncoder(d_bottleneck=16, H=32, H_prime=64, n_layers=2)
        T = enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert T.shape == (4, 16)

    def test_permutation_invariance(self, spatial_field):
        Y, S = spatial_field
        enc = SpatialEncoder(d_bottleneck=8, H=32, H_prime=64, n_layers=2)
        enc.eval()

        g1 = build_spatial_graph(Y, S)
        batch1 = Batch.from_data_list([g1])
        T1 = enc(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)

        perm = np.random.default_rng(99).permutation(len(Y))
        g2 = build_spatial_graph(Y[perm], S[perm])
        batch2 = Batch.from_data_list([g2])
        T2 = enc(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

        torch.testing.assert_close(T1, T2, atol=1e-5, rtol=1e-5)


# ── SpatialIQN tests ────────────────────────────────────────────────

class TestSpatialIQN:
    def test_output_shape(self):
        iqn = SpatialIQN(d_T=16, d_out=4, H=64, n_cosine=32, d_tau=64)
        T = torch.randn(8, 16)
        tau = torch.rand(8)
        out = iqn(T, tau)
        assert out.shape == (8, 4)

    def test_exp_activation(self):
        iqn = SpatialIQN(
            d_T=8, d_out=2, H=32, n_cosine=16, d_tau=32,
            out_activations=["exp", "exp"],
        )
        T = torch.randn(5, 8)
        tau = torch.rand(5)
        out = iqn(T, tau)
        assert (out > 0).all(), "exp activation must produce positive outputs"


# ── GBCSpatial joint model tests ────────────────────────────────────

class TestGBCSpatial:
    def test_end_to_end(self, spatial_batch):
        Ys, Ss = spatial_batch
        batch = build_batch(Ys, Ss)
        model = GBCSpatial(
            encoder_kwargs=dict(d_bottleneck=8, H=32, H_prime=64, n_layers=2),
            iqn_kwargs=dict(d_out=4, H=64, n_cosine=16, d_tau=32),
        )
        B = 4
        N_tau = 3
        tau = torch.rand(B * N_tau)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, tau)
        assert out.shape == (B * N_tau, 4)

    def test_train_step(self, spatial_batch):
        Ys, Ss = spatial_batch
        batch_data = build_batch(Ys, Ss)
        model = GBCSpatial(
            encoder_kwargs=dict(d_bottleneck=8, H=32, H_prime=64, n_layers=2),
            iqn_kwargs=dict(d_out=4, H=64, n_cosine=16, d_tau=32),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        psi_true = torch.randn(4, 4)

        loss = train_step(model, optimizer, batch_data, psi_true, N_tau=4)
        assert isinstance(loss, float)
        assert loss > 0


# ── Loss function tests ─────────────────────────────────────────────

class TestHuberQuantileLoss:
    def test_positive(self):
        y = torch.randn(10, 3)
        pred = torch.randn(10, 3)
        tau = torch.rand(10)
        loss = huber_quantile_loss(y, pred, tau)
        assert loss.item() > 0

    def test_zero_at_perfect(self):
        y = torch.ones(10, 2)
        pred = torch.ones(10, 2)
        tau = torch.rand(10)
        loss = huber_quantile_loss(y, pred, tau)
        assert loss.item() < 1e-6


# ── Posterior sampling test ──────────────────────────────────────────

class TestPosteriorSamples:
    def test_shape(self, spatial_field):
        Y, S = spatial_field
        model = GBCSpatial(
            encoder_kwargs=dict(d_bottleneck=8, H=32, H_prime=64, n_layers=2),
            iqn_kwargs=dict(d_out=4, H=64, n_cosine=16, d_tau=32),
        )
        samples = posterior_samples(model, Y, S, B=100)
        assert samples.shape == (100, 4)
