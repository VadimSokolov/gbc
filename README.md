# gbc — Generative Bayesian Computation

GBC replaces MCMC with a neural network trained by SGD. The core idea:
simulate `(θ, y)` pairs from the prior and likelihood, train an
**Implicit Quantile Network (IQN)** on the pinball loss, and obtain
posterior quantiles for any observed dataset via a single forward pass —
no chains, no burn-in, no likelihood evaluation.

This package implements all methods developed in the book, with each
module cross-referenced to its corresponding chapter and section.

## Installation

```bash
pip install git+https://github.com/VadimSokolov/gbc.git
```

Or clone and install in editable mode (recommended for following along with the book):

```bash
git clone https://github.com/VadimSokolov/gbc.git
cd gbc
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 2.0, SciPy ≥ 1.10,
scikit-learn ≥ 1.2, matplotlib ≥ 3.7, pandas ≥ 2.0.

## Quick Start

```python
from gbc import IQN, train_iqn, sample_iqn
from gbc.metrics import crps_samples, coverage
from gbc.plotting import quantile_fan
from gbc.data import load_motorcycle

# Load data
X, y = load_motorcycle()           # 133 observations, Ch 5

# Train IQN (Ch 5 §sec-iqn-training)
model, xm, xs, ym, ys = train_iqn(X, y, epochs=3000)

# Draw 500 posterior predictive samples
samples = sample_iqn(model, X, xm, xs, ym, ys, B=500)   # (500, 133)

print(f"CRPS  : {crps_samples(y, samples):.3f}")
print(f"90% coverage: {coverage(y, samples, alpha=0.90):.3f}")
```

### Multivariate Posterior (MGBC)

```python
from gbc.multivariate import (
    train_multivariate_iqn, sample_multivariate_iqn,
    order_by_variance, cholesky_precondition, cholesky_inverse,
)
from gbc.metrics import energy_score

# Y is (n, d) multivariate targets, X is (n, p) features
order = order_by_variance(Y)                  # order by decreasing variance
Z, L, mu = cholesky_precondition(Y[:, order]) # whiten

chain = train_multivariate_iqn(X, Z, epochs=3000)
Z_samples = sample_multivariate_iqn(chain, X_test, B=500)  # (500, n_test, d)
Y_samples = cholesky_inverse(Z_samples, L, mu)              # de-whiten

print(f"Energy Score: {energy_score(Y_test[:, order], Z_samples):.4f}")
```

## Module Map

| Module | Book chapter(s) | Purpose |
|--------|-----------------|---------|
| `gbc.iqn` | Ch 5 §sec-iqn | IQN class, `train_iqn`, `sample_iqn`, `predict_iqn` |
| `gbc.loss` | Ch 3, Ch 5 §sec-iqn-loss | Pinball, composite, Gaussian NLL |
| `gbc.multivariate` | Ch 4, 10; Lopes et al. (2026) | MGBC autoregressive chain, Cholesky preconditioning |
| `gbc.ensemble` | Ch 14 §sec-cal-het-mlp | Heteroskedastic MLP ensemble |
| `gbc.augment` | Ch 8 §sec-jumps | Aug-IQN pipeline for jump discontinuities |
| `gbc.active_learning` | Ch 9 §sec-al-randprior | Randomized-prior ensemble, acquisition |
| `gbc.causal` | Ch 13 §sec-causal | CausalIQN/v2 for CATE/ATE estimation |
| `gbc.welfare` | Ch 13; Polson et al. (2024) | MEU, Yaari distortion weighting, portfolio allocation |
| `gbc.sensitivity` | Ch 7 | Partial effects, elasticities, feature importance |
| `gbc.spatial` | Ch 10 | Spatial lags, Moran eigenvectors, panel features |
| `gbc.conformal` | Ch 14 §sec-cal-conformal | Conformal calibration, per-stratum PI |
| `gbc.data` | Ch 2, 5, 8, 9, 10 | Dataset loaders (motorcycle, Friedman, etc.) |
| `gbc.metrics` | Ch 5, 7, 8, 14 | CRPS, energy score, coverage, PI width, RMSE |
| `gbc.plotting` | All chapters | Quantile fan chart, PIT calibration plot |
| `gbc.utils` | App. B §sec-computing | Seeding, device selection, LR scheduler |

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v       # 125 tests, ~9 seconds
```

Test files cover each use case with synthetic data:

| Test file | Use case |
|-----------|----------|
| `test_prediction.py` | IQN pipeline, metrics, sensitivity, plotting |
| `test_jumps.py` | Aug-IQN for jump discontinuities |
| `test_causal.py` | Treatment effects, welfare analysis |
| `test_spatial.py` | Spatial feature engineering |
| `test_multivariate.py` | Autoregressive IQN, Cholesky preconditioning |
| `test_active_learning.py` | Ensemble disagreement acquisition |
| `test_ensemble.py` | HetMLP, conformal calibration |
| `test_financial.py` | MGBC on multivariate returns, portfolio allocation |
| `test_core.py` | Losses, cosine embedding, utils, data loaders |

## Key Results Reproduced by This Package

| Benchmark | GBC result | Competitor | Chapter |
|-----------|-----------|------------|---------|
| Motorcycle (CRPS) | 12.49 | hetGP 12.56 | Ch 5 |
| Phantom (CRPS) | 0.009 | MJGP 0.010 | Ch 8 |
| Star (CRPS) | 0.013 | MJGP 0.024 | Ch 8 |
| Rocket AL (RMSE) | 0.00723 | DGP 0.02134 | Ch 9 |
| Lake temp. (coverage) | 90.2% | GPBC ~85% | Ch 14 |

## Citing

If you use this package, please cite the book:

```bibtex
@book{polson2026gbc,
  title     = {Generative {B}ayesian Computation: Quantile Neural Networks
               for Inference and Surrogates},
  author    = {Polson, Nicholas and Sokolov, Vadim},
  year      = {2026},
  publisher = {(in preparation)}
}
```
