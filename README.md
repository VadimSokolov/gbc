# gbc — Generative Bayesian Computation

Python companion package for the textbook:

> **Generative Bayesian Computation: Quantile Neural Networks for Inference and Surrogates**
> Nicholas Polson (University of Chicago) & Vadim Sokolov (George Mason University), 2026.

## What This Is

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

## Module Map

| Module | Book chapter(s) | Purpose |
|--------|-----------------|---------|
| `gbc.iqn` | Ch 5 §sec-iqn | IQN class, `train_iqn`, `sample_iqn` |
| `gbc.loss` | Ch 3 §sec-quant-check-loss, Ch 5 §sec-iqn-loss | Pinball, composite, Gaussian NLL |
| `gbc.ensemble` | Ch 14 §sec-cal-het-mlp | Heteroskedastic MLP ensemble |
| `gbc.augment` | Ch 8 §sec-jumps | Aug-IQN pipeline for jump discontinuities |
| `gbc.active_learning` | Ch 9 §sec-al-randprior | Randomized-prior ensemble, acquisition |
| `gbc.causal` | Ch 13 §sec-causal | CausalIQN for CATE/ATE estimation |
| `gbc.conformal` | Ch 14 §sec-cal-conformal | Conformal calibration, per-stratum PI |
| `gbc.data` | Ch 2, 5, 8, 9, 10 | Dataset loaders (motorcycle, Friedman, etc.) |
| `gbc.metrics` | Ch 5, 7, 8, 14 | CRPS, coverage, PI width, RMSE |
| `gbc.plotting` | All chapters | Quantile fan chart, PIT calibration plot |
| `gbc.utils` | App. B §sec-computing | Seeding, device selection, LR scheduler |

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
