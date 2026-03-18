# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GBC (Generative Bayesian Computation) is a PyTorch research package companion to the textbook "Generative Bayesian Computation: Quantile Neural Networks for Inference and Surrogates" (Polson & Sokolov, 2026). The core idea: an Implicit Quantile Network (IQN) trained by SGD replaces MCMC for posterior sampling and surrogate construction.

## Installation & Development

```bash
pip install -e .                    # editable install (recommended)
pip install -e ".[dev]"             # includes pytest
```

Python ≥ 3.10 required. Core deps: torch, numpy, scipy, scikit-learn, matplotlib, pandas.

## Running Tests & Validation

No formal test suite exists. Validation is done via reproduction scripts:

```bash
python scripts/ch07_surrogates.py          # BGP & Friedman surrogate benchmarks
python scripts/ch08_jumps.py               # Aug-IQN on phantom/star jump functions
python scripts/ch09_active_learning.py     # RandomPriorNet ensemble on rocket data
python scripts/ch14_lake.py                # Lake temperature forecasting + conformal
```

Scripts support SLURM array jobs (`--rep $SLURM_ARRAY_TASK_ID`) and batch replication (`--reps 50`).

## Package Layout

All source modules live in the `gbc/` subdirectory (standard Python package layout).

Import as: `from gbc import IQN, train_iqn, sample_iqn` or `from gbc.data import load_motorcycle`.

## Architecture

### Core IQN Pipeline (iqn.py + loss.py)

The IQN class has a two-output head:
- **Col 0:** mean anchor (L1 loss) for training stability
- **Col 1:** quantile estimate (pinball loss) for actual prediction

Training uses a three-term composite loss: `w₀·L1_anchor + w₁·monotonicity_penalty + w₂·pinball_loss`

All IQN training functions return a 5-tuple `(model, xm, xs, ym, ys)` where the last four are normalization statistics (mean/std for inputs and output). These must be passed to `predict_iqn()` and `sample_iqn()` for denormalization.

Quantile embedding: `cosine_embed(tau, nh, device, dtype)` in `iqn.py` — shared by IQN, CausalIQN, and CausalIQNv2.

### Module Dependency Graph

```
loss.py              (pinball, composite, gaussian_nll)
  ↑
iqn.py               (IQN, cosine_embed, train/sample/predict)
  ↑
causal.py ──→ loss.py, iqn.py, utils.py
augment.py ─→ loss.py
ensemble.py → loss.py
sensitivity.py → iqn.py
multivariate.py → iqn.py
plotting.py → metrics.py

Independent: data.py, metrics.py, spatial.py, conformal.py, welfare.py, utils.py, active_learning.py
```

### Key Design Patterns

- **Optimizer:** Adam + CosineAnnealingLR (eta_min = lr × 0.01) throughout
- **Device:** `get_device()` auto-selects CUDA/CPU
- **Causal variants:** CausalIQN (multiplicative τ⊙TE) vs CausalIQNv2 (additive with skip connection, preserves heterogeneity better)
- **Multivariate:** autoregressive decomposition θ₁, θ₂, …, θ_d with optional Cholesky preconditioning
- **Conformal:** temporal CV (not random CV), per-stratum quantile calibration

### Each module maps to a book chapter

Modules are cross-referenced to chapters (see `__init__.py` docstrings and README module map). When modifying a module, the corresponding book chapter provides the mathematical specification.
