"""
gbc — Generative Bayesian Computation
======================================

Python package for the textbook:
  "Generative Bayesian Computation: Quantile Neural Networks
   for Inference and Surrogates"
  Nicholas Polson & Vadim Sokolov (2026)

Module → Book chapter mapping
------------------------------
gbc.loss            Ch 3 §sec-quant-check-loss, Ch 5 §sec-iqn-loss,
                    Ch 14 §sec-cal-nll
gbc.iqn             Ch 5 §sec-iqn (core IQN class, train, sample)
gbc.ensemble        Ch 14 §sec-cal-het-mlp (HetMLP + ensemble_predict)
gbc.augment         Ch 8 §sec-jumps-em/classifier/augiqn (Aug-IQN pipeline)
gbc.active_learning Ch 9 §sec-al-randprior, §sec-al-acquisition
gbc.causal          Ch 13 §sec-causal (CausalIQN, CATE/ATE estimation)
gbc.conformal       Ch 14 §sec-cal-conformal, §sec-cal-per-horizon
gbc.data            Ch 2 §sec-datasets (all dataset loaders)
gbc.metrics         Ch 5 §sec-iqn-diagnostics (CRPS, coverage, RMSE)
gbc.plotting        All chapters (quantile fan, calibration, PIT plots)
gbc.utils           App. B §sec-computing (seeding, device, scheduler)

Usage::

    from gbc import IQN, train_iqn, sample_iqn
    from gbc.loss import pinball_loss, composite_loss
    from gbc.ensemble import HetMLP
    from gbc.metrics import crps_samples, coverage
"""

from gbc.iqn import IQN, train_iqn, sample_iqn
from gbc.ensemble import HetMLP
from gbc.metrics import crps_samples, crps_gaussian, coverage, pi_width

__version__ = "0.1.0"
__all__ = [
    "IQN", "train_iqn", "sample_iqn",
    "HetMLP",
    "crps_samples", "crps_gaussian", "coverage", "pi_width",
]
