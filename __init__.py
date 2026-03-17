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
gbc.causal          Ch 13 §sec-causal (CausalIQN/v2, CausalEnsemble)
gbc.welfare         MEU + Yaari distortion weighting (Polson, Ruggeri & Sokolov 2024)
gbc.conformal       Ch 14 §sec-cal-conformal, §sec-cal-per-horizon
gbc.data            Ch 2 §sec-datasets (all dataset loaders)
gbc.metrics         Ch 5 §sec-iqn-diagnostics (CRPS, coverage, RMSE)
gbc.plotting        All chapters (quantile fan, calibration, PIT plots)
gbc.utils           App. B §sec-computing (seeding, device, scheduler)

Usage::

    from gbc import IQN, train_iqn, sample_iqn
    from gbc.causal import CausalIQN, CausalIQNv2, CausalEnsemble
    from gbc.welfare import meu, yaari_weighted, distortion_cvar
    from gbc.metrics import crps_samples, coverage
"""

from gbc.iqn import IQN, train_iqn, sample_iqn
from gbc.ensemble import HetMLP
from gbc.metrics import crps_samples, crps_gaussian, coverage, pi_width
from gbc.causal import CausalIQN, CausalIQNv2, CausalEnsemble

__version__ = "0.2.0"
__all__ = [
    "IQN", "train_iqn", "sample_iqn",
    "HetMLP",
    "CausalIQN", "CausalIQNv2", "CausalEnsemble",
    "crps_samples", "crps_gaussian", "coverage", "pi_width",
]
