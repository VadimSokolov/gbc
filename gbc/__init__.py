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
gbc.sensitivity     Ch 7 (partial effects, elasticities, feature importance)
gbc.spatial         Ch 10 (spatial lags, Moran eigenvectors, panel features)
gbc.multivariate    Ch 4, 10 (autoregressive IQN, Cholesky preconditioning)
gbc.portfolio       Portfolio dynamics: Kelly, (s,S) corridor, SPT, universal,
                    Bayesian selection (Polson & Sokolov 2026)
gbc.abc             ABC-MCMC sampler (Marjoram et al. 2003)
gbc.ssm             Ch 11 §sec-ssm (particle filters, GBC smoothers)
gbc.utils           App. B §sec-computing (seeding, device, scheduler)

Usage::

    from gbc import IQN, train_iqn, sample_iqn
    from gbc.causal import CausalIQN, CausalIQNv2, CausalEnsemble
    from gbc.welfare import meu, yaari_weighted, distortion_cvar
    from gbc.metrics import crps_samples, coverage
"""

from gbc.iqn import IQN, train_iqn, sample_iqn, predict_iqn
from gbc.ensemble import HetMLP
from gbc.metrics import crps_samples, crps_gaussian, coverage, pi_width, pit_values, energy_score
from gbc.causal import CausalIQN, CausalIQNv2, CausalEnsemble
from gbc.sensitivity import partial_effect, elasticity, feature_effects
from gbc.spatial import (
    spatial_lag, spatial_features, moran_eigenvectors, spatial_panel_features,
)
try:
    from gbc.spatial_gnn import (
        GBCSpatial, SpatialEncoder, SpatialIQN,
        build_spatial_graph, build_batch, posterior_samples as spatial_posterior_samples,
        huber_quantile_loss, train_step as spatial_train_step,
    )
except ImportError:
    pass  # torch_geometric not installed
from gbc.multivariate import (
    train_multivariate_iqn, sample_multivariate_iqn,
    predict_multivariate_iqn, order_by_variance,
    cholesky_precondition, cholesky_inverse,
)
from gbc.portfolio import (
    kelly_fraction, sS_corridor, bayesian_sS_corridor,
    excess_growth_rate, diversity_weights, universal_portfolio,
    bayesian_predictive_weights,
    garman_klass_vol, rogers_satchell_vol,
    simulate_gbm, simulate_sS_policy,
)
from gbc.ssm import (
    ParticleFilter, LTGBCFilter, AugmentedGBCFilter, BackwardSmoother,
    compute_ess, weighted_quantiles, multinomial_resample,
    linear_gaussian_ssm, stochastic_volatility, student_t_ssm,
)
from gbc.abc import abc_mcmc, ABCResult, summary_rates_by_decile
from gbc.testing import (
    build_slab_marginal, cauchy_slab_marginal, gaussian_slab_marginal,
    estimate_sparsity, local_fdr, log_bayes_factor,
    tweedie_select, bh_reject, fdp_power,
    mog_score, tweedie_stat,
)
try:
    from gbc.plotting import screening_histogram
except ImportError:
    pass  # matplotlib not installed (e.g. headless HPC compute node); core IQN still usable

__version__ = "0.2.0"
__all__ = [
    "IQN", "train_iqn", "sample_iqn", "predict_iqn",
    "HetMLP",
    "CausalIQN", "CausalIQNv2", "CausalEnsemble",
    "crps_samples", "crps_gaussian", "coverage", "pi_width", "pit_values",
    "energy_score",
    "partial_effect", "elasticity", "feature_effects",
    "spatial_lag", "spatial_features", "moran_eigenvectors",
    "spatial_panel_features",
    "train_multivariate_iqn", "sample_multivariate_iqn",
    "predict_multivariate_iqn", "order_by_variance",
    "cholesky_precondition", "cholesky_inverse",
    "ParticleFilter", "LTGBCFilter", "AugmentedGBCFilter", "BackwardSmoother",
    "compute_ess", "weighted_quantiles", "multinomial_resample",
    "linear_gaussian_ssm", "stochastic_volatility", "student_t_ssm",
    "abc_mcmc", "ABCResult", "summary_rates_by_decile",
    "kelly_fraction", "sS_corridor", "bayesian_sS_corridor",
    "excess_growth_rate", "diversity_weights", "universal_portfolio",
    "bayesian_predictive_weights",
    "garman_klass_vol", "rogers_satchell_vol",
    "simulate_gbm", "simulate_sS_policy",
    "build_slab_marginal", "cauchy_slab_marginal", "gaussian_slab_marginal",
    "estimate_sparsity", "local_fdr", "log_bayes_factor",
    "tweedie_select", "bh_reject", "fdp_power",
    "mog_score", "tweedie_stat",
]
