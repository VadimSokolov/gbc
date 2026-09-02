"""
gbc: Generative Bayesian Computation
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
gbc.survivor        Survivor function estimation and tail functionals from IQN
gbc.evt             Extreme Value Theory distributions and simulation for GBC
gbc.evt_inference   Classical EVT baselines (GEV MLE/MCMC, Hill) and GBC-QNN estimators
gbc.shrinkage       Horseshoe and hierarchical shrinkage priors
gbc.utils           App. B §sec-computing (seeding, device, scheduler)

Usage::

    from gbc import IQN, train_iqn, sample_iqn
    from gbc.causal import CausalIQN, CausalIQNv2, CausalEnsemble
    from gbc.welfare import meu, yaari_weighted, distortion_cvar
    from gbc.metrics import crps_samples, coverage
    from gbc.survivor import survivor_curve, return_level, expected_shortfall
    from gbc.evt import gev_quantile, simulate_gev_training_data
    from gbc.shrinkage import horseshoe_posterior, horseshoe_batch
"""

from gbc.iqn import IQN, train_iqn, sample_iqn, predict_iqn
from gbc.ensemble import HetMLP
from gbc.metrics import crps_samples, crps_gaussian, coverage, pi_width, pit_values, energy_score
from gbc.causal import CausalIQN, CausalIQNv2, CausalEnsemble
from gbc.sensitivity import partial_effect, elasticity, feature_effects
from gbc.spatial import (
    spatial_lag, spatial_features, moran_eigenvectors, spatial_panel_features,
)
from gbc.multivariate import (
    train_multivariate_iqn, sample_multivariate_iqn,
    predict_multivariate_iqn, order_by_variance,
    cholesky_precondition, cholesky_inverse,
)
from gbc.survivor import (
    survivor_curve, posterior_mean, return_level,
    exceedance_prob, expected_shortfall, tail_weight_crps,
    meu_optimal_action,
)
from gbc.evt import (
    gev_quantile, gev_cdf, gev_survivor, gev_density, gev_return_level,
    gpd_quantile, gpd_survivor, gpd_density,
    poisson_loglik,
    simulate_gev_training_data, simulate_gpd_training_data,
    simulate_nonstationary_gev_training_data,
)
from gbc.shrinkage import (
    horseshoe_draw, horseshoe_batch, shrinkage_factor,
    horseshoe_posterior, simulate_hierarchical_training_data,
)
from gbc.evt_inference import (
    fit_stationary_gev, fit_ns_gev, ns_params_at, hill, gev_mcmc,
    return_level_ci_delta, gev_nll,
    gbc_priors_from_data, train_predictive_iqn, train_functional_iqn,
    gbc_return_level_posterior, gbc_predictive_samples, gbc_crps_coverage_loyo,
)

__version__ = "0.6.0"
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
    # survivor module
    "survivor_curve", "posterior_mean", "return_level",
    "exceedance_prob", "expected_shortfall", "tail_weight_crps",
    "meu_optimal_action",
    # evt module
    "gev_quantile", "gev_cdf", "gev_survivor", "gev_density",
    "gev_return_level",
    "gpd_quantile", "gpd_survivor", "gpd_density",
    "poisson_loglik",
    "simulate_gev_training_data", "simulate_gpd_training_data",
    "simulate_nonstationary_gev_training_data",
    # shrinkage module
    "horseshoe_draw", "horseshoe_batch", "shrinkage_factor",
    "horseshoe_posterior", "simulate_hierarchical_training_data",
    # evt_inference module
    "fit_stationary_gev", "fit_ns_gev", "ns_params_at", "hill", "gev_mcmc",
    "return_level_ci_delta", "gev_nll",
    "gbc_priors_from_data", "train_predictive_iqn", "train_functional_iqn",
    "gbc_return_level_posterior", "gbc_predictive_samples", "gbc_crps_coverage_loyo",
]
