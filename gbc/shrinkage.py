"""Hierarchical shrinkage priors for GBC.

Implements the horseshoe prior and related shrinkage priors for
hierarchical parameter estimation across groups (e.g. stations, units,
subjects).  These priors adaptively pool parameters toward a global mean
while allowing genuinely different values to escape shrinkage.

The horseshoe prior (Carvalho, Polson & Scott, 2010) places:

    theta_s | lambda_s, tau ~ N(mu, lambda_s^2 * tau^2)
    lambda_s ~ C+(0, 1)       (half-Cauchy local scale)
    tau ~ C+(0, tau_0)         (half-Cauchy global scale)

The shrinkage factor kappa_s = 1 / (1 + lambda_s^2) concentrates
near 0 or 1, adaptively separating signals from noise.

Book references
---------------
- Carvalho, Polson & Scott (2010): horseshoe estimator
- Polson & Scott (2012): half-Cauchy prior for global scale
"""

import numpy as np
from typing import Callable


def horseshoe_draw(
    n_groups: int,
    global_tau: float = 0.1,
    mu: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Draw parameters from the horseshoe prior.

    Parameters
    ----------
    n_groups : number of groups (e.g. spatial stations).
    global_tau : global shrinkage scale (or drawn from half-Cauchy
        if negative; use abs value as scale).
    mu : global mean.
    rng : numpy random generator.

    Returns
    -------
    Dict with 'theta' (n_groups,), 'lambda_' (n_groups,),
    'tau' (scalar), 'kappa' (n_groups,) shrinkage factors.
    """
    if rng is None:
        rng = np.random.default_rng()
    tau = abs(rng.standard_cauchy()) * global_tau
    lambda_ = np.abs(rng.standard_cauchy(size=n_groups))
    theta = mu + lambda_ * tau * rng.standard_normal(size=n_groups)
    kappa = 1.0 / (1.0 + lambda_ ** 2)
    return {"theta": theta, "lambda_": lambda_, "tau": tau, "kappa": kappa}


def horseshoe_batch(
    n_draws: int,
    n_groups: int,
    global_tau: float = 0.1,
    mu: float = 0.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Draw a batch of horseshoe prior samples.

    Useful for generating the prior-predictive training distribution
    in GBC when parameters are shared across groups with sparsity.

    Parameters
    ----------
    n_draws : number of prior draws.
    n_groups : number of groups per draw.
    global_tau : global shrinkage scale.
    mu : global mean.
    seed : random seed.

    Returns
    -------
    Dict with 'theta' (n_draws, n_groups), 'kappa' (n_draws, n_groups).
    """
    rng = np.random.default_rng(seed)
    theta = np.empty((n_draws, n_groups))
    kappa = np.empty((n_draws, n_groups))
    for i in range(n_draws):
        draw = horseshoe_draw(n_groups, global_tau, mu, rng)
        theta[i] = draw["theta"]
        kappa[i] = draw["kappa"]
    return {"theta": theta, "kappa": kappa}


def shrinkage_factor(
    mle: np.ndarray,
    se: np.ndarray,
    global_mean: float | None = None,
) -> np.ndarray:
    r"""Empirical horseshoe shrinkage factor from MLEs and standard errors.

    .. math::
        \hat\kappa_s = 1 - \frac{\text{SE}_s^2}{\text{SE}_s^2 + (\hat\theta_s - \bar\theta)^2}

    This is the posterior mean of the shrinkage factor under a
    normal-normal model, which approximates horseshoe shrinkage
    for moderate signals.

    Parameters
    ----------
    mle : (S,) maximum likelihood estimates per group.
    se : (S,) standard errors per group.
    global_mean : if None, uses mean(mle).

    Returns
    -------
    (S,) estimated shrinkage factors in [0, 1].
    """
    if global_mean is None:
        global_mean = np.mean(mle)
    signal = (mle - global_mean) ** 2
    noise = se ** 2
    return signal / (signal + noise + 1e-30)


def horseshoe_posterior(
    mle: np.ndarray,
    se: np.ndarray,
    global_tau: float = 0.1,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    r"""Gibbs sampler for horseshoe posterior given MLEs and standard errors.

    Treats the observation model as:
        mle_s | theta_s ~ N(theta_s, se_s^2)
        theta_s | lambda_s, tau ~ N(mu, lambda_s^2 * tau^2)
        lambda_s ~ C+(0, 1)
        tau ~ C+(0, global_tau)

    Parameters
    ----------
    mle : (S,) observed MLEs per group.
    se : (S,) standard errors per group.
    global_tau : prior scale for tau.
    n_iter : number of Gibbs iterations.
    seed : random seed.

    Returns
    -------
    Dict with:
        'theta_mean' (S,) : posterior mean of theta.
        'theta_lower' (S,) : 5th percentile.
        'theta_upper' (S,) : 95th percentile.
        'kappa_mean' (S,) : posterior mean shrinkage factor.
        'theta_samples' (n_iter // 2, S) : post-burnin samples.
    """
    rng = np.random.default_rng(seed)
    S = len(mle)
    mu = np.mean(mle)
    se2 = se ** 2

    # Initialize
    theta = mle.copy()
    lambda2 = np.ones(S)
    tau2 = global_tau ** 2
    nu = np.ones(S)  # auxiliary for half-Cauchy
    eta = 1.0        # auxiliary for global tau

    theta_samples = np.empty((n_iter, S))

    for it in range(n_iter):
        # Update theta | lambda, tau, data
        prior_prec = 1.0 / (lambda2 * tau2 + 1e-30)
        data_prec = 1.0 / (se2 + 1e-30)
        post_var = 1.0 / (prior_prec + data_prec)
        post_mean = post_var * (prior_prec * mu + data_prec * mle)
        theta = rng.normal(post_mean, np.sqrt(post_var))

        # Update lambda^2 | theta, tau  (slice via inverse-gamma auxiliary)
        rate_lambda = 1.0 / nu + (theta - mu) ** 2 / (2.0 * tau2 + 1e-30)
        lambda2 = 1.0 / rng.gamma(1.0, 1.0 / (rate_lambda + 1e-30))

        # Update nu (half-Cauchy auxiliary for lambda)
        nu = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / (lambda2 + 1e-30)))

        # Update tau^2 | theta, lambda
        rate_tau = 1.0 / eta + 0.5 * np.sum((theta - mu) ** 2 / (lambda2 + 1e-30))
        tau2 = 1.0 / rng.gamma(0.5 * (S + 1), 1.0 / (rate_tau + 1e-30))
        tau2 = max(tau2, 1e-30)

        # Update eta (half-Cauchy auxiliary for tau)
        eta = 1.0 / rng.gamma(1.0, 1.0 / (1.0 + 1.0 / (tau2 / (global_tau ** 2) + 1e-30)))

        theta_samples[it] = theta

    burnin = n_iter // 2
    post = theta_samples[burnin:]
    kappa_samples = 1.0 / (1.0 + np.outer(np.ones(len(post)), np.ones(S)))
    # Approximate kappa from the posterior variance ratio
    kappa_mean = 1.0 - np.var(post, axis=0) / (np.var(post, axis=0) + se2 + 1e-30)

    return {
        "theta_mean": np.mean(post, axis=0),
        "theta_lower": np.percentile(post, 5, axis=0),
        "theta_upper": np.percentile(post, 95, axis=0),
        "kappa_mean": np.clip(kappa_mean, 0, 1),
        "theta_samples": post,
    }


def simulate_hierarchical_training_data(
    n_sim: int,
    n_groups: int,
    n_obs_per_group: int = 50,
    group_simulator: Callable | None = None,
    summary_fn: Callable | None = None,
    global_tau: float = 0.1,
    mu: float = 0.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate hierarchical prior-predictive data for GBC with shrinkage.

    For each simulation: draw group-level parameters from the horseshoe
    prior, simulate data per group, compute summary statistics, and
    pair with a target functional.

    Parameters
    ----------
    n_sim : number of training samples.
    n_groups : number of groups.
    n_obs_per_group : observations per group.
    group_simulator : callable(theta_s, n_obs, rng) -> (n_obs,) data.
        Default: N(theta_s, 1) samples.
    summary_fn : callable(data) -> (d,) summary statistics.
        Default: (mean, std, median, max, n).
    global_tau : horseshoe global scale.
    mu : horseshoe global mean.
    seed : random seed.

    Returns
    -------
    Dict with:
        'X' : (n_sim, n_groups * d_summary) flattened group summaries.
        'y' : (n_sim, n_groups) target parameters.
        'theta_true' : (n_sim, n_groups) true parameters.
        'kappa_true' : (n_sim, n_groups) true shrinkage factors.
    """
    rng = np.random.default_rng(seed)

    if group_simulator is None:
        def group_simulator(theta_s, n_obs, rng_):
            return rng_.normal(theta_s, 1.0, size=n_obs)

    if summary_fn is None:
        def summary_fn(data):
            return np.array([np.mean(data), np.std(data), np.median(data),
                             np.max(data), float(len(data))])

    # Determine summary dimension
    test_data = group_simulator(0.0, n_obs_per_group, rng)
    d_summary = len(summary_fn(test_data))

    X_list = []
    y_list = []
    theta_list = []
    kappa_list = []

    for _ in range(n_sim):
        draw = horseshoe_draw(n_groups, global_tau, mu, rng)
        theta = draw["theta"]
        kappa = draw["kappa"]

        summaries = []
        for s in range(n_groups):
            data = group_simulator(theta[s], n_obs_per_group, rng)
            summaries.append(summary_fn(data))

        X_list.append(np.concatenate(summaries))
        y_list.append(theta)
        theta_list.append(theta)
        kappa_list.append(kappa)

    return {
        "X": np.array(X_list),
        "y": np.array(y_list),
        "theta_true": np.array(theta_list),
        "kappa_true": np.array(kappa_list),
    }
