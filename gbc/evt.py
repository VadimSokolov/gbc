"""Extreme Value Theory distributions and prior-predictive simulation for GBC.

Provides GEV and GPD quantile/survivor functions, Poisson process
likelihood, and simulation routines for generating prior-predictive
training data.  These tools enable GBC-IQN inference for any EVT problem.

Book references
---------------
- Fisher-Tippett-Gnedenko theorem: GEV as limit of block maxima
- Balkema-de Haan-Pickands theorem: GPD for threshold exceedances
- Smith (1989): Poisson process unification
- Coles (2001): standard EVT reference

Key distributions
-----------------
GEV(mu, sigma, xi):
    G(x) = exp{-(1 + xi*(x-mu)/sigma)^{-1/xi}}

GPD(sigma, xi):
    H(y) = 1 - (1 + xi*y/sigma)^{-1/xi},  y > 0
"""

import math

import numpy as np
from typing import Callable


# ── GEV functions ──────────────────────────────────────────────────

def gev_quantile(
    tau: np.ndarray | float,
    mu: float,
    sigma: float,
    xi: float,
) -> np.ndarray:
    r"""GEV quantile function Q(tau) = G^{-1}(tau).

    .. math::
        Q(\tau) = \mu + \frac{\sigma}{\xi}[(-\log\tau)^{-\xi} - 1]

    For xi = 0 (Gumbel): Q(tau) = mu - sigma * log(-log(tau)).

    Parameters
    ----------
    tau : quantile level(s) in (0, 1).
    mu : location parameter.
    sigma : scale parameter (> 0).
    xi : shape parameter.

    Returns
    -------
    Array of quantile values.
    """
    tau = np.asarray(tau, dtype=float)
    if abs(xi) < 1e-8:
        return mu - sigma * np.log(-np.log(tau))
    return mu + (sigma / xi) * ((-np.log(tau)) ** (-xi) - 1.0)


def gev_cdf(
    x: np.ndarray | float,
    mu: float,
    sigma: float,
    xi: float,
) -> np.ndarray:
    r"""GEV cumulative distribution function G(x).

    Parameters
    ----------
    x : evaluation point(s).
    mu, sigma, xi : GEV parameters.

    Returns
    -------
    Array of CDF values in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        return np.exp(-np.exp(-z))
    t = 1.0 + xi * z
    t = np.maximum(t, 1e-300)
    return np.exp(-t ** (-1.0 / xi))


def gev_survivor(
    t: np.ndarray | float,
    mu: float,
    sigma: float,
    xi: float,
) -> np.ndarray:
    """GEV survivor function S(t) = 1 - G(t)."""
    return 1.0 - gev_cdf(t, mu, sigma, xi)


def gev_density(
    x: np.ndarray | float,
    mu: float,
    sigma: float,
    xi: float,
) -> np.ndarray:
    """GEV probability density function g(x)."""
    x = np.asarray(x, dtype=float)
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        return (1.0 / sigma) * np.exp(-z - np.exp(-z))
    t = 1.0 + xi * z
    valid = t > 0
    result = np.zeros_like(x)
    t_v = t[valid]
    result[valid] = (
        (1.0 / sigma) * t_v ** (-1.0 / xi - 1.0) * np.exp(-t_v ** (-1.0 / xi))
    )
    return result


def gev_return_level(N: float, mu: float, sigma: float, xi: float) -> float:
    """N-period return level from GEV parameters.

    Parameters
    ----------
    N : return period (e.g. 100).
    mu, sigma, xi : GEV parameters.

    Returns
    -------
    Scalar return level z_N.
    """
    return float(gev_quantile(1.0 - 1.0 / N, mu, sigma, xi))


# ── GPD functions ──────────────────────────────────────────────────

def gpd_quantile(
    tau: np.ndarray | float,
    sigma: float,
    xi: float,
) -> np.ndarray:
    r"""GPD quantile function for exceedances.

    .. math::
        Q(\tau) = \frac{\sigma}{\xi}[(1-\tau)^{-\xi} - 1]

    For xi = 0: Q(tau) = -sigma * log(1 - tau).

    Parameters
    ----------
    tau : quantile level(s) in (0, 1).
    sigma : scale parameter (> 0).
    xi : shape parameter.
    """
    tau = np.asarray(tau, dtype=float)
    if abs(xi) < 1e-8:
        return -sigma * np.log(1.0 - tau)
    return (sigma / xi) * ((1.0 - tau) ** (-xi) - 1.0)


def gpd_survivor(
    y: np.ndarray | float,
    sigma: float,
    xi: float,
) -> np.ndarray:
    r"""GPD survivor function S(y) = (1 + xi*y/sigma)^{-1/xi}.

    Parameters
    ----------
    y : exceedance value(s) (> 0).
    sigma : scale parameter (> 0).
    xi : shape parameter.
    """
    y = np.asarray(y, dtype=float)
    if abs(xi) < 1e-8:
        return np.exp(-y / sigma)
    t = 1.0 + xi * y / sigma
    t = np.maximum(t, 1e-300)
    return t ** (-1.0 / xi)


def gpd_density(
    y: np.ndarray | float,
    sigma: float,
    xi: float,
) -> np.ndarray:
    """GPD probability density function."""
    y = np.asarray(y, dtype=float)
    if abs(xi) < 1e-8:
        return (1.0 / sigma) * np.exp(-y / sigma)
    t = 1.0 + xi * y / sigma
    valid = t > 0
    result = np.zeros_like(y)
    result[valid] = (1.0 / sigma) * (t[valid]) ** (-1.0 / xi - 1.0)
    return result


# ── Poisson process likelihood ─────────────────────────────────────

def poisson_loglik(
    y: np.ndarray,
    mu: float,
    sigma: float,
    xi: float,
    u: float,
) -> float:
    r"""Poisson process log-likelihood unifying GEV and GPD.

    Implements the Smith (1989) Poisson process representation:
    exceedances above threshold u follow a Poisson process with
    GPD marks.

    Reference implementation, not a code path behind any published number.
    Every study in the WSC 2026 paper is a block-maxima study, so no driver
    calls this and no network is trained on exceedances; the paper says as
    much.  It is kept, and covered by ``TestPoissonExponentMeasure``, because
    the formulation is one of the paper's claims and a claim you cannot run is
    worth less than one you can.  Anyone extending the work to daily series
    starts here.

    Parameters
    ----------
    y : (n,) all observations.
    mu, sigma, xi : GEV parameters.
    u : threshold.

    Returns
    -------
    Scalar log-likelihood (may be -inf for invalid parameters).
    """
    if sigma <= 0:
        return -np.inf
    exc = y[y > u]
    n_u = len(exc)
    n = len(y)

    sigma_u = sigma + xi * (u - mu)
    if sigma_u <= 0:
        return -np.inf

    # Check support
    if abs(xi) > 1e-8:
        z = 1.0 + xi * (exc - mu) / sigma
        if np.any(z <= 0):
            return -np.inf

    # Expected number of exceedances per block: the exponent measure
    #   Lambda(u) = (1 + xi (u - mu) / sigma)^(-1/xi),
    # which is the intensity of the Poisson process representation.  It is NOT
    # the GEV survivor 1 - exp(-Lambda): that is the probability a block maximum
    # exceeds u, the binomial view, and it understates the Poisson mean by 5% at
    # a 0.90 threshold and more at lower ones.
    if abs(xi) < 1e-8:
        lam_u = float(np.exp(-(u - mu) / sigma))
    else:
        t_u = 1.0 + xi * (u - mu) / sigma
        if t_u <= 0:
            return -np.inf
        lam_u = float(t_u ** (-1.0 / xi))
    if not np.isfinite(lam_u) or lam_u <= 0:
        return -np.inf

    # GPD density of exceedances
    exc_shifted = exc - u
    log_gpd = np.sum(np.log(np.maximum(gpd_density(exc_shifted, sigma_u, xi), 1e-300)))

    # Poisson count term: N(u) ~ Poisson(n * Lambda(u)), constants in n_u dropped
    ll = -n * lam_u + n_u * np.log(lam_u) + log_gpd
    return float(ll)


# ── Prior-predictive simulation for GBC training ───────────────────

def _truncated_normal(rng, mean: float, sd: float,
                      lo: float, hi: float) -> float:
    """One draw from N(mean, sd^2) conditioned on [lo, hi].

    Clipping is not truncation: it moves the rejected tail mass onto the
    endpoints as point masses.  With xi ~ N(0, 0.3^2) on [-0.8, 0.5] that is a
    4.8% atom at the upper bound, which the prior-predictive training set then
    teaches the network.  Inverse-CDF sampling gives the density the docstrings
    and the paper actually claim.
    """
    a, b = (lo - mean) / sd, (hi - mean) / sd
    lo_p, hi_p = _std_norm_cdf(a), _std_norm_cdf(b)
    u = rng.uniform(lo_p, hi_p)
    return float(mean + sd * _std_norm_ppf(u))


def _std_norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _std_norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, ~1e-9)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def simulate_gev_training_data(
    n_sim: int,
    n_obs: int = 50,
    mu_prior: tuple[float, float] = (0.0, 5.0),
    log_sigma_prior: tuple[float, float] = (0.0, 0.5),
    xi_prior: tuple[float, float] = (0.0, 0.3),
    xi_bounds: tuple[float, float] = (-1.0, 2.0),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    r"""Simulate GEV prior-predictive training data for GBC-IQN.

    Generates (summary_statistics, target_quantile, tau) triples by:
    1. Drawing GEV parameters from the prior.
    2. Simulating block maxima.
    3. Computing summary statistics.
    4. Drawing a random quantile level and its true value.

    Parameters
    ----------
    n_sim : number of training samples.
    n_obs : block maxima sample size per simulation.
    mu_prior : (mean, std) for N(mean, std^2) prior on mu.
    log_sigma_prior : (mean, std) for N(mean, std^2) prior on log(sigma).
    xi_prior : (mean, std) for truncated N(mean, std^2) prior on xi.
    xi_bounds : (lo, hi) truncation for xi.
    seed : random seed.

    Returns
    -------
    Dict with keys:
        'X' : (n_sim, n_features) summary statistics
        'y' : (n_sim,) target quantile values
        'tau' : (n_sim,) quantile levels
        'params' : (n_sim, 3) true (mu, sigma, xi)
    """
    rng = np.random.default_rng(seed)

    X_list = []
    y_list = []
    tau_list = []
    params_list = []

    for _ in range(n_sim):
        mu = rng.normal(mu_prior[0], mu_prior[1])
        log_sigma = rng.normal(log_sigma_prior[0], log_sigma_prior[1])
        sigma = np.exp(log_sigma)
        xi = _truncated_normal(rng, xi_prior[0], xi_prior[1],
                              xi_bounds[0], xi_bounds[1])

        # Simulate block maxima
        u = rng.uniform(0, 1, size=n_obs)
        data = gev_quantile(u, mu, sigma, xi)

        # Summary statistics
        stats = _block_maxima_summary(data)

        # Random quantile level and true quantile value
        tau = rng.uniform(0.01, 0.99)
        q_true = gev_quantile(tau, mu, sigma, xi)

        X_list.append(stats)
        y_list.append(float(q_true))
        tau_list.append(tau)
        params_list.append([mu, sigma, xi])

    return {
        "X": np.array(X_list),
        "y": np.array(y_list),
        "tau": np.array(tau_list),
        "params": np.array(params_list),
    }


def simulate_gpd_training_data(
    n_sim: int,
    n_exc: int = 30,
    log_sigma_prior: tuple[float, float] = (0.0, 0.5),
    xi_prior: tuple[float, float] = (0.0, 0.3),
    xi_bounds: tuple[float, float] = (-1.0, 2.0),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    r"""Simulate GPD prior-predictive training data for threshold exceedances.

    Parameters
    ----------
    n_sim : number of training samples.
    n_exc : number of exceedances per simulation.
    log_sigma_prior : (mean, std) for prior on log(sigma_u).
    xi_prior : (mean, std) for prior on xi.
    xi_bounds : truncation bounds for xi.
    seed : random seed.

    Returns
    -------
    Dict with 'X', 'y', 'tau', 'params' (sigma, xi).
    """
    rng = np.random.default_rng(seed)

    X_list = []
    y_list = []
    tau_list = []
    params_list = []

    for _ in range(n_sim):
        log_sigma = rng.normal(log_sigma_prior[0], log_sigma_prior[1])
        sigma = np.exp(log_sigma)
        xi = _truncated_normal(rng, xi_prior[0], xi_prior[1],
                              xi_bounds[0], xi_bounds[1])

        u = rng.uniform(0, 1, size=n_exc)
        data = gpd_quantile(u, sigma, xi)

        stats = _exceedance_summary(data)
        tau = rng.uniform(0.01, 0.99)
        q_true = gpd_quantile(tau, sigma, xi)

        X_list.append(stats)
        y_list.append(float(q_true))
        tau_list.append(tau)
        params_list.append([sigma, xi])

    return {
        "X": np.array(X_list),
        "y": np.array(y_list),
        "tau": np.array(tau_list),
        "params": np.array(params_list),
    }


def simulate_nonstationary_gev_training_data(
    n_sim: int,
    n_obs: int = 50,
    mu0_prior: tuple[float, float] = (0.0, 5.0),
    mu1_prior: tuple[float, float] = (0.0, 1.0),
    log_sigma_prior: tuple[float, float] = (0.0, 0.5),
    xi_prior: tuple[float, float] = (0.0, 0.3),
    xi_bounds: tuple[float, float] = (-1.0, 2.0),
    covariate_range: tuple[float, float] = (-1.0, 1.0),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    r"""Simulate non-stationary GEV training data with a linear covariate.

    The location parameter varies as mu(t) = mu0 + mu1 * T(t), where
    T(t) is a covariate (e.g. temperature anomaly).

    Parameters
    ----------
    n_sim : number of training samples.
    n_obs : observations per simulation.
    mu0_prior : (mean, std) for intercept prior.
    mu1_prior : (mean, std) for slope prior.
    log_sigma_prior : (mean, std) for log-scale prior.
    xi_prior : (mean, std) for shape prior.
    xi_bounds : truncation bounds for xi.
    covariate_range : (lo, hi) for uniform covariate distribution.
    seed : random seed.

    Returns
    -------
    Dict with 'X', 'y', 'tau', 'params' (mu0, mu1, sigma, xi).
    """
    rng = np.random.default_rng(seed)

    X_list = []
    y_list = []
    tau_list = []
    params_list = []

    for _ in range(n_sim):
        mu0 = rng.normal(mu0_prior[0], mu0_prior[1])
        mu1 = rng.normal(mu1_prior[0], mu1_prior[1])
        sigma = np.exp(rng.normal(log_sigma_prior[0], log_sigma_prior[1]))
        xi = _truncated_normal(rng, xi_prior[0], xi_prior[1],
                              xi_bounds[0], xi_bounds[1])

        # Covariate values
        T = rng.uniform(covariate_range[0], covariate_range[1], size=n_obs)
        mu_t = mu0 + mu1 * T

        # Simulate non-stationary block maxima
        u = rng.uniform(0, 1, size=n_obs)
        data = np.array([
            float(gev_quantile(u[j], mu_t[j], sigma, xi)) for j in range(n_obs)
        ])

        # Summary: block-maxima stats + covariate stats
        stats = np.concatenate([_block_maxima_summary(data), _covariate_summary(T)])

        # Target: quantile at a random covariate value
        T_star = rng.uniform(covariate_range[0], covariate_range[1])
        mu_star = mu0 + mu1 * T_star
        tau = rng.uniform(0.01, 0.99)
        q_true = gev_quantile(tau, mu_star, sigma, xi)

        X_list.append(np.append(stats, T_star))
        y_list.append(float(q_true))
        tau_list.append(tau)
        params_list.append([mu0, mu1, sigma, xi])

    return {
        "X": np.array(X_list),
        "y": np.array(y_list),
        "tau": np.array(tau_list),
        "params": np.array(params_list),
    }


# ── Summary statistics ─────────────────────────────────────────────

def _block_maxima_summary(data: np.ndarray) -> np.ndarray:
    """Compute summary statistics for block maxima data.

    Returns a vector of order statistics, moments, and quantile-based
    features that are approximately sufficient for GEV inference.
    """
    n = len(data)
    s = np.sort(data)
    return np.array([
        np.mean(data),
        np.std(data),
        np.median(data),
        s[-1],                         # maximum
        s[-min(2, n)],                 # second largest
        s[-min(5, n)],                 # fifth largest
        np.quantile(data, 0.75) - np.quantile(data, 0.25),  # IQR
        np.quantile(data, 0.90),
        np.quantile(data, 0.95),
        np.quantile(data, 0.99) if n >= 100 else s[-1],
        float(n),
    ])


def _exceedance_summary(data: np.ndarray) -> np.ndarray:
    """Compute summary statistics for GPD exceedance data."""
    n = len(data)
    s = np.sort(data)
    return np.array([
        np.mean(data),
        np.std(data),
        np.median(data),
        s[-1],
        s[-min(2, n)],
        np.quantile(data, 0.75),
        np.quantile(data, 0.90),
        np.quantile(data, 0.95),
        np.log(np.mean(data) + 1e-10),  # log mean excess
        float(n),
    ])


def _covariate_summary(T: np.ndarray) -> np.ndarray:
    """Summary statistics for a covariate vector."""
    return np.array([
        np.mean(T),
        np.std(T),
        np.min(T),
        np.max(T),
    ])
