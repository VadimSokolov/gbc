"""EVT inference layer: classical baselines and GBC-QNN amortized estimators.

Complements ``evt.py`` (GEV/GPD distributions, Poisson likelihood, prior-predictive
simulation) and ``survivor.py`` (functionals of a trained IQN) with the estimators
used to compare GBC against classical extreme-value inference:

Classical
    fit_stationary_gev, fit_ns_gev, ns_params_at, hill, gev_logprior, gev_mcmc,
    return_level_ci_delta
GBC-QNN (amortized: train once on prior-predictive draws, apply by a forward pass)
    gbc_priors_from_data, train_predictive_iqn, train_functional_iqn,
    gbc_return_level_posterior, gbc_predictive_samples, gbc_crps_coverage_loyo
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from .evt import (gev_return_level, simulate_gev_training_data,
                  _block_maxima_summary)
from .iqn import train_iqn, predict_iqn, sample_iqn
from .metrics import crps_samples

_EPS = 1e-8

# ── Classical GEV likelihood and estimators ────────────────────────────────


def gev_nll(params, x):
    """Negative log-likelihood of a stationary GEV (Coles 2001 parameterisation)."""
    mu, sigma, xi = params
    if sigma <= 0:
        return 1e12
    z = (np.asarray(x, float) - mu) / sigma
    if abs(xi) < _EPS:
        return float(np.sum(np.log(sigma) + z + np.exp(-z)))
    t = 1 + xi * z
    if np.any(t <= _EPS):
        return 1e12
    return float(np.sum(np.log(sigma) + (1 + 1 / xi) * np.log(t) + t ** (-1 / xi)))


def _init(x):
    sd = np.std(x, ddof=1)
    return np.array([np.mean(x) - 0.45 * sd, max(sd * np.sqrt(6) / np.pi, 0.1), 0.1])


def fit_stationary_gev(x):
    """Maximum-likelihood (mu, sigma, xi) for block maxima x."""
    x = np.asarray(x, float)
    res = optimize.minimize(gev_nll, _init(x), args=(x,), method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    mu, sigma, xi = res.x
    return {"mu": float(mu), "sigma": float(sigma), "xi": float(xi), "nll": float(gev_nll(res.x, x))}


def ns_gev_nll(params, x, T):
    """Non-stationary GEV NLL: mu(t)=mu0+mu1 T, log sigma(t)=s0+s1 T, xi constant."""
    mu0, mu1, s0, s1, xi = params
    x, T = np.asarray(x, float), np.asarray(T, float)
    mu = mu0 + mu1 * T
    sigma = np.exp(s0 + s1 * T)
    z = (x - mu) / sigma
    if abs(xi) < _EPS:
        return float(np.sum(np.log(sigma) + z + np.exp(-z)))
    t = 1 + xi * z
    if np.any(t <= _EPS):
        return 1e12
    return float(np.sum(np.log(sigma) + (1 + 1 / xi) * np.log(t) + t ** (-1 / xi)))


def fit_ns_gev(x, T):
    """ML fit of the linear-covariate non-stationary GEV."""
    x, T = np.asarray(x, float), np.asarray(T, float)
    s = fit_stationary_gev(x)
    p0 = np.array([s["mu"], 0.0, np.log(s["sigma"]), 0.0, s["xi"]])
    res = optimize.minimize(ns_gev_nll, p0, args=(x, T), method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 20000})
    mu0, mu1, s0, s1, xi = res.x
    return {"mu0": float(mu0), "mu1": float(mu1), "s0": float(s0), "s1": float(s1), "xi": float(xi)}


def ns_params_at(fit, T):
    """Marginal (mu, sigma, xi) implied by a non-stationary fit at covariate value T."""
    return fit["mu0"] + fit["mu1"] * T, float(np.exp(fit["s0"] + fit["s1"] * T)), fit["xi"]


def hill(x, k):
    """Hill tail-index estimate from the top-k order statistics (positive data)."""
    xs = np.sort(np.asarray(x, float))[::-1]
    k = min(k, len(xs) - 1)
    return float(np.mean(np.log(xs[:k])) - np.log(xs[k]))


def gev_logprior(p, mu_ref, sig_ref):
    """Log prior density of (mu, sigma, xi), as a density in sigma.

    The priors are stated on mu, *log* sigma and xi, but the sampler walks on
    sigma, so this must be a density with respect to Lebesgue measure on sigma:
    p(sigma) = N(log sigma; log sigma_ref, 0.5^2) / sigma.  Dropping the
    1/sigma Jacobian shifts the effective prior on log sigma to
    N(log sigma_ref + 0.25, 0.5^2), which centres sigma 28% high.  Exposed as a
    function so the prior can be tested on its own rather than only through the
    chain, where a long record hides a prior this wrong.
    """
    mu, sigma, xi = p
    if sigma <= 0 or xi < -1 or xi > 2:
        return -np.inf
    return float(-0.5 * ((mu - mu_ref) / 5.0) ** 2
                 - 0.5 * ((np.log(sigma) - np.log(sig_ref)) / 0.5) ** 2
                 - np.log(sigma)
                 - 0.5 * (xi / 0.3) ** 2)


def gev_mcmc(x, n_iter=30000, burn=10000, thin=5, seed=0):
    """Random-walk Metropolis for the GEV with weakly-informative priors (Coles 1996).

    Priors: mu ~ N(mu_ref, 5^2), log sigma ~ N(log sigma_ref, 0.5^2),
    xi ~ N(0, 0.3^2) truncated to [-1, 2]; mu_ref, sigma_ref set from the data.
    Returns an (S, 3) array of posterior (mu, sigma, xi) samples.
    """
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    mu_ref, sig_ref = np.mean(x), np.std(x, ddof=1)

    def logpost(p):
        lp = gev_logprior(p, mu_ref, sig_ref)
        if not np.isfinite(lp):
            return -np.inf
        return lp - gev_nll(p, x)

    cur = _init(x)
    lc = logpost(cur)
    step = np.array([0.4, 0.25, 0.05])
    samples, acc = [], 0
    for it in range(n_iter):
        prop = cur + step * rng.standard_normal(3)
        lp = logpost(prop)
        if np.log(rng.uniform()) < lp - lc:
            cur, lc, acc = prop, lp, acc + 1
        if 0 < it < burn and it % 500 == 0:
            rate = acc / it
            step *= 1.15 if rate > 0.35 else 0.87 if rate < 0.2 else 1.0
        if it >= burn and (it - burn) % thin == 0:
            samples.append(cur.copy())
    return np.array(samples)


def return_level_ci_delta(x, N=100, alpha=0.10):
    """Delta-method z_N point estimate and (1-alpha) CI via a numerical Hessian."""
    x = np.asarray(x, float)
    fit = fit_stationary_gev(x)
    p = np.array([fit["mu"], fit["sigma"], fit["xi"]])
    zN = gev_return_level(N, *p)
    h = 1e-4 * (np.abs(p) + 1)
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            pp = p.copy(); pp[i] += h[i]; pp[j] += h[j]
            pm = p.copy(); pm[i] += h[i]; pm[j] -= h[j]
            mp = p.copy(); mp[i] -= h[i]; mp[j] += h[j]
            mm = p.copy(); mm[i] -= h[i]; mm[j] -= h[j]
            H[i, j] = (gev_nll(pp, x) - gev_nll(pm, x) - gev_nll(mp, x) + gev_nll(mm, x)) / (4 * h[i] * h[j])
    cov = np.linalg.pinv(H)
    g = np.zeros(3)
    for i in range(3):
        pp = p.copy(); pp[i] += h[i]
        pm = p.copy(); pm[i] -= h[i]
        g[i] = (gev_return_level(N, *pp) - gev_return_level(N, *pm)) / (2 * h[i])
    se = float(np.sqrt(max(g @ cov @ g, 0.0)))
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"zN": float(zN), "se": se, "lo": float(zN - zc * se), "hi": float(zN + zc * se)}


# ── GBC-QNN amortized estimators ───────────────────────────────────────────


def gbc_priors_from_data(x):
    """Empirical-Bayes reference values for the weakly-informative GBC priors."""
    x = np.asarray(x, float)
    return {"mu_prior": (float(np.mean(x)), 5.0),
            "log_sigma_prior": (float(np.log(np.std(x, ddof=1))), 0.5),
            "xi_prior": (0.0, 0.3), "xi_bounds": (-1.0, 2.0)}


def _sim(n_sim, n_obs, pr, seed):
    return simulate_gev_training_data(n_sim=n_sim, n_obs=n_obs, mu_prior=pr["mu_prior"],
                                      log_sigma_prior=pr["log_sigma_prior"],
                                      xi_prior=pr["xi_prior"], xi_bounds=pr["xi_bounds"], seed=seed)


def train_predictive_iqn(x, n_sim=40000, epochs=1500, seed=0, tail=True, tau0=0.9):
    """Train the predictive IQN (target = a fresh GEV draw) for CRPS/coverage.

    The tail head is on by default here and off in ``train_functional_iqn``, and
    the asymmetry is deliberate: this network models the data distribution, so
    tau indexes the tail of a new annual maximum and the far tail must
    extrapolate.  The functional network's target is the scalar z_N, so its tau
    indexes posterior uncertainty about a parameter, which is concentrated and
    needs no parametric tail.
    """
    d = _sim(n_sim, len(x), gbc_priors_from_data(x), seed)
    return train_iqn(d["X"], d["y"], epochs=epochs, seed=seed, tail=tail, tau0=tau0)


def train_functional_iqn(x, N=100, n_sim=40000, epochs=1500, seed=1):
    """Train an IQN whose target is z_N(theta); its quantiles are the z_N posterior."""
    d = _sim(n_sim, len(x), gbc_priors_from_data(x), seed)
    zN = np.array([gev_return_level(N, *p) for p in d["params"]])
    return train_iqn(d["X"], zN, epochs=epochs, seed=seed)


def _feat(x):
    return _block_maxima_summary(np.asarray(x, float)).reshape(1, -1)


def gbc_return_level_posterior(trained, x, alpha=0.10):
    """Posterior median + (1-alpha) CI for z_N from a functional IQN (train_functional_iqn)."""
    model, xm, xs, ym, ys = trained
    q = predict_iqn(model, _feat(x), xm, xs, ym, ys,
                    np.array([alpha / 2, 0.5, 1 - alpha / 2])).ravel()
    return {"zN": float(q[1]), "lo": float(q[0]), "hi": float(q[2])}


def gbc_predictive_samples(trained, x, B=2000):
    """Deterministic predictive quantile grid for a new block maximum."""
    model, xm, xs, ym, ys = trained
    return sample_iqn(model, _feat(x), xm, xs, ym, ys, B=B).ravel()


def gbc_crps_coverage_loyo(x, n_sim=20000, epochs=1500, seed=0, alpha=0.10, B=1000):
    """Leave-one-year-out grid CRPS and central grid-interval coverage."""
    x = np.asarray(x, float)
    model, xm, xs, ym, ys = train_predictive_iqn(x, n_sim=n_sim, epochs=epochs, seed=seed)
    crps_list, hits = [], 0
    lo_t, hi_t = alpha / 2, 1 - alpha / 2
    for i in range(len(x)):
        xho = np.delete(x, i)
        samp = sample_iqn(model, _feat(xho), xm, xs, ym, ys, B=B).ravel()
        crps_list.append(crps_samples(np.array([x[i]]), samp.reshape(-1, 1)))
        lo, hi = np.quantile(samp, [lo_t, hi_t])
        hits += int(lo <= x[i] <= hi)
    return {"crps": float(np.mean(crps_list)), "coverage": hits / len(x)}
