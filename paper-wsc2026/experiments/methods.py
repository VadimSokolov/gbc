"""Estimators for the GBC-EVT paper: classical baselines + Bayesian MCMC.

All return-level math uses the Coles (2001) GEV parameterisation.
The GBC-QNN estimators live in gbc_qnn.py; this file is the classical comparison set.
"""

import os
import sys

import numpy as np
from scipy import optimize, stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _cand in (os.path.join(_ROOT, "gbc"), _ROOT, os.path.dirname(_ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break
from gbc.evt import gev_return_level  # noqa: E402

_EPS = 1e-8


def gev_nll(params, x):
    """Negative log-likelihood of a stationary GEV (mu, sigma, xi)."""
    mu, sigma, xi = params
    if sigma <= 0:
        return 1e12
    z = (x - mu) / sigma
    if abs(xi) < _EPS:
        return np.sum(np.log(sigma) + z + np.exp(-z))
    t = 1 + xi * z
    if np.any(t <= _EPS):
        return 1e12
    return np.sum(np.log(sigma) + (1 + 1 / xi) * np.log(t) + t ** (-1 / xi))


def _init(x):
    sd = np.std(x, ddof=1)
    return np.array([np.mean(x) - 0.45 * sd, max(sd * np.sqrt(6) / np.pi, 0.1), 0.1])


def fit_stationary_gev(x):
    x = np.asarray(x, float)
    res = optimize.minimize(gev_nll, _init(x), args=(x,), method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    mu, sigma, xi = res.x
    return {"mu": mu, "sigma": sigma, "xi": xi, "nll": float(gev_nll(res.x, x))}


def ns_gev_nll(params, x, T):
    """Non-stationary GEV: mu(t)=mu0+mu1 T, log sigma(t)=s0+s1 T, xi constant."""
    mu0, mu1, s0, s1, xi = params
    mu = mu0 + mu1 * T
    sigma = np.exp(s0 + s1 * T)
    z = (x - mu) / sigma
    if abs(xi) < _EPS:
        return np.sum(np.log(sigma) + z + np.exp(-z))
    t = 1 + xi * z
    if np.any(t <= _EPS):
        return 1e12
    return np.sum(np.log(sigma) + (1 + 1 / xi) * np.log(t) + t ** (-1 / xi))


def fit_ns_gev(x, T):
    x, T = np.asarray(x, float), np.asarray(T, float)
    s = fit_stationary_gev(x)
    p0 = np.array([s["mu"], 0.0, np.log(s["sigma"]), 0.0, s["xi"]])
    res = optimize.minimize(ns_gev_nll, p0, args=(x, T), method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 20000})
    mu0, mu1, s0, s1, xi = res.x
    return {"mu0": mu0, "mu1": mu1, "s0": s0, "s1": s1, "xi": xi}


def ns_params_at(fit, T):
    """Marginal GEV (mu,sigma,xi) implied by a NS fit at covariate value T."""
    return fit["mu0"] + fit["mu1"] * T, np.exp(fit["s0"] + fit["s1"] * T), fit["xi"]


def hill(x, k):
    """Hill tail-index estimate from the top-k order statistics (positive data)."""
    xs = np.sort(np.asarray(x, float))[::-1]
    k = min(k, len(xs) - 1)
    return float(np.mean(np.log(xs[:k])) - np.log(xs[k]))


def gev_mcmc(x, n_iter=30000, burn=10000, thin=5, seed=0):
    """Random-walk Metropolis for the GEV with weakly-informative priors (Coles 1996).

    Priors: mu ~ N(mu_ref, 5^2), log sigma ~ N(log sigma_ref, 0.5^2),
            xi ~ N(0, 0.3^2) truncated to [-1, 2].   mu_ref, sigma_ref from data.
    Returns an (S, 3) array of (mu, sigma, xi) posterior samples.
    """
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    mu_ref, sig_ref = np.mean(x), np.std(x, ddof=1)

    def logpost(p):
        mu, sigma, xi = p
        if sigma <= 0 or xi < -1 or xi > 2:
            return -np.inf
        lp = (-0.5 * ((mu - mu_ref) / 5.0) ** 2
              - 0.5 * ((np.log(sigma) - np.log(sig_ref)) / 0.5) ** 2
              - 0.5 * (xi / 0.3) ** 2)
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
        if it < burn and it > 0 and it % 500 == 0:  # adapt during burn-in
            rate = acc / it
            step *= 1.15 if rate > 0.35 else 0.87 if rate < 0.2 else 1.0
        if it >= burn and (it - burn) % thin == 0:
            samples.append(cur.copy())
    return np.array(samples)


def return_level_ci_delta(x, N=100, alpha=0.10):
    """Profile/delta-method (mu,sigma,xi) -> z_N point and (1-alpha) CI via numerical Hessian."""
    x = np.asarray(x, float)
    fit = fit_stationary_gev(x)
    p = np.array([fit["mu"], fit["sigma"], fit["xi"]])
    zN = gev_return_level(N, *p)
    h = 1e-4 * (np.abs(p) + 1)
    H = np.zeros((3, 3))
    f0 = gev_nll(p, x)
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


if __name__ == "__main__":  # smoke test on real Seattle data
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "data", "pnw_jja_maxima.csv"), index_col="year")
    x = df["SEA"].dropna()
    print(f"SEA n={len(x)} years, max={x.max():.1f}C")
    s = fit_stationary_gev(x.values)
    print(f"stationary GEV: mu={s['mu']:.2f} sigma={s['sigma']:.2f} xi={s['xi']:.3f}")
    ci = return_level_ci_delta(x.values, 100)
    print(f"z100 = {ci['zN']:.1f}C  90% CI [{ci['lo']:.1f}, {ci['hi']:.1f}]  width={ci['hi']-ci['lo']:.1f}")
    print(f"Hill (k=20) gamma = {hill(x.values, 20):.3f}")
    sm = gev_mcmc(x.values, seed=1)
    z = np.array([gev_return_level(100, *p) for p in sm])
    print(f"MCMC: {len(sm)} samples, xi post mean={sm[:,2].mean():.3f}, "
          f"z100 post median={np.median(z):.1f} 90% CI [{np.quantile(z,.05):.1f}, {np.quantile(z,.95):.1f}]")
