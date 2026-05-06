"""Portfolio dynamics: Kelly criterion, (s,S) corridors, and GBC synthesis.

Implements the core portfolio optimization tools from
Polson & Sokolov (2026), "Generative Bayesian Computation for
Optimal Portfolio Dynamics":

- Kelly fraction and (s,S) no-trade corridor computation
- Bayesian posterior over corridor boundaries
- Fernholz excess growth rate (Stochastic Portfolio Theory)
- Cover's universal portfolio algorithm
- Bayesian predictive portfolio selection (normal-inverse-Wishart)
- Garman-Klass and Rogers-Satchell volatility estimators

These are general-purpose functions for any portfolio problem.
For GBC-specific portfolio allocation via quantile functions,
see :mod:`gbc.welfare.portfolio_meu`.

Book references
---------------
- Ch 7 of Polson & Sokolov (2026): GBC for portfolio dynamics
- Kelly (1956): log-optimal criterion
- Taksar, Klass & Assaf (1988): singular stochastic control
- Fernholz (2002): stochastic portfolio theory
- Cover (1991): universal portfolios
- Jacquier & Polson (2012): Bayesian predictive selection
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


# ── Kelly Criterion and (s,S) Corridors ──────────────────────────


def kelly_fraction(mu: float, sigma: float, r: float = 0.0) -> float:
    r"""Kelly (log-optimal) fraction for a single risky asset.

    .. math::
        \pi^* = \frac{\mu - r}{\sigma^2}

    Parameters
    ----------
    mu : expected return (annualized).
    sigma : volatility (annualized), must be > 0.
    r : risk-free rate (annualized).

    Returns
    -------
    Optimal fraction of wealth in the risky asset.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (mu - r) / sigma**2


def sS_corridor(
    mu: float,
    sigma: float,
    r: float = 0.0,
    lam: float = 0.005,
    clip: bool = True,
) -> dict[str, float]:
    r"""No-trade corridor boundaries for proportional transaction costs.

    Uses the Davis-Norman asymptotic expansion for the corridor
    half-width around the Kelly fraction:

    .. math::
        \frac{S^* - s^*}{2} \approx
        \left(\frac{3\lambda}{2}\right)^{1/3}
        \frac{\pi^*(1 - \pi^*)}{\sigma}

    Parameters
    ----------
    mu : expected return (annualized).
    sigma : volatility (annualized), must be > 0.
    r : risk-free rate.
    lam : proportional transaction cost (symmetric).
    clip : if True, clip pi* to [0.01, 0.99] for long-only feasibility.

    Returns
    -------
    Dict with keys ``pi_star``, ``s``, ``S``, ``half_width``.
    """
    pi = kelly_fraction(mu, sigma, r)
    if clip:
        pi = np.clip(pi, 0.01, 0.99)
    hw = (3 * lam / 2) ** (1 / 3) * pi * (1 - pi) / sigma
    return {
        "pi_star": float(pi),
        "s": float(max(0, pi - hw)),
        "S": float(min(1, pi + hw)),
        "half_width": float(hw),
    }


def bayesian_sS_corridor(
    log_returns: ArrayLike,
    r: float = 0.0,
    lam: float = 0.005,
    n_draw: int = 5000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    r"""Posterior distribution over (s,S) corridor boundaries.

    Draws from the conjugate normal-gamma posterior for daily
    log-return parameters :math:`(\mu, \sigma^2)` and computes
    corridor boundaries for each draw.

    Parameters
    ----------
    log_returns : (T,) array of daily log-returns.
    r : annualized risk-free rate.
    lam : proportional transaction cost.
    n_draw : number of posterior draws.
    seed : random seed.

    Returns
    -------
    Dict with keys ``pi_star``, ``s``, ``S`` (each (n_draw,) arrays),
    ``mu_annual``, ``sigma_annual`` (posterior draws of annualized params),
    and ``plug_in`` (dict from sS_corridor at MLE estimates).
    """
    y = np.asarray(log_returns, dtype=float)
    n = len(y)
    ybar = y.mean()
    s2 = y.var(ddof=1)

    rng = np.random.default_rng(seed)

    # Conjugate normal-gamma posterior
    alpha_post = (n - 1) / 2
    beta_post = (n - 1) * s2 / 2
    sigma2_draws = 1.0 / rng.gamma(alpha_post, 1.0 / beta_post, size=n_draw)
    mu_draws = rng.normal(ybar, np.sqrt(sigma2_draws / n))

    mu_ann = mu_draws * 252
    sig_ann = np.sqrt(sigma2_draws * 252)

    # Compute corridor for each draw
    pi_arr = np.clip((mu_ann - r) / sig_ann**2, 0.01, 0.99)
    hw_arr = (3 * lam / 2) ** (1 / 3) * pi_arr * (1 - pi_arr) / sig_ann
    s_arr = np.maximum(0, pi_arr - hw_arr)
    S_arr = np.minimum(1, pi_arr + hw_arr)

    plug = sS_corridor(ybar * 252, np.sqrt(s2 * 252), r, lam)

    return {
        "pi_star": pi_arr,
        "s": s_arr,
        "S": S_arr,
        "mu_annual": mu_ann,
        "sigma_annual": sig_ann,
        "plug_in": plug,
    }


# ── Stochastic Portfolio Theory ──────────────────────────────────


def excess_growth_rate(
    weights: ArrayLike, Sigma: ArrayLike, annualize: int = 252
) -> float:
    r"""Fernholz excess growth rate for a portfolio.

    .. math::
        g(\pi) = \frac{1}{2}\left(
            \sum_i \pi_i \sigma_{ii} - \pi^\top \Sigma \pi
        \right)

    This measures the volatility-harvesting advantage of the
    portfolio from rebalancing. Higher values indicate greater
    diversification benefit.

    Parameters
    ----------
    weights : (n,) portfolio weight vector.
    Sigma : (n, n) covariance matrix of log-returns.
    annualize : trading days per year (set to 1 for no annualization).

    Returns
    -------
    Annualized excess growth rate (scalar).
    """
    w = np.asarray(weights, dtype=float)
    S = np.asarray(Sigma, dtype=float)
    sigma_port = float(w @ S @ w)
    egr = 0.5 * (float(w @ np.diag(S)) - sigma_port)
    return egr * annualize


def diversity_weights(market_weights: ArrayLike, p: float = 0.5) -> np.ndarray:
    r"""Diversity-weighted portfolio from Fernholz SPT.

    .. math::
        w_i = \frac{\mu_i^p}{\sum_j \mu_j^p}

    Parameters
    ----------
    market_weights : (n,) market capitalization weights, must be positive.
    p : diversity parameter in (0, 1). Smaller p → more diversified.

    Returns
    -------
    (n,) diversity-weighted portfolio.
    """
    mu = np.asarray(market_weights, dtype=float)
    w = mu**p
    return w / w.sum()


# ── Cover's Universal Portfolios ─────────────────────────────────


def universal_portfolio(
    returns: ArrayLike, n_grid: int = 200
) -> dict[str, np.ndarray | float]:
    r"""Cover (1991) universal portfolio for two assets.

    Implements the Bayesian mixture over constant rebalanced
    portfolios (CRPs) with a uniform prior on the simplex.

    Parameters
    ----------
    returns : (T, 2) matrix of gross return factors (not log-returns).
    n_grid : number of grid points for the CRP weight.

    Returns
    -------
    Dict with ``weights`` (T,), ``wealth`` (T,), ``best_b`` (float),
    ``best_crp_wealth`` (float).
    """
    ret = np.asarray(returns, dtype=float)
    if ret.ndim != 2 or ret.shape[1] != 2:
        raise ValueError("returns must be (T, 2) for two-asset universal portfolio")
    T = ret.shape[0]

    b_grid = np.linspace(0.01, 0.99, n_grid)
    crp_wealth = np.ones(n_grid)
    univ_weight = np.empty(T)
    univ_wealth = np.empty(T + 1)
    univ_wealth[0] = 1.0

    for t in range(T):
        r1, r2 = ret[t, 0], ret[t, 1]
        post = crp_wealth / crp_wealth.sum()
        univ_weight[t] = float(b_grid @ post)
        b = univ_weight[t]
        univ_wealth[t + 1] = univ_wealth[t] * (b * r1 + (1 - b) * r2)
        crp_wealth *= b_grid * r1 + (1 - b_grid) * r2

    best_idx = np.argmax(crp_wealth)
    return {
        "weights": univ_weight,
        "wealth": univ_wealth[1:],
        "best_b": float(b_grid[best_idx]),
        "best_crp_wealth": float(crp_wealth[best_idx]),
    }


# ── Bayesian Portfolio Selection ─────────────────────────────────


def bayesian_predictive_weights(
    R: ArrayLike,
    gamma: float = 3.0,
    kappa0: float = 1.0,
    nu0: int | None = None,
    Psi0: ArrayLike | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    r"""Bayesian predictive portfolio weights (normal-inverse-Wishart).

    Computes optimal portfolio weights by maximizing expected utility
    under the posterior predictive distribution, which is multivariate-t:

    .. math::
        R_{T+1} \mid \mathcal{D}_T \sim
        t_{\nu_T - n + 1}\!\left(\bar\mu_T,\,
        \frac{\kappa_T + 1}{\kappa_T(\nu_T - n + 1)}\Psi_T\right)

    Parameters
    ----------
    R : (T, n) matrix of historical returns.
    gamma : risk aversion coefficient (gamma=1 is log utility).
    kappa0 : prior precision for the mean.
    nu0 : prior degrees of freedom (default: n + 2).
    Psi0 : prior scale matrix (default: scaled identity).
    seed : random seed.

    Returns
    -------
    Dict with ``weights`` (n,), ``mu_posterior`` (n,),
    ``Sigma_predictive`` (n, n), ``df_predictive`` (int).
    """
    R = np.asarray(R, dtype=float)
    T_obs, n = R.shape
    R_bar = R.mean(axis=0)
    S_mat = (T_obs - 1) * np.cov(R, rowvar=False)

    mu0 = R_bar.copy()
    if nu0 is None:
        nu0 = n + 2
    if Psi0 is None:
        Psi0 = np.eye(n) * R.var(axis=0).mean() * (nu0 - n - 1)
    else:
        Psi0 = np.asarray(Psi0, dtype=float)

    kappa_T = kappa0 + T_obs
    nu_T = nu0 + T_obs
    mu_T = (kappa0 * mu0 + T_obs * R_bar) / kappa_T
    Psi_T = Psi0 + S_mat + (kappa0 * T_obs / kappa_T) * np.outer(
        R_bar - mu0, R_bar - mu0
    )

    df_pred = nu_T - n + 1
    Sigma_pred = (kappa_T + 1) / (kappa_T * df_pred) * Psi_T

    # Analytical mean-variance Bayesian weights
    try:
        w_raw = np.linalg.solve(Sigma_pred, mu_T) / gamma
    except np.linalg.LinAlgError:
        w_raw = np.ones(n) / n

    w_pos = np.maximum(w_raw, 0)
    w_opt = w_pos / w_pos.sum() if w_pos.sum() > 0 else np.ones(n) / n

    return {
        "weights": w_opt,
        "mu_posterior": mu_T,
        "Sigma_predictive": Sigma_pred,
        "df_predictive": int(df_pred),
    }


# ── Volatility Estimators ────────────────────────────────────────


def garman_klass_vol(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    trading_days: int = 252,
) -> float:
    r"""Garman-Klass volatility estimator from OHLC data.

    Achieves approximately 8x efficiency over the close-to-close
    estimator by exploiting intraday range information.

    .. math::
        \hat\sigma^2_{\text{GK}} = \frac{1}{n}\sum_i
        \left[\tfrac{1}{2}(H_i - L_i)^2
              - (2\ln 2 - 1) C_i^2\right]

    where :math:`H = \ln(\text{high}/\text{open})`,
    :math:`L = \ln(\text{low}/\text{open})`,
    :math:`C = \ln(\text{close}/\text{open})`.

    Parameters
    ----------
    open, high, low, close : (T,) OHLC price arrays.
    trading_days : days per year for annualization.

    Returns
    -------
    Annualized volatility estimate.

    References
    ----------
    Garman, M.B. and Klass, M.J. (1980). "On the estimation of
    security price volatilities from historical data." *Journal
    of Business*, 53, 67-78.
    """
    o, h, l, c = (np.asarray(x, dtype=float) for x in (open, high, low, close))
    H = np.log(h / o)
    L = np.log(l / o)
    C = np.log(c / o)
    gk_daily = 0.5 * (H - L) ** 2 - (2 * np.log(2) - 1) * C**2
    return float(np.sqrt(max(gk_daily.mean() * trading_days, 0)))


def rogers_satchell_vol(
    open: ArrayLike,
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    trading_days: int = 252,
) -> float:
    r"""Rogers-Satchell volatility estimator (drift-corrected).

    Unlike Garman-Klass, this estimator is unbiased for non-zero drift.

    .. math::
        \hat\sigma^2_{\text{RS}} = \frac{1}{n}\sum_i
        \left[\ln\frac{h_i}{c_i}\ln\frac{h_i}{o_i}
              + \ln\frac{l_i}{c_i}\ln\frac{l_i}{o_i}\right]

    Parameters
    ----------
    open, high, low, close : (T,) OHLC price arrays.
    trading_days : days per year for annualization.

    Returns
    -------
    Annualized volatility estimate.

    References
    ----------
    Rogers, L.C.G. and Satchell, S.E. (1991). "Estimating variance
    from high, low and closing prices." *Annals of Applied Probability*,
    1, 504-512.
    """
    o, h, l, c = (np.asarray(x, dtype=float) for x in (open, high, low, close))
    rs_daily = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return float(np.sqrt(max(rs_daily.mean() * trading_days, 0)))


# ── Simulation Utilities ─────────────────────────────────────────


def simulate_gbm(
    n_days: int,
    mu: float = 0.08,
    sigma: float = 0.30,
    S0: float = 100.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate geometric Brownian motion with OHLC data.

    Parameters
    ----------
    n_days : number of trading days.
    mu : annualized drift.
    sigma : annualized volatility.
    S0 : initial price.
    seed : random seed.

    Returns
    -------
    Dict with ``open``, ``high``, ``low``, ``close`` (each (n_days,)),
    ``log_returns`` (n_days-1,), and ``prices`` (n_days,).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252
    n_intra = 100

    open_p = np.empty(n_days)
    high_p = np.empty(n_days)
    low_p = np.empty(n_days)
    close_p = np.empty(n_days)

    price = S0
    for i in range(n_days):
        open_p[i] = price
        steps = rng.normal(mu * dt / n_intra, sigma * np.sqrt(dt / n_intra), n_intra)
        path = np.cumsum(np.concatenate([[0], steps[:-1]]))
        high_p[i] = open_p[i] * np.exp(path.max())
        low_p[i] = open_p[i] * np.exp(path.min())
        close_p[i] = open_p[i] * np.exp(path[-1])
        price = close_p[i]

    log_rets = np.diff(np.log(close_p))

    return {
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "log_returns": log_rets,
        "prices": close_p,
    }


def simulate_sS_policy(
    n_steps: int = 2520,
    mu: float = 0.08,
    sigma: float = 0.30,
    r: float = 0.0,
    lam: float = 0.005,
    seed: int = 42,
) -> dict[str, np.ndarray | dict]:
    """Simulate portfolio dynamics under (s,S) corridor policy.

    Parameters
    ----------
    n_steps : number of daily time steps.
    mu : annualized drift.
    sigma : annualized volatility.
    r : annualized risk-free rate.
    lam : proportional transaction cost.
    seed : random seed.

    Returns
    -------
    Dict with ``wealth`` (n_steps+1,), ``rho`` (n_steps+1,),
    ``n_trades`` (int), ``corridor`` (dict from sS_corridor).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252
    corr = sS_corridor(mu, sigma, r, lam)
    s, S = corr["s"], corr["S"]

    W = np.empty(n_steps + 1)
    rho_path = np.empty(n_steps + 1)
    W[0] = 1.0
    rho_path[0] = corr["pi_star"]
    rho = corr["pi_star"]
    w = 1.0
    n_trades = 0

    for i in range(n_steps):
        dZ = rng.standard_normal()
        R_s = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dZ)
        R_b = np.exp(r * dt)

        S_pos = rho * w * R_s
        B_pos = (1 - rho) * w * R_b
        w_new = S_pos + B_pos
        rho_new = S_pos / w_new

        if rho_new < s:
            w_new -= lam * (s - rho_new) * w_new
            rho_new = s
            n_trades += 1
        elif rho_new > S:
            w_new -= lam * (rho_new - S) * w_new
            rho_new = S
            n_trades += 1

        w = w_new
        rho = rho_new
        W[i + 1] = w
        rho_path[i + 1] = rho

    return {
        "wealth": W,
        "rho": rho_path,
        "n_trades": n_trades,
        "corridor": corr,
    }
