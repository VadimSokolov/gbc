"""Approximate Bayesian Computation with MCMC.

Provides a general-purpose ABC-MCMC sampler for simulation-based
Bayesian inference when the likelihood is intractable but forward
simulation from the generative model is cheap.

The ABC-MCMC algorithm (Marjoram et al., 2003) replaces likelihood
evaluation with a distance check between simulated and observed summary
statistics:

    1. Propose θ* from a proposal kernel q(·|θ_t).
    2. Simulate y* ~ p(y|θ*) and compute summary S(y*).
    3. Accept θ* if d(S(y*), S(y_obs)) < ε and a Metropolis ratio check
       passes; otherwise keep θ_t.

As ε → 0, the ABC posterior converges to the exact posterior (given
sufficient summary statistics).

References
----------
Marjoram, P. et al. (2003). MCMC without likelihoods. *PNAS*, 100(26).
Beaumont, M. A. et al. (2002). Approximate Bayesian computation in
  population genetics. *Genetics*, 162(4).
Sisson, S. A. et al. (2018). *Handbook of Approximate Bayesian
  Computation*. Chapman & Hall/CRC.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Protocol


# ── Type protocols ─────────────────────────────────────────────


class Simulator(Protocol):
    """Callable that generates synthetic data from parameters."""
    def __call__(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        ...


class SummaryStatistic(Protocol):
    """Callable that reduces data to a low-dimensional summary."""
    def __call__(self, data: np.ndarray) -> np.ndarray:
        ...


class Prior(Protocol):
    """Object with ``logpdf`` and ``rvs`` methods (scipy-style)."""
    def logpdf(self, theta: np.ndarray) -> float: ...
    def rvs(self, random_state: np.random.Generator) -> np.ndarray: ...


# ── Result container ───────────────────────────────────────────


@dataclass
class ABCResult:
    """Container for ABC-MCMC output.

    Attributes
    ----------
    chain : (n_iter,) or (n_iter, d) accepted parameter values.
    acceptance_rate : fraction of proposals accepted.
    distances : (n_iter,) distance at each iteration (0 if rejected).
    """
    chain: np.ndarray
    acceptance_rate: float
    distances: np.ndarray

    def posterior(self, burn_in: float = 0.3) -> np.ndarray:
        """Return post-burn-in samples.

        Parameters
        ----------
        burn_in : fraction of chain to discard.  Default 0.3.
        """
        start = int(len(self.chain) * burn_in)
        return self.chain[start:]

    def posterior_mean(self, burn_in: float = 0.3) -> np.ndarray:
        """Posterior mean after burn-in."""
        return np.mean(self.posterior(burn_in), axis=0)

    def credible_interval(
        self, alpha: float = 0.05, burn_in: float = 0.3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Equal-tailed credible interval.

        Parameters
        ----------
        alpha : significance level. Default 0.05 → 95% CI.
        burn_in : fraction of chain to discard.
        """
        samples = self.posterior(burn_in)
        lo = np.percentile(samples, 100 * alpha / 2, axis=0)
        hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
        return lo, hi

    def geweke(self, burn_in: float = 0.3, frac: float = 0.1) -> float:
        """Geweke convergence diagnostic (z-score).

        Compares the mean of the first ``frac`` of post-burn-in samples
        to the mean of the last ``frac``.  Under convergence, the
        z-score should be in [-2, 2].

        Parameters
        ----------
        burn_in : fraction of chain to discard.
        frac : fraction of remaining chain for each window.
        """
        samples = self.posterior(burn_in)
        n = len(samples)
        n_window = max(int(n * frac), 2)
        first = samples[:n_window]
        last = samples[-n_window:]
        se = np.sqrt(np.var(first) / n_window + np.var(last) / n_window)
        if se < 1e-15:
            return 0.0
        return float((np.mean(first) - np.mean(last)) / se)


# ── ABC-MCMC sampler ──────────────────────────────────────────


def abc_mcmc(
    simulator: Simulator,
    summary_obs: np.ndarray,
    prior: Prior,
    n_iter: int = 5000,
    epsilon: float = 0.05,
    proposal_sd: float | np.ndarray = 0.05,
    theta_init: np.ndarray | None = None,
    summary_fn: SummaryStatistic | None = None,
    distance_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> ABCResult:
    r"""Run ABC-MCMC (Marjoram et al. 2003).

    Parameters
    ----------
    simulator : callable(theta, rng) → data.
        Generates synthetic data given parameters and a RNG.
    summary_obs : (d_s,) observed summary statistic.
        If ``summary_fn`` is None, the simulator is assumed to return
        summaries directly.
    prior : object with ``.logpdf(theta)`` and ``.rvs(random_state=)``.
        Scipy distribution objects work out of the box.
    n_iter : number of MCMC iterations (including burn-in).
    epsilon : ABC tolerance.  Proposals are accepted only if
        distance(S_sim, S_obs) < epsilon.
    proposal_sd : standard deviation of Gaussian random-walk proposal.
        Scalar or array of same shape as theta.
    theta_init : starting value.  If None, drawn from the prior.
    summary_fn : callable(data) → summary.  Applied to simulator output.
        If None, the simulator must return the summary directly.
    distance_fn : callable(s1, s2) → scalar distance.
        Default: Euclidean (L2) norm.
    bounds : (lower, upper) arrays.  Proposals outside bounds are
        reflected back.  None means unbounded.
    seed : random seed.
    verbose : if True, print progress every 500 iterations.

    Returns
    -------
    ABCResult with chain, acceptance_rate, distances.

    Examples
    --------
    >>> from scipy.stats import beta
    >>> prior = beta(6.5, 3.5)
    >>> result = abc_mcmc(my_simulator, s_obs, prior, n_iter=5000)
    >>> print(result.posterior_mean())
    """
    rng = np.random.default_rng(seed)

    if distance_fn is None:
        distance_fn = lambda s1, s2: float(np.linalg.norm(s1 - s2))

    # Initialize
    if theta_init is None:
        theta = np.atleast_1d(prior.rvs(random_state=rng))
    else:
        theta = np.atleast_1d(np.asarray(theta_init, dtype=float))

    proposal_sd = np.atleast_1d(np.asarray(proposal_sd, dtype=float))
    if proposal_sd.shape != theta.shape:
        proposal_sd = np.broadcast_to(proposal_sd, theta.shape).copy()

    chain = np.empty((n_iter, theta.size))
    distances = np.zeros(n_iter)
    n_accept = 0

    for t in range(n_iter):
        # Propose
        theta_star = theta + rng.normal(0, proposal_sd, size=theta.shape)

        # Reflect off bounds
        if bounds is not None:
            lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
            theta_star = _reflect(theta_star, lo, hi)

        # Simulate and summarize
        data_star = simulator(theta_star, rng)
        s_star = summary_fn(data_star) if summary_fn is not None else data_star
        s_star = np.atleast_1d(s_star)

        dist = distance_fn(s_star, summary_obs)

        # Accept/reject
        if dist < epsilon:
            log_prior_ratio = prior.logpdf(theta_star) - prior.logpdf(theta)
            if np.log(rng.uniform()) < log_prior_ratio:
                theta = theta_star
                n_accept += 1

        chain[t] = theta
        distances[t] = dist

        if verbose and (t + 1) % 500 == 0:
            rate = n_accept / (t + 1)
            print(f"  iter {t+1}/{n_iter}  accept={rate:.3f}  dist={dist:.4f}")

    chain = chain.squeeze()  # (n_iter,) if scalar
    return ABCResult(
        chain=chain,
        acceptance_rate=n_accept / n_iter,
        distances=distances,
    )


def _reflect(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Reflect values into [lo, hi]."""
    for i in range(len(x)):
        while x[i] < lo[i] or x[i] > hi[i]:
            if x[i] < lo[i]:
                x[i] = 2 * lo[i] - x[i]
            if x[i] > hi[i]:
                x[i] = 2 * hi[i] - x[i]
    return x


# ── Utility: summary statistics ────────────────────────────────


def summary_rates_by_decile(
    outcomes: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Compute outcome rates by decile of a grouping variable.

    Useful as an ABC summary statistic: bin observations into deciles
    of ``groups`` and compute the mean of ``outcomes`` in each bin.

    Parameters
    ----------
    outcomes : (n,) binary or continuous outcomes.
    groups : (n,) continuous variable used to form decile bins.

    Returns
    -------
    (10,) array of mean outcomes per decile.
    """
    deciles = np.digitize(
        groups, np.percentile(groups, np.arange(10, 100, 10))
    )
    rates = np.array([
        np.mean(outcomes[deciles == d]) if np.any(deciles == d) else 0.0
        for d in range(10)
    ])
    return rates
