"""State-space model particle filtering via Generative Bayesian Computation.

Bootstrap particle filter (BPF), likelihood-tempered GBC, augmented GBC for
scale-mixture models, and backward smoothing.  All computations use log-space
weights for numerical stability.

Book references
---------------
- Ch 11 §sec-ssm-pf       : bootstrap particle filter as GBC
- Ch 11 §sec-ssm-lt       : likelihood-tempered GBC filter
- Ch 11 §sec-ssm-augment  : augmented GBC for scale-mixture observations
- Ch 11 §sec-ssm-smoother : particle GBC backward smoother (PGBS)

Notes
-----
All weight arithmetic is done in log-space with ``logsumexp`` normalisation.
The module depends only on numpy/scipy — no torch dependency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_DEFAULT_QUANTILES = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def compute_ess(log_weights: np.ndarray) -> float:
    """Effective sample size from normalised log-weights.

    Parameters
    ----------
    log_weights : (N,) log-normalised importance weights.

    Returns
    -------
    Scalar ESS in [1, N].
    """
    log_w = log_weights - logsumexp(log_weights)
    return float(np.exp(-logsumexp(2.0 * log_w)))


def weighted_quantiles(
    values: np.ndarray,
    log_weights: np.ndarray,
    quantiles: np.ndarray = _DEFAULT_QUANTILES,
) -> np.ndarray:
    """Weighted quantile computation via sorted cumulative weights.

    Parameters
    ----------
    values : (N,) particle values (scalar state).
    log_weights : (N,) log-normalised weights.
    quantiles : (Q,) quantile levels in (0, 1).

    Returns
    -------
    (Q,) array of weighted quantiles.
    """
    weights = np.exp(log_weights - logsumexp(log_weights))
    idx = np.argsort(values)
    sorted_vals = values[idx]
    cum_w = np.cumsum(weights[idx])
    # Interpolate: find first cumulative weight >= each quantile level
    out = np.interp(quantiles, cum_w, sorted_vals)
    return out


def multinomial_resample(
    particles: np.ndarray,
    log_weights: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial resampling.

    Parameters
    ----------
    particles : (N,) or (N, d) particle array.
    log_weights : (N,) log-normalised weights.
    rng : numpy random generator (default: module-level default).

    Returns
    -------
    (resampled_particles, uniform_log_weights) tuple.
    """
    if rng is None:
        rng = np.random.default_rng()
    weights = np.exp(log_weights - logsumexp(log_weights))
    N = particles.shape[0]
    indices = rng.choice(N, size=N, replace=True, p=weights)
    new_particles = particles[indices].copy()
    new_log_weights = np.full(N, -np.log(N))
    return new_particles, new_log_weights


# ---------------------------------------------------------------------------
# Bootstrap Particle Filter (BPF as GBC)
# ---------------------------------------------------------------------------

class ParticleFilter:
    """Bootstrap particle filter (Ch 11 §sec-ssm-pf).

    The simplest GBC filter: propagate through the transition, then
    re-weight by the observation likelihood.

    Parameters
    ----------
    transition_fn : callable(particles, t) -> new_particles
        State transition: draw x_t ~ p(x_t | x_{t-1}).
    log_likelihood_fn : callable(y_t, particles) -> (N,) log-likelihoods
        Observation model: log p(y_t | x_t).
    n_particles : int
        Number of particles N.
    resample_threshold : float
        Resample when ESS / N drops below this threshold.
    seed : int
        Random seed.

    Examples
    --------
    >>> pf = ParticleFilter(transition, log_lik, n_particles=1000)
    >>> result = pf.filter(y)
    >>> result["mean"]     # (T,) posterior mean at each step
    >>> result["quantiles"]  # (T, 5) quantile summaries
    """

    def __init__(
        self,
        transition_fn: Callable,
        log_likelihood_fn: Callable,
        n_particles: int = 1000,
        resample_threshold: float = 0.5,
        seed: int = 42,
    ):
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.N = n_particles
        self.resample_threshold = resample_threshold
        self.rng = np.random.default_rng(seed)

        # State
        self.particles: np.ndarray | None = None
        self.log_weights: np.ndarray | None = None
        self._t = 0

    def _init_particles(self, particles: np.ndarray):
        """Set initial particle cloud (called internally or by user)."""
        self.particles = particles.copy()
        self.log_weights = np.full(self.N, -np.log(self.N))
        self._t = 0

    def step(self, y_t: float | np.ndarray) -> dict:
        """One filtering step: propagate, weight, (optionally) resample.

        Parameters
        ----------
        y_t : scalar or array observation at time t.

        Returns
        -------
        dict with keys ``particles``, ``log_weights``, ``ess``,
        ``mean``, ``quantiles``.
        """
        if self.particles is None:
            raise RuntimeError("Particles not initialised. Call filter() "
                               "or _init_particles() first.")

        # --- propagate ---
        self.particles = self.transition_fn(self.particles, self._t)

        # --- re-weight ---
        log_lik = self.log_likelihood_fn(y_t, self.particles)
        self.log_weights = self.log_weights + log_lik
        self.log_weights -= logsumexp(self.log_weights)

        ess = compute_ess(self.log_weights)

        # --- summaries ---
        weights = np.exp(self.log_weights)
        mean = float(np.dot(weights, self.particles))
        quantiles = weighted_quantiles(
            self.particles, self.log_weights, _DEFAULT_QUANTILES
        )

        # --- resample ---
        if ess / self.N < self.resample_threshold:
            self.particles, self.log_weights = multinomial_resample(
                self.particles, self.log_weights, self.rng
            )

        self._t += 1
        return {
            "particles": self.particles.copy(),
            "log_weights": self.log_weights.copy(),
            "ess": ess,
            "mean": mean,
            "quantiles": quantiles,
        }

    def filter(
        self,
        y: np.ndarray,
        x0: np.ndarray | None = None,
    ) -> dict:
        """Run the full filter over an observation sequence.

        Parameters
        ----------
        y : (T,) or (T, dy) observation sequence.
        x0 : (N,) or (N, d) initial particles.  If *None*, the first call
            to ``transition_fn(zeros, 0)`` serves as the prior draw.

        Returns
        -------
        dict with arrays ``particles`` (T, N), ``log_weights`` (T, N),
        ``ess`` (T,), ``mean`` (T,), ``quantiles`` (T, 5).
        """
        T = y.shape[0]
        if x0 is None:
            x0 = np.zeros(self.N)
        self._init_particles(x0)

        out = {
            "particles": np.empty((T, self.N)),
            "log_weights": np.empty((T, self.N)),
            "ess": np.empty(T),
            "mean": np.empty(T),
            "quantiles": np.empty((T, len(_DEFAULT_QUANTILES))),
        }

        for t in range(T):
            res = self.step(y[t])
            out["particles"][t] = res["particles"]
            out["log_weights"][t] = res["log_weights"]
            out["ess"][t] = res["ess"]
            out["mean"][t] = res["mean"]
            out["quantiles"][t] = res["quantiles"]

        return out


# ---------------------------------------------------------------------------
# Likelihood-Tempered GBC Filter
# ---------------------------------------------------------------------------

class LTGBCFilter:
    """Likelihood-Tempered GBC filter (Ch 11 §sec-ssm-lt).

    Uses a tempered proposal that partially accounts for the current
    observation, yielding better-placed particles than the bootstrap filter.

    The importance weight correction is:

        w_t = p(y_t | x_t) * p(x_t | x_{t-1}) / q_alpha(x_t | x_{t-1}, y_t)

    Parameters
    ----------
    transition_fn : callable(particles, t) -> new_particles
        Prior state transition.
    log_transition_density_fn : callable(new_particles, old_particles, t) -> (N,)
        Log-density of the transition evaluated at proposed particles.
    log_likelihood_fn : callable(y_t, particles) -> (N,)
        Observation log-likelihood.
    proposal_fn : callable(particles, y_t, alpha) -> (new_particles, log_q)
        Tempered proposal: returns new particles and their log-density.
    n_particles : int
        Number of particles.
    alpha : float
        Tempering parameter in (0, 1].
    resample_threshold : float
        ESS / N threshold for resampling.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        transition_fn: Callable,
        log_transition_density_fn: Callable,
        log_likelihood_fn: Callable,
        proposal_fn: Callable,
        n_particles: int = 1000,
        alpha: float = 0.5,
        resample_threshold: float = 0.5,
        seed: int = 42,
    ):
        self.transition_fn = transition_fn
        self.log_transition_density_fn = log_transition_density_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.proposal_fn = proposal_fn
        self.N = n_particles
        self.alpha = alpha
        self.resample_threshold = resample_threshold
        self.rng = np.random.default_rng(seed)

        self.particles: np.ndarray | None = None
        self.log_weights: np.ndarray | None = None
        self._t = 0

    def _init_particles(self, particles: np.ndarray):
        self.particles = particles.copy()
        self.log_weights = np.full(self.N, -np.log(self.N))
        self._t = 0

    def step(self, y_t: float | np.ndarray) -> dict:
        """One filtering step with tempered proposal.

        Parameters
        ----------
        y_t : observation at time t.

        Returns
        -------
        dict with ``particles``, ``log_weights``, ``ess``, ``mean``,
        ``quantiles``.
        """
        if self.particles is None:
            raise RuntimeError("Particles not initialised.")

        old_particles = self.particles.copy()

        # --- propose from tempered proposal ---
        new_particles, log_q = self.proposal_fn(
            old_particles, y_t, self.alpha
        )

        # --- importance weight correction ---
        log_lik = self.log_likelihood_fn(y_t, new_particles)
        log_trans = self.log_transition_density_fn(
            new_particles, old_particles, self._t
        )
        log_inc = log_lik + log_trans - log_q
        self.log_weights = self.log_weights + log_inc
        self.log_weights -= logsumexp(self.log_weights)

        self.particles = new_particles
        ess = compute_ess(self.log_weights)

        weights = np.exp(self.log_weights)
        mean = float(np.dot(weights, self.particles))
        quantiles = weighted_quantiles(
            self.particles, self.log_weights, _DEFAULT_QUANTILES
        )

        if ess / self.N < self.resample_threshold:
            self.particles, self.log_weights = multinomial_resample(
                self.particles, self.log_weights, self.rng
            )

        self._t += 1
        return {
            "particles": self.particles.copy(),
            "log_weights": self.log_weights.copy(),
            "ess": ess,
            "mean": mean,
            "quantiles": quantiles,
        }

    def filter(
        self,
        y: np.ndarray,
        x0: np.ndarray | None = None,
    ) -> dict:
        """Run the full filter over an observation sequence.

        Parameters
        ----------
        y : (T,) or (T, dy) observation sequence.
        x0 : (N,) initial particles.

        Returns
        -------
        dict with arrays keyed by ``particles``, ``log_weights``, ``ess``,
        ``mean``, ``quantiles``.
        """
        T = y.shape[0]
        if x0 is None:
            x0 = np.zeros(self.N)
        self._init_particles(x0)

        out = {
            "particles": np.empty((T, self.N)),
            "log_weights": np.empty((T, self.N)),
            "ess": np.empty(T),
            "mean": np.empty(T),
            "quantiles": np.empty((T, len(_DEFAULT_QUANTILES))),
        }

        for t in range(T):
            res = self.step(y[t])
            out["particles"][t] = res["particles"]
            out["log_weights"][t] = res["log_weights"]
            out["ess"][t] = res["ess"]
            out["mean"][t] = res["mean"]
            out["quantiles"][t] = res["quantiles"]

        return out


# ---------------------------------------------------------------------------
# Augmented GBC Filter (scale-mixture observations)
# ---------------------------------------------------------------------------

class AugmentedGBCFilter:
    """Augmented GBC for scale-mixture observation models (Ch 11 §sec-ssm-augment).

    Introduces auxiliary variables (e.g. inverse-chi-squared scales) that
    allow a conditionally Gaussian proposal even when the marginal
    observation model is heavy-tailed (Student-t, Laplace, etc.).

    Parameters
    ----------
    transition_fn : callable(particles, t) -> new_particles
        State transition.
    log_likelihood_fn : callable(y_t, particles) -> (N,)
        Marginal observation log-likelihood (used for weight correction).
    augment_fn : callable(N, rng) -> auxiliary_variables
        Draw auxiliary variables from their prior or conditional.
    conditional_proposal_fn : callable(particles, y_t, aux) ->
        (new_particles, log_proposal_density)
        Conditionally Gaussian (or otherwise tractable) proposal given
        the auxiliary variables.
    n_particles : int
        Number of particles.
    resample_threshold : float
        ESS / N threshold for resampling.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        transition_fn: Callable,
        log_likelihood_fn: Callable,
        augment_fn: Callable,
        conditional_proposal_fn: Callable,
        log_transition_density_fn: Callable | None = None,
        log_conditional_likelihood_fn: Callable | None = None,
        n_particles: int = 1000,
        resample_threshold: float = 0.5,
        seed: int = 42,
    ):
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.augment_fn = augment_fn
        self.conditional_proposal_fn = conditional_proposal_fn
        self.log_transition_density_fn = log_transition_density_fn
        self.log_conditional_likelihood_fn = log_conditional_likelihood_fn
        self.N = n_particles
        self.resample_threshold = resample_threshold
        self.rng = np.random.default_rng(seed)

        self.particles: np.ndarray | None = None
        self.log_weights: np.ndarray | None = None
        self._t = 0

    def _init_particles(self, particles: np.ndarray):
        self.particles = particles.copy()
        self.log_weights = np.full(self.N, -np.log(self.N))
        self._t = 0

    def step(self, y_t: float | np.ndarray) -> dict:
        """One augmented filtering step.

        Parameters
        ----------
        y_t : observation at time t.

        Returns
        -------
        dict with ``particles``, ``log_weights``, ``ess``, ``mean``,
        ``quantiles``.
        """
        if self.particles is None:
            raise RuntimeError("Particles not initialised.")

        # --- draw auxiliary variables ---
        aux = self.augment_fn(self.N, self.rng)

        # --- conditional proposal given old particles and auxiliaries ---
        # The proposal receives the *previous* particles (x_{t-1}), not
        # propagated draws, so it can compute the correct Kalman update.
        new_particles, log_q = self.conditional_proposal_fn(
            self.particles, y_t, aux
        )

        # --- importance weight ---
        # When conditional likelihood is provided (augmented scheme):
        #   w = p(y_t | x_t, aux) * p(x_t | x_{t-1}) / q(x_t | x_{t-1}, y_t, aux)
        # The p(aux) terms cancel between target and proposal.
        # Falls back to marginal likelihood if no conditional is given.
        if self.log_conditional_likelihood_fn is not None:
            log_lik = self.log_conditional_likelihood_fn(y_t, new_particles, aux)
        else:
            log_lik = self.log_likelihood_fn(y_t, new_particles)
        log_trans = self.log_transition_density_fn(
            new_particles, self.particles, self._t
        ) if self.log_transition_density_fn is not None else 0.0
        log_inc = log_lik + log_trans - log_q
        self.log_weights = self.log_weights + log_inc
        self.log_weights -= logsumexp(self.log_weights)

        self.particles = new_particles
        ess = compute_ess(self.log_weights)

        weights = np.exp(self.log_weights)
        mean = float(np.dot(weights, self.particles))
        quantiles = weighted_quantiles(
            self.particles, self.log_weights, _DEFAULT_QUANTILES
        )

        if ess / self.N < self.resample_threshold:
            self.particles, self.log_weights = multinomial_resample(
                self.particles, self.log_weights, self.rng
            )

        self._t += 1
        return {
            "particles": self.particles.copy(),
            "log_weights": self.log_weights.copy(),
            "ess": ess,
            "mean": mean,
            "quantiles": quantiles,
        }

    def filter(
        self,
        y: np.ndarray,
        x0: np.ndarray | None = None,
    ) -> dict:
        """Run the full augmented filter over an observation sequence.

        Parameters
        ----------
        y : (T,) or (T, dy) observation sequence.
        x0 : (N,) initial particles.

        Returns
        -------
        dict with arrays keyed by ``particles``, ``log_weights``, ``ess``,
        ``mean``, ``quantiles``.
        """
        T = y.shape[0]
        if x0 is None:
            x0 = np.zeros(self.N)
        self._init_particles(x0)

        out = {
            "particles": np.empty((T, self.N)),
            "log_weights": np.empty((T, self.N)),
            "ess": np.empty(T),
            "mean": np.empty(T),
            "quantiles": np.empty((T, len(_DEFAULT_QUANTILES))),
        }

        for t in range(T):
            res = self.step(y[t])
            out["particles"][t] = res["particles"]
            out["log_weights"][t] = res["log_weights"]
            out["ess"][t] = res["ess"]
            out["mean"][t] = res["mean"]
            out["quantiles"][t] = res["quantiles"]

        return out


# ---------------------------------------------------------------------------
# Backward Smoother (PGBS)
# ---------------------------------------------------------------------------

class BackwardSmoother:
    """Particle GBC backward smoother (Ch 11 §sec-ssm-smoother).

    Given forward filtering output, draw backward-smoothing trajectories
    using a backward kernel.

    Examples
    --------
    >>> fwd = pf.filter(y, x0)
    >>> smoother = BackwardSmoother()
    >>> paths = smoother.smooth(fwd, log_bk_fn, M=100)
    >>> paths.shape   # (M, T)
    """

    def smooth(
        self,
        forward_output: dict,
        log_backward_kernel_fn: Callable,
        M: int = 100,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate M backward-smoothing trajectories.

        At time T-1, sample from the filtering distribution.  Then for
        t = T-2, ..., 0 compute backward weights:

            w_{t|t+1}^i = w_t^i * p(x_{t+1} | x_t^i)

        and resample from the weighted particles.

        Parameters
        ----------
        forward_output : dict
            Output of ``ParticleFilter.filter`` (or compatible), containing
            ``particles`` (T, N) and ``log_weights`` (T, N).
        log_backward_kernel_fn : callable(x_next, particles_t, t) -> (N,)
            Log backward transition kernel:
            log p(x_{t+1} | x_t) evaluated for each particle at time t.
        M : int
            Number of backward trajectories to draw.
        seed : int
            Random seed.

        Returns
        -------
        (M, T) array of smoothed state trajectories.
        """
        rng = np.random.default_rng(seed)
        particles = forward_output["particles"]   # (T, N)
        log_weights = forward_output["log_weights"]  # (T, N)
        T, N = particles.shape

        paths = np.empty((M, T))

        # --- terminal time: sample from filtering distribution ---
        w_T = np.exp(log_weights[T - 1] - logsumexp(log_weights[T - 1]))
        idx = rng.choice(N, size=M, replace=True, p=w_T)
        paths[:, T - 1] = particles[T - 1, idx]

        # --- backward pass ---
        for t in range(T - 2, -1, -1):
            log_w_t = log_weights[t]
            for m in range(M):
                x_next = paths[m, t + 1]
                log_bk = log_backward_kernel_fn(
                    x_next, particles[t], t
                )
                log_w_sm = log_w_t + log_bk
                log_w_sm -= logsumexp(log_w_sm)
                w_sm = np.exp(log_w_sm)
                j = rng.choice(N, p=w_sm)
                paths[m, t] = particles[t, j]

        return paths


# ---------------------------------------------------------------------------
# Factory functions for common models
# ---------------------------------------------------------------------------

def linear_gaussian_ssm(
    F: float,
    H: float,
    Q: float,
    R: float,
    m0: float,
    P0: float,
    n_particles: int = 1000,
    seed: int = 42,
) -> ParticleFilter:
    """Configure a bootstrap particle filter for a linear-Gaussian SSM.

    Model::

        x_t = F * x_{t-1} + e_t,   e_t ~ N(0, Q)
        y_t = H * x_t     + v_t,   v_t ~ N(0, R)
        x_0 ~ N(m0, P0)

    Parameters
    ----------
    F : state transition coefficient.
    H : observation coefficient.
    Q : state noise variance.
    R : observation noise variance.
    m0 : prior mean for x_0.
    P0 : prior variance for x_0.
    n_particles : number of particles.
    seed : random seed.

    Returns
    -------
    Configured ``ParticleFilter`` instance.

    Examples
    --------
    >>> pf = linear_gaussian_ssm(F=0.9, H=1.0, Q=1.0, R=1.0, m0=0, P0=10)
    >>> result = pf.filter(y, x0=rng.normal(m0, np.sqrt(P0), size=1000))
    """
    rng_factory = np.random.default_rng(seed)

    def transition_fn(particles, t):
        return F * particles + rng_factory.normal(0, np.sqrt(Q), size=particles.shape)

    def log_likelihood_fn(y_t, particles):
        return norm.logpdf(y_t, loc=H * particles, scale=np.sqrt(R))

    pf = ParticleFilter(
        transition_fn=transition_fn,
        log_likelihood_fn=log_likelihood_fn,
        n_particles=n_particles,
        seed=seed,
    )
    return pf


def stochastic_volatility(
    phi: float = 0.98,
    sigma: float = 0.16,
    beta: float = 0.65,
    n_particles: int = 1000,
    seed: int = 42,
) -> ParticleFilter:
    """Configure a bootstrap particle filter for the stochastic volatility model.

    Model::

        x_t = phi * x_{t-1} + sigma * e_t,   e_t ~ N(0, 1)
        y_t = beta * exp(x_t / 2) * v_t,     v_t ~ N(0, 1)
        x_0 ~ N(0, sigma^2 / (1 - phi^2))

    Parameters
    ----------
    phi : persistence parameter (|phi| < 1).
    sigma : state noise standard deviation.
    beta : observation scale.
    n_particles : number of particles.
    seed : random seed.

    Returns
    -------
    Configured ``ParticleFilter`` instance.

    Examples
    --------
    >>> pf = stochastic_volatility(phi=0.98, sigma=0.16, beta=0.65)
    >>> P0 = sigma**2 / (1 - phi**2)
    >>> x0 = rng.normal(0, np.sqrt(P0), size=1000)
    >>> result = pf.filter(y, x0)
    """
    rng_factory = np.random.default_rng(seed)

    def transition_fn(particles, t):
        return phi * particles + sigma * rng_factory.standard_normal(particles.shape)

    def log_likelihood_fn(y_t, particles):
        vol = beta * np.exp(particles / 2.0)
        return norm.logpdf(y_t, loc=0.0, scale=vol)

    pf = ParticleFilter(
        transition_fn=transition_fn,
        log_likelihood_fn=log_likelihood_fn,
        n_particles=n_particles,
        seed=seed,
    )
    return pf


def student_t_ssm(
    phi: float = 0.9,
    Q: float = 1.0,
    nu: float = 5.0,
    n_particles: int = 1000,
    seed: int = 42,
) -> AugmentedGBCFilter:
    r"""Configure an augmented GBC filter for an SSM with Student-t observations.

    Model::

        x_t = phi * x_{t-1} + e_t,   e_t ~ N(0, Q)
        y_t | x_t, lambda_t ~ N(x_t, 1 / lambda_t)
        lambda_t ~ Gamma(nu/2, nu/2)

    The marginal observation model integrates to y_t | x_t ~ t_nu(x_t, 1).

    The augmentation variable lambda_t is drawn from its Gamma prior, then
    the conditional proposal is Gaussian with precision lambda_t.

    Parameters
    ----------
    phi : state transition coefficient.
    Q : state noise variance.
    nu : degrees of freedom for Student-t observations.
    n_particles : number of particles.
    seed : random seed.

    Returns
    -------
    Configured ``AugmentedGBCFilter`` instance.

    Examples
    --------
    >>> filt = student_t_ssm(phi=0.9, Q=1.0, nu=5.0)
    >>> result = filt.filter(y, x0=rng.normal(0, 2, size=1000))
    """
    from scipy.stats import t as student_t

    rng_factory = np.random.default_rng(seed)

    def transition_fn(particles, t_step):
        return phi * particles + rng_factory.normal(
            0, np.sqrt(Q), size=particles.shape
        )

    def log_likelihood_fn(y_t, particles):
        # Marginal: y_t | x_t ~ t_nu(loc=x_t, scale=1)
        return student_t.logpdf(y_t, df=nu, loc=particles, scale=1.0)

    def augment_fn(N, rng):
        # lambda ~ Gamma(nu/2, rate=nu/2)  =>  shape=nu/2, scale=2/nu
        return rng.gamma(shape=nu / 2.0, scale=2.0 / nu, size=N)

    def conditional_proposal_fn(old_particles, y_t, aux):
        # old_particles = x_{t-1}; compute predictive mean mu = phi * x_{t-1}
        mu_pred = phi * old_particles
        precision_prior = 1.0 / Q
        precision_lik = aux  # lambda_t
        precision_post = precision_prior + precision_lik
        var_post = 1.0 / precision_post
        mean_post = var_post * (precision_prior * mu_pred + precision_lik * y_t)
        new_particles = rng_factory.normal(mean_post, np.sqrt(var_post))
        log_q = norm.logpdf(new_particles, loc=mean_post, scale=np.sqrt(var_post))
        return new_particles, log_q

    def log_transition_density_fn(new_particles, old_particles, t_step):
        mu_pred = phi * old_particles
        return norm.logpdf(new_particles, loc=mu_pred, scale=np.sqrt(Q))

    def log_conditional_likelihood_fn(y_t, particles, aux):
        # p(y_t | x_t, lambda_t) = N(y_t; x_t, 1/lambda_t)
        return norm.logpdf(y_t, loc=particles, scale=1.0 / np.sqrt(aux))

    filt = AugmentedGBCFilter(
        transition_fn=transition_fn,
        log_likelihood_fn=log_likelihood_fn,
        augment_fn=augment_fn,
        conditional_proposal_fn=conditional_proposal_fn,
        log_transition_density_fn=log_transition_density_fn,
        log_conditional_likelihood_fn=log_conditional_likelihood_fn,
        n_particles=n_particles,
        seed=seed,
    )
    return filt
