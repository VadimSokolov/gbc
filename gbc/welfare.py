"""Welfare analysis via quantile functions.

Implements the MEU (Maximum Expected Utility) framework for computing
expected utility and welfare changes directly from estimated quantile
functions, and Yaari dual-theory distortion weighting for distributional
analysis.

Book references
---------------
- Ch 13 §sec-causal-qte-motivation : quantile treatment effects for welfare
- Polson, Ruggeri & Sokolov (2024) : MEU identity E(U) = ∫₀¹ F⁻¹(z) dz
- Yaari (1987) : dual theory of choice under risk

Key identity
------------
For any random variable U with quantile function Q_U = F_U^{-1}:

    E(U) = ∫₀¹ Q_U(z) dz

This means expected utility (or any expected value) can be computed
directly from quantile estimates without recovering the density.

Yaari distortion weighting applies a monotone transformation h to the
quantile levels, enabling welfare measures that emphasise particular
parts of the distribution (e.g. worst outcomes):

    W_h = ∫₀¹ Q_U(z) · h'(z) dz

Different h give different welfare criteria:
  h(z) = z                : standard expected value
  h(z) = min(z/α, 1)      : CVaR at level α (worst α fraction)
  h(z) = z^γ, γ > 1       : power distortion (pessimistic)
  h(z) = Φ(Φ⁻¹(z) + λ)   : Wang transform (insurance pricing)
"""

import numpy as np
from typing import Callable


def meu(quantile_values: np.ndarray, quantile_levels: np.ndarray) -> float:
    r"""Expected value from a quantile function via trapezoidal integration.

    .. math::
        E(U) = \int_0^1 F_U^{-1}(z)\, dz \approx \sum_j Q(z_j) \Delta z

    Parameters
    ----------
    quantile_values : (n_q,) values of the quantile function Q(z).
    quantile_levels : (n_q,) quantile levels z ∈ (0, 1), sorted ascending.

    Returns
    -------
    Scalar expected value.
    """
    return float(np.trapezoid(quantile_values, quantile_levels))


def welfare_change(
    qf_treated: np.ndarray,
    qf_control: np.ndarray,
    quantile_levels: np.ndarray,
) -> dict[str, float | np.ndarray]:
    r"""Welfare change from quantile treatment effects.

    .. math::
        \Delta W = E(U \mid Z=1) - E(U \mid Z=0)
                 = \int_0^1 [Q_1(z) - Q_0(z)]\, dz

    Parameters
    ----------
    qf_treated : (n_q,) quantile function under treatment.
    qf_control : (n_q,) quantile function under control.
    quantile_levels : (n_q,) quantile levels.

    Returns
    -------
    Dict with ``welfare_change``, ``eu_treated``, ``eu_control``, ``qte``.
    """
    qte = qf_treated - qf_control
    return {
        "welfare_change": meu(qte, quantile_levels),
        "eu_treated": meu(qf_treated, quantile_levels),
        "eu_control": meu(qf_control, quantile_levels),
        "qte": qte,
    }


def yaari_weighted(
    quantile_values: np.ndarray,
    quantile_levels: np.ndarray,
    distortion: Callable[[float], float],
) -> float:
    r"""Yaari dual-theory weighted expected value.

    .. math::
        W_h = \int_0^1 F_U^{-1}(z) \cdot h'(z)\, dz

    where *h* is a distortion function mapping [0,1] → [0,1].

    Parameters
    ----------
    quantile_values : (n_q,) quantile function values.
    quantile_levels : (n_q,) quantile levels.
    distortion : callable h(z) → [0, 1], must be non-decreasing with h(0)=0, h(1)=1.

    Returns
    -------
    Scalar distortion-weighted expected value.
    """
    h_vals = np.array([distortion(z) for z in quantile_levels])
    h_prime = np.gradient(h_vals, quantile_levels)
    return float(np.trapezoid(quantile_values * h_prime, quantile_levels))


def individual_welfare(
    qte_matrix: np.ndarray, quantile_levels: np.ndarray
) -> np.ndarray:
    r"""Per-unit welfare change from individual quantile treatment effects.

    .. math::
        \Delta W_i = \int_0^1 \text{QTE}_i(z)\, dz

    Parameters
    ----------
    qte_matrix : (n, n_q) quantile treatment effects per unit.
    quantile_levels : (n_q,) quantile levels.

    Returns
    -------
    (n,) welfare changes.
    """
    return np.array([
        np.trapezoid(qte_matrix[i], quantile_levels)
        for i in range(qte_matrix.shape[0])
    ])


# ── Standard distortion functions ──────────────────────────────


def distortion_identity(z: float) -> float:
    """No distortion — standard expected value."""
    return z


def distortion_cvar(alpha: float) -> Callable[[float], float]:
    r"""CVaR (Conditional Value at Risk) distortion.

    Focuses on the worst α fraction of outcomes.

    .. math::
        h(z) = \min(z / \alpha, 1)

    Parameters
    ----------
    alpha : fraction in (0, 1). Smaller α → more pessimistic.
    """
    def h(z: float) -> float:
        return min(z / alpha, 1.0)
    return h


def distortion_power(gamma: float) -> Callable[[float], float]:
    r"""Power distortion.

    .. math::
        h(z) = z^\gamma

    γ > 1 → pessimistic (overweights low quantiles).
    γ < 1 → optimistic (overweights high quantiles).
    """
    def h(z: float) -> float:
        return z ** gamma
    return h


def distortion_wang(lam: float) -> Callable[[float], float]:
    r"""Wang transform distortion.

    .. math::
        h(z) = \Phi(\Phi^{-1}(z) + \lambda)

    λ > 0 → risk aversion; λ < 0 → risk seeking.

    Requires scipy.
    """
    from scipy.stats import norm

    def h(z: float) -> float:
        z = np.clip(z, 1e-10, 1 - 1e-10)
        return float(norm.cdf(norm.ppf(z) + lam))
    return h
