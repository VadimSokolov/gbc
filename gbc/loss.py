"""Loss functions for GBC models.

- Pinball (quantile) loss
- Three-term composite loss (L1 anchor + monotonicity + pinball)
- Gaussian negative log-likelihood

Book references
---------------
- Ch 3 §sec-quant-check-loss : pinball (check) loss definition and subgradients
- Ch 5 §sec-iqn-loss         : three-term composite IQN loss
- Ch 14 §sec-cal-nll         : Gaussian NLL for heteroskedastic MLP
"""

import torch
import torch.nn as nn


def _sample_quantile_pair(
    local_probability: float = 0.5,
    local_radius: float = 0.05,
) -> tuple[float, float]:
    """Sample distinct quantile levels for pinball and crossing losses.

    The first level has a Uniform(0, 1) marginal distribution. Local pairs
    make the crossing term sensitive to short oscillations, while independent
    global pairs still compare distant parts of the curve.

    Args:
        local_probability: Probability of drawing the second level near the
            first.
        local_radius: Maximum distance between levels for a local pair. It must
            be no greater than 0.5 so one direction always remains inside the
            interval.
    """
    if not 0.0 <= local_probability <= 1.0:
        raise ValueError("local_probability must be between 0 and 1")
    if not 0.0 < local_radius <= 0.5:
        raise ValueError("local_radius must be in (0, 0.5]")

    eps = float(torch.finfo(torch.float32).eps)

    def draw_level() -> float:
        return float(torch.rand(()).clamp(eps, 1.0 - eps).item())

    tau = draw_level()
    if torch.rand(()).item() < local_probability:
        distance = local_radius * (0.5 + 0.5 * torch.rand(()).item())
        direction = -1.0 if torch.rand(()).item() < 0.5 else 1.0
        tau_other = tau + direction * distance
        if not eps < tau_other < 1.0 - eps:
            tau_other = tau - direction * distance
    else:
        tau_other = draw_level()

    if tau_other == tau:
        tau_other = tau + eps if tau <= 0.5 else tau - eps
    return tau, float(tau_other)


def pinball_loss(y: torch.Tensor, y_hat: torch.Tensor, tau: float) -> torch.Tensor:
    r"""Pinball (check) loss for quantile regression.

    Implements eq. (3.x) from Ch 3 §sec-quant-check-loss.

    .. math::
        \rho_\tau(e) = \max(\tau e, (\tau - 1) e)

    Parameters
    ----------
    y : (n,) observed values.
    y_hat : (n,) predicted quantiles.
    tau : quantile level in (0, 1).

    Returns
    -------
    Scalar mean pinball loss.
    """
    e = y - y_hat
    return torch.mean(torch.maximum(tau * e, (tau - 1) * e))


def composite_loss(
    y: torch.Tensor,
    f: torch.Tensor,
    tau: float,
    w: tuple[float, float, float] = (0.3, 0.3, 0.4),
    f_other: torch.Tensor | None = None,
    tau_other: float | None = None,
) -> torch.Tensor:
    """Three-term IQN loss (Ch 5 §sec-iqn-loss).

    The three terms are:
      1. L1 location anchor (col 0 of f), which stabilises training. Its
         population target under absolute error is a conditional median.
      2. Monotonicity penalty on quantile crossing.
      3. Pinball loss on the quantile estimate (col 1 of f).

    Term 2 compares the model at *two* levels, so it needs a second evaluation:
    pass ``f_other = model(x, tau_other)``.  The penalty is
    ``mean(relu(Q(tau_lo) - Q(tau_hi)))`` for the correctly ordered pair, which
    is zero whenever the quantile curve is non-decreasing in tau.  Being zero
    (with zero subgradient) at every non-crossing configuration, it leaves the
    pinball minimiser, the tau-quantile, exactly where it was.

    Both pair arguments are required when the crossing weight is nonzero. This
    prevents callers from silently requesting a regularizer that is not
    evaluated. In particular, term 2 is *not* replaced by a one-sided penalty
    on the sign of the residual ``y - Q(tau)``: such a term never compares two
    levels, so it cannot detect a crossing, and it is active at the correct
    quantile, which shifts the minimiser to a different level than tau.

    Parameters
    ----------
    y : (n,) targets.
    f : (n, 2) model output; col 0 = location anchor, col 1 = quantile estimate.
    tau : quantile level at which ``f`` was evaluated; scores the pinball term.
    w : weights for (L1 anchor, monotonicity, pinball).
    f_other : (n, 2) model output at ``tau_other``; enables term 2.
    tau_other : the second quantile level. Either side of ``tau`` is fine; the
        pair is ordered here.

    Returns
    -------
    Scalar loss.
    """
    has_other = f_other is not None
    has_other_tau = tau_other is not None
    if has_other != has_other_tau:
        raise ValueError("f_other and tau_other must be supplied together")
    if w[1] != 0.0 and not has_other:
        raise ValueError(
            "f_other and tau_other are required for a nonzero crossing weight"
        )

    e = y.view(-1, 1) - f
    # L1 location anchor; absolute error targets a conditional median.
    loss = w[0] * torch.mean(torch.abs(e[:, 0]))
    # Monotonicity penalty: a genuine crossing between two quantile levels
    if f_other is not None and tau_other is not None and w[1] != 0.0:
        q_lo, q_hi = (f[:, 1], f_other[:, 1]) if tau <= tau_other else (
            f_other[:, 1], f[:, 1])
        loss = loss + w[1] * torch.mean(torch.relu(q_lo - q_hi))
    # Pinball loss
    loss = loss + w[2] * torch.mean(
        torch.maximum(tau * e[:, 1], (tau - 1) * e[:, 1])
    )
    return loss


def gaussian_nll(
    mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    r"""Gaussian negative log-likelihood (heteroskedastic).

    .. math::
        \ell = \frac{1}{2}\left[\log\sigma^2 + \frac{(y - \mu)^2}{\sigma^2}\right]

    Parameters
    ----------
    mu : (n, 1) predicted mean.
    logvar : (n, 1) predicted log-variance.
    y : (n, 1) targets.
    """
    var = torch.exp(logvar).clamp(min=1e-6)
    return 0.5 * (logvar + (y - mu) ** 2 / var).mean()
