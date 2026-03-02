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
) -> torch.Tensor:
    """Three-term IQN loss (Ch 5 §sec-iqn-loss).

    The three terms are:
      1. L1 anchor on the conditional mean (col 0 of f) — stabilises training.
      2. Monotonicity regularisation — penalises quantile crossings.
      3. Pinball loss on the quantile estimate (col 1 of f).

    Parameters
    ----------
    y : (n,) targets.
    f : (n, 2) model output — col 0 = mean estimate, col 1 = quantile estimate.
    tau : sampled quantile level.
    w : weights for (L1 anchor, monotonicity, pinball).

    Returns
    -------
    Scalar loss.
    """
    e = y.view(-1, 1) - f
    # L1 anchor on conditional mean
    loss = w[0] * torch.mean(torch.abs(e[:, 0]))
    # Monotonicity regularization
    tauind = float(tau < 0.5)
    mono = tauind * torch.mean(torch.relu(-e[:, 1])) + (1 - tauind) * torch.mean(
        torch.relu(e[:, 1])
    )
    loss = loss + w[1] * abs(tau - 0.5) * mono
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
