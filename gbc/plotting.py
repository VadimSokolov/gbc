"""Plotting utilities for the GBC book.

Standard figure theme, quantile fan plots, calibration plots.

Book references
---------------
- Ch 5 §sec-iqn-motorcycle : quantile_fan used for motorcycle fan chart
- Ch 5 §sec-iqn-diagnostics: calibration_plot (PIT histogram) for IQN validation
- Ch 14 §sec-cal-conformal : calibration_plot for conformal coverage check
  (figures appear throughout all chapters)
"""

import numpy as np
import matplotlib.pyplot as plt

from gbc.metrics import pit_values


def set_theme():
    """Apply GBC book figure theme."""
    plt.rcParams.update({
        "figure.figsize": (7, 4),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def quantile_fan(
    x: np.ndarray,
    samples: np.ndarray,
    y_true: np.ndarray | None = None,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
    ax: plt.Axes | None = None,
    color: str = "steelblue",
    title: str = "",
) -> plt.Axes:
    """Quantile fan chart showing nested prediction intervals.

    Parameters
    ----------
    x : (n,) x-axis values for test points.
    samples : (B, n) quantile samples.
    y_true : (n,) true function values (optional).
    X_train : (n_train,) training x values for scatter (optional).
    y_train : (n_train,) training y values for scatter (optional).
    levels : coverage levels for nested bands.
    ax : matplotlib axes (created if None).
    color : band color.
    title : plot title.
    """
    if ax is None:
        _, ax = plt.subplots()

    alphas = np.linspace(0.15, 0.35, len(levels))
    for alpha_val, level in zip(alphas, sorted(levels, reverse=True)):
        lo = np.quantile(samples, (1 - level) / 2, axis=0)
        hi = np.quantile(samples, 1 - (1 - level) / 2, axis=0)
        ax.fill_between(x, lo, hi, alpha=alpha_val, color=color,
                        label=f"{int(level*100)}% PI")

    mu = np.median(samples, axis=0)
    ax.plot(x, mu, color=color, lw=2, label="Median")

    if y_true is not None:
        ax.plot(x, y_true, "k--", lw=1, label="True")
    if X_train is not None and y_train is not None:
        ax.scatter(X_train, y_train, s=12, c="k", zorder=5, label="Data")

    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def calibration_plot(
    y: np.ndarray,
    samples: np.ndarray,
    ax: plt.Axes | None = None,
    n_bins: int = 20,
    title: str = "Calibration",
) -> plt.Axes:
    """PIT histogram / calibration plot.

    Parameters
    ----------
    y : (n,) observed values.
    samples : (B, n) quantile samples.
    n_bins : number of histogram bins.
    """
    if ax is None:
        _, ax = plt.subplots()

    pit = pit_values(y, samples)
    ax.hist(pit, bins=n_bins, density=True, alpha=0.7, color="steelblue",
            edgecolor="white")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="Ideal (Uniform)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return ax
