"""Dataset loaders for GBC book examples.

Book references
---------------
- Ch 2 §sec-datasets      : overview of all motivating datasets
- Ch 5 §sec-iqn-motorcycle: MASS::mcycle data (load_motorcycle)
- Ch 5 §sec-iqn-friedman  : Friedman #1 function (friedman1)
- Ch 8 §sec-jumps-problem : 1D jump function (jump_fn)
- Ch 9 §sec-al-problem    : Latin hypercube initialisation (lhs_1d)
- Ch 10 §sec-bayesian-inference: bimodal ABC benchmark (make_bimodal)
"""

import numpy as np
import os


def load_motorcycle() -> tuple[np.ndarray, np.ndarray]:
    """MASS::mcycle dataset (n=133).

    Returns
    -------
    X : (133, 1) time in ms.
    y : (133,) acceleration in g.
    """
    times = np.array([
        2.4,2.6,3.2,3.6,4,6.2,6.6,6.8,7.8,8.2,8.8,8.8,9.6,10,10.2,10.6,11,
        11.4,13.2,13.6,13.8,14.6,14.6,14.6,14.6,14.6,14.6,14.8,15.4,15.4,15.4,
        15.4,15.6,15.6,15.8,15.8,16,16,16.2,16.2,16.2,16.4,16.4,16.6,16.8,16.8,
        16.8,17.6,17.6,17.6,17.6,17.8,17.8,18.6,18.6,19.2,19.4,19.4,19.6,20.2,
        20.4,21.2,21.4,21.8,22,23.2,23.4,24,24.2,24.2,24.6,25,25,25.4,25.4,25.6,
        26,26.2,26.2,26.4,27,27.2,27.2,27.2,27.6,28.2,28.4,28.4,28.6,29.4,30.2,
        31,31.2,32,32,32.8,33.4,33.8,34.4,34.8,35.2,35.2,35.4,35.6,35.6,36.2,
        36.2,38,38,39.2,39.4,40,40.4,41.6,41.6,42.4,42.8,42.8,43,44,44.4,45,
        46.6,47.8,47.8,48.8,50.6,52,53.2,55,55,55.4,57.6])
    accel = np.array([
        0,-1.3,-2.7,0,-2.7,-2.7,-2.7,-1.3,-2.7,-2.7,-1.3,-2.7,-2.7,-2.7,-5.4,
        -2.7,-5.4,0,-2.7,-2.7,0,-13.3,-5.4,-5.4,-9.3,-16,-22.8,-2.7,-22.8,-32.1,
        -53.5,-54.9,-40.2,-21.5,-21.5,-50.8,-42.9,-26.8,-21.5,-50.8,-61.7,-5.4,
        -80.4,-59,-71,-91.1,-77.7,-37.5,-85.6,-123.1,-101.9,-99.1,-104.4,-112.5,
        -50.8,-123.1,-85.6,-72.3,-127.2,-123.1,-117.9,-134,-101.9,-108.4,-123.1,
        -123.1,-128.5,-112.5,-95.1,-81.8,-53.5,-64.4,-57.6,-72.3,-44.3,-26.8,
        -5.4,-107.1,-21.5,-65.6,-16,-45.6,-24.2,9.5,4,12,-21.5,37.5,46.9,-17.4,
        36.2,75,8.1,54.9,48.2,46.9,16,45.6,1.3,75,-16,-54.9,69.6,34.8,32.1,
        -37.5,22.8,46.9,10.7,5.4,-1.3,-21.5,-13.3,30.8,-10.7,29.4,0,-10.7,14.7,
        -1.3,0,10.7,10.7,-26.8,-14.7,-13.3,0,10.7,-14.7,-2.7,10.7,-2.7,10.7])
    n = min(len(times), len(accel))
    return times[:n].reshape(-1, 1), accel[:n]


def friedman1(X: np.ndarray) -> np.ndarray:
    r"""Friedman #1 function (d=10, only first 5 active).

    .. math::
        f(x) = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5
    """
    return (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )


def jump_fn(x: np.ndarray) -> np.ndarray:
    r"""1D jump function: :math:`f(x) = \sin(x) + 10 \cdot \mathbf{1}_{[50,70]}(x)`."""
    return np.sin(x) + 10.0 * ((x >= 50) & (x < 70)).astype(float)


def make_bimodal(
    n: int = 100000,
    n_obs: int = 10,
    theta_true: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate bimodal posterior ABC benchmark.

    Model: y_i | theta ~ N(theta^2, 1), theta ~ U(-3, 3).
    Posterior is bimodal around +/- sqrt(mean(y)).

    Parameters
    ----------
    n : number of simulation pairs.
    n_obs : observations per simulation.
    theta_true : true parameter for observed data.
    seed : random seed.

    Returns
    -------
    (S_y, theta_sim, y_obs) — summary stats, parameters, and observed data.
    """
    rng = np.random.default_rng(seed)
    y_obs = theta_true**2 + rng.normal(0, 1, n_obs)
    theta_sim = rng.uniform(-3, 3, n)
    y_sim = theta_sim[:, np.newaxis] ** 2 + rng.normal(0, 1, (n, n_obs))
    S_y = y_sim.mean(axis=1, keepdims=True)
    return S_y, theta_sim, y_obs


def lhs_1d(
    n: int, lo: float = 0.0, hi: float = 100.0, seed: int = 0
) -> np.ndarray:
    """1D Latin-hypercube-like stratified sample."""
    rng = np.random.default_rng(seed)
    cuts = np.linspace(lo, hi, n + 1)
    return rng.permutation(rng.uniform(cuts[:-1], cuts[1:]))
