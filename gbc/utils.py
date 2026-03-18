"""Utilities: seeding, device selection, cosine annealing.

Book references
---------------
- Ch 5 §sec-iqn-training  : cosine_schedule used in train_iqn (App. B §sec-computing)
- App. B §sec-computing   : reproducibility via set_seed; GPU selection via get_device
"""

import torch
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_schedule(optimizer, T_max: int, eta_min_ratio: float = 0.01):
    """Create CosineAnnealingLR scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    T_max : int
        Total number of epochs.
    eta_min_ratio : float
        Minimum LR as fraction of initial LR.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=base_lr * eta_min_ratio
    )
