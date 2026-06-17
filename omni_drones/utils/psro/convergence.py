"""
Convergence detection for PSRO training.
"""

import numpy as np
from typing import List


def has_converged(
    payoff_history: List[np.ndarray],
    window: int = 10,
    threshold: float = 1e-3,
) -> bool:
    """
    Check whether the meta-game has converged by examining the
    stability of recent payoff matrices.

    Args:
        payoff_history: List of payoff matrices over PSRO iterations.
        window: Number of recent iterations to check.
        threshold: Maximum allowed difference between consecutive matrices.

    Returns:
        True if converged, False otherwise.
    """
    if len(payoff_history) < window:
        return False

    recent = payoff_history[-window:]
    for i in range(len(recent) - 1):
        diff = np.abs(recent[i] - recent[i + 1]).max()
        if diff > threshold:
            return False
    return True
