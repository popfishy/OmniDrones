"""
Meta-game solvers for Policy-Space Response Oracles (PSRO).

Provides solvers that compute meta-strategies (distributions over policies)
from an empirical payoff matrix.
"""

import numpy as np
from typing import Tuple


def nash_solver(payoff_matrix: np.ndarray, iterations: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an approximate Nash equilibrium of a two-player zero-sum game
    using the fictitious play algorithm.

    Args:
        payoff_matrix: Square payoff matrix of shape (N, N).
        iterations: Number of fictitious play iterations.

    Returns:
        Tuple of (row_strategy, col_strategy) as numpy probability distributions.
    """
    N = payoff_matrix.shape[0]
    row_strategy = np.zeros(N)
    col_strategy = np.zeros(N)

    row_counts = np.zeros(N, dtype=int)
    col_counts = np.zeros(N, dtype=int)

    for t in range(1, iterations + 1):
        row_best = np.argmax(payoff_matrix @ col_strategy)
        col_best = np.argmin(row_strategy @ payoff_matrix)

        row_counts[row_best] += 1
        col_counts[col_best] += 1

        row_strategy = row_counts / t
        col_strategy = col_counts / t

    return row_strategy, col_strategy


def uniform_solver(payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a uniform meta-strategy over all policies.

    Args:
        payoff_matrix: Square payoff matrix (only used for shape).

    Returns:
        Tuple of uniform distributions.
    """
    N = payoff_matrix.shape[0]
    dist = np.ones(N) / N
    return dist, dist
