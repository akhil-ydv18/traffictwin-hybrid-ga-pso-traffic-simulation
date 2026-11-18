"""
common.py - shared result container and helpers for GA / PSO / Hybrid.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class OptResult:
    name: str
    best_x: np.ndarray
    best_f: float                              # best (lowest) objective J found
    history: List[float]                       # best-so-far per generation / iteration
    eval_history: List[Tuple[int, float]]      # (objective evaluations used, best-so-far)
    n_evals: int
    runtime: float                             # seconds
    extra: dict = field(default_factory=dict)


def check_bounds(lower, upper):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1 or lower.size == 0:
        raise ValueError("lower and upper must be 1-D arrays of equal length")
    if np.any(lower >= upper):
        raise ValueError("every lower bound must be smaller than its upper bound")
    return lower, upper


def diversity(positions: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Population diversity: mean (over dimensions) of the standard deviation of
    the positions, scaled by the search range.  0 = everybody identical."""
    return float(np.mean(np.std(positions, axis=0) / (upper - lower)))
