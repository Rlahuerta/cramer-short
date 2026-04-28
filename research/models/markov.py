"""Markov regime model.

Mirrors TS logic:
  - 3-state regime classification (bull/bear/sideways)
  - Adaptive threshold = 0.5 * median(|returns|)
  - Transition matrix with Dirichlet smoothing + exponential decay
  - Structural break detection via Frobenius norm
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

RegimeState = Literal["bull", "bear", "sideways"]
REGIME_STATES: list[RegimeState] = ["bull", "bear", "sideways"]
STATE_INDEX: dict[RegimeState, int] = {"bull": 0, "bear": 1, "sideways": 2}
NUM_STATES = len(REGIME_STATES)


def classify_regime(
    daily_return: float,
    return_threshold: float = 0.01,
) -> RegimeState:
    """Classify a single return into a regime state."""
    if daily_return > return_threshold:
        return "bull"
    if daily_return < -return_threshold:
        return "bear"
    return "sideways"


def classify_regime_series(
    returns: np.ndarray | pd.Series | list[float],
    return_threshold_multiplier: float = 0.5,
) -> list[RegimeState]:
    """Classify a return series into regime states with adaptive threshold.

    The threshold is set to 0.5 * median(|returns|), ensuring ~30-40%
    of days are bull, ~30-40% bear, regardless of asset volatility.
    """
    arr = np.asarray(returns)
    if len(arr) == 0:
        return []

    abs_returns = np.sort(np.abs(arr))
    median_abs = float(abs_returns[len(abs_returns) // 2])
    threshold = max(0.001, return_threshold_multiplier * median_abs)

    return [classify_regime(float(r), threshold) for r in arr]


def estimate_transition_matrix(
    states: list[RegimeState],
    alpha: float | None = None,
    min_observations: int = 30,
    decay_rate: float = 0.97,
) -> np.ndarray:
    """Estimate transition matrix with Dirichlet smoothing and exponential decay.

    Parameters
    ----------
    states : list[RegimeState]
        Sequence of regime states (oldest first).
    alpha : float | None
        Dirichlet prior. Auto-tuned as max(0.01, 5/N) if None.
    min_observations : int
        Minimum observations before estimating (returns default matrix otherwise).
    decay_rate : float
        Exponential decay: recent transitions weighted more.

    Returns
    -------
    np.ndarray
        3x3 transition matrix (rows sum to 1).
    """
    if len(states) < min_observations:
        return _default_matrix()

    effective_alpha = alpha if alpha is not None else max(0.01, 5.0 / len(states))

    counts = np.full((NUM_STATES, NUM_STATES), effective_alpha, dtype=float)

    n = len(states) - 1
    for i in range(n):
        from_idx = STATE_INDEX[states[i]]
        to_idx = STATE_INDEX[states[i + 1]]
        age = n - 1 - i  # 0 = most recent
        weight = math.pow(decay_rate, age)
        counts[from_idx][to_idx] += weight

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid div by zero
    return counts / row_sums


def _default_matrix(diagonal: float = 0.6) -> np.ndarray:
    """Identity-like default matrix with correct row sums."""
    off_diag = (1.0 - diagonal) / (NUM_STATES - 1)
    return np.eye(NUM_STATES) * (diagonal - off_diag) + np.full((NUM_STATES, NUM_STATES), off_diag)


def detect_structural_break(
    states: list[RegimeState],
    divergence_threshold: float = 0.05,
    alpha: float = 0.1,
    decay_rate: float = 0.97,
    min_length: int = 60,
) -> dict:
    """Detect structural break by comparing first/second half transition matrices.

    Each half must have enough observations for a stable transition estimate.
    With ``NUM_STATES**2 = 9`` cells and the ≥5-expected-counts rule of thumb,
    each half needs ≥45 transitions; rounded up to 60 so the divergence
    statistic isn't dominated by Dirichlet smoothing (the TS counterpart applies
    the same guard).

    Returns
    -------
    dict
        detected (bool), divergence (float), first_half_matrix, second_half_matrix.
    """
    if len(states) < min_length:
        fallback = _default_matrix()
        return {
            "detected": False,
            "divergence": 0.0,
            "first_half_matrix": fallback,
            "second_half_matrix": fallback,
        }

    mid = len(states) // 2
    first_half = states[:mid]
    second_half = states[mid:]

    first_matrix = estimate_transition_matrix(first_half, alpha, 10, decay_rate)
    second_matrix = estimate_transition_matrix(second_half, alpha, 10, decay_rate)

    divergence = float(np.sum((first_matrix - second_matrix) ** 2))

    return {
        "detected": divergence > divergence_threshold,
        "divergence": divergence,
        "first_half_matrix": first_matrix,
        "second_half_matrix": second_matrix,
    }


def compute_markov_forecast(
    transition_matrix: np.ndarray,
    current_regime: RegimeState,
    horizon: int,
) -> dict[RegimeState, float]:
    """Compute regime probabilities at a given horizon via matrix exponentiation.

    Parameters
    ----------
    transition_matrix : np.ndarray
        3x3 transition matrix.
    current_regime : RegimeState
        Starting regime state.
    horizon : int
        Number of steps ahead.

    Returns
    -------
    dict[RegimeState, float]
        Probability of each regime at the horizon.
    """
    P_n = np.linalg.matrix_power(transition_matrix, horizon)
    idx = STATE_INDEX[current_regime]
    probs = P_n[idx]
    return {
        "bull": float(probs[0]),
        "bear": float(probs[1]),
        "sideways": float(probs[2]),
    }
