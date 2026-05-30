"""Transition matrix estimation and structural break detection.

Mirrors TS logic:
  - Dirichlet smoothing + exponential decay
  - Structural break detection via Frobenius norm
"""

from __future__ import annotations

import math

import numpy as np

from research.models.markov.core import NUM_STATES, RegimeState, STATE_INDEX
from research.models.markov.policies import resolve_forecast_lab_markov_parameter_defaults


def estimate_transition_matrix(
    states: list[RegimeState],
    alpha: float | None = None,
    min_observations: int | None = None,
    decay_rate: float | None = None,
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
    defaults = resolve_forecast_lab_markov_parameter_defaults()
    effective_min_observations = int(
        min_observations
        if min_observations is not None
        else defaults["transitionMinObservations"]
    )
    effective_decay_rate = float(
        decay_rate if decay_rate is not None else defaults["transitionDecay"]
    )

    if len(states) < effective_min_observations:
        return _default_matrix()

    effective_alpha = alpha if alpha is not None else max(0.01, 5.0 / len(states))

    counts = np.full((NUM_STATES, NUM_STATES), effective_alpha, dtype=float)

    n = len(states) - 1
    for i in range(n):
        from_idx = STATE_INDEX[states[i]]
        to_idx = STATE_INDEX[states[i + 1]]
        age = n - 1 - i  # 0 = most recent
        weight = math.pow(effective_decay_rate, age)
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
    decay_rate: float | None = None,
    min_length: int | None = None,
) -> dict:
    """Detect structural break by comparing first/second half transition matrices.

    Each half must have enough observations for a stable transition estimate.
    With ``NUM_STATES**2 = 9`` cells and the >=5-expected-counts rule of thumb,
    each half needs >=45 transitions; the TS default of 36 provides a practical
    floor that balances sensitivity against false alarms.

    Returns
    -------
    dict
        detected (bool), divergence (float), first_half_matrix, second_half_matrix.
    """
    defaults = resolve_forecast_lab_markov_parameter_defaults()
    effective_decay_rate = float(
        decay_rate if decay_rate is not None else defaults["transitionDecay"]
    )
    effective_min_length = int(
        min_length if min_length is not None else defaults["structuralBreakMinLength"]
    )

    if len(states) < effective_min_length:
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

    first_matrix = estimate_transition_matrix(first_half, alpha, 10, effective_decay_rate)
    second_matrix = estimate_transition_matrix(second_half, alpha, 10, effective_decay_rate)

    divergence = float(np.sum((first_matrix - second_matrix) ** 2))

    return {
        "detected": divergence > divergence_threshold,
        "divergence": divergence,
        "first_half_matrix": first_matrix,
        "second_half_matrix": second_matrix,
    }
