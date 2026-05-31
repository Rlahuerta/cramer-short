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


def stationary_distribution(
    P: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Compute the stationary distribution of an ergodic Markov chain.

    Uses repeated application of the transition matrix (power iteration).
    For any row-stochastic P, πP = π when the chain is ergodic.

    Parameters
    ----------
    P : np.ndarray
        Square, row-stochastic transition matrix.
    max_iterations : int
        Maximum power-iteration steps.
    tolerance : float
        L1-convergence threshold for π.

    Returns
    -------
    np.ndarray
        1D probability vector summing to 1.

    Raises
    ------
    ValueError
        If the chain does not converge within ``max_iterations``.
    """
    n = P.shape[0]
    pi = np.full(n, 1.0 / n, dtype=float)

    for _ in range(max_iterations):
        next_pi = pi @ P
        if np.sum(np.abs(next_pi - pi)) < tolerance:
            return next_pi
        pi = next_pi

    raise ValueError(
        f"Stationary distribution did not converge in {max_iterations} iterations"
    )


def second_largest_eigenvalue(
    P: np.ndarray,
    iterations: int = 100,
) -> float:
    """Compute the second-largest absolute eigenvalue via power iteration + deflation.

    ρ determines mixing time: ``exp(-ρ * n)`` is how quickly the chain
    forgets its initial state.  Small ρ → fast mixing; Markov signal
    decays quickly.

    Mirrors ``src/tools/finance/markov-distribution/transition.ts``.

    Parameters
    ----------
    P : np.ndarray
        Square, row-stochastic transition matrix.
    iterations : int
        Power-iteration rounds for each eigenvector.

    Returns
    -------
    float
        Value in ``[0, 1]``.
    """
    n = P.shape[0]

    # --- First eigenvector (stationary distribution) via power iteration ---
    v = np.full(n, 1.0 / n, dtype=float)
    for _ in range(iterations):
        nxt = v @ P
        norm = float(np.sum(nxt))
        v = nxt / norm if norm > 1e-12 else v

    # L2-normalise v for correct orthogonal projection in deflation
    v_l2 = float(np.linalg.norm(v))
    v_unit = v if v_l2 < 1e-12 else v / v_l2

    # --- Deflate: remove first eigenvector, find second via power iteration ---
    # Start with a basis vector and project out the first eigenvector so w ⟂ v
    w = np.zeros(n, dtype=float)
    w[0] = 1.0
    dot = float(np.dot(w, v_unit))
    w = w - dot * v_unit
    w_norm = float(np.linalg.norm(w))
    if w_norm < 1e-12:
        # v_unit was exactly e_0, try e_1
        w = np.zeros(n, dtype=float)
        w[1] = 1.0
        dot = float(np.dot(w, v_unit))
        w = w - dot * v_unit
        w_norm = float(np.linalg.norm(w))
    w = w / w_norm if w_norm > 1e-12 else w

    for _ in range(iterations):
        nxt = w @ P
        dot = float(np.dot(nxt, v_unit))
        deflated = nxt - dot * v_unit
        norm = float(np.linalg.norm(deflated))
        if norm < 1e-10:
            return 0.0
        w = deflated / norm

    Pw = w @ P
    lambda2 = float(np.dot(w, Pw))
    return min(1.0, max(0.0, abs(lambda2)))


def is_irreducible(P: np.ndarray, tol: float = 1e-12) -> bool:
    """Check whether a transition matrix is irreducible (strongly connected).

    Uses repeated squaring: a chain is irreducible iff there exists some
    power ``P^m`` with no zero entries.  For an n-state chain, if
    ``P^(2^(ceil(log2(n))))`` has all positive entries, the chain is
    irreducible (Chapman-Kolmogorov).

    Parameters
    ----------
    P : np.ndarray
        Square, row-stochastic transition matrix.
    tol : float
        Threshold below which an entry is considered zero.

    Returns
    -------
    bool
        True if every state can reach every other state.
    """
    n = P.shape[0]
    m = int(np.ceil(np.log2(max(n, 2))))
    P_m = np.linalg.matrix_power(P, 2 ** m)
    return bool(np.all(P_m > tol))


def mixing_time_scale(
    P: np.ndarray,
    horizon: int = 30,
) -> float:
    """Return the mixing-time scale factor ``exp(-ρ * horizon)``.

    A value near 0 means the chain has essentially forgotten its initial
    state after ``horizon`` steps; a value near 1 means the initial state
    still heavily influences the forecast.

    Parameters
    ----------
    P : np.ndarray
        Square, row-stochastic transition matrix.
    horizon : int
        Forecast horizon in days.

    Returns
    -------
    float
        Mixing weight in ``[0, 1]``.
    """
    rho = second_largest_eigenvalue(P)
    return math.exp(-rho * horizon)
