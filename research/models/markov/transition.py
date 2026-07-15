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
    stickiness_shrinkage: bool = False,
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
    stickiness_shrinkage : bool
        If True, apply the xiphos continuous stickiness penalty and row-wise Bayesian shrinkage.

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
    result = counts / row_sums

    # Post-estimation validation: ensure result is a valid stochastic matrix.
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Transition matrix rows must sum to 1")
    if np.any(result < -1e-12):
        raise ValueError("Transition matrix entries must be nonnegative")

    if stickiness_shrinkage:
        max_diag = float(np.max(np.diag(result)))
        d_start = 0.8
        s = float(np.square(max(0.0, (max_diag - d_start) / (1.0 - d_start))))
        raw_row_sums = counts.sum(axis=1) - NUM_STATES * effective_alpha
        raw_row_sums = np.maximum(0.0, raw_row_sums)
        prior_strength = 10.0 * (1.0 + 9.0 * s)
        default = _default_matrix()
        for i in range(NUM_STATES):
            lambda_i = raw_row_sums[i] / (raw_row_sums[i] + prior_strength)
            result[i] = lambda_i * result[i] + (1.0 - lambda_i) * default[i]

        if not np.allclose(result.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("Transition matrix rows must sum to 1")
        if np.any(result < -1e-12):
            raise ValueError("Transition matrix entries must be nonnegative")

    return result


def estimate_conditional_transition_matrices(
    regime_sequence: list[RegimeState],
    environment_sequence: list[str],
    alpha: float | None = None,
    decay_rate: float | None = None,
) -> dict[str, np.ndarray]:
    """Estimate per-environment transition matrices (AH-HMM)."""
    if len(regime_sequence) != len(environment_sequence):
        raise ValueError("Length mismatch")

    pooled = (
        estimate_transition_matrix(regime_sequence, alpha=alpha, decay_rate=decay_rate)
        if len(regime_sequence) >= 2
        else _default_matrix()
    )
    defaults = resolve_forecast_lab_markov_parameter_defaults()
    sparse_observation_count = max(2, int(defaults["transitionMinObservations"]))

    pairs_by_env: dict[str, list[RegimeState]] = {env: [] for env in environment_sequence}
    for i in range(len(regime_sequence) - 1):
        env = environment_sequence[i]
        pairs_by_env.setdefault(env, []).append(regime_sequence[i])
        pairs_by_env[env].append(regime_sequence[i + 1])

    matrices: dict[str, np.ndarray] = {}
    for env, env_regimes in pairs_by_env.items():
        matrices[env] = (
            pooled
            if len(env_regimes) < sparse_observation_count
            else estimate_transition_matrix(env_regimes, alpha=alpha, decay_rate=decay_rate)
        )
    return matrices


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
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Transition matrix rows must sum to 1")
    if np.any(P < -1e-12):
        raise ValueError("Transition matrix entries must be nonnegative")
    if not is_irreducible(P):
        raise ValueError("Transition matrix must be irreducible")

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
) -> float:
    """Compute the second-largest absolute eigenvalue."""
    eigenvalues = np.linalg.eigvals(P)
    if len(eigenvalues) < 2:
        return 0.0
    magnitudes = np.sort(np.abs(eigenvalues))
    return min(1.0, max(0.0, float(magnitudes[-2])))


def is_irreducible(P: np.ndarray, tol: float = 1e-12) -> bool:
    """Check whether a transition matrix is irreducible (strongly connected).

    A chain is irreducible iff every state can reach every other state through
    positive-probability transitions. Periodic chains can be irreducible even
    when no single power of ``P`` has all-positive entries.

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
    reachable = np.asarray(P > tol, dtype=bool)
    np.fill_diagonal(reachable, True)
    for k in range(n):
        reachable = reachable | (reachable[:, [k]] & reachable[[k], :])
    return bool(np.all(reachable))


def mixing_time_scale(
    P: np.ndarray,
    horizon: int = 30,
) -> float:
    """Return the mixing-time scale factor ``ρ ** horizon``.

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
    return math.pow(rho, horizon)
