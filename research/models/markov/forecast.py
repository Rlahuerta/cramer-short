"""Markov horizon forecasts and drift/volatility computations.

Mirrors TS logic:
  - Matrix exponentiation for regime probabilities at horizon
  - Regime-weighted drift and volatility
"""

from __future__ import annotations

import math

import numpy as np

from research.models.markov.core import NUM_STATES, RegimeState, REGIME_STATES, STATE_INDEX
from research.models.soft_regime import blend_regime_mixtures


def compute_markov_forecast(
    transition_matrix: np.ndarray,
    current_regime: RegimeState,
    horizon: int,
    *,
    start_mixture: dict[RegimeState, float] | None = None,
    forecast_mixture: dict[RegimeState, float] | None = None,
    soft_transition_blend_weight: float = 0.0,
) -> dict[RegimeState, float]:
    """Compute regime probabilities at a given horizon via matrix exponentiation."""
    P_n = np.linalg.matrix_power(transition_matrix, horizon)
    if start_mixture is None:
        idx = STATE_INDEX[current_regime]
        probs = np.array(P_n[idx], dtype=float)
    else:
        probs = np.zeros(NUM_STATES, dtype=float)
        for state, weight in start_mixture.items():
            probs += float(weight) * P_n[STATE_INDEX[state]]

    if forecast_mixture is not None:
        weight = min(1.0, max(0.0, float(soft_transition_blend_weight)))
        blended = blend_regime_mixtures(
            {
                "bull": float(probs[0]),
                "bear": float(probs[1]),
                "sideways": float(probs[2]),
            },
            forecast_mixture,
            weight,
        )
        probs = np.array([blended["bull"], blended["bear"], blended["sideways"]], dtype=float)

    total = float(np.sum(probs))
    if total > 0:
        probs = probs / total
    return {
        "bull": float(probs[0]),
        "bear": float(probs[1]),
        "sideways": float(probs[2]),
    }


def _compute_terminal_state_weights(
    horizon: int,
    P: np.ndarray,
    initial_state: RegimeState,
    start_mixture: dict[RegimeState, float] | None = None,
) -> np.ndarray:
    """Compute regime-state probability weights at a horizon via matrix power."""
    P_n = np.linalg.matrix_power(P, horizon)
    if start_mixture is None:
        idx = STATE_INDEX[initial_state]
        return np.array(P_n[idx], dtype=float)
    weights = np.zeros(NUM_STATES, dtype=float)
    for state, w in start_mixture.items():
        weights += float(w) * P_n[STATE_INDEX[state]]
    return weights


def _compute_horizon_drift_vol(
    horizon: int,
    P: np.ndarray,
    regime_stats: dict[RegimeState, dict[str, float]],
    initial_state: RegimeState,
    start_mixture: dict[RegimeState, float] | None = None,
) -> dict[str, float]:
    """Compute horizon-ahead drift and volatility from regime-weighted stats."""
    state_weights = _compute_terminal_state_weights(horizon, P, initial_state, start_mixture)

    mu_obs = sum(
        state_weights[i] * regime_stats[state]["meanReturn"]
        for i, state in enumerate(REGIME_STATES)
    )
    var_of_means = sum(
        state_weights[i] * (regime_stats[state]["meanReturn"] - mu_obs) ** 2
        for i, state in enumerate(REGIME_STATES)
    )
    mixture_var = sum(
        state_weights[i] * regime_stats[state]["stdReturn"] ** 2
        for i, state in enumerate(REGIME_STATES)
    ) + var_of_means
    sigma_obs = math.sqrt(mixture_var)
    sigma_n = sigma_obs * math.sqrt(horizon)
    mu_n = horizon * mu_obs
    return {"mu_n": mu_n, "sigma_n": sigma_n}
