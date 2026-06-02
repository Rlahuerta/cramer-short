"""Horizon drift, volatility and state-weight computation.

Mirrors TS logic from src/tools/finance/markov-distribution.ts:
  - Regime-weighted drift/vol via Markov matrix powers
  - Momentum adjustment, HMM override, GARCH scaling
  - Mixture variance with dominant-state sigma option
"""

from __future__ import annotations

import math

import numpy as np

from research.models.markov import NUM_STATES, RegimeState, REGIME_STATES, STATE_INDEX
from research.models.trajectory.types import RegimeStats


def _mat_pow(P: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.eye(P.shape[0])
    return np.linalg.matrix_power(P, n)


def _normalize_state_weight_vector(weights: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.size != NUM_STATES:
        return np.full(NUM_STATES, 1 / NUM_STATES, dtype=float)
    sanitized = np.where(np.isfinite(arr) & (arr >= 0), arr, 0.0)
    total = float(np.sum(sanitized))
    if total <= 0:
        return np.full(NUM_STATES, 1 / NUM_STATES, dtype=float)
    return sanitized / total


def compute_mixing_weight(second_eigenvalue: float, horizon: int) -> float:
    """Compute the mixing-time weight ρ^horizon where ρ = |λ₂|.

    The exponential-decay formulation exp(-ρ·h) is numerically unstable for
    negative eigenvalues (which can occur in oscillatory Markov chains),
    where it grows unbounded instead of decaying.  Using the power formulation
    guarantees the result stays in [0, 1] for any valid second eigenvalue.
    """
    rho = max(1e-15, float(abs(second_eigenvalue)))
    return math.pow(rho, horizon)


def compute_horizon_drift_vol(
    horizon: int,
    P: np.ndarray,
    regime_stats: dict[RegimeState, RegimeStats],
    initial_state: RegimeState,
    momentum_adjustment: float = 0.0,
    start_mixture: dict[RegimeState, float] | None = None,
    hmm_override: dict[str, float] | None = None,
    regime_specific_sigma: bool = False,
    regime_specific_sigma_threshold: float | None = None,
    garch_scales: list[float] | None = None,
    terminal_state_weights: list[float] | np.ndarray | None = None,
) -> dict[str, float]:
    if terminal_state_weights is not None:
        state_weights = _normalize_state_weight_vector(terminal_state_weights)
    else:
        Pn = _mat_pow(P, horizon)

        if start_mixture:
            state_weights = np.zeros(NUM_STATES)
            for state, w in start_mixture.items():
                idx = STATE_INDEX[state]
                state_weights += w * Pn[idx]
            state_weights = _normalize_state_weight_vector(state_weights)
        else:
            state_weights = _normalize_state_weight_vector(Pn[STATE_INDEX[initial_state]])

    mu_obs = sum(
        state_weights[i] * regime_stats[state].mean_return
        for i, state in enumerate(REGIME_STATES)
    )

    var_of_means = sum(
        state_weights[i] * (regime_stats[state].mean_return - mu_obs) ** 2
        for i, state in enumerate(REGIME_STATES)
    )
    mixture_sigma = math.sqrt(
        sum(
            state_weights[i] * regime_stats[state].std_return ** 2
            for i, state in enumerate(REGIME_STATES)
        )
        + var_of_means
    )

    mu_eff = mu_obs
    dominant_idx = int(np.argmax(state_weights))
    dominant_sigma = regime_stats[REGIME_STATES[dominant_idx]].std_return
    threshold = 0.60 if regime_specific_sigma_threshold is None else regime_specific_sigma_threshold
    sigma_eff = dominant_sigma if regime_specific_sigma and float(np.max(state_weights)) > threshold else mixture_sigma

    mu_n = horizon * (mu_eff + momentum_adjustment)
    sigma_n = sigma_eff * math.sqrt(horizon)

    if hmm_override:
        w = hmm_override.get("weight", 0.0)
        hmm_drift = hmm_override.get("drift", mu_eff)
        hmm_vol = hmm_override.get("vol", sigma_eff)
        mu_n = w * (horizon * hmm_drift) + (1 - w) * mu_n
        sigma_n = w * (hmm_vol * math.sqrt(horizon)) + (1 - w) * sigma_n

    if garch_scales:
        variance_scale = 0.0
        for day in range(horizon):
            scale = garch_scales[day] if day < len(garch_scales) else 1.0
            variance_scale += scale * scale if math.isfinite(scale) and scale > 0 else 1.0
        # Scale the already-blended sigma_n (which may include HMM contribution)
        # rather than replacing it with the pre-blend sigma_eff.
        sigma_n *= math.sqrt(variance_scale / horizon) if horizon > 0 else 1.0

    return {
        "mu_n": mu_n,
        "sigma_n": sigma_n,
    }
