"""Monte Carlo trajectory simulation engine.

Mirrors TS logic from src/tools/finance/markov-distribution.ts:
  - Shared MC paths with per-day mixture drift/vol
  - Student-t innovations with jump-diffusion support
  - Monotonically widening confidence intervals
"""

from __future__ import annotations

import math

import numpy as np

from research.models.markov import NUM_STATES, RegimeState, REGIME_STATES, STATE_INDEX
from research.models.jump_diffusion import JumpEventSpec, jump_drift_compensator
from research.models.trajectory.distributions import student_t_ppf, student_t_survival
from research.models.trajectory.driftvol import _mat_pow
from research.models.trajectory.types import RegimeStats, TrajectoryPoint


def compute_trajectory(
    current_price: float,
    days: int,
    P: np.ndarray,
    regime_stats: dict[RegimeState, RegimeStats],
    initial_state: RegimeState,
    momentum_adjustment: float = 0.0,
    n_samples: int = 1000,
    nu: int = 5,
    empirical_daily_vol: float | None = None,
    start_mixture: dict[RegimeState, float] | None = None,
    hmm_override: dict[str, float] | None = None,
    jump_spec: list[JumpEventSpec] | None = None,
    garch_scales: list[float] | None = None,
    uncertainty_ci_scale: float = 1.0,
) -> list[TrajectoryPoint]:
    initial_idx = STATE_INDEX[initial_state]
    trajectory: list[TrajectoryPoint] = []

    regime_weights_per_day: list[np.ndarray] = []
    for d in range(1, days + 1):
        Pd = _mat_pow(P, d)
        if start_mixture:
            weights = np.zeros(NUM_STATES)
            for state, w in start_mixture.items():
                idx = STATE_INDEX[state]
                weights += w * Pd[idx]
            regime_weights_per_day.append(weights)
        else:
            regime_weights_per_day.append(Pd[initial_idx])

    daily_drifts = np.zeros(days)
    daily_vols = np.zeros(days)

    for d in range(days):
        weights = regime_weights_per_day[d]

        mu_obs = sum(
            weights[i] * regime_stats[state].mean_return
            for i, state in enumerate(REGIME_STATES)
        )

        var_of_means = sum(
            weights[i] * (regime_stats[state].mean_return - mu_obs) ** 2
            for i, state in enumerate(REGIME_STATES)
        )
        expected_var = sum(
            weights[i] * regime_stats[state].std_return ** 2
            for i, state in enumerate(REGIME_STATES)
        )
        sigma_obs = math.sqrt(expected_var + var_of_means)

        mu_obs += momentum_adjustment

        if hmm_override:
            w = hmm_override.get("weight", 0.0)
            hmm_drift = hmm_override.get("drift", mu_obs)
            hmm_vol = hmm_override.get("vol", sigma_obs)
            mu_obs = w * hmm_drift + (1 - w) * mu_obs
            sigma_obs = w * hmm_vol + (1 - w) * sigma_obs

        if empirical_daily_vol:
            sigma_obs = max(sigma_obs, empirical_daily_vol)

        daily_drifts[d] = mu_obs
        daily_vols[d] = sigma_obs

    if garch_scales:
        for d in range(min(days, len(garch_scales))):
            k = garch_scales[d]
            if math.isfinite(k) and k > 0:
                daily_vols[d] *= k

    has_jumps = bool(jump_spec)
    if has_jumps:
        compensator = jump_drift_compensator(jump_spec)
        daily_drifts -= compensator

    paths = np.zeros((n_samples, days))
    for s in range(n_samples):
        cum_log_return = 0.0
        for d in range(days):
            u = np.random.random()
            z = student_t_ppf(u, nu)
            scaled_vol = daily_vols[d] * math.sqrt((nu - 2) / nu) if nu > 2 else daily_vols[d]
            cum_log_return += daily_drifts[d] + z * scaled_vol

            if has_jumps:
                for e in jump_spec:
                    if np.random.random() < e.daily_intensity:
                        u1 = max(1e-12, np.random.random())
                        u2 = np.random.random()
                        z_j = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                        cum_log_return += e.mean_log_jump + z_j * e.std_log_jump

            paths[s, d] = cum_log_return

    for d in range(1, days + 1):
        day_idx = d - 1
        state_weights = regime_weights_per_day[day_idx]

        mu_n = float(np.sum(daily_drifts[:d]))
        sigma_n = math.sqrt(float(np.sum(daily_vols[:d] ** 2)))

        prices = current_price * np.exp(paths[:, day_idx])
        prices_sorted = np.sort(prices)

        p5_idx = max(0, int(n_samples * 0.05) - 1)
        p50_idx = int(n_samples * 0.5)
        p95_idx = min(n_samples - 1, int(np.ceil(n_samples * 0.95)))

        lower_bound = float(prices_sorted[p5_idx])
        upper_bound = float(prices_sorted[p95_idx])

        expected_price = current_price * math.exp(mu_n)

        # Apply uncertainty CI scale from the original asymmetric bounds around
        # expected_price. Re-centering total width can shrink one side of a
        # skewed lognormal interval even when the caller requested widening.
        if abs(uncertainty_ci_scale - 1.0) > 1e-9:
            scaled_lower = expected_price - (expected_price - lower_bound) * uncertainty_ci_scale
            scaled_upper = expected_price + (upper_bound - expected_price) * uncertainty_ci_scale
            if uncertainty_ci_scale >= 1.0:
                scaled_lower = min(lower_bound, scaled_lower)
                scaled_upper = max(upper_bound, scaled_upper)
            lower_bound = max(0.01, scaled_lower)
            upper_bound = scaled_upper

        p_up = student_t_survival(current_price, current_price, mu_n, sigma_n, nu)

        ret = (expected_price - current_price) / current_price
        cumulative_return = f"{(ret * 100):+.1f}%"

        max_weight = -1.0
        regime: RegimeState = initial_state
        for i, state in enumerate(REGIME_STATES):
            if state_weights[i] > max_weight:
                max_weight = float(state_weights[i])
                regime = state

        trajectory.append(
            TrajectoryPoint(
                day=d,
                expected_price=round(expected_price, 2),
                lower_bound=round(lower_bound, 2),
                upper_bound=round(upper_bound, 2),
                p_up=round(p_up, 3),
                cumulative_return=cumulative_return,
                regime=regime,
            )
        )

    return trajectory
