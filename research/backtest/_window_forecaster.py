"""Per-window Markov forecaster: regime → transition → trajectory → forecast.

Takes a price window and model parameters, returns the forecast payload
with p_up, predicted_return, CI bounds, entropy, and break metadata.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np

from research.models.markov import (
    classify_regime_series,
    detect_structural_break,
    estimate_transition_matrix,
    compute_markov_forecast,
)
from research.models.garch_scales import GarchClampOptions, compute_garch_scales
from research.models.hmm import ASSET_PROFILES, baum_welch, fit_volatility_hmm, predict
from research.models.transition_entropy import compute_transition_entropy
from research.models.trajectory import RegimeStats, compute_trajectory


class WindowForecast(TypedDict):
    p_up: float
    predicted_return: float
    ci_lower: float
    ci_upper: float
    garch_vol_applied: bool
    entropy: object
    break_result: object
    break_rerun_triggered: bool
    original_break_result: object


def compute_window_forecast(
    window_prices: list[float],
    horizon: int,
    return_threshold_multiplier: float,
    decay_rate: float,
    break_divergence_threshold: float,
    use_hmm: bool,
    asset_profile: str,
    enable_garch_vol: bool,
    garch_horizon: int | None,
    garch_ceiling: tuple[float, float] | None,
) -> WindowForecast:
    active_returns = np.array(
        [(window_prices[i] - window_prices[i - 1]) / window_prices[i - 1]
         for i in range(1, len(window_prices))]
    )
    regimes = classify_regime_series(
        active_returns, return_threshold_multiplier=return_threshold_multiplier
    )
    P = estimate_transition_matrix(regimes, decay_rate=decay_rate)

    break_result = detect_structural_break(
        regimes,
        divergence_threshold=break_divergence_threshold,
        decay_rate=decay_rate,
    )

    current_regime = regimes[-1] if regimes else "sideways"
    forecast = compute_markov_forecast(P, current_regime, horizon)

    regime_stats: dict[str, RegimeStats] = {}
    for state in ["bull", "bear", "sideways"]:
        mask = [r == state for r in regimes]
        if any(mask):
            state_returns = active_returns[mask]
            regime_stats[state] = RegimeStats(
                mean_return=float(np.mean(state_returns)),
                std_return=float(np.std(state_returns, ddof=1))
                if len(state_returns) > 1
                else 0.01,
            )
        else:
            regime_stats[state] = RegimeStats(mean_return=0.0, std_return=0.01)

    up_rates: dict[str, float] = {}

    for state in ["bull", "bear", "sideways"]:
        mask = [r == state for r in regimes]
        if any(mask):
            state_returns = active_returns[mask]
            up_rates[state] = float(np.mean(state_returns > 0))
        else:
            up_rates[state] = 0.5

    p_up = sum(
        forecast[state] * up_rates[state]
        for state in ["bull", "bear", "sideways"]
    )

    hmm_override: dict[str, float] | None = None
    if use_hmm:
        hmm_result = baum_welch(
            active_returns,
            n_states=3,
            max_iterations=50,
            tolerance=1e-3,
        )

        if hmm_result.converged:
            hmm_pred = predict(
                active_returns, hmm_result.params, forecast_horizon=horizon
            )
            vol_scale = fit_volatility_hmm(
                active_returns, vol_window=5, n_states=2
            )
            profile = ASSET_PROFILES.get(asset_profile, ASSET_PROFILES["crypto"])
            hmm_weight = np.clip(profile.hmm_weight_multiplier * 0.5, 0.0, 1.0)
            hmm_override = {
                "drift": hmm_pred.expected_return,
                "vol": hmm_pred.expected_volatility * vol_scale,
                "weight": float(hmm_weight),
            }
        else:
            return {
                "p_up": 0.5,
                "predicted_return": 0.0,
                "ci_lower": window_prices[-1] * 0.9,
                "ci_upper": window_prices[-1] * 1.1,
                "garch_vol_applied": False,
                "entropy": compute_transition_entropy(P),
                "break_result": break_result,
                "break_rerun_triggered": False,
                "original_break_result": break_result,
            }

    garch_scales: list[float] | None = None
    garch_vol_applied = False

    if enable_garch_vol:
        log_returns = [
            math.log(window_prices[i] / window_prices[i - 1])
            for i in range(1, len(window_prices))
        ]
        opts: GarchClampOptions | None = None
        if garch_horizon is not None or garch_ceiling is not None:
            opts = GarchClampOptions(
                horizon_cap=garch_horizon,
                ceiling=garch_ceiling or (1.5, 3.0),
            )
        scales = compute_garch_scales(log_returns, horizon, opts)
        if scales:
            garch_scales = scales
            garch_vol_applied = True

    traj = compute_trajectory(
        window_prices[-1],
        horizon,
        P,
        regime_stats,
        current_regime,
        hmm_override=hmm_override,
        n_samples=500,
        garch_scales=garch_scales,
    )
    horizon_point = traj[-1]
    predicted_return = (
        horizon_point.expected_price - window_prices[-1]
    ) / window_prices[-1]

    return {
        "p_up": float(p_up),
        "predicted_return": float(predicted_return),
        "ci_lower": float(horizon_point.lower_bound),
        "ci_upper": float(horizon_point.upper_bound),
        "garch_vol_applied": bool(garch_vol_applied),
        "entropy": compute_transition_entropy(P),
        "break_result": break_result,
        "break_rerun_triggered": False,
        "original_break_result": break_result,
    }
