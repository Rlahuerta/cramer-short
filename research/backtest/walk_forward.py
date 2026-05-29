"""Walk-forward backtest harness.

Slides a window over historical prices, estimates the Markov model at each
step, records predictions vs realised outcomes, and aggregates results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from research.data.prices import fetch_historical_prices
from research.models.markov import (
    classify_regime_series,
    detect_structural_break,
    estimate_transition_matrix,
    compute_markov_forecast,
    get_btc_short_horizon_live_policy,
)
from research.models.garch_scales import GarchClampOptions, compute_garch_scales
from research.models.hmm import ASSET_PROFILES, baum_welch, fit_volatility_hmm, predict
from research.models.transition_entropy import (
    EntropyZScoreTracker,
    compute_transition_entropy,
    entropy_z_to_ci_scale,
)
from research.models.trajectory import compute_trajectory, RegimeStats
from research.utils.forecast_lab_runtime_defaults import (
    forecast_lab_runtime_asset_scope,
    resolve_forecast_lab_runtime_asset_scope_for_ticker,
)


@dataclass
class BacktestStep:
    start_idx: int
    predicted_prob: float
    predicted_return: float
    ci_lower: float
    ci_upper: float
    realised_return: float
    realised_price: float
    direction_correct: bool
    in_ci: bool
    garch_vol_applied: bool | None = None
    transition_entropy: float | None = None
    transition_entropy_norm: float | None = None
    transition_entropy_z: float | None = None
    entropy_ci_scale: float | None = None
    entropy_ci_modulation_applied: bool | None = None
    structural_break_detected: bool | None = None
    structural_break_rerun_triggered: bool | None = None
    original_structural_break_detected: bool | None = None
    original_structural_break_divergence: float | None = None


@dataclass
class WalkForwardResult:
    steps: list[BacktestStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def walk_forward(
    prices: list[float],
    horizon: int = 7,
    warmup: int = 120,
    stride: int = 10,
    ticker: str | None = None,
    return_threshold_multiplier: float = 0.5,
    decay_rate: float = 0.97,
    break_divergence_threshold: float = 0.05,
    btc_break_divergence_threshold: float | None = None,
    post_break_short_window: bool | None = None,
    post_break_window_size: int | None = None,
    use_live_btc_short_horizon_policy: bool = False,
    use_hmm: bool = False,
    asset_profile: str = "crypto",
    enable_garch_vol: bool = False,
    garch_horizon_cap: int | None = None,
    garch_regime_ceiling: tuple[float, float] | None = None,
    enable_entropy_ci_modulation: bool = False,
    entropy_window_size: int = 60,
    entropy_kappa: float = 0.15,
) -> WalkForwardResult:
    """Run a walk-forward backtest on a price series.

    Parameters
    ----------
    prices : list[float]
        Historical close prices (oldest first).
    horizon : int
        Forecast horizon in days.
    warmup : int
        Minimum history for regime estimation.
    stride : int
        Days between consecutive test windows.
    return_threshold_multiplier : float
        Adaptive threshold multiplier for regime classification.
    decay_rate : float
        Exponential decay for transition matrix weighting.
    use_hmm : bool
        Whether to blend HMM predictions into the forecast.
    asset_profile : str
        Asset profile key (etf, equity, crypto, commodity) for HMM weight tuning.

    Returns
    -------
    WalkForwardResult
        Steps and any errors encountered.
    """
    runtime_asset_scope = resolve_forecast_lab_runtime_asset_scope_for_ticker(ticker or "")

    with forecast_lab_runtime_asset_scope(runtime_asset_scope):
        result = WalkForwardResult()
        entropy_tracker = EntropyZScoreTracker(max(5, entropy_window_size))
        btc_live_policy = (
            get_btc_short_horizon_live_policy(ticker or "", horizon)
            if use_live_btc_short_horizon_policy
            else None
        )
        effective_warmup = btc_live_policy.history_days if btc_live_policy else warmup
        effective_break_divergence_threshold = (
            btc_break_divergence_threshold
            if btc_break_divergence_threshold is not None
            else (
                btc_live_policy.break_divergence_threshold
                if btc_live_policy
                else break_divergence_threshold
            )
        )
        effective_post_break_short_window = (
            post_break_short_window
            if post_break_short_window is not None
            else (btc_live_policy.rerun_on_break if btc_live_policy else False)
        )
        effective_post_break_window_size = (
            post_break_window_size
            if post_break_window_size is not None
            else (
                btc_live_policy.rerun_window_days
                if btc_live_policy and btc_live_policy.rerun_window_days is not None
                else 60
            )
        )

        if len(prices) < effective_warmup + horizon + 10:
            result.errors.append(
                f"Insufficient data: {len(prices)} prices, need {effective_warmup + horizon + 10}"
            )
            return result

        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]

        def _compute_window_forecast(window_prices: list[float]) -> dict[str, object]:
            active_returns = np.array(
                [
                    (window_prices[i] - window_prices[i - 1]) / window_prices[i - 1]
                    for i in range(1, len(window_prices))
                ]
            )
            regimes = classify_regime_series(
                active_returns, return_threshold_multiplier=return_threshold_multiplier
            )
            P = estimate_transition_matrix(regimes, decay_rate=decay_rate)
            break_result = detect_structural_break(
                regimes,
                divergence_threshold=effective_break_divergence_threshold,
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
                    result.errors.append(
                        "HMM did not converge for one walk-forward window; using pure Markov forecast"
                    )

            garch_scales: list[float] | None = None
            garch_vol_applied = False
            if enable_garch_vol:
                log_returns = [
                    math.log(window_prices[i] / window_prices[i - 1])
                    for i in range(1, len(window_prices))
                ]
                opts: GarchClampOptions | None = None
                if garch_horizon_cap is not None or garch_regime_ceiling is not None:
                    opts = GarchClampOptions(
                        horizon_cap=garch_horizon_cap,
                        ceiling=garch_regime_ceiling or (1.5, 3.0),
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
            }

        for start in range(effective_warmup, len(prices) - horizon, stride):
            try:
                current_price = prices[start]
                realised_price = prices[start + horizon]
                realised_return = (realised_price - current_price) / current_price
                window_prices = prices[start - effective_warmup : start + 1]
                forecast_payload = _compute_window_forecast(window_prices)
                original_break_result = forecast_payload["break_result"]
                original_structural_break_detected = bool(original_break_result["detected"])
                original_structural_break_divergence = float(original_break_result["divergence"])
                structural_break_rerun_triggered = False

                if (
                    effective_post_break_short_window
                    and original_structural_break_detected
                    and len(window_prices) > max(30, effective_post_break_window_size)
                ):
                    structural_break_rerun_triggered = True
                    forecast_payload = _compute_window_forecast(
                        window_prices[-max(30, effective_post_break_window_size) :]
                    )

                break_result = forecast_payload["break_result"]
                entropy = forecast_payload["entropy"]
                entropy_z: float | None = None
                entropy_ci_scale = 1.0

                if enable_entropy_ci_modulation:
                    entropy_z = entropy_tracker.z_score(entropy.entropy_norm)
                    if entropy_z is not None:
                        entropy_ci_scale = entropy_z_to_ci_scale(entropy_z, entropy_kappa)

                p_up = float(forecast_payload["p_up"])
                predicted_return = float(forecast_payload["predicted_return"])
                ci_lower = float(forecast_payload["ci_lower"])
                ci_upper = float(forecast_payload["ci_upper"])

                if bool(break_result["detected"]):
                    center = (ci_lower + ci_upper) / 2.0
                    half_width = (ci_upper - ci_lower) / 2.0 * 1.5
                    ci_lower = center - half_width
                    ci_upper = center + half_width

                if entropy_ci_scale != 1.0:
                    center = (ci_lower + ci_upper) / 2.0
                    half_width = (ci_upper - ci_lower) / 2.0
                    ci_lower = center - half_width * entropy_ci_scale
                    ci_upper = center + half_width * entropy_ci_scale

                direction_correct = (p_up > 0.5 and realised_return > 0) or (p_up <= 0.5 and realised_return <= 0)
                in_ci = ci_lower <= realised_price <= ci_upper

                result.steps.append(
                    BacktestStep(
                        start_idx=start,
                        predicted_prob=float(p_up),
                        predicted_return=float(predicted_return),
                        ci_lower=float(ci_lower),
                        ci_upper=float(ci_upper),
                        realised_return=float(realised_return),
                        realised_price=float(realised_price),
                        direction_correct=bool(direction_correct),
                        in_ci=bool(in_ci),
                        garch_vol_applied=bool(forecast_payload["garch_vol_applied"]),
                        transition_entropy=float(entropy.entropy_nats),
                        transition_entropy_norm=float(entropy.entropy_norm),
                        transition_entropy_z=None if entropy_z is None else float(entropy_z),
                        entropy_ci_scale=float(entropy_ci_scale),
                        entropy_ci_modulation_applied=bool(abs(entropy_ci_scale - 1.0) > 1e-12),
                        structural_break_detected=bool(break_result["detected"]),
                        structural_break_rerun_triggered=bool(structural_break_rerun_triggered),
                        original_structural_break_detected=bool(original_structural_break_detected),
                        original_structural_break_divergence=float(original_structural_break_divergence),
                    )
                )
                entropy_tracker.push(entropy.entropy_norm)
            except Exception as e:
                result.errors.append(f"Step {start}: {e}")

        return result
