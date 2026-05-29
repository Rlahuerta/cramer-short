"""Walk-forward backtest harness.

Slides a window over historical prices, estimates the Markov model at each
step, records predictions vs realised outcomes, and aggregates results.
"""

from __future__ import annotations

# import math
# import numpy as np

from research.backtest._config import BacktestStep, WalkForwardResult
from research.backtest._window_forecaster import compute_window_forecast
from research.backtest._ci_transformer import modulate_ci_by_entropy, widen_for_structural_break
from research.models.markov import get_btc_short_horizon_live_policy
from research.models.transition_entropy import (EntropyZScoreTracker, entropy_z_to_ci_scale,)
from research.utils.forecast_lab_runtime_defaults import (
    forecast_lab_runtime_asset_scope,
    resolve_forecast_lab_runtime_asset_scope_for_ticker,
)


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
        effective = _resolve_effective_config(
            btc_break_divergence_threshold=btc_break_divergence_threshold,
            break_divergence_threshold=break_divergence_threshold,
            post_break_short_window=post_break_short_window,
            post_break_window_size=post_break_window_size,
            warmup=warmup,
            btc_live_policy=btc_live_policy,
        )

        if len(prices) < effective["warmup"] + horizon + 10:
            result.errors.append(
                f"Insufficient data: {len(prices)} prices, "
                f"need {effective['warmup'] + horizon + 10}"
            )
            return result

        for start in range(effective["warmup"], len(prices) - horizon, stride):
            try:
                current_price = prices[start]
                realised_price = prices[start + horizon]
                realised_return = (realised_price - current_price) / current_price
                window_prices = prices[start - effective["warmup"] : start + 1]

                wf = compute_window_forecast(
                    window_prices,
                    horizon=horizon,
                    return_threshold_multiplier=return_threshold_multiplier,
                    decay_rate=decay_rate,
                    break_divergence_threshold=effective["break_divergence"],
                    use_hmm=use_hmm,
                    asset_profile=asset_profile,
                    enable_garch_vol=enable_garch_vol,
                    garch_horizon=garch_horizon_cap,
                    garch_ceiling=garch_regime_ceiling,
                )

                original_structural_break_detected = bool(wf["original_break_result"]["detected"])
                original_structural_break_divergence = float(wf["original_break_result"]["divergence"])
                structural_break_rerun_triggered = False

                if (
                    effective["post_break_short_window"]
                    and original_structural_break_detected
                    and len(window_prices) > max(30, effective["post_break_window_size"])
                ):
                    structural_break_rerun_triggered = True
                    wf = compute_window_forecast(
                        window_prices[-max(30, effective["post_break_window_size"]):],
                        horizon=horizon,
                        return_threshold_multiplier=return_threshold_multiplier,
                        decay_rate=decay_rate,
                        break_divergence_threshold=effective["break_divergence"],
                        use_hmm=use_hmm,
                        asset_profile=asset_profile,
                        enable_garch_vol=enable_garch_vol,
                        garch_horizon=garch_horizon_cap,
                        garch_ceiling=garch_regime_ceiling,
                    )
                    wf["break_rerun_triggered"] = True

                break_result = wf["break_result"]
                entropy = wf["entropy"]
                entropy_z = None
                entropy_ci_scale = 1.0

                if enable_entropy_ci_modulation:
                    entropy_z = entropy_tracker.z_score(entropy.entropy_norm)
                    if entropy_z is not None:
                        entropy_ci_scale = entropy_z_to_ci_scale(entropy_z, entropy_kappa)

                p_up = float(wf["p_up"])
                predicted_return = float(wf["predicted_return"])
                ci_lower = float(wf["ci_lower"])
                ci_upper = float(wf["ci_upper"])

                if bool(break_result["detected"]):
                    ci_lower, ci_upper = widen_for_structural_break(ci_lower, ci_upper)

                ci_lower, ci_upper = modulate_ci_by_entropy(ci_lower, ci_upper, entropy_ci_scale)

                direction_correct = (
                    (p_up > 0.5 and realised_return > 0)
                    or (p_up <= 0.5 and realised_return <= 0)
                )
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
                        garch_vol_applied=bool(wf["garch_vol_applied"]),
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


def _resolve_effective_config(
    btc_break_divergence_threshold: float | None,
    break_divergence_threshold: float,
    post_break_short_window: bool | None,
    post_break_window_size: int | None,
    warmup: int,
    btc_live_policy: object | None,
) -> dict:
    """Merge explicit args with live-policy defaults into effective backtest config."""
    effective_warmup = btc_live_policy.history_days if btc_live_policy else warmup
    effective_break_divergence = (
        btc_break_divergence_threshold
        if btc_break_divergence_threshold is not None
        else (btc_live_policy.break_divergence_threshold if btc_live_policy else break_divergence_threshold)
    )
    effective_post_break_short_window = (
        post_break_short_window
        if post_break_short_window is not None
        else (btc_live_policy.rerun_on_break if btc_live_policy else False)
    )
    effective_post_break_window_size = (
        post_break_window_size
        if post_break_window_size is not None
        else (btc_live_policy.rerun_window_days
              if btc_live_policy and btc_live_policy.rerun_window_days is not None
              else 60)
    )
    return {
        "warmup": effective_warmup,
        "break_divergence": effective_break_divergence,
        "post_break_short_window": effective_post_break_short_window,
        "post_break_window_size": effective_post_break_window_size,
    }
