"""Walk-forward backtest harness for the Markov regime forecast tool.

Slides a rolling window across historical price data, generates a Markov
forecast at each position, and scores every prediction against the known
realised outcome.

Architecture
------------
The module delegates heavy computation to two sub-modules and one shared
config layer so the orchestrator stays focused on window-sliding and step
recording:

    research/backtest/walk_forward.py       ← this file (orchestrator)
    research/backtest/_config.py            → BacktestStep, WalkForwardResult
    research/backtest/_window_forecaster.py → per-window Markov → trajectory
    research/backtest/_ci_transformer.py    → break widening, entropy modulation

Optional features activated by boolean flags include HMM blending
(``use_hmm``), GARCH volatility scaling (``enable_garch_vol``), and
transition-entropy CI modulation (``enable_entropy_ci_modulation``).

This module is a Python mirror of the TypeScript engine at
``src/tools/finance/backtest/walk-forward.ts``.  The two implementations
produce comparable results on the same price fixtures.
"""

from __future__ import annotations

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
        Historical close prices, oldest first.
    horizon : int
        Forecast horizon in trading days (default 7).
    warmup : int
        Minimum history in days before the first prediction (default 120).
    stride : int
        Days between consecutive test windows (default 10).
    ticker : str or None
        Ticker label used for live-policy selection and runtime defaults.
    return_threshold_multiplier : float
        Multiplier on median absolute return for regime classification
        (default 0.5).
    decay_rate : float
        Exponential decay weight for transition matrix counts
        (default 0.97).
    break_divergence_threshold : float
        Frobenius-divergence threshold for structural-break detection
        (default 0.05).
    btc_break_divergence_threshold : float or None
        BTC-specific override for the break detection threshold.  When
        provided, it takes precedence over ``break_divergence_threshold``
        and any live-policy default.
    post_break_short_window : bool or None
        When True (and a break is detected), rerun the forecast on a
        shorter recent window.  When None, inherits from the live policy.
    post_break_window_size : int or None
        Window size in days for the post-break rerun.  When None,
        inherits from the live policy (defaults to 60).
    use_live_btc_short_horizon_policy : bool
        Apply mirrored BTC short-horizon parameter defaults that override
        warmup, break thresholds, and rerun settings (default False).
    use_hmm : bool
        Blend a 3-state Gaussian HMM (Baum-Welch) into the trajectory
        drift and volatility (default False).
    asset_profile : str
        HMM weight profile: ``"etf"``, ``"equity"``, ``"crypto"``, or
        ``"commodity"`` (default ``"crypto"``).
    enable_garch_vol : bool
        Apply GARCH(1,1) volatility scaling in the trajectory Monte Carlo
        (default False).
    garch_horizon_cap : int or None
        Beyond this day count the GARCH scalar soft-blends toward 1.0
        (default None — no cap).
    garch_regime_ceiling : tuple[float, float] or None
        ``(calm_ceiling, turbulent_ceiling)`` caps for the GARCH scalar.
        None means the static 3.0 cap is used.
    enable_entropy_ci_modulation : bool
        Scale confidence intervals by a rolling z-score of the transition
        entropy (default False).
    entropy_window_size : int
        Rolling history size for the entropy z-score tracker (default 60).
    entropy_kappa : float
        Sensitivity of CI width to the entropy z-score (default 0.15).

    Returns
    -------
    WalkForwardResult
        Collected ``BacktestStep`` records and any errors encountered.

    Notes
    -----
    The pipeline inside the loop has four phases:

    1. **Forecast** — ``_window_forecaster.compute_window_forecast`` runs
       the full Markov pipeline (regime, transition, trajectory).
    2. **Break rerun** — if a structural break is detected and
       ``post_break_short_window`` is active, the forecast is rerun on a
       shorter recent window.
    3. **CI transformation** — if a break was detected, the CI is
       widened 1.5×.  If entropy modulation is enabled, the CI half-width
       is scaled by the entropy-derived factor.
    4. **Scoring** — the forecast is compared against the realised
       outcome and recorded as a ``BacktestStep``.

    Error handling
    --------------
    Each window step is wrapped in try/except.  Unhandled exceptions are
    appended to ``WalkForwardResult.errors`` with the failing step index.
    The loop continues to the next window — a single bad step does not
    abort the full backtest.

    Examples
    --------
    >>> result = walk_forward(prices, horizon=7, warmup=120, stride=10)
    >>> assert len(result.errors) == 0
    >>> print(f"{len(result.steps)} steps, Brier={brier_score(result.steps):.3f}")
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
    """Resolve effective backtest parameters from explicit args and live policy.

    Parameters follow a precedence chain: explicit argument > live-policy
    default > hard-coded fallback.

    Parameters
    ----------
    btc_break_divergence_threshold : float or None
        Explicit BTC break-threshold override.
    break_divergence_threshold : float
        Default break threshold when no override is active.
    post_break_short_window : bool or None
        Explicit toggle for post-break short-window reruns.
    post_break_window_size : int or None
        Explicit short-window size.
    warmup : int
        Default warmup window length.
    btc_live_policy : object or None
        BTC live-policy dataclass with ``history_days``,
        ``break_divergence_threshold``, ``rerun_on_break``, and
        ``rerun_window_days`` attributes; ``None`` when the flag is off.

    Returns
    -------
    dict
        Keys: ``warmup``, ``break_divergence``, ``post_break_short_window``,
        ``post_break_window_size``.
    """
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
