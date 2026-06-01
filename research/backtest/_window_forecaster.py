"""Per-window Markov forecaster: regime -> transition -> trajectory -> forecast.

This module owns the single-window forecasting stage used by the
walk-forward harness.  It treats recent price returns as observations of a
finite-state Markov process, classifies each return into a bull/bear/sideways
regime, estimates an exponentially weighted transition matrix, and propagates
that transition structure through the Monte Carlo trajectory engine.  The
returned ``p_up``, expected return, and confidence interval therefore come from
the same terminal trajectory distribution rather than from separate probability
and interval estimators.

The Markov-probability literature review under ``references/markov-probability``
motivates three design cautions reflected here: transition matrices are
estimates rather than exact probabilities, latent-state models such as HMMs
should be treated as optional noisy-emission overlays, and nonstationarity or
structural breaks must be surfaced as diagnostics rather than hidden behind a
single point forecast.
"""

from __future__ import annotations

import math
import re
from typing import TypedDict

import numpy as np

from research.models.markov import (
    classify_regime_series,
    detect_structural_break,
    estimate_transition_matrix,
    compute_markov_forecast,
    estimate_regime_stats,
)
from research.models.garch_scales import GarchClampOptions, compute_garch_scales
from research.models.hmm import ASSET_PROFILES, baum_welch, fit_volatility_hmm, predict
from research.models.transition_entropy import compute_transition_entropy
from research.models.trajectory import RegimeStats, compute_trajectory
from research.models.transition_entropy import EntropyZScoreTracker, entropy_z_to_ci_scale

HMM_OPTIONAL_ERRORS = (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError)
HMM_OPTIONAL_ERROR_PATTERN = re.compile(
    r"\b(hmm|hidden markov|baum[- ]?welch|transition matrix|covariance)\b",
    re.IGNORECASE,
)


class WindowForecast(TypedDict):
    """Forecast payload emitted for one rolling backtest window.

    Fields carry both the tradable forecast outputs and the diagnostics needed
    by the orchestrator.  ``p_up`` is the horizon probability of a positive
    return, ``predicted_return`` is the horizon expected return from the
    simulated trajectory, and ``ci_lower``/``ci_upper`` are price-level bounds.
    ``entropy`` summarizes transition-matrix uncertainty, while
    ``break_result`` and ``original_break_result`` preserve structural-break
    metadata for any downstream confidence-interval widening or short-window
    rerun.
    """

    p_up: float
    predicted_return: float
    ci_lower: float
    ci_upper: float
    garch_vol_applied: bool
    entropy: object
    break_result: object
    break_rerun_triggered: bool
    original_break_result: object
    hmm_status: str


def _positive_window_std_floor(log_returns: np.ndarray) -> float:
    finite_returns = log_returns[np.isfinite(log_returns)]
    if finite_returns.size >= 2:
        window_std = float(np.std(finite_returns))
        if math.isfinite(window_std) and window_std > 0:
            return window_std
    return 1e-6


def _adapt_sparse_regime_stats(
    raw_stats: dict[str, dict[str, float]],
    log_returns: np.ndarray,
) -> dict[str, RegimeStats]:
    std_fallback = _positive_window_std_floor(log_returns)
    regime_stats: dict[str, RegimeStats] = {}
    for state in ["bull", "bear", "sideways"]:
        std_return = float(raw_stats[state]["stdReturn"])
        if not math.isfinite(std_return) or std_return <= 0:
            std_return = std_fallback
        regime_stats[state] = RegimeStats(
            mean_return=raw_stats[state]["meanReturn"],
            std_return=std_return,
        )
    return regime_stats


def _usable_hmm_overlay_components(
    drift: float,
    volatility: float,
    scale: float,
    weight: float,
) -> bool:
    return (
        math.isfinite(drift)
        and math.isfinite(volatility)
        and volatility >= 0
        and math.isfinite(scale)
        and scale > 0
        and math.isfinite(volatility * scale)
        and math.isfinite(weight)
        and 0 <= weight <= 1
    )


def _is_recoverable_hmm_overlay_error(exc: BaseException) -> bool:
    if not isinstance(exc, HMM_OPTIONAL_ERRORS):
        return False
    return bool(HMM_OPTIONAL_ERROR_PATTERN.search(str(exc)))


def _combine_uncertainty_ci_scale(
    structural_scale: float = 1.0,
    entropy_scale: float = 1.0,
) -> float:
    """Combine CI scales without letting entropy undo required widening."""
    if not math.isfinite(structural_scale) or structural_scale <= 0:
        structural_scale = 1.0
    if not math.isfinite(entropy_scale) or entropy_scale <= 0:
        entropy_scale = 1.0
    combined = structural_scale * entropy_scale
    return max(1.0, structural_scale, combined)


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
    entropy_tracker: EntropyZScoreTracker | None = None,
    entropy_kappa: float = 0.15,
) -> WindowForecast:
    """Compute one horizon forecast from a rolling price window.

    Parameters
    ----------
    window_prices : list[float]
        Ordered price history for the current backtest window.  The final
        element is the forecast origin price used to convert trajectory output
        into ``predicted_return``.
    horizon : int
        Number of future bars/days to project the Markov state distribution and
        price trajectory.
    return_threshold_multiplier : float
        Multiplier passed to regime classification.  It controls how large a
        return must be, relative to the rolling return scale, before it is
        labeled bull or bear rather than sideways.
    decay_rate : float
        Exponential decay factor for transition counts and empirical up-rates.
        Values closer to 1.0 preserve more long-window history; lower values
        emphasize the most recent regime transitions.
    break_divergence_threshold : float
        Divergence threshold for detecting a structural break between recent
        and longer-window transition behavior.
    use_hmm : bool
        When True, fit a 3-state Gaussian HMM with Baum-Welch and blend its
        drift/volatility forecast into the trajectory if the fit converges.
    asset_profile : str
        Asset profile key used to scale the optional HMM overlay weight.
    enable_garch_vol : bool
        When True, apply GARCH-derived volatility scaling to the Monte Carlo
        trajectory.
    garch_horizon : int or None
        Optional cap after which GARCH scaling soft-blends back toward 1.0.
    garch_ceiling : tuple[float, float] or None
        Optional calm/turbulent ceiling pair for GARCH volatility scaling.
    entropy_tracker : EntropyZScoreTracker or None
        Optional rolling entropy tracker from walk-forward orchestrator.
        When provided, enables CI modulation via entropy z-score.
    entropy_kappa : float
        Sensitivity of CI width to the entropy z-score (default 0.15).

    Returns
    -------
    WindowForecast
        Horizon probability, expected return, price confidence interval, and
        diagnostics for transition entropy and structural breaks.

    Notes
    -----
    The forecast uses a finite-state Markov abstraction: once returns are
    classified into regimes, the current regime and estimated transition matrix
    drive the n-step state distribution.  The returned confidence interval is a
    simulation product, not a posterior credible interval over the transition
    matrix itself; callers should use the entropy and structural-break metadata
    when interpreting forecast reliability.
    """
    # Single setup path: compute simple returns for regime classification,
    # log returns for stats/up-rate/trajectory consistency
    active_returns = np.array(
        [(window_prices[i] - window_prices[i - 1]) / window_prices[i - 1]
         for i in range(1, len(window_prices))]
    )
    log_returns = np.log(1.0 + active_returns)

    # Classify regimes once
    regimes = classify_regime_series(active_returns, return_threshold_multiplier=return_threshold_multiplier)
    current_regime = regimes[-1] if regimes else "sideways"

    # Estimate transition matrix once
    P = estimate_transition_matrix(regimes, decay_rate=decay_rate)

    # Detect structural break once
    break_result = detect_structural_break(
        regimes,
        divergence_threshold=break_divergence_threshold,
        decay_rate=decay_rate,
    )

    # Compute Markov forecast once
    # forecast = compute_markov_forecast(P, current_regime, horizon)

    # Use robust estimate_regime_stats with log returns
    raw_stats = estimate_regime_stats(log_returns, regimes, min_obs_per_state=1)
    regime_stats = _adapt_sparse_regime_stats(raw_stats, log_returns)

    hmm_override: dict[str, float] | None = None
    hmm_status = "disabled"

    if use_hmm:
        hmm_status = "not_attempted"
        try:
            hmm_result = baum_welch(
                active_returns,
                n_states=3,
                max_iterations=50,
                tolerance=1e-3,
            )
        except HMM_OPTIONAL_ERRORS as exc:
            if not _is_recoverable_hmm_overlay_error(exc):
                raise
            hmm_status = f"error:{type(exc).__name__}"

        else:
            hmm_status = "non_converged"

            if hmm_result.converged:
                try:
                    hmm_pred = predict(
                        active_returns, hmm_result.params, forecast_horizon=horizon
                    )
                    vol_scale = fit_volatility_hmm(
                        active_returns, vol_window=5, n_states=2
                    )
                except HMM_OPTIONAL_ERRORS as exc:
                    if not _is_recoverable_hmm_overlay_error(exc):
                        raise
                    hmm_status = f"error:{type(exc).__name__}"
                else:
                    profile = ASSET_PROFILES.get(asset_profile, ASSET_PROFILES["crypto"])
                    hmm_weight = float(np.clip(profile.hmm_weight_multiplier * 0.5, 0.0, 1.0))
                    if _usable_hmm_overlay_components(
                        hmm_pred.expected_return,
                        hmm_pred.expected_volatility,
                        vol_scale,
                        hmm_weight,
                    ):
                        # HMM predict returns per-step emission values (daily drift/vol)
                        hmm_override = {
                            "drift": hmm_pred.expected_return,
                            "vol": hmm_pred.expected_volatility * vol_scale,
                            "weight": hmm_weight,
                        }
                        hmm_status = "override_applied"
                    else:
                        hmm_status = "non_finite"

    garch_scales: list[float] | None = None
    garch_vol_applied = False

    if enable_garch_vol:
        log_returns_for_garch = [
            math.log(window_prices[i] / window_prices[i - 1])
            for i in range(1, len(window_prices))
        ]
        opts: GarchClampOptions | None = None
        if garch_horizon is not None or garch_ceiling is not None:
            opts = GarchClampOptions(horizon_cap=garch_horizon, ceiling=garch_ceiling or (1.5, 3.0),)

        scales = compute_garch_scales(log_returns_for_garch, horizon, opts)
        if scales:
            garch_scales = scales
            garch_vol_applied = True

    # Phase 4 — derive combined uncertainty CI scale from structural break
    # divergence and transition entropy z-score. This scale will be passed
    # directly into compute_trajectory to widen CIs at source, avoiding
    # double-widening in walk_forward.py.
    transition_entropy_result = compute_transition_entropy(P)
    structural_scale = 1.0
    if break_result["detected"]:
        # Structural break contributes monotonic widening based on divergence.
        # Formula: clamp(1 + divergence * 2.5, 1.0, 1.5) maps threshold 0.05 → 1.125, 0.20 → 1.5.
        # This matches legacy widen_for_structural_break (1.5x) at high divergence.
        divergence = float(break_result["divergence"])
        structural_scale = max(1.0, min(1.5, 1.0 + divergence * 2.5))

    entropy_scale = 1.0
    if entropy_tracker is not None:
        # Entropy z-score contributes via entropy_z_to_ci_scale (bounded [0.7, 1.4]).
        # Uses rolling z-score from orchestrator's history tracker.
        entropy_z = entropy_tracker.z_score(transition_entropy_result.entropy_norm)
        if entropy_z is not None:
            entropy_scale = entropy_z_to_ci_scale(entropy_z, kappa=entropy_kappa)

    uncertainty_ci_scale = _combine_uncertainty_ci_scale(
        structural_scale=structural_scale,
        entropy_scale=entropy_scale,
    )

    traj = compute_trajectory(
        window_prices[-1],
        horizon,
        P,
        regime_stats,
        current_regime,
        hmm_override=hmm_override,
        n_samples=500,
        garch_scales=garch_scales,
        uncertainty_ci_scale=uncertainty_ci_scale,
    )
    horizon_point = traj[-1]

    predicted_return = (horizon_point.expected_price - window_prices[-1]) / window_prices[-1]

    # Use trajectory's p_up directly for probability coherence
    p_up = horizon_point.p_up

    return {
        "p_up": float(p_up),
        "predicted_return": float(predicted_return),
        "ci_lower": float(horizon_point.lower_bound),
        "ci_upper": float(horizon_point.upper_bound),
        "garch_vol_applied": bool(garch_vol_applied),
        "entropy": transition_entropy_result,
        "break_result": break_result,
        "break_rerun_triggered": False,
        "original_break_result": break_result,
        "hmm_status": hmm_status,
    }
