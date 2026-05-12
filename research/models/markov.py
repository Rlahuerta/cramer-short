"""Markov regime model.

Mirrors TS logic:
  - 3-state regime classification (bull/bear/sideways)
  - Adaptive threshold = 0.5 * median(|returns|)
  - Transition matrix with Dirichlet smoothing + exponential decay
  - Structural break detection via Frobenius norm
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from research.models.soft_regime import blend_regime_mixtures
from research.utils.forecast_lab_runtime_defaults import (
    create_forecast_lab_asset_scoped_runtime_defaults,
    ForecastLabRuntimeAssetScope,
)

RegimeState = Literal["bull", "bear", "sideways"]
REGIME_STATES: list[RegimeState] = ["bull", "bear", "sideways"]
STATE_INDEX: dict[RegimeState, int] = {"bull": 0, "bear": 1, "sideways": 2}
NUM_STATES = len(REGIME_STATES)

BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS = 252
BTC_SHORT_HORIZON_LIVE_RERUN_WINDOW_DAYS = 60
BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT = 0.15

GOLD_SHORT_HORIZON_LIVE_HISTORY_DAYS = 252
GOLD_ULTRA_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT = 0.12
GOLD_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT = 0.15

FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS: dict[str, float | int | bool] = {
    "recommendedConfidenceThreshold": 0.22,
    "transitionMinObservations": 30,
    "transitionDecay": 0.97,
    "structuralBreakMinLength": 36,
    "momentumLookback": 10,
    "momentumAdjustmentScale": 0.25,
    "momentumAdjustmentClamp": 0.003,
    "trendPenaltyOnlyBreakConfidence": True,
    "divergenceWeightedBreakConfidence": False,
}

PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS: dict[str, float | int | bool] = {
    "transitionMinObservations": 31,
    "structuralBreakMinLength": 28,
    "momentumLookback": 9,
    "momentumAdjustmentScale": 0.252,
    "momentumAdjustmentClamp": 0.00305,
}

PROMOTED_HYPE_MARKOV_RUNTIME_DEFAULTS: dict[str, float | int | bool] = {
    "recommendedConfidenceThreshold": 0.15,
    "momentumAdjustmentScale": 0.48,
    "momentumAdjustmentClamp": 0.0058,
}

_forecast_lab_markov_runtime_defaults = create_forecast_lab_asset_scoped_runtime_defaults(
    FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS
)
_forecast_lab_markov_runtime_defaults.set("sol", PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS)
_forecast_lab_markov_runtime_defaults.set("hype", PROMOTED_HYPE_MARKOV_RUNTIME_DEFAULTS)


def resolve_forecast_lab_markov_parameter_defaults(
    asset_scope: ForecastLabRuntimeAssetScope | None = None,
) -> dict[str, float | int | bool]:
    return _forecast_lab_markov_runtime_defaults.resolve(asset_scope)


def get_forecast_lab_markov_runtime_defaults(
    asset_scope: ForecastLabRuntimeAssetScope,
) -> dict[str, float | int | bool] | None:
    return _forecast_lab_markov_runtime_defaults.get(asset_scope)


def set_forecast_lab_markov_runtime_defaults(
    asset_scope: ForecastLabRuntimeAssetScope,
    overrides: dict[str, float | int | bool] | None = None,
) -> None:
    _forecast_lab_markov_runtime_defaults.set(asset_scope, overrides)


@dataclass(frozen=True)
class BtcShortHorizonLivePolicy:
    history_days: int
    break_divergence_threshold: float
    rerun_on_break: bool
    rerun_window_days: int | None = None


@dataclass(frozen=True)
class GoldShortHorizonLivePolicy:
    history_days: int
    break_divergence_threshold: float
    rerun_on_break: Literal[False]


def is_btc_ticker_symbol(ticker: str) -> bool:
    upper = ticker.strip().upper()
    return upper in {"BTC", "BTC-USD"}


def get_btc_short_horizon_live_policy(
    ticker: str,
    horizon: int,
) -> BtcShortHorizonLivePolicy | None:
    if not is_btc_ticker_symbol(ticker) or horizon < 1 or horizon > 14:
        return None

    if horizon == 1:
        return BtcShortHorizonLivePolicy(
            history_days=BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
            break_divergence_threshold=0.10,
            rerun_on_break=True,
            rerun_window_days=BTC_SHORT_HORIZON_LIVE_RERUN_WINDOW_DAYS,
        )

    if horizon == 2:
        return BtcShortHorizonLivePolicy(
            history_days=BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
            break_divergence_threshold=BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
            rerun_on_break=True,
            rerun_window_days=120,
        )

    if horizon == 3:
        return BtcShortHorizonLivePolicy(
            history_days=BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
            break_divergence_threshold=BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
            rerun_on_break=True,
            rerun_window_days=45,
        )

    return BtcShortHorizonLivePolicy(
        history_days=BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
        break_divergence_threshold=0.08 if horizon == 14 else BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
        rerun_on_break=False,
    )


def is_gold_ticker_symbol(ticker: str) -> bool:
    upper = ticker.strip().upper()
    return upper in {"GLD", "GOLD"}


def get_gold_short_horizon_live_policy(
    ticker: str,
    horizon: int,
) -> GoldShortHorizonLivePolicy | None:
    if not is_gold_ticker_symbol(ticker) or horizon < 1 or horizon > 14:
        return None
    return GoldShortHorizonLivePolicy(
        history_days=GOLD_SHORT_HORIZON_LIVE_HISTORY_DAYS,
        break_divergence_threshold=(
            GOLD_ULTRA_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT
            if horizon <= 3
            else GOLD_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT
        ),
        rerun_on_break=False,
    )


def classify_regime(
    daily_return: float,
    return_threshold: float = 0.01,
) -> RegimeState:
    """Classify a single return into a regime state."""
    if daily_return > return_threshold:
        return "bull"
    if daily_return < -return_threshold:
        return "bear"
    return "sideways"


def classify_regime_series(
    returns: np.ndarray | pd.Series | list[float],
    return_threshold_multiplier: float = 0.5,
) -> list[RegimeState]:
    """Classify a return series into regime states with adaptive threshold.

    The threshold is set to 0.5 * median(|returns|), ensuring ~30-40%
    of days are bull, ~30-40% bear, regardless of asset volatility.
    """
    arr = np.asarray(returns)
    if len(arr) == 0:
        return []

    abs_returns = np.sort(np.abs(arr))
    median_abs = float(abs_returns[len(abs_returns) // 2])
    threshold = max(0.001, return_threshold_multiplier * median_abs)

    return [classify_regime(float(r), threshold) for r in arr]


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


def _winsorize(values: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """Clip values beyond n_sigma standard deviations from the mean."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return values
    lo = mean - n_sigma * std
    hi = mean + n_sigma * std
    return np.clip(values, lo, hi)


def estimate_regime_stats(
    returns: np.ndarray,
    states: list[RegimeState],
    max_daily_drift: float | None = None,
    min_obs_per_state: int = 5,
) -> dict[RegimeState, dict[str, float]]:
    """Bin returns by regime state, compute winsorized mean/std per state."""
    defaults: dict[RegimeState, dict[str, float]] = {
        "bull": {"meanReturn": 0.005, "stdReturn": 0.010},
        "bear": {"meanReturn": -0.005, "stdReturn": 0.012},
        "sideways": {"meanReturn": 0.000, "stdReturn": 0.006},
    }
    bins: dict[RegimeState, list[float]] = {"bull": [], "bear": [], "sideways": []}
    n = min(len(returns), len(states))
    for i in range(n):
        bins[states[i]].append(float(returns[i]))

    result = dict(defaults)
    for state, vals in bins.items():
        if len(vals) >= min_obs_per_state:
            arr = np.asarray(vals, dtype=float)
            cleaned = _winsorize(arr)
            mean = float(np.mean(cleaned))
            variance = float(np.mean((cleaned - mean) ** 2))
            if max_daily_drift is not None and max_daily_drift > 0:
                mean = max(-max_daily_drift, min(max_daily_drift, mean))
            result[state] = {"meanReturn": mean, "stdReturn": math.sqrt(variance)}
    return result


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


def compute_r2_os(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
) -> float:
    """Out-of-sample R²: 1 - SS_res / SS_tot.

    Returns 0 if fewer than 2 observations or zero total variance.
    """
    actuals = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)
    if len(actuals) < 2:
        return 0.0
    mean = float(np.mean(actuals))
    ss_res = float(np.sum((actuals - predicted) ** 2))
    ss_tot = float(np.sum((actuals - mean) ** 2))
    if ss_tot < 1e-14:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_validation_r2_os(
    asset_type: str,
    horizon: int,
    regime_seq: list[RegimeState],
    log_returns: np.ndarray,
    max_daily_drift: float | None = None,
    transition_decay_override: float | None = None,
) -> dict:
    """Walk-forward excess R²: Markov R² minus historical-mean baseline R².

    Two validation paths:
    - Horizon-return (crypto, horizon 7-14): expanding windows, cumulative returns
    - Daily-return (default): hold out last 20 observations, multi-step prediction
    """
    effective_decay = transition_decay_override if transition_decay_override is not None else 0.97
    returns = np.asarray(log_returns, dtype=float)

    use_horizon_validator = asset_type == "crypto" and 7 <= horizon <= 14
    if use_horizon_validator:
        heldout_days = min(len(returns) - 30, max(12 * horizon, 84))
        if len(returns) >= 30 + heldout_days:
            actuals: list[float] = []
            predicted: list[float] = []
            baseline_preds: list[float] = []
            start_idx = len(returns) - heldout_days
            step = max(1, horizon)

            for start in range(start_idx, len(returns) - horizon + 1, step):
                train_states = regime_seq[:start]
                train_returns = returns[:start]
                if len(train_states) < 20 or len(train_returns) < 20:
                    continue

                train_P = estimate_transition_matrix(train_states, None, 30, effective_decay)
                train_stats = estimate_regime_stats(train_returns, train_states, max_daily_drift)
                last_state = train_states[-1]
                train_mean = float(np.mean(train_returns))
                realized = float(np.sum(returns[start:start + horizon]))

                hv = _compute_horizon_drift_vol(horizon, train_P, train_stats, last_state)
                actuals.append(realized)
                predicted.append(hv["mu_n"])
                baseline_preds.append(horizon * train_mean)

            if len(actuals) >= 6:
                return {
                    "r2os": compute_r2_os(np.array(actuals), np.array(predicted))
                            - compute_r2_os(np.array(actuals), np.array(baseline_preds)),
                    "validation_metric": "horizon_return",
                }

    # Daily-return validator (default)
    min_held_out = 20
    if len(regime_seq) >= min_held_out + 30:
        train_states = regime_seq[:-min_held_out]
        train_returns = returns[:-min_held_out]
        test_returns = returns[-min_held_out:]

        train_P = estimate_transition_matrix(train_states, None, 30, effective_decay)
        train_stats = estimate_regime_stats(train_returns, train_states, max_daily_drift)
        train_mean = float(np.mean(train_returns))
        last_train_state = train_states[-1]

        preds = []
        for i in range(len(test_returns)):
            weights = _compute_terminal_state_weights(i + 1, train_P, last_train_state)
            p = sum(
                weights[j] * train_stats[state]["meanReturn"]
                for j, state in enumerate(REGIME_STATES)
            )
            preds.append(p)

        baseline = [train_mean] * min_held_out
        return {
            "r2os": compute_r2_os(test_returns, np.array(preds))
                    - compute_r2_os(test_returns, np.array(baseline)),
            "validation_metric": "daily_return",
        }

    return {"r2os": None, "validation_metric": "daily_return"}


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
    With ``NUM_STATES**2 = 9`` cells and the ≥5-expected-counts rule of thumb,
    each half needs ≥45 transitions; the TS default of 36 provides a practical
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


def compute_markov_forecast(
    transition_matrix: np.ndarray,
    current_regime: RegimeState,
    horizon: int,
    *,
    start_mixture: dict[RegimeState, float] | None = None,
    forecast_mixture: dict[RegimeState, float] | None = None,
    soft_transition_blend_weight: float = 0.0,
) -> dict[RegimeState, float]:
    """Compute regime probabilities at a given horizon via matrix exponentiation.

    Parameters
    ----------
    transition_matrix : np.ndarray
        3x3 transition matrix.
    current_regime : RegimeState
        Starting regime state.
    horizon : int
        Number of steps ahead.

    Additional Parameters
    ---------------------
    start_mixture : dict[RegimeState, float] | None
        Optional soft start-state mixture. When provided, replaces the hard
        one-hot current regime with a weighted combination of all start states.
    forecast_mixture : dict[RegimeState, float] | None
        Optional horizon-state mixture from an HMM posterior forecast.
    soft_transition_blend_weight : float
        Blend weight in [0, 1] used when combining the Markov transition
        forecast with the soft HMM forecast mixture.

    Returns
    -------
    dict[RegimeState, float]
        Probability of each regime at the horizon.
    """
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


# ---------------------------------------------------------------------------
# Entropy-based soft regime blending (Blake et al. 2510.03236, matching TS)
# ---------------------------------------------------------------------------


def compute_regime_entropy(
    mixture: dict[RegimeState, float],
) -> float:
    """Normalized entropy of a regime probability mixture in [0, 1].

    Uses ln(3) as the maximum-entropy baseline so the output is
    directly comparable across all 3-state mixture configurations.
    """
    import math
    denom = math.log(3)
    if denom <= 0:
        return 0.0
    entropy = 0.0
    for prob in mixture.values():
        if prob > 1e-12:
            entropy -= prob * math.log(prob)
    return min(1.0, entropy / denom)


def soft_regime_confidence_multiplier(entropy: float) -> float:
    """Confidence floor scaling: max(0.65, 1 - entropy * 0.35)."""
    return max(0.65, 1.0 - float(entropy) * 0.35)


def soft_regime_ci_scale(entropy: float) -> float:
    """CI width scaling: 1 + entropy * 0.35."""
    return 1.0 + float(entropy) * 0.35


def adjust_hmm_weight(
    hmm_weight: float,
    entropy: float,
) -> float:
    """Attenuate HMM influence under high posterior entropy.

    Formula: hmmWeight * max(0.5, 1 - entropy * 0.4).
    """
    return float(hmm_weight) * max(0.5, 1.0 - float(entropy) * 0.4)


# ---------------------------------------------------------------------------
# Model-only bypass + abstain decision functions (matching TS markov-distribution.ts)
# ---------------------------------------------------------------------------

COMMODITY_MODEL_ONLY_MIN_R2 = -0.02
COMMODITY_MODEL_ONLY_MIN_CONFIDENCE = 0.15
CRYPTO_MODEL_ONLY_MIN_R2 = -0.03
CRYPTO_MODEL_ONLY_MIN_CONFIDENCE = 0.18


def evaluate_validation_acceptability(
    out_of_sample_r2: float | None,
    asset_type: str,
    horizon: int,
    validation_metric: str,
    anchor_quality: str,
    trusted_anchors: int,
) -> bool:
    """Check whether Markov forecast passes out-of-sample validation.

    Matches TS ``validationAcceptable`` logic:
      - R² >= -0.01 (any asset)
      - R² >= -0.04 for crypto short-horizon with good anchors
      - R² >= -0.05 for crypto short-horizon with sparse anchors
    """
    has_r2 = out_of_sample_r2 is not None and math.isfinite(out_of_sample_r2)
    has_positive_r2 = has_r2 and out_of_sample_r2 >= -0.01  # type: ignore[operator]

    crypto_short = asset_type == "crypto" and 1 <= horizon <= 14
    r2_neutral_crypto = (
        crypto_short
        and validation_metric == "horizon_return"
        and anchor_quality == "good"
        and trusted_anchors >= 2
        and has_r2
        and out_of_sample_r2 >= -0.04  # type: ignore[operator]
    )
    sparse_crypto_allowed = (
        crypto_short
        and validation_metric == "horizon_return"
        and anchor_quality == "sparse"
        and trusted_anchors >= 1
        and has_r2
        and out_of_sample_r2 >= -0.05  # type: ignore[operator]
    )
    return has_positive_r2 or r2_neutral_crypto or sparse_crypto_allowed


def evaluate_model_only_bypass(
    asset_type: str,
    horizon: int,
    trusted_anchors: int,
    out_of_sample_r2: float | None,
    prediction_confidence: float,
) -> dict[str, bool]:
    """Determine whether commodity or crypto model-only emission bypass applies.

    Returns ``{"commodity_model_only": bool, "crypto_model_only": bool}``.
    """
    has_r2 = out_of_sample_r2 is not None and math.isfinite(out_of_sample_r2)

    commodity_model_only = (
        asset_type == "commodity"
        and trusted_anchors == 0
        and has_r2
        and out_of_sample_r2 >= COMMODITY_MODEL_ONLY_MIN_R2  # type: ignore[operator]
        and prediction_confidence >= COMMODITY_MODEL_ONLY_MIN_CONFIDENCE
    )

    crypto_model_only = (
        asset_type == "crypto"
        and 1 <= horizon <= 14
        and trusted_anchors == 0
        and has_r2
        and out_of_sample_r2 >= CRYPTO_MODEL_ONLY_MIN_R2  # type: ignore[operator]
        and prediction_confidence >= CRYPTO_MODEL_ONLY_MIN_CONFIDENCE
    )

    return {"commodity_model_only": commodity_model_only, "crypto_model_only": crypto_model_only}


def is_composite_validation_acceptable(
    out_of_sample_r2: float | None,
    asset_type: str,
    horizon: int,
    validation_metric: str,
    anchor_quality: str,
    trusted_anchors: int,
    prediction_confidence: float,
    goodness_of_fit_passes: bool | None = None,
) -> bool:
    """Full composite validation gate matching TS ``isCompositeValidationAcceptable``.

    Includes the ``compositeCryptoValidation`` path (R² ≥ -0.06,
    confidence ≥ 0.16, goodness-of-fit passing) that the simpler
    ``evaluate_validation_acceptability`` omits.
    """
    has_r2 = out_of_sample_r2 is not None and math.isfinite(out_of_sample_r2)
    has_positive_r2 = has_r2 and out_of_sample_r2 >= -0.01  # type: ignore[operator]
    if has_positive_r2:
        return True

    crypto_short = asset_type == "crypto" and 1 <= horizon <= 14
    r2_neutral_crypto = (
        crypto_short
        and validation_metric == "horizon_return"
        and anchor_quality == "good"
        and trusted_anchors >= 2
        and has_r2
        and out_of_sample_r2 >= -0.04  # type: ignore[operator]
    )
    if r2_neutral_crypto:
        return True

    sparse_crypto_allowed = (
        crypto_short
        and validation_metric == "horizon_return"
        and anchor_quality == "sparse"
        and trusted_anchors >= 1
        and has_r2
        and out_of_sample_r2 >= -0.05  # type: ignore[operator]
    )
    if sparse_crypto_allowed:
        return True

    composite_crypto_validation = (
        crypto_short
        and validation_metric == "horizon_return"
        and anchor_quality in ("good", "sparse")
        and trusted_anchors >= 1
        and has_r2
        and out_of_sample_r2 >= -0.06  # type: ignore[operator]
        and prediction_confidence >= 0.16
        and (goodness_of_fit_passes if goodness_of_fit_passes is not None else True)
    )
    return composite_crypto_validation


def evaluate_can_emit_canonical(
    trusted_anchors: int,
    anchor_quality: str,
    validation_acceptable: bool,
    commodity_model_only: bool,
    crypto_model_only: bool,
    sparse_crypto_anchor_allowed: bool = False,
) -> bool:
    """Final emission gate: can the Markov forecast be emitted?

    Matches TS ``canEmitCanonical``:
      - Good anchors + trusted count > 0 + validation acceptable, OR
      - Sparse crypto anchor allowed, OR
      - Model-only bypass (commodity or crypto)
    """
    return (
        (trusted_anchors > 0 and anchor_quality == "good" and validation_acceptable)
        or sparse_crypto_anchor_allowed
        or commodity_model_only
        or crypto_model_only
    )
