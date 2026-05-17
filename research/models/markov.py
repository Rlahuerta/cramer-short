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
    resolve_forecast_lab_runtime_asset_scope_for_ticker,
)

RegimeState = Literal["bull", "bear", "sideways"]
PredictionConfidenceMode = Literal["legacy", "rebalanced"]
BreakConfidencePolicy = Literal["default", "trend_penalty_only", "divergence_weighted"]
ActionRecommendation = Literal["BUY", "HOLD", "SELL"]
ActionConfidence = Literal["HIGH", "MEDIUM", "LOW"]
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

DEFAULT_DIVERGENCE_PENALTY_SCHEDULE: dict[str, float] = {
    "mild": 0.80,
    "medium": 0.70,
    "high": 0.60,
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
    return upper == "GLD"


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


def _is_finite_number(value: float | int | None) -> bool:
    return value is not None and math.isfinite(float(value))


def compute_divergence_penalty(
    divergence: float,
    schedule: dict[str, float] | None = None,
) -> float:
    effective_schedule = schedule or DEFAULT_DIVERGENCE_PENALTY_SCHEDULE
    if divergence < 0.05:
        return 1.0
    if divergence < 0.10:
        return float(effective_schedule["mild"])
    if divergence < 0.20:
        return float(effective_schedule["medium"])
    return float(effective_schedule["high"])


def should_emit_context_only_canonical(
    ticker: str,
    asset_type: str,
    horizon: int,
    prediction_confidence: float,
    out_of_sample_r2: float | None,
    anchor_quality: str,
    trusted_anchors: int,
    goodness_of_fit_passes: bool | None = None,
) -> bool:
    crypto_short_horizon = asset_type == "crypto" and horizon <= 14
    validation_unavailable = not _is_finite_number(out_of_sample_r2)
    supportive_anchors = anchor_quality == "good" and trusted_anchors >= 2
    fit_supportive = goodness_of_fit_passes if goodness_of_fit_passes is not None else True
    recommended_confidence_threshold = float(
        resolve_forecast_lab_markov_parameter_defaults(
            resolve_forecast_lab_runtime_asset_scope_for_ticker(ticker)
        )["recommendedConfidenceThreshold"]
    )

    return (
        crypto_short_horizon
        and validation_unavailable
        and supportive_anchors
        and fit_supportive
        and prediction_confidence >= recommended_confidence_threshold
    )


def _read_distribution_bound(
    point: dict,
    camel_key: str,
    snake_key: str,
    fallback: float,
) -> float:
    value = point.get(camel_key, point.get(snake_key, fallback))
    if value is None or not math.isfinite(float(value)):
        return fallback
    return float(value)


def _with_distribution_bounds(point: dict, lower: float, upper: float) -> dict:
    updated = {
        **point,
        "lowerBound": lower,
        "upperBound": upper,
    }
    if "lower_bound" in point:
        updated["lower_bound"] = lower
    if "upper_bound" in point:
        updated["upper_bound"] = upper
    return updated


def calibrate_probabilities(
    distribution: list[dict],
    *,
    ensemble_consensus: int = 0,
    historical_days: int = 60,
    hmm_converged: bool = False,
    base_rate: float = 0.5,
    kappa_multiplier: float = 1.0,
    current_regime: str | None = None,
    mature_bull_calibration_active: bool = False,
    current_price: float | None = None,
    drift_n: float | None = None,
    vol_n: float | None = None,
    nu: int = 5,
) -> list[dict]:
    from research.models.trajectory import student_t_ppf, student_t_survival

    consensus = ensemble_consensus
    n_days = historical_days
    center = max(0.25, min(0.80, base_rate))

    kappa = 0.45
    kappa -= consensus * 0.07
    if n_days > 60:
        kappa -= min(0.08, 0.04 * math.log2(n_days / 60))
    if hmm_converged:
        kappa -= 0.03
    kappa *= kappa_multiplier

    if mature_bull_calibration_active:
        kappa += 0.10
    elif current_regime in {"bull", "bear"}:
        kappa -= 0.03
    elif current_regime == "sideways":
        kappa += 0.03

    kappa = max(0.15, min(0.55, kappa))

    if (
        current_price is not None
        and drift_n is not None
        and vol_n is not None
        and vol_n > 0
    ):
        raw_p_up = student_t_survival(current_price, current_price, drift_n, vol_n, nu)
        target_p_up = max(0.01, min(0.99, kappa * center + (1 - kappa) * raw_p_up))
        scaled_vol = vol_n * math.sqrt((nu - 2) / nu) if nu > 2 else vol_n
        z_target = student_t_ppf(1 - target_p_up, nu)
        calibrated_drift = -z_target * scaled_vol

        calibrated: list[dict] = []
        for point in distribution:
            probability = float(point["probability"])
            source = point.get("source", "markov")
            if source == "markov":
                new_prob = student_t_survival(
                    float(point["price"]),
                    current_price,
                    calibrated_drift,
                    vol_n,
                    nu,
                )
            else:
                old_markov = student_t_survival(
                    float(point["price"]),
                    current_price,
                    drift_n,
                    vol_n,
                    nu,
                )
                new_markov = student_t_survival(
                    float(point["price"]),
                    current_price,
                    calibrated_drift,
                    vol_n,
                    nu,
                )
                new_prob = max(0.0, min(1.0, probability + (new_markov - old_markov)))

            delta = new_prob - probability
            new_lower = max(
                0.0,
                min(
                    1.0,
                    _read_distribution_bound(point, "lowerBound", "lower_bound", new_prob)
                    + delta,
                ),
            )
            new_upper = max(
                0.0,
                min(
                    1.0,
                    _read_distribution_bound(point, "upperBound", "upper_bound", new_prob)
                    + delta,
                ),
            )
            calibrated.append(
                _with_distribution_bounds(
                    {
                        **point,
                        "probability": new_prob,
                    },
                    min(new_lower, new_prob),
                    max(new_upper, new_prob),
                )
            )

        for index in range(len(calibrated) - 2, -1, -1):
            if calibrated[index]["probability"] < calibrated[index + 1]["probability"]:
                calibrated[index]["probability"] = calibrated[index + 1]["probability"]
                if calibrated[index]["upperBound"] < calibrated[index]["probability"]:
                    calibrated[index]["upperBound"] = calibrated[index]["probability"]
                if calibrated[index]["lowerBound"] > calibrated[index]["probability"]:
                    calibrated[index]["lowerBound"] = calibrated[index]["probability"]
                if "upper_bound" in calibrated[index]:
                    calibrated[index]["upper_bound"] = calibrated[index]["upperBound"]
                if "lower_bound" in calibrated[index]:
                    calibrated[index]["lower_bound"] = calibrated[index]["lowerBound"]
        return calibrated

    calibrated = []
    for point in distribution:
        probability = float(point["probability"])
        new_prob = kappa * center + (1 - kappa) * probability
        delta = new_prob - probability
        new_lower = max(
            0.0,
            min(
                1.0,
                _read_distribution_bound(point, "lowerBound", "lower_bound", new_prob)
                + delta,
            ),
        )
        new_upper = max(
            0.0,
            min(
                1.0,
                _read_distribution_bound(point, "upperBound", "upper_bound", new_prob)
                + delta,
            ),
        )
        calibrated.append(
            _with_distribution_bounds(
                {
                    **point,
                    "probability": new_prob,
                },
                min(new_lower, new_prob),
                max(new_upper, new_prob),
            )
        )

    for index in range(len(calibrated) - 2, -1, -1):
        if calibrated[index]["probability"] < calibrated[index + 1]["probability"]:
            calibrated[index]["probability"] = calibrated[index + 1]["probability"]

    return calibrated


def compute_prediction_confidence_breakdown(
    *,
    p_up: float,
    ensemble_consensus: int,
    hmm_converged: bool,
    regime_run_length: int,
    structural_break: bool,
    asset_type: Literal["etf", "equity", "crypto", "commodity"] | None = None,
    recent_vol: float | None = None,
    momentum_agreement: float | None = None,
    calibrated_p_up: float | None = None,
    base_rate: float | None = None,
    trusted_anchors: int | None = None,
    horizon_days: int | None = None,
    out_of_sample_r2: float | None = None,
    break_confidence_policy: BreakConfidencePolicy = "default",
    skip_sideways_break_penalty: bool = False,
    regime_state: RegimeState | None = None,
    structural_break_divergence: float | None = None,
    divergence_penalty_schedule: dict[str, float] | None = None,
    confidence_mode: PredictionConfidenceMode = "legacy",
    posterior_entropy: float | None = None,
) -> dict:
    mode = confidence_mode
    crypto_short_horizon = asset_type == "crypto" and horizon_days is not None and horizon_days <= 14
    anchors_helpful = (trusted_anchors or 0) >= 2
    r2 = out_of_sample_r2
    r2_unavailable = not _is_finite_number(r2)
    r2_clearly_bad = _is_finite_number(r2) and float(r2) < -0.05
    r2_near_zero = _is_finite_number(r2) and -0.02 <= float(r2) <= 0.02

    decisiveness = min(1.0, abs(p_up - 0.5) * 2)
    consensus_score = ensemble_consensus / 3
    hmm_score = 1.0 if hmm_converged else 0.0
    stability_score = min(1.0, regime_run_length / 20)
    momentum_agr = momentum_agreement or 0.0

    components = {
        "decisiveness": 0.30 * decisiveness,
        "ensembleConsensus": 0.15 * consensus_score,
        "hmmConvergence": 0.10 * hmm_score,
        "regimeStability": 0.15 * stability_score,
        "momentumAgreement": 0.10 * momentum_agr,
        "baseRateAlignment": 0.0,
        "nearZeroR2Bonus": 0.0,
        "anchorSupport": 0.0,
    }

    if calibrated_p_up is not None and base_rate is not None:
        pred_direction = 1 if calibrated_p_up >= 0.5 else -1
        base_direction = 1 if base_rate >= 0.5 else -1
        base_strength = abs(base_rate - 0.5) * 2
        if pred_direction == base_direction:
            components["baseRateAlignment"] = 0.20 * base_strength
        else:
            components["baseRateAlignment"] = -0.08 * base_strength

    if crypto_short_horizon and r2_near_zero:
        components["nearZeroR2Bonus"] = 0.08

    if mode == "rebalanced":
        if crypto_short_horizon:
            components["anchorSupport"] = 0.05 if anchors_helpful else 0.03
        elif anchors_helpful:
            components["anchorSupport"] = 0.02

    confidence = sum(components.values())

    structural_break_multiplier = 1.0
    if structural_break:
        skip_break_penalty = regime_state == "sideways" and (
            break_confidence_policy == "trend_penalty_only"
            or skip_sideways_break_penalty
        )
        if not skip_break_penalty:
            if break_confidence_policy == "divergence_weighted":
                break_penalty = compute_divergence_penalty(
                    structural_break_divergence or 0.20,
                    divergence_penalty_schedule,
                )
            else:
                break_penalty = (
                    0.85
                    if crypto_short_horizon and anchors_helpful and r2_unavailable
                    else 0.8
                    if crypto_short_horizon and anchors_helpful and not r2_clearly_bad
                    else 0.6
                )
            structural_break_multiplier = break_penalty
            confidence *= structural_break_multiplier

    asset_type_multiplier = 1.0
    if asset_type == "crypto":
        if mode == "rebalanced":
            if crypto_short_horizon:
                if anchors_helpful and r2_unavailable:
                    asset_type_multiplier = 0.95
                elif anchors_helpful and not r2_clearly_bad:
                    asset_type_multiplier = 0.92
                else:
                    asset_type_multiplier = 0.82
            else:
                asset_type_multiplier = 0.74
        else:
            asset_type_multiplier = 0.85 if crypto_short_horizon and anchors_helpful else 0.7
    elif asset_type == "commodity":
        asset_type_multiplier = 0.85
    elif asset_type == "etf":
        asset_type_multiplier = 1.1
    confidence *= asset_type_multiplier

    volatility_multiplier = 1.0
    if recent_vol is not None and recent_vol > 0.02:
        if mode == "rebalanced" and crypto_short_horizon:
            volatility_multiplier = max(0.85, 1 - (recent_vol - 0.02) * 3)
        else:
            volatility_multiplier = max(0.7, 1 - (recent_vol - 0.02) * 5)
        confidence *= volatility_multiplier

    normalization_multiplier = (
        1.6
        if mode == "rebalanced" and asset_type == "crypto" and crypto_short_horizon
        else 1.35
        if mode == "rebalanced" and asset_type == "crypto"
        else 1.0
    )
    confidence *= normalization_multiplier

    posterior_uncertainty_multiplier = (
        max(0.65, 1 - posterior_entropy * 0.35)
        if posterior_entropy is not None
        else 1.0
    )
    confidence *= posterior_uncertainty_multiplier

    return {
        "mode": mode,
        "total": max(0.0, min(1.0, confidence)),
        "components": components,
        "multipliers": {
            "structuralBreak": structural_break_multiplier,
            "assetType": asset_type_multiplier,
            "volatility": volatility_multiplier,
            "normalization": normalization_multiplier,
            "posteriorUncertainty": posterior_uncertainty_multiplier,
        },
    }


def compute_prediction_confidence(**kwargs) -> float:
    return float(compute_prediction_confidence_breakdown(**kwargs)["total"])


def compute_action_levels(
    distribution: list[dict],
    current_price: float,
) -> dict[str, float]:
    def find_price_at_prob(target_prob: float) -> float:
        if not distribution:
            return current_price
        for index in range(len(distribution) - 1):
            hi = distribution[index]
            lo = distribution[index + 1]
            hi_prob = float(hi["probability"])
            lo_prob = float(lo["probability"])
            if hi_prob >= target_prob >= lo_prob:
                if abs(hi_prob - lo_prob) < 1e-10:
                    return float(hi["price"])
                t = (hi_prob - target_prob) / (hi_prob - lo_prob)
                return float(hi["price"]) + t * (float(lo["price"]) - float(hi["price"]))
        if target_prob >= float(distribution[0]["probability"]):
            return float(distribution[0]["price"])
        return float(distribution[-1]["price"])

    return {
        "medianPrice": find_price_at_prob(0.50),
        "targetPrice": find_price_at_prob(0.30),
        "stopLoss": find_price_at_prob(0.90),
        "bullCase": find_price_at_prob(0.20),
        "bearCase": find_price_at_prob(0.80),
    }


def compute_action_signal(
    distribution: list[dict],
    current_price: float,
    buy_threshold: float = 0.05,
    sell_threshold: float = 0.03,
    horizon: int = 30,
    recent_vol: float | None = None,
    scenarios: dict | None = None,
    asset_type: Literal["etf", "equity", "crypto", "commodity"] | None = None,
) -> dict:
    from research.models.trajectory import interpolate_survival

    if not distribution:
        action_levels = compute_action_levels([], current_price)
        return {
            "buyProbability": 0.0,
            "holdProbability": 1.0,
            "sellProbability": 0.0,
            "recommendation": "HOLD",
            "baseRecommendation": "HOLD",
            "recommendationSource": "expected_return",
            "confidence": "LOW",
            "expectedReturn": 0.0,
            "riskRewardRatio": 1.0,
            "buyThreshold": buy_threshold,
            "sellThreshold": sell_threshold,
            "actionLevels": action_levels,
        }

    p_above_buy = interpolate_survival(distribution, current_price * (1 + buy_threshold))
    p_above_sell = interpolate_survival(distribution, current_price * (1 - sell_threshold))
    p_below_sell = 1 - p_above_sell
    p_hold = max(0.0, 1 - p_above_buy - p_below_sell)

    expected_price = 0.0
    for index in range(len(distribution) - 1):
        mass = float(distribution[index]["probability"]) - float(distribution[index + 1]["probability"])
        mid = (float(distribution[index]["price"]) + float(distribution[index + 1]["price"])) / 2
        expected_price += mass * mid
    expected_price += (1 - float(distribution[0]["probability"])) * float(distribution[0]["price"])
    expected_price += float(distribution[-1]["probability"]) * float(distribution[-1]["price"])

    expected_return = (
        (expected_price - current_price) / current_price
        if current_price > 0 and math.isfinite(expected_price)
        else 0.0
    )

    expected_upside = 0.0
    expected_downside = 0.0
    for index in range(len(distribution) - 1):
        mass = float(distribution[index]["probability"]) - float(distribution[index + 1]["probability"])
        mid = (float(distribution[index]["price"]) + float(distribution[index + 1]["price"])) / 2
        expected_upside += mass * max(0.0, mid - current_price)
        expected_downside += mass * max(0.0, current_price - mid)
    expected_downside += (1 - float(distribution[0]["probability"])) * max(
        0.0, current_price - float(distribution[0]["price"])
    )
    expected_upside += float(distribution[-1]["probability"]) * max(
        0.0, float(distribution[-1]["price"]) - current_price
    )

    raw_rrr = expected_upside / expected_downside if expected_downside > 0 else 1.0
    risk_reward_ratio = raw_rrr if math.isfinite(raw_rrr) else 1.0

    if recent_vol is not None and recent_vol > 0:
        vol_scaled = recent_vol * math.sqrt(horizon)
        action_buy_thr = max(0.001, 0.08 * vol_scaled)
        action_sell_thr = max(0.001, 0.06 * vol_scaled)
    else:
        action_buy_thr = 0.003 if horizon <= 7 else 0.005 if horizon <= 30 else 0.008
        action_sell_thr = 0.002 if horizon <= 7 else 0.003 if horizon <= 30 else 0.005

    if expected_return > action_buy_thr:
        recommendation: ActionRecommendation = "BUY"
    elif expected_return < -action_sell_thr:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    base_recommendation = recommendation
    recommendation_source = "expected_return"

    short_horizon_crypto = asset_type == "crypto" and horizon <= 14
    if short_horizon_crypto and recommendation == "HOLD" and scenarios:
        scenarios_p_up = float(scenarios.get("pUp", scenarios.get("p_up", 0.5)))
        if scenarios_p_up >= 0.55 and expected_return >= 0 and risk_reward_ratio >= 1:
            scenario_recommendation: ActionRecommendation = "BUY"
        elif scenarios_p_up <= 0.45 and expected_return <= 0 and risk_reward_ratio <= 1:
            scenario_recommendation = "SELL"
        else:
            scenario_recommendation = "HOLD"
        recommendation = scenario_recommendation
        if recommendation != base_recommendation:
            recommendation_source = "short_horizon_scenario"

    if scenarios:
        scenario_p_up = float(scenarios.get("pUp", scenarios.get("p_up", 0.5)))
        buckets = scenarios.get("buckets", [])
        up_scenarios = sum(float(bucket.get("probability", 0.0)) for bucket in buckets[3:5])
        down_scenarios = sum(float(bucket.get("probability", 0.0)) for bucket in buckets[0:2])

        if recommendation == "BUY":
            if scenario_p_up < 0.45 and risk_reward_ratio < 1:
                recommendation = "HOLD"
            elif down_scenarios > up_scenarios + 0.05:
                recommendation = "HOLD"
        elif recommendation == "SELL":
            if scenario_p_up > 0.55 and risk_reward_ratio > 1:
                recommendation = "HOLD"
            elif up_scenarios > down_scenarios + 0.05:
                recommendation = "HOLD"

    active_thr = action_buy_thr if recommendation == "BUY" else action_sell_thr
    conviction = abs(expected_return)
    confidence: ActionConfidence = (
        "HIGH"
        if conviction >= 2 * active_thr
        else "MEDIUM"
        if conviction >= active_thr
        else "LOW"
    )

    action_levels = compute_action_levels(distribution, current_price)
    median_return = (action_levels["medianPrice"] - current_price) / current_price if current_price > 0 else 0.0
    if (
        (expected_return > 0 and median_return < -0.005)
        or (expected_return < 0 and median_return > 0.005)
    ) and confidence == "HIGH":
        confidence = "MEDIUM"

    return {
        "buyProbability": p_above_buy,
        "holdProbability": p_hold,
        "sellProbability": p_below_sell,
        "recommendation": recommendation,
        "baseRecommendation": base_recommendation,
        "recommendationSource": recommendation_source,
        "confidence": confidence,
        "expectedReturn": expected_return,
        "riskRewardRatio": risk_reward_ratio,
        "buyThreshold": buy_threshold,
        "sellThreshold": sell_threshold,
        "actionLevels": action_levels,
    }


def cap_btc_short_horizon_confidence(
    ticker: str,
    horizon: int,
    structural_break_detected: bool,
    out_of_sample_r2: float | None,
    prediction_confidence: float,
    confidence: ActionConfidence,
) -> ActionConfidence:
    is_btc_short_horizon = is_btc_ticker_symbol(ticker) and horizon <= 14
    weak_validation = (
        _is_finite_number(out_of_sample_r2) and float(out_of_sample_r2) < 0.05
    ) or prediction_confidence < 0.40

    if (
        is_btc_short_horizon
        and structural_break_detected
        and weak_validation
        and confidence == "HIGH"
    ):
        return "MEDIUM"
    return confidence


def should_apply_btc14d_bearish_break_sell_gate(
    ticker: str,
    horizon: int,
    recommendation: ActionRecommendation,
    raw_recommendation: ActionRecommendation | None,
    structural_break_detected: bool,
    regime_state: RegimeState,
    prediction_confidence: float,
    raw_predicted_prob: float,
    predicted_prob: float,
    expected_return: float,
) -> bool:
    is_target_slice = is_btc_ticker_symbol(ticker) and horizon == 14
    return (
        is_target_slice
        and recommendation == "BUY"
        and raw_recommendation == "SELL"
        and structural_break_detected
        and regime_state != "bull"
        and prediction_confidence <= 0.09
        and raw_predicted_prob <= 0.50
        and predicted_prob <= 0.54
        and expected_return <= 0.025
    )
