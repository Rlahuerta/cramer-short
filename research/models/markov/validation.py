"""Out-of-sample R² and validation gates.

Mirrors TS logic:
  - Walk-forward excess R²
  - Asset-specific validation thresholds
"""

from __future__ import annotations

import math

import numpy as np

from research.models.markov.core import RegimeState
from research.models.markov.forecast import _compute_horizon_drift_vol, _compute_terminal_state_weights
from research.models.markov.regime import estimate_regime_stats
from research.models.markov.transition import estimate_transition_matrix


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
    """Walk-forward excess R²: Markov R² minus historical-mean baseline R²."""
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
                for j, state in enumerate({"bull": 0, "bear": 1, "sideways": 2})
            )
            preds.append(p)

        baseline = [train_mean] * min_held_out
        return {
            "r2os": compute_r2_os(test_returns, np.array(preds))
                    - compute_r2_os(test_returns, np.array(baseline)),
            "validation_metric": "daily_return",
        }

    return {"r2os": None, "validation_metric": "daily_return"}


def evaluate_validation_acceptability(
    out_of_sample_r2: float | None,
    asset_type: str,
    horizon: int,
    validation_metric: str,
    anchor_quality: str,
    trusted_anchors: int,
) -> bool:
    """Check whether Markov forecast passes out-of-sample validation."""
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
    """Full composite validation gate."""
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


def evaluate_model_only_bypass(
    asset_type: str,
    horizon: int,
    trusted_anchors: int,
    out_of_sample_r2: float | None,
    prediction_confidence: float,
) -> dict[str, bool]:
    """Determine whether commodity or crypto model-only emission bypass applies."""
    from research.models.markov.core import (
        COMMODITY_MODEL_ONLY_MIN_R2,
        COMMODITY_MODEL_ONLY_MIN_CONFIDENCE,
        CRYPTO_MODEL_ONLY_MIN_R2,
        CRYPTO_MODEL_ONLY_MIN_CONFIDENCE,
    )

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
