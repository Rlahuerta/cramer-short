"""Emission and evaluation gates.

Mirrors TS logic:
  - Validation acceptability
  - Model-only bypass
  - Context-only canonical emission
  - Divergence penalty
  - BTC confidence cap and sell gate
"""

from __future__ import annotations

import math

from research.models.markov.core import (
    ActionConfidence,
    DEFAULT_DIVERGENCE_PENALTY_SCHEDULE,
    RegimeState,
)
from research.models.markov.policies import (
    is_btc_ticker_symbol,
    resolve_forecast_lab_markov_parameter_defaults,
    resolve_forecast_lab_runtime_asset_scope_for_ticker,
)


def compute_divergence_penalty(
    divergence: float,
    schedule: dict[str, float] | None = None,
) -> float:
    """Map a Frobenius divergence value to a confidence penalty multiplier."""
    effective_schedule = schedule or DEFAULT_DIVERGENCE_PENALTY_SCHEDULE
    if divergence < 0.05:
        return 1.0
    if divergence < 0.10:
        return float(effective_schedule["mild"])
    if divergence < 0.20:
        return float(effective_schedule["medium"])
    return float(effective_schedule["high"])


def evaluate_can_emit_canonical(
    trusted_anchors: int,
    anchor_quality: str,
    validation_acceptable: bool,
    commodity_model_only: bool,
    crypto_model_only: bool,
    sparse_crypto_anchor_allowed: bool = False,
) -> bool:
    """Final emission gate: can the Markov forecast be emitted?"""
    return (
        (trusted_anchors > 0 and anchor_quality == "good" and validation_acceptable)
        or sparse_crypto_anchor_allowed
        or commodity_model_only
        or crypto_model_only
    )


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
    """Determine whether to emit a context-only canonical forecast."""
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


def cap_btc_short_horizon_confidence(
    ticker: str,
    horizon: int,
    structural_break_detected: bool,
    out_of_sample_r2: float | None,
    prediction_confidence: float,
    confidence: ActionConfidence,
) -> ActionConfidence:
    """Demote HIGH to MEDIUM confidence for BTC h<=14 when validation is weak."""
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
    recommendation: str,
    raw_recommendation: str | None,
    structural_break_detected: bool,
    regime_state: RegimeState,
    prediction_confidence: float,
    raw_predicted_prob: float,
    predicted_prob: float,
    expected_return: float,
) -> bool:
    """Flip a BTC 14d BUY to SELL when the raw signal says SELL under a break."""
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


def _is_finite_number(value: float | int | None) -> bool:
    return value is not None and math.isfinite(float(value))
