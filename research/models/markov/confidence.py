"""Prediction confidence scoring.

Mirrors TS logic:
  - Multi-component confidence with asset-specific multipliers
"""

from __future__ import annotations

import math
from typing import Literal

from research.models.markov.core import (
    BreakConfidencePolicy,
    PredictionConfidenceMode,
    RegimeState,
)
from research.models.markov.gates import compute_divergence_penalty


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
    """Detailed prediction confidence breakdown."""
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
    """Simplified prediction confidence score."""
    return float(compute_prediction_confidence_breakdown(**kwargs)["total"])


def _is_finite_number(value: float | int | None) -> bool:
    return value is not None and math.isfinite(float(value))
