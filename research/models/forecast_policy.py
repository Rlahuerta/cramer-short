"""Forecast trust-policy engine — determines the trust level of model predictions.

Decides whether a forecast should be trusted at full strength, emitted with
a context-only label, or rejected entirely.  The policy considers:

- Ensemble consensus strength (Polymarket agreement)
- Structural-break presence
- Conformal confidence
- HMM convergence and posterior entropy
- Domain/horizon trust scores

Mirrors ``src/agent/forecast-policy.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ForecastTrustPolicyLevel = Literal["full", "context-only", "abstain"]
SemanticPrimary = Literal["terminal", "barrier_touch", "path_dependent"]
ConformalMode = Literal["normal", "break"]


@dataclass(frozen=True)
class ForecastTrustPolicy:
    level: ForecastTrustPolicyLevel
    horizon_eligible: bool
    trade_eligible: bool
    reasons: list[str]


@dataclass(frozen=True)
class ForecastTrustPolicyInput:
    leverage: int
    semantic_primary: SemanticPrimary
    is_divergent: bool
    markov_confidence: float
    structural_break: bool
    flat_probability: float
    has_terminal_support: bool
    anchor_quality: str | None = None
    trusted_anchors: int | None = None
    conformal_applied: bool = False
    conformal_radius: float | None = None
    conformal_coverage: float | None = None
    conformal_mode: ConformalMode = "normal"


def compute_forecast_trust_policy(params: ForecastTrustPolicyInput) -> ForecastTrustPolicy:
    reasons: list[str] = []
    weak_confidence = params.markov_confidence < 0.45
    severe_confidence = params.markov_confidence < 0.2
    conformal_break_mode = params.conformal_mode == "break"
    weak_conformal = params.conformal_applied and (
        conformal_break_mode
        or (params.conformal_coverage is not None and params.conformal_coverage < 0.75)
        or (params.conformal_radius is not None and params.conformal_radius >= 0.08)
    )
    severe_conformal = params.conformal_applied and (
        (params.conformal_coverage is not None and params.conformal_coverage < 0.6)
        or (params.conformal_radius is not None and params.conformal_radius >= 0.12)
        or (conformal_break_mode and params.conformal_coverage is not None and params.conformal_coverage < 0.65)
    )
    missing_trusted_support = (
        (params.trusted_anchors is not None and params.trusted_anchors <= 0)
        or params.anchor_quality in {"none", "weak"}
        or (
            not params.has_terminal_support
            and params.structural_break
            and params.flat_probability >= 0.8
            and severe_confidence
        )
    )

    if params.is_divergent:
        reasons.append("Markov and Polymarket disagree on direction, so this horizon is context-only until the signals realign.")
    if params.semantic_primary in {"barrier_touch", "path_dependent"}:
        reasons.append("Prediction-market support is barrier/path dependent rather than a clean terminal anchor.")
    if params.structural_break:
        reasons.append("A structural-break flag is active, so regime trust is reduced.")
    if params.flat_probability >= 0.7:
        reasons.append("Flat-probability is elevated, which weakens immediate directional edge.")
    if params.leverage >= 8:
        reasons.append(f"{params.leverage}x leverage is too unforgiving for the current forecast quality.")
    if weak_confidence:
        reasons.append("Markov prediction confidence is too weak to treat as a standalone trade trigger.")
    if weak_conformal:
        reasons.append("Conformal diagnostics are stressed, so keep the forecast as regime context rather than a full trade signal.")
    if missing_trusted_support:
        reasons.append("Trusted horizon support is missing, so the forecast should abstain instead of manufacturing a calibrated edge.")

    level: ForecastTrustPolicyLevel = "full"
    if (
        missing_trusted_support
        or (params.structural_break and severe_confidence and params.flat_probability >= 0.82)
        or (params.structural_break and severe_conformal and not params.has_terminal_support)
    ):
        level = "abstain"
    elif (
        params.is_divergent
        or params.flat_probability >= 0.7
        or params.leverage >= 8
        or weak_confidence
        or weak_conformal
        or params.semantic_primary in {"barrier_touch", "path_dependent"}
    ):
        level = "context-only"

    if not reasons:
        reasons.append(
            "Evidence is aligned and regime diagnostics are healthy enough for full guidance."
            if level == "full"
            else (
                "Use the forecast as context only until trust diagnostics improve."
                if level == "context-only"
                else "No calibrated edge is available for this horizon."
            )
        )

    return ForecastTrustPolicy(
        level=level,
        horizon_eligible=level != "abstain",
        trade_eligible=level == "full",
        reasons=reasons,
    )
