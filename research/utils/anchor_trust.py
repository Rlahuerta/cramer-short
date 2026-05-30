"""Anchor-to-Polymarket trust policy — decides when to trust market anchors.

The Markov forecaster can blend Polymarket prediction-market prices as
probability anchors.  This module evaluates whether a given anchor is
trustworthy based on:

- Trading volume and market maturity
- Whether the horizon is too short for the anchor to be informative
- Whether the anchor is near target resolution (prices collapse to 0/1)

When trust is low, the forecaster falls back to pure Markov-model
probabilities instead of blending in the anchor signal.

Mirrors ``src/tools/finance/markov-distribution.ts`` (anchor-trust section).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AnchorLowTrustReason = Literal["young_market", "resolution_mismatch", "missing_volume"]


@dataclass(frozen=True)
class AnchorTrustEvaluation:
    trust_score: Literal["high", "low"]
    trust_weight: float
    low_trust_reasons: list[AnchorLowTrustReason]


def evaluate_anchor_trust(
    *,
    has_volume: bool,
    is_young: bool,
    is_short_horizon_crypto: bool,
    is_long_horizon_crypto: bool,
    is_near_target_resolution: bool,
) -> AnchorTrustEvaluation:
    needs_resolution_match = is_long_horizon_crypto or (is_short_horizon_crypto and is_young)
    is_non_crypto = not is_short_horizon_crypto and not is_long_horizon_crypto
    trust_weight = 0.0
    if has_volume:
        if is_non_crypto:
            trust_weight = 0.75 if is_young else 1.0
        elif is_long_horizon_crypto:
            if not is_young and is_near_target_resolution:
                trust_weight = 0.9
            elif not is_young and not is_near_target_resolution:
                trust_weight = 0.35
            elif is_young and is_near_target_resolution:
                trust_weight = 0.45
            else:
                trust_weight = 0.2
        elif is_short_horizon_crypto:
            if not is_young and is_near_target_resolution:
                trust_weight = 0.9
            elif not is_young and not is_near_target_resolution:
                trust_weight = 0.7
            elif is_young and is_near_target_resolution:
                trust_weight = 0.7
            else:
                trust_weight = 0.35

    reasons: list[AnchorLowTrustReason] = []
    if not has_volume:
        reasons.append("missing_volume")
    if is_young:
        reasons.append("young_market")
    if needs_resolution_match and not is_near_target_resolution:
        reasons.append("resolution_mismatch")

    return AnchorTrustEvaluation("high" if trust_weight >= 0.7 else "low", trust_weight, reasons)
