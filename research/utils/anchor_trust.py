"""Python mirror of the TypeScript anchor-trust policy helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AnchorLowTrustReason = Literal["young_market", "resolution_mismatch", "missing_volume"]


@dataclass(frozen=True)
class AnchorTrustEvaluation:
    trust_score: Literal["high", "low"]
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
    trusted = has_volume and (
        (is_long_horizon_crypto and not is_young and is_near_target_resolution)
        or (
            not is_long_horizon_crypto
            and (not is_young or (is_short_horizon_crypto and is_near_target_resolution))
        )
    )

    reasons: list[AnchorLowTrustReason] = []
    if not has_volume:
        reasons.append("missing_volume")
    if is_young:
        reasons.append("young_market")
    if needs_resolution_match and not is_near_target_resolution:
        reasons.append("resolution_mismatch")

    return AnchorTrustEvaluation("high" if trusted else "low", reasons)
