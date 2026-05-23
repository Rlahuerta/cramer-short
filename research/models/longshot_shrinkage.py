"""R5 Idea #11 — Longshot odds shrinkage.

Python mirror of applyLongshotShrinkage in
src/tools/finance/rnd-integration.ts.

Source: docs/forecast-improvement-ideas-round5-2026-04-29.md (Idea #11).

When a calibrated probability is in the extreme tail (p < low_threshold
or p > high_threshold), shrink it toward 0.5 using a weight w:

    p_shrunk = w * 0.5 + (1 - w) * p

Defaults: low_threshold=0.05, high_threshold=0.95, weight=0.5.

Apply this *after* transform_q_to_p to avoid double-shrinking the Q->P
shift.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongshotShrinkResult:
    """Result of apply_longshot_shrinkage."""

    p: float
    """Possibly shrunk probability, clamped to [0, 1]."""

    applied: bool
    """True if the shrinkage was applied (input was in a tail zone)."""

    tail_distance: float
    """Absolute distance from 0.5 in the *original* probability."""


def apply_longshot_shrinkage(
    p: float,
    low_threshold: float = 0.05,
    high_threshold: float = 0.95,
    weight: float = 0.5,
) -> LongshotShrinkResult:
    """Shrink extreme tail probabilities toward 0.5.

    Returns the (possibly modified) probability together with diagnostics.
    """
    w = max(0.0, min(1.0, weight))
    tail_distance = abs(p - 0.5)
    if low_threshold < p < high_threshold:
        return LongshotShrinkResult(p=p, applied=False, tail_distance=tail_distance)
    shrunk = w * 0.5 + (1.0 - w) * p
    return LongshotShrinkResult(
        p=max(0.0, min(1.0, shrunk)),
        applied=True,
        tail_distance=tail_distance,
    )
