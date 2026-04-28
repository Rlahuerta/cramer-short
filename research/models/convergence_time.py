"""Convergence-Time Confidence Signal — Python mirror.

Reference: Voigt 2025 (Polymarket Beta-HMM paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

NO_SPEEDUP = 0.6
FAST_THRESHOLD_DAYS = 7
SLOW_THRESHOLD_DAYS = 30
FAST_BOOST = 0.15
SLOW_DAMP = 0.10


@dataclass(frozen=True)
class ConvergenceResult:
    converged: bool
    days_to_converge: Optional[int]
    direction: Optional[str]  # "yes" | "no" | None


def convergence_time(prices: Sequence[float], epsilon: float = 0.05) -> ConvergenceResult:
    if not prices:
        return ConvergenceResult(False, None, None)
    upper = 1 - epsilon
    for i, p in enumerate(prices):
        if p > upper:
            return ConvergenceResult(True, i, "yes")
        if p < epsilon:
            return ConvergenceResult(True, i, "no")
    return ConvergenceResult(False, None, None)


def convergence_time_factor(r: ConvergenceResult) -> float:
    if not r.converged or r.days_to_converge is None:
        return 1.0
    days = r.days_to_converge * NO_SPEEDUP if r.direction == "no" else float(r.days_to_converge)
    if days <= FAST_THRESHOLD_DAYS:
        return 1 + FAST_BOOST
    if days >= SLOW_THRESHOLD_DAYS:
        return 1 - SLOW_DAMP
    t = (days - FAST_THRESHOLD_DAYS) / (SLOW_THRESHOLD_DAYS - FAST_THRESHOLD_DAYS)
    return (1 + FAST_BOOST) + t * ((1 - SLOW_DAMP) - (1 + FAST_BOOST))
