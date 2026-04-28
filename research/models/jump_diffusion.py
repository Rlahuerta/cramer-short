"""Jump-diffusion helpers (Idea 2 — Polymarket-informed Merton jumps).

Mirrors ``src/tools/finance/jump-diffusion.ts``. Pure math, no I/O.

The Merton (1976) jump-diffusion model adds a compound Poisson term to GBM::

    dS_t / S_t = (μ − λ·κ) dt + σ dW_t + (J − 1) dN_t

with ``log(J) ~ N(μ_J, σ_J²)`` and ``κ = exp(μ_J + σ_J²/2) − 1`` the
expected percentage jump.  ``λ·κ`` is the **drift compensator** that keeps
``E[dS/S] = μ dt``; without it, the simulated drift is biased by the
expected jump impact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from .rnd import transform_q_to_p

# ---------------------------------------------------------------------------
# Per-asset-class jump-magnitude defaults
# ---------------------------------------------------------------------------
# Calibration source: rolling 90-day max-abs-daily-log-return percentiles
# across SPY/QQQ (etf, equity), BTC/ETH (crypto), GLD/USO (commodity)
# over 2020-01-01..2024-12-31.  See TS mirror for design notes.

JUMP_DEFAULTS: Mapping[str, Mapping[str, float]] = {
    "etf":         {"mean_log_jump": -0.04, "std_log_jump": 0.02},
    "equity":      {"mean_log_jump": -0.05, "std_log_jump": 0.03},
    "crypto":      {"mean_log_jump": -0.08, "std_log_jump": 0.05},
    "commodity":   {"mean_log_jump": -0.05, "std_log_jump": 0.03},
    # War, sanctions, political shock — spec: ±10% expected impact, wide uncertainty.
    "geopolitics": {"mean_log_jump": -0.10, "std_log_jump": 0.06},
}


@dataclass(frozen=True)
class JumpEventSpec:
    """A single Polymarket-implied jump event in physical measure."""

    id: str
    daily_intensity: float  # λ_e per day (already Q→P-converted, capped at 0.95)
    mean_log_jump: float
    std_log_jump: float


def polymarket_prob_to_hazard(p: float, horizon_days: int) -> float:
    """Convert total settlement probability ``p`` to a per-day Poisson hazard.

    Uses the survival relation ``1 − p = exp(−λ_total)`` ⇒ ``λ_total = −ln(1 − p)``,
    then splits uniformly across the ``horizon_days`` window.

    The result is capped at 0.95 to keep the daily Bernoulli draw well-behaved.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 0.95
    days = max(1, horizon_days)
    lambda_total = -math.log(1.0 - p)
    lambda_daily = lambda_total / days
    return min(0.95, max(0.0, lambda_daily))


def build_jump_event_spec(
    raw: float,
    horizon_days: int,
    historical_drift_annual: float,
    risk_free_rate: float,
    volatility_annual: float,
    prior: Mapping[str, float],
    id: str,
) -> JumpEventSpec:
    """Convert a raw Polymarket Q-prob into a fully-specified JumpEventSpec.

    Composes Q→P transformation with hazard derivation, then attaches the
    asset-class jump-size prior.
    """
    p_physical = transform_q_to_p(
        raw,
        historical_drift=historical_drift_annual,
        risk_free_rate=risk_free_rate,
        volatility=volatility_annual,
        days_to_expiry=horizon_days,
    )
    daily_intensity = polymarket_prob_to_hazard(p_physical, horizon_days)
    return JumpEventSpec(
        id=id,
        daily_intensity=daily_intensity,
        mean_log_jump=float(prior["mean_log_jump"]),
        std_log_jump=float(prior["std_log_jump"]),
    )


def jump_drift_compensator(events: Iterable[JumpEventSpec]) -> float:
    """Daily Merton drift compensator ``Σ_e λ_e · (exp(μ_J,e + σ_J,e²/2) − 1)``.

    Subtract from ``μ_t·Δt`` (Δt = 1 day) so the post-jump expected return
    remains equal to ``μ_t``.
    """
    total = 0.0
    for e in events:
        kappa = math.exp(e.mean_log_jump + (e.std_log_jump ** 2) / 2.0) - 1.0
        total += e.daily_intensity * kappa
    return total


__all__ = [
    "JUMP_DEFAULTS",
    "JumpEventSpec",
    "polymarket_prob_to_hazard",
    "build_jump_event_spec",
    "jump_drift_compensator",
]
