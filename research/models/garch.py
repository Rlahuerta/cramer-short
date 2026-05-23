"""P3a — GARCH(1,1) interim volatility helper (Python mirror).

Source: docs/polymarket-prediction-improvements-research-2026-07.md §10.3

    h_t  = ω + α · z²_{t-1} · h_{t-1} + β · h_{t-1}
    σ_t  = √h_t

Mirrors ``src/utils/garch.ts``. Uses fixed industry priors
(α = 0.10, β = 0.85) and matches the unconditional variance to the
sample variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List, Sequence

GARCH_DEFAULT_ALPHA = 0.10
GARCH_DEFAULT_BETA = 0.85


@dataclass(frozen=True)
class Garch11Params:
    omega: float
    alpha: float
    beta: float
    h0: float


def fit_garch11(
    returns: Sequence[float],
    alpha: float = GARCH_DEFAULT_ALPHA,
    beta: float = GARCH_DEFAULT_BETA,
) -> Garch11Params:
    """Moment-matching GARCH(1,1) estimator (mirrors fitGarch11 TS)."""
    if len(returns) < 5:
        raise ValueError(f"fit_garch11 requires >= 5 observations, got {len(returns)}")
    if alpha + beta >= 1:
        raise ValueError(f"fit_garch11 requires alpha + beta < 1, got {alpha + beta}")
    sse = sum(r * r for r in returns)
    sample_var = sse / len(returns)
    omega = sample_var * (1 - alpha - beta)
    return Garch11Params(omega=omega, alpha=alpha, beta=beta, h0=sample_var)


def garch_step(prev_h: float, prev_z: float, p: Garch11Params) -> float:
    return p.omega + p.alpha * prev_z * prev_z * prev_h + p.beta * prev_h


def garch_forecast(p: Garch11Params, horizon_days: int) -> List[float]:
    """k-step σ forecast using E[z²] = 1 substitution.

    Returns σ_t for t = 1..horizon_days. Mirrors garchForecast TS.
    """
    if horizon_days <= 0:
        return []
    persistence = p.alpha + p.beta
    out: List[float] = []
    h = garch_step(p.h0, 1.0, p)
    out.append(sqrt(max(0.0, h)))
    for _ in range(1, horizon_days):
        h = p.omega + persistence * h
        out.append(sqrt(max(0.0, h)))
    return out
