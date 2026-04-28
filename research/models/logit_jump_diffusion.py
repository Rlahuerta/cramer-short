"""Logit Jump-Diffusion for prediction-market prices.

Mirrors src/tools/finance/logit-jump-diffusion.ts.

Reference: *Toward Black–Scholes for Prediction Markets* (arXiv:2510.15205).
Models the price p_t ∈ (0, 1) of a binary prediction-market contract by
working in log-odds space x = logit(p) where the support is unbounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

Z_MAX = 30.0


def logit(p: float) -> float:
    eps = 1e-12
    clipped = min(1 - eps, max(eps, p))
    return math.log(clipped / (1 - clipped))


def inv_logit(z: float) -> float:
    if z >= Z_MAX:
        return 1 - 1e-12
    if z <= -Z_MAX:
        return 1e-12
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def ito_martingale_drift(p: float, sigma: float, lam: float, jump_mean: float, jump_std: float) -> float:
    """Itô-Jensen drift correction such that p = sigmoid(x) is a martingale.

    μ_x = -½ σ² (1 - 2p) - λ · ⟨Δp⟩ / (p(1-p))

    Jump compensator integral approximated via Gauss-Hermite 3-point quadrature.
    """
    safe_p = min(1 - 1e-12, max(1e-12, p))
    diffusion_drift = -0.5 * sigma * sigma * (1 - 2 * safe_p)
    if lam <= 0 or (jump_mean == 0 and jump_std == 0):
        return diffusion_drift
    nodes = (-math.sqrt(1.5), 0.0, math.sqrt(1.5))
    weights = (1 / 6, 2 / 3, 1 / 6)
    x = logit(safe_p)
    expected = 0.0
    for w, node in zip(weights, nodes):
        j = jump_mean + math.sqrt(2.0) * jump_std * node
        expected += w * inv_logit(x + j)
    mean_dp = expected - safe_p
    denom = safe_p * (1 - safe_p)
    if denom < 1e-12:
        return diffusion_drift
    return diffusion_drift - (lam * mean_dp) / denom


@dataclass
class LogitJumpDiffusionResult:
    terminal: np.ndarray  # shape (n_paths,)
    paths: Optional[np.ndarray]  # shape (n_paths, days) if requested
    total_jumps: int
    effective_lambda: float


def simulate_logit_jump_diffusion(
    initial_price: float,
    days: int,
    sigma_per_day: float,
    n_paths: int,
    *,
    jump_intensity_per_day: float = 0.0,
    polymarket_jump_prob: Optional[float] = None,
    jump_logit_mean: float = 0.0,
    jump_logit_std: float = 0.0,
    rng: Optional[Callable[[], float]] = None,
    store_paths: bool = False,
) -> LogitJumpDiffusionResult:
    if days < 1:
        raise ValueError("days must be ≥ 1")
    if n_paths < 1:
        raise ValueError("n_paths must be ≥ 1")

    if polymarket_jump_prob is not None:
        lam = max(0.0, min(1.0, polymarket_jump_prob)) / days
    else:
        lam = max(0.0, jump_intensity_per_day)

    if rng is None:
        nprng = np.random.default_rng()
        rng = nprng.random  # type: ignore[assignment]

    initial_logit = logit(initial_price)
    terminal = np.empty(n_paths)
    paths = np.empty((n_paths, days)) if store_paths else None
    total_jumps = 0

    for s in range(n_paths):
        x = initial_logit
        p = inv_logit(x)
        for d in range(days):
            drift = ito_martingale_drift(p, sigma_per_day, lam, jump_logit_mean, jump_logit_std)
            z = _box_muller(rng)
            x += drift + sigma_per_day * z
            if lam > 0 and rng() < lam:
                jz = _box_muller(rng)
                x += jump_logit_mean + jump_logit_std * jz
                total_jumps += 1
            p = inv_logit(x)
            if paths is not None:
                paths[s, d] = p
        terminal[s] = p

    return LogitJumpDiffusionResult(terminal=terminal, paths=paths, total_jumps=total_jumps, effective_lambda=lam)


def _box_muller(rng: Callable[[], float]) -> float:
    u1 = max(rng(), 1e-12)
    u2 = rng()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
