"""Hawkes self-exciting point process — mirror of TS hawkes.ts.

λ(t) = μ + Σ_{t_i ≤ t} α · exp(−β · (t − t_i))

Stability requires α/β < 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class HawkesParams:
    mu: float
    alpha: float
    beta: float


@dataclass(frozen=True)
class HawkesFit:
    mu: float
    alpha: float
    beta: float
    log_likelihood: float
    is_stable: bool


class HawkesIntensity:
    def __init__(self, mu: float, alpha: float, beta: float) -> None:
        if not (mu >= 0) or not math.isfinite(mu):
            raise ValueError("Hawkes: mu must be >= 0")
        if alpha < 0:
            raise ValueError("Hawkes: alpha must be >= 0")
        if not (beta > 0):
            raise ValueError("Hawkes: beta must be > 0")
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def intensity(self, t: float, history: Sequence[float]) -> float:
        s = 0.0
        for ti in history:
            if ti <= t:
                s += math.exp(-self.beta * (t - ti))
        return self.mu + self.alpha * s

    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    def is_stable(self) -> bool:
        return self.branching_ratio() < 1.0

    def log_likelihood(self, events: Sequence[float], horizon: float) -> float:
        if horizon <= 0:
            return 0.0
        log_sum = 0.0
        recursive = 0.0
        prev = 0.0
        for i, ti in enumerate(events):
            if i > 0:
                recursive = (recursive + 1) * math.exp(-self.beta * (ti - prev))
            lam = self.mu + self.alpha * recursive
            if lam <= 0:
                return -math.inf
            log_sum += math.log(lam)
            prev = ti
        compensator = self.mu * horizon
        ratio = self.alpha / self.beta
        for ti in events:
            if ti < horizon:
                compensator += ratio * (1 - math.exp(-self.beta * (horizon - ti)))
        return log_sum - compensator


def simulate_hawkes(
    params: HawkesParams, T: float, rng: Callable[[], float]
) -> list[float]:
    h = HawkesIntensity(params.mu, params.alpha, params.beta)
    events: list[float] = []
    t = 0.0
    while t < T:
        lam_bar = h.intensity(t, events) + 1e-12
        u = rng()
        w = -math.log(u) / lam_bar
        t = t + w
        if t >= T:
            break
        d = rng()
        lam_t = h.intensity(t, events)
        if d * lam_bar <= lam_t:
            events.append(t)
    return events


def fit_hawkes_mle(
    events: Sequence[float],
    horizon: float,
    initial_mu: float | None = None,
    initial_alpha: float = 0.1,
    initial_beta: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> HawkesFit:
    if initial_mu is None:
        initial_mu = max(1e-3, len(events) / max(horizon, 1e-6))
    mu, alpha, beta = initial_mu, initial_alpha, initial_beta

    def ll(m: float, a: float, b: float) -> float:
        if m <= 0 or a < 0 or b <= 0:
            return -math.inf
        if a / b >= 0.999:
            return -math.inf
        return HawkesIntensity(m, a, b).log_likelihood(events, horizon)

    def golden(lo: float, hi: float, fn: Callable[[float], float], iters: int = 40) -> float:
        phi = (math.sqrt(5) - 1) / 2
        a, b = lo, hi
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        for _ in range(iters):
            if fn(c) > fn(d):
                b = d
            else:
                a = c
            c = b - phi * (b - a)
            d = a + phi * (b - a)
        return (a + b) / 2

    prev_ll = ll(mu, alpha, beta)
    for _ in range(max_iter):
        mu = golden(1e-6, max(initial_mu * 10, len(events) / horizon + 1), lambda m: ll(m, alpha, beta))
        beta = golden(1e-3, 50, lambda b: ll(mu, alpha, b))
        alpha = golden(0, 0.999 * beta, lambda a: ll(mu, a, beta))
        next_ll = ll(mu, alpha, beta)
        if abs(next_ll - prev_ll) < tol:
            break
        prev_ll = next_ll

    final = HawkesIntensity(mu, alpha, beta)
    return HawkesFit(
        mu=mu,
        alpha=alpha,
        beta=beta,
        log_likelihood=final.log_likelihood(events, horizon),
        is_stable=final.is_stable(),
    )
