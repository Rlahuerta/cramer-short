"""R5 Idea #14 — Markov transition-entropy CI modulator.

Python mirror of src/tools/finance/transition-entropy.ts.

Source: docs/forecast-improvement-ideas-round5-2026-04-29.md (Idea #14),
arXiv:2511.05621 Chen et al. 2025.

Hypothesis: when the row-entropy of the empirical Markov transition
matrix spikes (high uncertainty about which regime is next), the
forecast CI should widen.  When the matrix becomes near-deterministic
(low entropy), the CI can tighten.

Formula:
    H = -sum_i pi_i * sum_j P_ij * log(P_ij)   # nats
    H_max = log(K)
    H_norm = H / H_max in [0, 1]

The caller maintains a rolling Z-score over the last `window_size`
H_norm values and applies:
    ci_scale = clamp(1 + kappa * z, 0.7, 1.4)

Pure functions only — no I/O, no global state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TransitionEntropyResult:
    """Result of compute_transition_entropy."""

    entropy_nats: float
    """Stationary-weighted row entropy in nats."""

    entropy_norm: float
    """H normalized to [0, 1] by log(K)."""

    K: int
    """Number of states."""


def approximate_stationary(P: Sequence[Sequence[float]]) -> list[float]:
    """Compute stationary distribution by power-iteration on P^T.

    Falls back to uniform if P is degenerate.
    Converges in at most 100 iterations (tolerance 1e-9).
    """
    K = len(P)
    if K == 0:
        return []
    pi = [1.0 / K] * K
    for _ in range(100):
        nxt = [0.0] * K
        for j in range(K):
            s = 0.0
            for i in range(K):
                row = P[i]
                s += pi[i] * (row[j] if j < len(row) else 0.0)
            nxt[j] = s
        total = sum(nxt)
        if not (total > 0):
            return [1.0 / K] * K
        nxt = [x / total for x in nxt]
        delta = sum(abs(nxt[j] - pi[j]) for j in range(K))
        pi = nxt
        if delta < 1e-9:
            break
    return pi


def compute_transition_entropy(P: Sequence[Sequence[float]]) -> TransitionEntropyResult:
    """Compute stationary-weighted Shannon entropy of transition matrix P."""
    K = len(P)
    if K == 0:
        return TransitionEntropyResult(entropy_nats=0.0, entropy_norm=0.0, K=0)
    pi = approximate_stationary(P)
    H = 0.0
    for i in range(K):
        row = P[i]
        row_h = 0.0
        for j in range(K):
            p = row[j] if j < len(row) else 0.0
            if p > 0:
                row_h -= p * math.log(p)
        H += pi[i] * row_h
    Hmax = math.log(max(2, K))
    norm = H / Hmax if Hmax > 0 else 0.0
    return TransitionEntropyResult(
        entropy_nats=H,
        entropy_norm=max(0.0, min(1.0, norm)),
        K=K,
    )


class EntropyZScoreTracker:
    """Online rolling z-score for transition entropy values.

    Maintains a reservoir of the last `window_size` H_norm values.
    Returns None until at least 5 values have been observed.
    """

    def __init__(self, window_size: int = 60) -> None:
        if window_size < 5:
            raise ValueError("EntropyZScoreTracker window_size must be >= 5")
        self._window_size = window_size
        self._buf: deque[float] = deque(maxlen=window_size)

    def push(self, value: float) -> None:
        """Add a new H_norm value to the rolling buffer."""
        self._buf.append(value)

    def z_score(self, value: float) -> float | None:
        """Compute the z-score of `value` against the current buffer.

        Returns None until at least 5 values have been pushed.
        """
        if len(self._buf) < 5:
            return None
        mean = sum(self._buf) / len(self._buf)
        sse = sum((v - mean) ** 2 for v in self._buf)
        std = math.sqrt(sse / len(self._buf))
        if not (std > 1e-9):
            return 0.0
        return (value - mean) / std

    def size(self) -> int:
        """Current number of values in the buffer."""
        return len(self._buf)


def entropy_z_to_ci_scale(
    z_norm: float,
    kappa: float = 0.15,
    bounds: tuple[float, float] = (0.7, 1.4),
) -> float:
    """Convert a transition-entropy z-score to a CI width scalar.

    High z (unusually uncertain) -> ciScale > 1 (wider CI).
    Low z (unusually deterministic) -> ciScale < 1 (tighter CI).

    Formula: ciScale = clamp(1 + kappa * z, bounds[0], bounds[1])
    """
    raw = 1.0 + kappa * z_norm
    return max(bounds[0], min(bounds[1], raw))
