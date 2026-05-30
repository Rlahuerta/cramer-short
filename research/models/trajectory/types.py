"""Core types for price trajectory and scenario probability computation.

Mirrors TS logic from src/tools/finance/markov-distribution.ts:
  - TrajectoryPoint dataclass for per-day forecast ribbons
  - RegimeStats for regime-specific return parameters
"""

from __future__ import annotations

from dataclasses import dataclass

from research.models.markov import RegimeState


@dataclass
class RegimeStats:
    mean_return: float
    std_return: float


@dataclass
class TrajectoryPoint:
    day: int
    expected_price: float
    lower_bound: float
    upper_bound: float
    p_up: float
    cumulative_return: str
    regime: RegimeState
