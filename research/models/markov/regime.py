"""Regime classification and per-state statistics.

Mirrors TS logic:
  - Adaptive threshold = 0.5 * median(|returns|)
  - Winsorized mean/std per regime
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from research.models.markov.core import RegimeState

if TYPE_CHECKING:
    from research.models.markov.policies import resolve_forecast_lab_markov_parameter_defaults


def classify_regime(
    daily_return: float,
    return_threshold: float = 0.01,
) -> RegimeState:
    """Classify a single return into a regime state."""
    if daily_return > return_threshold:
        return "bull"
    if daily_return < -return_threshold:
        return "bear"
    return "sideways"


def classify_regime_series(
    returns: np.ndarray | pd.Series | list[float],
    return_threshold_multiplier: float = 0.5,
) -> list[RegimeState]:
    """Classify a return series into regime states with adaptive threshold.

    The threshold is set to 0.5 * median(|returns|), ensuring ~30-40%
    of days are bull, ~30-40% bear, regardless of asset volatility.
    """
    arr = np.asarray(returns)
    if len(arr) == 0:
        return []

    abs_returns = np.sort(np.abs(arr))
    median_abs = float(abs_returns[len(abs_returns) // 2])
    threshold = max(0.001, return_threshold_multiplier * median_abs)

    return [classify_regime(float(r), threshold) for r in arr]


def _winsorize(values: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """Clip values beyond n_sigma standard deviations from the mean."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return values
    lo = mean - n_sigma * std
    hi = mean + n_sigma * std
    return np.clip(values, lo, hi)


def estimate_regime_stats(
    returns: np.ndarray,
    states: list[RegimeState],
    max_daily_drift: float | None = None,
    min_obs_per_state: int = 5,
) -> dict[RegimeState, dict[str, float]]:
    """Bin returns by regime state, compute winsorized mean/std per state."""
    from research.models.markov.core import NUM_STATES, REGIME_STATES

    defaults: dict[RegimeState, dict[str, float]] = {
        "bull": {"meanReturn": 0.005, "stdReturn": 0.010},
        "bear": {"meanReturn": -0.005, "stdReturn": 0.012},
        "sideways": {"meanReturn": 0.000, "stdReturn": 0.006},
    }
    bins: dict[RegimeState, list[float]] = {"bull": [], "bear": [], "sideways": []}
    n = min(len(returns), len(states))
    for i in range(n):
        bins[states[i]].append(float(returns[i]))

    result = dict(defaults)
    for state, vals in bins.items():
        if len(vals) >= min_obs_per_state:
            arr = np.asarray(vals, dtype=float)
            cleaned = _winsorize(arr)
            mean = float(np.mean(cleaned))
            variance = float(np.mean((cleaned - mean) ** 2))
            if max_daily_drift is not None and max_daily_drift > 0:
                mean = max(-max_daily_drift, min(max_daily_drift, mean))
            result[state] = {"meanReturn": mean, "stdReturn": math.sqrt(variance)}
    return result
