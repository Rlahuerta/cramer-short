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

from research.models.markov.core import RegimeState, REGIME_STATES

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


def compute_regime_up_rates(
    regime_seq: list[RegimeState],
    log_returns: np.ndarray | list[float],
    horizon: int,
    decay_rate: float | None = None,
) -> dict[RegimeState, float]:
    """Compute empirical P(up | regime) for each regime state from historical data.

    For each day where ``regime_seq[i]`` corresponds to ``log_returns[i]``,
    look forward ``horizon`` days and check whether the cumulative log return
    was positive.  This yields the actual frequency of "up" outcomes
    following each regime — much more accurate than a lossy
    regime → mean return → Student-t survival mapping.

    The result can be combined with the n-step transition probabilities to
    obtain:

    .. math::
        P(up) = \\sum_{s} P(regime_s\\ at\\ horizon) \\times P(up\\ |\\ regime_s)

    Parameters
    ----------
    regime_seq : list[RegimeState]
        Sequence of regime states (oldest first).  Each index ``i``
        corresponds to ``log_returns[i]``.
    log_returns : np.ndarray or list[float]
        Log returns, i.e. ``log(price[t] / price[t-1])``.  Must have the
        same length as ``regime_seq``.
    horizon : int
        Number of days to look ahead when accumulating returns.
    decay_rate : float or None
        Exponential decay factor.  When provided, recent observations are
        weighted more heavily: ``weight = decay_rate^(max_start - 1 - i)``.

    Returns
    -------
    dict[RegimeState, float]
        Mapping from each regime state to the empirical probability that the
        cumulative return over the next ``horizon`` days is positive.
        A state with no observations returns 0.5 (uninformative).

    Notes
    -----
    Log returns are used because the sign of ``Σ log(1+r)`` equals the sign
    of the cumulative simple return.  Summing simple returns directly is
    only an approximation and biases the up-rate downward at multi-day
    horizons because compound returns satisfy
    ``exp(Σ log(1+r)) - 1 ≠ Σ r`` when ``|r|`` is non-trivial.

    Examples
    --------
    >>> regimes = ["bull", "bull", "bear", "sideways", "bull"]
    >>> log_returns = np.array([0.01, 0.02, -0.01, 0.00, 0.015])
    >>> rates = compute_regime_up_rates(regimes, log_returns, horizon=2)
    >>> rates["bull"]  # first bull at i=0 -> cum = 0.01+0.02 = 0.03 > 0 -> up
    1.0
    """
    log_ret = np.asarray(log_returns, dtype=float)
    max_start = min(len(regime_seq), len(log_ret)) - horizon

    counts: dict[RegimeState, dict[str, float]] = {
        "bull": {"up": 0.0, "total": 0.0},
        "bear": {"up": 0.0, "total": 0.0},
        "sideways": {"up": 0.0, "total": 0.0},
    }

    for i in range(max(0, max_start)):
        regime = regime_seq[i]
        # Cumulative log return over the next `horizon` days (future returns only)
        cum_log_return = float(np.sum(log_ret[i + 1 : i + 1 + horizon]))

        # Bounded exponential weighting: recent observations get more weight
        weight = (
            math.pow(decay_rate, max_start - 1 - i)
            if decay_rate is not None
            else 1.0
        )

        counts[regime]["total"] += weight
        if cum_log_return > 0:
            counts[regime]["up"] += weight

    return {
        state: (
            counts[state]["up"] / counts[state]["total"]
            if counts[state]["total"] > 0
            else 0.5
        )
        for state in REGIME_STATES
    }


def compute_regime_expected_returns(
    regime_seq: list[RegimeState],
    log_returns: np.ndarray | list[float],
    horizon: int,
    decay_rate: float | None = None,
) -> dict[RegimeState, float]:
    """Compute the empirical mean cumulative log return per regime.

    For each day where ``regime_seq[i]`` corresponds to ``log_returns[i]``,
    look forward ``horizon`` days and record the cumulative log return.
    Returns the exponentially weighted mean per regime — the empirical
    expected cumulative log return given the regime at time i.

    Parameters
    ----------
    regime_seq : list[RegimeState]
        Sequence of regime states (oldest first).
    log_returns : np.ndarray or list[float]
        Log returns, same length as ``regime_seq``.
    horizon : int
        Number of days to look ahead.
    decay_rate : float or None
        Exponential decay factor for recency weighting.

    Returns
    -------
    dict[RegimeState, float]
        Mapping from each regime state to the mean cumulative log return.
        States with no observations return 0.0 (no expected drift).
    """
    log_ret = np.asarray(log_returns, dtype=float)
    max_start = min(len(regime_seq), len(log_ret)) - horizon

    accum: dict[RegimeState, dict[str, float]] = {
        "bull": {"sum_weighted": 0.0, "total": 0.0},
        "bear": {"sum_weighted": 0.0, "total": 0.0},
        "sideways": {"sum_weighted": 0.0, "total": 0.0},
    }

    for i in range(max(0, max_start)):
        regime = regime_seq[i]
        cum_log_return = float(np.sum(log_ret[i + 1 : i + 1 + horizon]))
        weight = (
            math.pow(decay_rate, max_start - 1 - i)
            if decay_rate is not None
            else 1.0
        )
        accum[regime]["total"] += weight
        accum[regime]["sum_weighted"] += weight * cum_log_return

    return {
        state: (
            accum[state]["sum_weighted"] / accum[state]["total"]
            if accum[state]["total"] > 0
            else 0.0
        )
        for state in REGIME_STATES
    }
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
