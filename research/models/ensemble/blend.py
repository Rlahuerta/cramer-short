"""Signal ensemble blending across multiple auxiliary sources.

Mirrors TS logic from src/utils/ensemble.ts:
  - PM, sentiment, fundamental, options, Markov signal weighting
  - PM weight scaled by average market quality before blending
"""

from __future__ import annotations

import math

from research.models.ensemble.types import OtherSignals


def compute_ensemble(
    pm_signal: float,
    pm_avg_quality: float,
    others: OtherSignals,
) -> dict:
    """Blend Polymarket signal with auxiliary sources using quality-scaled weights.

    Combines up to five signal sources — Polymarket, sentiment, fundamentals,
    options skew, and Markov forecast — into a single expected return. The
    Polymarket weight is scaled by the average market quality (0.40 × quality).
    Only sources with valid (non-NaN) data are included; weights are normalized
    to sum to 1. If all weights are zero, falls back to equal weighting.

    Parameters
    ----------
    pm_signal : float
        Quality-weighted average conditional return from Polymarket markets.
    pm_avg_quality : float
        Average market quality score across all Polymarket inputs.
    others : OtherSignals
        Auxiliary signals with per-source data and horizon days.

    Returns
    -------
    dict
        ``forecast_return`` (float) — blended expected return.
        ``weights`` (dict[str, float]) — normalized weight per source.
    """
    w_pm_eff = 0.40 * pm_avg_quality

    available: dict[str, dict[str, float]] = {}
    available["pm"] = {"weight": w_pm_eff, "signal": pm_signal}

    if others.sentiment_score is not None and not math.isnan(others.sentiment_score):
        available["sentiment"] = {
            "weight": 0.20,
            "signal": others.sentiment_score * 0.04,
        }

    if others.fundamental_return is not None and not math.isnan(others.fundamental_return):
        available["fundamental"] = {
            "weight": 0.25,
            "signal": others.fundamental_return * (others.horizon_days / 365),
        }

    if others.options_skew is not None and not math.isnan(others.options_skew):
        available["options"] = {
            "weight": 0.15,
            "signal": others.options_skew * 0.03,
        }

    if others.markov_return is not None and not math.isnan(others.markov_return):
        available["markov"] = {
            "weight": 0.20,
            "signal": others.markov_return,
        }

    total_raw = sum(e["weight"] for e in available.values())

    weights: dict[str, float] = {}
    forecast_return = 0.0

    if total_raw == 0:
        n = len(available)
        for key, entry in available.items():
            w = 1.0 / n if n > 0 else 0.0
            weights[key] = w
            forecast_return += w * entry["signal"]
    else:
        for key, entry in available.items():
            w = entry["weight"] / total_raw
            weights[key] = w
            forecast_return += w * entry["signal"]

    return {"forecast_return": forecast_return, "weights": weights}
