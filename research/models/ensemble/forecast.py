"""End-to-end ensemble forecast orchestration.

Mirrors TS logic from src/utils/ensemble.ts:
  - Full pipeline: PM signal → blend → variance → CI → quality score
  - Horizon-aware sigma floor (10% annualized sqrt scaling)
  - Signal data counting and whale penalty scoring
"""

from __future__ import annotations

import math

from research.models.ensemble.blend import compute_ensemble
from research.models.ensemble.polymarket import compute_polymarket_signal
from research.models.ensemble.scoring import compute_quality_score, score_to_grade
from research.models.ensemble.types import EnsembleResult, MarketInput, OtherSignals
from research.models.ensemble.variance import compute_ci, compute_variance


def run_ensemble(
    current_price: float,
    markets: list[MarketInput],
    others: OtherSignals,
) -> EnsembleResult:
    pm = compute_polymarket_signal(markets)
    pm_signal = pm["signal"]
    avg_quality = pm["avg_quality"]
    warnings = pm["warnings"]

    ensemble = compute_ensemble(pm_signal, avg_quality, others)
    forecast_return = ensemble["forecast_return"]
    weights = ensemble["weights"]

    pm_eff = 0.40 * avg_quality

    forecast_price = current_price * (1 + forecast_return)

    raw_sigma = compute_variance(
        markets,
        weights.get("pm", 0.0),
        weights.get("sentiment", 0.0),
        others.sentiment_score,
    )

    horizon_frac = max(1, others.horizon_days) / 252
    sigma_floor = 0.10 * math.sqrt(horizon_frac)
    sigma = max(sigma_floor, raw_sigma)

    ci = compute_ci(forecast_price, sigma)

    signals_with_data = sum([
        1 if markets else 0,
        1 if (others.sentiment_score is not None and not math.isnan(others.sentiment_score)) else 0,
        1 if (others.fundamental_return is not None and not math.isnan(others.fundamental_return)) else 0,
        1 if (others.options_skew is not None and not math.isnan(others.options_skew)) else 0,
        1 if (others.markov_return is not None and not math.isnan(others.markov_return)) else 0,
    ])

    whale_count = sum(1 for m in markets if m.price_spike_detected)

    quality_score = compute_quality_score(
        markets, avg_quality, sigma, signals_with_data, whale_count
    )
    quality_grade = score_to_grade(quality_score)

    return EnsembleResult(
        forecast_return=forecast_return,
        forecast_price=forecast_price,
        ci_low95=ci["low"],
        ci_high95=ci["high"],
        sigma=sigma,
        quality_score=quality_score,
        quality_grade=quality_grade,
        pm_signal=pm_signal,
        pm_effective_weight=pm_eff,
        pm_normalized_weight=weights.get("pm", 0.0),
        avg_market_quality=avg_quality,
        warnings=warnings,
        weights=weights,
    )
