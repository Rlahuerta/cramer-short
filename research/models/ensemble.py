"""Polymarket weighted ensemble forecast engine.

Mirrors TS logic from src/utils/ensemble.ts:
  - YES-bias correction (additive and multiplicative)
  - Market quality weighting (age, liquidity, tier, whale penalty)
  - Conditional return computation
  - Signal ensemble blending
  - Variance and confidence interval
  - Quality scoring (0-100, A-D grade)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from research.utils.calibration import adjust_yes_bias, YES_BIAS_MULTIPLIER

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Tier = Literal["macro", "geopolitical", "electoral"]


@dataclass
class MarketInput:
    question: str
    probability: float  # raw [0, 1]
    volume24h_usd: float
    age_days: int | None = None
    price_spike_detected: bool = False
    transitory_move: bool = False
    signal_tier: Tier = "geopolitical"
    delta_yes: float = 0.06
    delta_no: float = -0.04


@dataclass
class OtherSignals:
    sentiment_score: float | None = None  # -1 to +1
    fundamental_return: float | None = None  # decimal
    options_skew: float | None = None  # -1/0/+1
    markov_return: float | None = None  # decimal
    horizon_days: int = 7


@dataclass
class EnsembleResult:
    forecast_return: float
    forecast_price: float
    ci_low95: float
    ci_high95: float
    sigma: float
    quality_score: float
    quality_grade: str
    pm_signal: float
    pm_effective_weight: float
    avg_market_quality: float
    warnings: list[str]
    weights: dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# YES-bias correction (imported from research.utils.calibration)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Market quality
# ---------------------------------------------------------------------------

def compute_market_quality(m: MarketInput) -> float:
    """Composite quality weight for a single Polymarket market.

    Factors: age, liquidity (log-volume), tier discount,
    whale penalty (50%), transitory discount (30%).
    """
    w_age = min(1.0, (m.age_days or 21) / 21)
    w_liq = min(1.0, math.log10(m.volume24h_usd + 1) / 6)

    if m.signal_tier == "macro":
        tau = 0.90
    elif m.signal_tier == "electoral":
        tau = 0.55
    else:
        tau = 0.75

    delta_whale = 1.0 if m.price_spike_detected else 0.0
    delta_transitory = 1.0 if (m.transitory_move and not m.price_spike_detected) else 0.0

    w = w_age * w_liq * tau * (1 - delta_whale * 0.5) * (1 - delta_transitory * 0.3)
    return _clamp(w, 0.0, 1.0)


def compute_conditional_return(p_adjusted: float, delta_yes: float, delta_no: float) -> float:
    """Expected asset return conditioned on adjusted YES probability."""
    return p_adjusted * delta_yes + (1 - p_adjusted) * delta_no


# ---------------------------------------------------------------------------
# Polymarket signal aggregation
# ---------------------------------------------------------------------------

def compute_polymarket_signal(markets: list[MarketInput]) -> dict:
    """Aggregate Polymarket signal across all markets.

    Returns quality-weighted average conditional return, mean quality,
    and warnings.
    """
    if not markets:
        return {
            "signal": 0.0,
            "avg_quality": 0.0,
            "warnings": ["No Polymarket markets found — PM signal omitted"],
        }

    warnings: list[str] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for m in markets:
        p_adj = adjust_yes_bias(m.probability)
        w = compute_market_quality(m)
        r = compute_conditional_return(p_adj, m.delta_yes, m.delta_no)
        weighted_sum += w * r
        total_weight += w

        if abs(m.probability - adjust_yes_bias(m.probability)) > 0.1:
            warnings.append(
                f'Market "{m.question}" has high YES bias '
                f"(raw p={m.probability:.3f})"
            )
        if m.price_spike_detected:
            warnings.append(
                f'Market "{m.question}" has a price spike '
                f"(possible whale activity) — quality discounted 50%"
            )
        if m.transitory_move:
            warnings.append(
                f'Market "{m.question}" shows a transitory 24-48h move '
                f"— quality discounted 30%"
            )

    signal = weighted_sum / total_weight if total_weight > 0 else 0.0
    avg_quality = total_weight / len(markets)

    return {"signal": signal, "avg_quality": avg_quality, "warnings": warnings}


# ---------------------------------------------------------------------------
# Ensemble blending
# ---------------------------------------------------------------------------

def compute_ensemble(
    pm_signal: float,
    pm_avg_quality: float,
    others: OtherSignals,
) -> dict:
    """Combine Polymarket signal with auxiliary signals.

    Base weights: PM=0.40, sentiment=0.20, fundamental=0.25, options=0.15, markov=0.20.
    PM weight is scaled by pm_avg_quality before normalization.
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


# ---------------------------------------------------------------------------
# Variance and CI
# ---------------------------------------------------------------------------

def compute_variance(
    markets: list[MarketInput],
    pm_weight: float,
    sent_weight: float,
    sent_signal: float | None,
) -> float:
    """Estimate total forecast standard deviation."""
    if not markets:
        return 0.05  # default 5% uncertainty

    weights = [compute_market_quality(m) for m in markets]
    total_weight = sum(weights)

    variance_pm = 0.0
    for i, m in enumerate(markets):
        p_adj = adjust_yes_bias(m.probability)
        norm_w = weights[i] / total_weight if total_weight > 0 else 0.0
        spread = m.delta_yes - m.delta_no
        variance_pm += norm_w * norm_w * p_adj * (1 - p_adj) * spread * spread

    variance_sent = (sent_weight * 0.04) ** 2 if sent_weight else 0.0

    variance_combined = pm_weight**2 * variance_pm + variance_sent
    return math.sqrt(variance_combined) * 1.2


def compute_ci(forecast_price: float, sigma: float) -> dict:
    """95% confidence interval around forecast price using 1.96σ."""
    return {
        "low": forecast_price * (1 - 1.96 * sigma),
        "high": forecast_price * (1 + 1.96 * sigma),
    }


# ---------------------------------------------------------------------------
# Quality score
# ---------------------------------------------------------------------------

def compute_quality_score(
    markets: list[MarketInput],
    avg_quality: float,
    sigma: float,
    signals_with_data: int,
    whale_count: int,
) -> float:
    """Composite quality score in [0, 100]."""
    s1 = 30 * min(len(markets), 5) / 5
    s2 = 25 * avg_quality
    s3 = 20 * max(0, 1 - sigma / 0.20)
    s4 = 15 * (signals_with_data / 4)
    s5 = 10 * (1 - whale_count / len(markets)) if markets else 0
    return round(min(100, max(0, s1 + s2 + s3 + s4 + s5)))


def score_to_grade(score: float) -> str:
    """Convert quality score to letter grade."""
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# End-to-end ensemble forecast
# ---------------------------------------------------------------------------

def run_ensemble(
    current_price: float,
    markets: list[MarketInput],
    others: OtherSignals,
) -> EnsembleResult:
    """End-to-end ensemble forecast.

    Parameters
    ----------
    current_price : float
        Current asset price.
    markets : list[MarketInput]
        Polymarket markets.
    others : OtherSignals
        Auxiliary signals.

    Returns
    -------
    EnsembleResult
        Full forecast result with price, CI, quality score, and metadata.
    """
    # Step 1: Polymarket signal
    pm = compute_polymarket_signal(markets)
    pm_signal = pm["signal"]
    avg_quality = pm["avg_quality"]
    warnings = pm["warnings"]

    # Step 2: Blend
    ensemble = compute_ensemble(pm_signal, avg_quality, others)
    forecast_return = ensemble["forecast_return"]
    weights = ensemble["weights"]

    # Step 3: Forecast price
    forecast_price = current_price * (1 + forecast_return)

    # Step 4: Uncertainty
    raw_sigma = compute_variance(
        markets,
        weights.get("pm", 0.0),
        weights.get("sentiment", 0.0),
        others.sentiment_score,
    )

    # Floor based on horizon
    horizon_frac = max(1, others.horizon_days) / 252
    sigma_floor = 0.10 * math.sqrt(horizon_frac)
    sigma = max(sigma_floor, raw_sigma)

    # Step 5: CI
    ci = compute_ci(forecast_price, sigma)

    # Step 6: Count available signals
    signals_with_data = sum([
        1 if markets else 0,
        1 if (others.sentiment_score is not None and not math.isnan(others.sentiment_score)) else 0,
        1 if (others.fundamental_return is not None and not math.isnan(others.fundamental_return)) else 0,
        1 if (others.options_skew is not None and not math.isnan(others.options_skew)) else 0,
        1 if (others.markov_return is not None and not math.isnan(others.markov_return)) else 0,
    ])

    # Step 7: Whale count
    whale_count = sum(1 for m in markets if m.price_spike_detected)

    # Step 8-9: Quality
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
        pm_effective_weight=weights.get("pm", 0.0),
        avg_market_quality=avg_quality,
        warnings=warnings,
        weights=weights,
    )
