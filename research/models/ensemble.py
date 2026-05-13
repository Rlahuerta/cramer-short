"""Polymarket weighted ensemble forecast engine.

Mirrors TS logic from src/utils/ensemble.ts:
  - YES-bias correction (additive for ensemble, multiplicative for Markov)
  - Market quality weighting (age, liquidity, tier, whale, transitory, depth-decay)
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

from research.utils.calibration import (
    adjust_yes_bias,
    YES_BIAS_MULTIPLIER,
    adjust_yes_bias_v2,
    compute_expiry_boost,
)

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
    # P1b — Days remaining to market resolution.
    days_to_expiry: float | None = None
    # P5 — Requested short-horizon expiry target in days.
    requested_horizon_days: int | None = None
    # P1d — Bid-ask spread on YES token in [0, 1].
    bid_ask_spread: float | None = None
    # P1e — Per-hour drift in pp (positive = momentum, negative = fading).
    price_velocity_pp_h: float | None = None
    # P3 — Preferred per-hour drift signal in log-odds units.
    price_velocity_logit_per_hour: float | None = None
    # P1e — Largest single-hour |Δp| over the prior 24h window.
    max_hourly_jump: float | None = None
    # P3 — Preferred single-hour jump signal in log-odds units.
    max_hourly_logit_jump: float | None = None
    # P2 — Semantic classification. 'ambiguous' applies a 40% quality discount.
    market_semantics: str | None = None
    # P4 — Stable multi-snapshot path; rewards persistent markets.
    stable_path: bool = False


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
    pm_normalized_weight: float
    avg_market_quality: float
    warnings: list[str]
    weights: dict[str, float]


# ---------------------------------------------------------------------------
# Phase 3 — Tier-aware spread benchmarks
# ---------------------------------------------------------------------------

TIER_SPREAD_BENCHMARKS: dict[str, float] = {
    "electoral": 0.07,
    "macro": 0.10,
    "geopolitical": 0.14,
}
"""Dubach (2026) Phase 3 — Category-aware spread benchmarks.

Different signal families carry structurally different expected bid-ask spreads.
Electoral markets are most actively traded (tight spreads, 7% benchmark).
Macro/policy markets are well-arbitraged (10% benchmark, legacy global default).
Geopolitical markets trade less frequently (wider spread structurally expected, 14%).
"""

# ---------------------------------------------------------------------------
# Phase 3 — Microstructure-aware weighting constants
# ---------------------------------------------------------------------------

_SPREAD_THINNESS_AMPLIFICATION = 0.40
_LONGSHOT_PROBABILITY_THRESHOLD = 0.07
_MAX_LONGSHOT_MICRO_PENALTY = 0.45
_LONGSHOT_MICRO_PENALTY_WARN_THRESHOLD = 0.08

# Legacy pp-space thresholds (fallback when logit-space unavailable)
_LEGACY_PRICE_VELOCITY_PPH_THRESHOLD = 2
_LEGACY_MAX_HOURLY_JUMP_THRESHOLD = 0.08
# Logit-space thresholds (preferred)
_LOGIT_PRICE_VELOCITY_THRESHOLD = 0.1
_LOGIT_MAX_HOURLY_JUMP_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float | np.floating, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


# ---------------------------------------------------------------------------
# YES-bias correction (imported from research.utils.calibration)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Market quality
# ---------------------------------------------------------------------------


def _base_liquidity_quality(volume24h_usd: float) -> float:
    """Raw liquidity quality from log-volume (before depth-decay haircut)."""
    return min(1.0, math.log10(volume24h_usd + 1) / 6)


def depth_decay_haircut(days_to_expiry: float | None) -> float:
    """W3 Idea 1a — Dubach (2026) depth-decay haircut.

    Empirical Polymarket depth shrinks materially as a contract approaches
    resolution (arXiv 2604.24366 §4). volume24hUsd is backward-looking, so
    this corrects a *liquidity* misestimate — independent of the
    information-value compute_expiry_boost.

      >= 30 d        →  1.00
      undefined/NaN  →  1.00
      day-by-day     →  (d/30)^0.55, floored at 0.5
    """
    if days_to_expiry is None or not math.isfinite(days_to_expiry):
        return 1.0
    if days_to_expiry >= 30:
        return 1.0
    ratio = max(0, days_to_expiry) / 30
    return max(0.5, ratio**0.55)


def tier_spread_benchmark(tier: str | None) -> float:
    """Returns the spread benchmark for a signal tier. Falls back to 'geopolitical'."""
    return TIER_SPREAD_BENCHMARKS.get(tier or "geopolitical", 0.14)  # type: ignore[arg-type]


def longshot_microstructure_score(
    age_days: int | None,
    volume24h_usd: float,
    bid_ask_spread: float | None,
    signal_tier: str | None,
) -> float:
    """Dubach (2026) Phase 1 — microstructure quality score for longshot/near-certain contracts.

    Longshot spread premium finding: low-probability contracts carry systematically
    wider bid-ask spreads. When spread data is available, it is the most informative
    live signal. Age and volume are supporting signals only.

    Weighting when spread is available: spread 0.60, age 0.25, volume 0.15.
    Fallback when spread is absent: age 0.60, volume 0.40.
    Returns a score in [0, 1]; higher = better microstructure.
    """
    w_age = min(1.0, (age_days or 21) / 21)
    vol_quality = _base_liquidity_quality(volume24h_usd)
    if bid_ask_spread is not None and math.isfinite(bid_ask_spread):
        spread_quality = max(0.0, 1.0 - bid_ask_spread / tier_spread_benchmark(signal_tier))
        raw_score = 0.60 * spread_quality + 0.25 * w_age + 0.15 * vol_quality
        return min(raw_score, spread_quality)
    return 0.60 * w_age + 0.40 * vol_quality


def compute_market_quality(m: MarketInput) -> float:
    """Composite quality weight for a single Polymarket market.

    Factors: age, liquidity (log-volume with depth-decay), tier discount,
    whale penalty (50%), transitory discount (30%), stablePath boost (1.1x),
    expiry boost, horizon gap penalty, tier-aware spread+thinness penalty,
    logit-space velocity/jump preference, ambiguous semantics penalty,
    longshot microstructure penalty.
    """
    w_age = min(1.0, (m.age_days or 21) / 21)
    w_liq_raw = _base_liquidity_quality(m.volume24h_usd)
    w_liq = w_liq_raw * depth_decay_haircut(m.days_to_expiry)

    if m.signal_tier == "macro":
        tau = 0.90
    elif m.signal_tier == "electoral":
        tau = 0.55
    else:
        tau = 0.75

    delta_whale = 1.0 if m.price_spike_detected else 0.0
    delta_transitory = 1.0 if (m.transitory_move and not m.price_spike_detected) else 0.0

    w = w_age * w_liq * tau * (1 - delta_whale * 0.5) * (1 - delta_transitory * 0.3)

    # P4 — Stable multi-snapshot path; modest quality boost.
    if m.stable_path and not m.price_spike_detected and not m.transitory_move:
        w *= 1.1

    # P1b — time-to-resolution boost.
    if m.days_to_expiry is not None:
        w *= compute_expiry_boost(m.days_to_expiry)

    # P5 — Requested horizon gap penalty.
    if m.requested_horizon_days is not None and m.days_to_expiry is not None:
        horizon_gap = abs(m.days_to_expiry - m.requested_horizon_days)
        w *= max(0.5, 1 - 0.25 * horizon_gap)

    # P1d (strengthened) — tier-aware bid-ask spread × thinness compound discount.
    if m.bid_ask_spread is not None and math.isfinite(m.bid_ask_spread):
        raw_spread_frac = m.bid_ask_spread / tier_spread_benchmark(m.signal_tier)
        # thinness uses raw (un-decayed) liquidity so near-expiry decay does not
        # artificially inflate thinness and compound the spread penalty.
        thinness = 1 - min(1.0, w_age * math.sqrt(w_liq_raw))
        amplification = 1 + _SPREAD_THINNESS_AMPLIFICATION * thinness
        w *= max(0.0, 1 - raw_spread_frac * amplification)

    # P3 — prefer logit-space microstructure when present.
    has_logit_velocity = (
        m.price_velocity_logit_per_hour is not None
        and math.isfinite(m.price_velocity_logit_per_hour)
    )
    has_logit_jump = (
        m.max_hourly_logit_jump is not None
        and math.isfinite(m.max_hourly_logit_jump)
    )

    if has_logit_velocity:
        if abs(m.price_velocity_logit_per_hour) > _LOGIT_PRICE_VELOCITY_THRESHOLD:  # type: ignore[arg-type]
            w *= 0.80
    elif m.price_velocity_pp_h is not None and abs(m.price_velocity_pp_h) > _LEGACY_PRICE_VELOCITY_PPH_THRESHOLD:
        w *= 0.80

    if has_logit_jump:
        if m.max_hourly_logit_jump > _LOGIT_MAX_HOURLY_JUMP_THRESHOLD:  # type: ignore[arg-type]
            w *= 0.70
    elif m.max_hourly_jump is not None and m.max_hourly_jump > _LEGACY_MAX_HOURLY_JUMP_THRESHOLD:
        w *= 0.70

    # P2 — ambiguous semantics penalty.
    if m.market_semantics == "ambiguous":
        w *= 0.6

    # P3b — Longshot microstructure penalty.
    p = m.probability
    if p < _LONGSHOT_PROBABILITY_THRESHOLD or p > 1 - _LONGSHOT_PROBABILITY_THRESHOLD:
        micro_score = longshot_microstructure_score(
            m.age_days, m.volume24h_usd, m.bid_ask_spread, m.signal_tier
        )
        w *= 1 - _MAX_LONGSHOT_MICRO_PENALTY * (1 - micro_score)

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
        if m.market_semantics == "ambiguous":
            warnings.append(
                f'Market "{m.question}" has ambiguous resolution semantics '
                f"— quality discounted 40%"
            )

        # P3b — Warn on longshot/favourite with poor microstructure.
        mp = m.probability
        if mp < _LONGSHOT_PROBABILITY_THRESHOLD or mp > 1 - _LONGSHOT_PROBABILITY_THRESHOLD:
            micro_score = longshot_microstructure_score(
                m.age_days, m.volume24h_usd, m.bid_ask_spread, m.signal_tier
            )
            penalty = _MAX_LONGSHOT_MICRO_PENALTY * (1 - micro_score)
            if penalty > _LONGSHOT_MICRO_PENALTY_WARN_THRESHOLD:
                range_label = "longshot" if mp < _LONGSHOT_PROBABILITY_THRESHOLD else "near-certain favourite"
                warnings.append(
                    f'Market "{m.question}" is a {range_label} (p={mp:.3f}) with poor microstructure — '
                    f"quality reduced by additional {penalty:.1%}"
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

    # PM effective weight (pre-normalization, for provenance display)
    pm_eff = 0.40 * avg_quality

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
        pm_effective_weight=pm_eff,
        pm_normalized_weight=weights.get("pm", 0.0),
        avg_market_quality=avg_quality,
        warnings=warnings,
        weights=weights,
    )
