"""Market quality scoring and microstructure-aware weighting.

Mirrors TS logic from src/utils/ensemble.ts:
  - Depth-decay haircut, tier discounts, spread thinness
  - Logit-space velocity/jump preference over legacy pp thresholds
  - Longshot microstructure penalties, ambiguous-semantic discounts
"""

from __future__ import annotations

import math

from research.utils.calibration import compute_expiry_boost
from research.models.ensemble.types import MarketInput


TIER_SPREAD_BENCHMARKS: dict[str, float] = {
    "electoral": 0.07,
    "macro": 0.10,
    "geopolitical": 0.14,
}

_SPREAD_THINNESS_AMPLIFICATION = 0.40
_LONGSHOT_PROBABILITY_THRESHOLD = 0.07
_MAX_LONGSHOT_MICRO_PENALTY = 0.45
_LONGSHOT_MICRO_PENALTY_WARN_THRESHOLD = 0.08

_LEGACY_PRICE_VELOCITY_PPH_THRESHOLD = 2
_LEGACY_MAX_HOURLY_JUMP_THRESHOLD = 0.08
_LOGIT_PRICE_VELOCITY_THRESHOLD = 0.1
_LOGIT_MAX_HOURLY_JUMP_THRESHOLD = 0.35


def _clamp(v: float | float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _base_liquidity_quality(volume24h_usd: float) -> float:
    return min(1.0, math.log10(volume24h_usd + 1) / 6)


def depth_decay_haircut(days_to_expiry: float | None) -> float:
    """Multiply quality by a depth-decay factor based on days to expiry.

    Markets far from resolution carry less near-term signal. The haircut is
    1.0 for markets ≥ 30 days out and decays smoothly to 0.5 at expiry.

    Parameters
    ----------
    days_to_expiry : float or None
        Days until the market resolves. None or non-finite → no haircut (1.0).

    Returns
    -------
    float
        Multiplicative quality factor in [0.5, 1.0].
    """
    if days_to_expiry is None or not math.isfinite(days_to_expiry):
        return 1.0
    if days_to_expiry >= 30:
        return 1.0
    ratio = max(0, days_to_expiry) / 30
    return max(0.5, ratio**0.55)


def tier_spread_benchmark(tier: str | None) -> float:
    """Return the benchmark spread for a market signal tier.

    Parameters
    ----------
    tier : str or None
        Market tier: ``"electoral"`` (0.07), ``"macro"`` (0.10), or
        ``"geopolitical"`` / unknown (0.14).

    Returns
    -------
    float
        Benchmark spread fraction for the tier.
    """
    return TIER_SPREAD_BENCHMARKS.get(tier or "geopolitical", 0.14)


def longshot_microstructure_score(
    age_days: int | None,
    volume24h_usd: float,
    bid_ask_spread: float | None,
    signal_tier: str | None,
) -> float:
    """Score microstructure quality for extreme-probability (longshot) markets.

    Very low or very high probability markets are especially sensitive to thin
    liquidity and wide spreads. This composite score combines spread tightness
    (60%), market age (25%), and volume (15%). When spread data is missing,
    falls back to age + volume only.

    Parameters
    ----------
    age_days : int or None
        Market age in days (defaults to 21 if None).
    volume24h_usd : float
        24-hour trading volume in USD.
    bid_ask_spread : float or None
        Current bid-ask spread as a fraction of price.
    signal_tier : str or None
        Market tier used to look up the benchmark spread.

    Returns
    -------
    float
        Microstructure quality score in [0, 1].
    """
    w_age = min(1.0, (age_days or 21) / 21)
    vol_quality = _base_liquidity_quality(volume24h_usd)
    if bid_ask_spread is not None and math.isfinite(bid_ask_spread):
        spread_quality = max(0.0, 1.0 - bid_ask_spread / tier_spread_benchmark(signal_tier))
        raw_score = 0.60 * spread_quality + 0.25 * w_age + 0.15 * vol_quality
        return min(raw_score, spread_quality)
    return 0.60 * w_age + 0.40 * vol_quality


def compute_market_quality(m: MarketInput) -> float:
    """Compute a composite market quality score in [0, 1].

    Combines ten multiplicative adjustment factors into a single quality metric
    used to weight Polymarket signals in ensemble blending. Higher values
    indicate more reliable market-implied probabilities.

    Parameters
    ----------
    m : MarketInput
        Market metadata including probability, volume, spread, age, tier,
        and microstructure signals.

    Returns
    -------
    float
        Quality score clamped to [0.0, 1.0].

    Notes
    -----
    Adjustments applied multiplicatively in order:
    1. Age weight — maturity bonus (0 at creation, 1 after 21 days)
    2. Liquidity — log-scaled volume capped at 1.0, with expiry depth decay
    3. Tier tau — electoral (0.55), macro (0.90), geopolitical/other (0.75)
    4. Whale penalty — 50% discount when ``price_spike_detected``
    5. Transitory penalty — 30% discount when ``transitory_move`` (without spike)
    6. Stable-path bonus — 10% boost when stable and no flags active
    7. Expiry boost — multiplicative factor from ``compute_expiry_boost``
    8. Horizon gap penalty — up to 50% discount for anchor/forecast mismatch
    9. Spread thinness — amplified by illiquidity (thinness × 0.40)
    10. Logit velocity/jump — 20%/30% discount for rapid price movement
    11. Ambiguous semantics — 40% discount for unclear resolution criteria
    12. Longshot microstructure — up to 45% penalty for extreme probabilities
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

    if m.stable_path and not m.price_spike_detected and not m.transitory_move:
        w *= 1.1

    if m.days_to_expiry is not None:
        w *= compute_expiry_boost(m.days_to_expiry)

    if m.requested_horizon_days is not None and m.days_to_expiry is not None:
        horizon_gap = abs(m.days_to_expiry - m.requested_horizon_days)
        w *= max(0.5, 1 - 0.25 * horizon_gap)

    if m.bid_ask_spread is not None and math.isfinite(m.bid_ask_spread):
        raw_spread_frac = m.bid_ask_spread / tier_spread_benchmark(m.signal_tier)
        thinness = 1 - min(1.0, w_age * math.sqrt(w_liq_raw))
        amplification = 1 + _SPREAD_THINNESS_AMPLIFICATION * thinness
        w *= max(0.0, 1 - raw_spread_frac * amplification)

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

    if m.market_semantics == "ambiguous":
        w *= 0.6

    p = m.probability
    if p < _LONGSHOT_PROBABILITY_THRESHOLD or p > 1 - _LONGSHOT_PROBABILITY_THRESHOLD:
        micro_score = longshot_microstructure_score(
            m.age_days, m.volume24h_usd, m.bid_ask_spread, m.signal_tier
        )
        w *= 1 - _MAX_LONGSHOT_MICRO_PENALTY * (1 - micro_score)

    return _clamp(w, 0.0, 1.0)
