"""Calibration utilities — YES-bias correction and probability calibration.

Polymarket probabilities systematically overestimate the true probability
of events (the "YES bias"), especially for long-duration markets.  This
module applies a multiplicative correction factor and an expiry-date boost
to bring Polymarket-implied probabilities closer to empirical frequencies.

References: Reichenbach & Walther (2025), Polymarket calibration analysis.
"""

from __future__ import annotations

import math


YES_BIAS_MULTIPLIER = 0.95
"""Multiplicative YES-bias discount factor (5% haircut).

Used by the Markov distribution module where survival probabilities are
log-spaced and a multiplicative discount is more natural for interpolation
across price levels.

Rationale: Reichenbach & Walther (2025) report ~5% aggregate YES overpricing
across 124M Polymarket trades.
"""


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def adjust_yes_bias(p: float, beta: float = 0.035) -> float:
    """Additive YES-bias correction for conditional return estimation.

    Reichenbach & Walther (2025): systematic YES-overtrading across 124M
    Polymarket trades. Uses an *additive offset* (-beta when p > 0.5)
    because ensemble conditional returns are linear in p: small absolute
    shifts in p map directly to small return shifts.

    The Markov distribution module uses a *multiplicative* correction
    (YES_BIAS_MULTIPLIER) instead, because survival probabilities are
    log-spaced and a multiplicative discount is more natural for
    interpolation across price levels.

    Both corrections target the same phenomenon (YES overpricing) but use
    the form best suited to their downstream math.
    """
    if p > 0.5:
        return _clamp(p - beta, 0.01, 0.99)
    return _clamp(p, 0.01, 0.99)


def adjust_yes_bias_v2(p: float) -> float:
    """P1a — Empirically calibrated YES-bias correction (longshot-aware).

    Replaces the flat shift of `adjust_yes_bias` with a U-shaped curve that
    addresses the favourite-longshot bias documented in:
      - Reichenbach & Walther (2025)
      - l-marque calibration study (GitHub, 2024)
      - docs/polymarket-prediction-improvements-research-2026-07.md §2

    Regimes::

        p < 0.05         → strong longshot discount   (× 0.70)
        0.05 ≤ p ≤ 0.15  → linear interpolation       (× 0.70 → × 0.95)
        0.15 < p ≤ 0.85  → legacy mid-range behaviour (−0.035 when p > 0.5)
        p > 0.85         → mild favourite haircut     (−0.025)

    Output is clamped to [0.001, 0.999].
    """
    if p is None or p != p or p <= 0:  # NaN-safe via p != p
        return 0.001
    if p >= 1:
        return 0.999

    if p < 0.05:
        adjusted = p * 0.70
    elif p <= 0.15:
        t = (p - 0.05) / 0.10
        mult = 0.70 + t * (0.95 - 0.70)
        adjusted = p * mult
    elif p <= 0.85:
        adjusted = (p - 0.035) if p > 0.50 else p
    else:
        adjusted = p - 0.025

    return _clamp(adjusted, 0.001, 0.999)


def compute_expiry_boost(days_to_expiry: float) -> float:
    """P1b — Time-to-resolution multiplicative boost for market quality.

    Prediction-market prices are martingales; near-expiry markets carry
    sharper information. Schedule mirrors TS ``computeExpiryBoost``.
    """
    if days_to_expiry is None or days_to_expiry != days_to_expiry:
        return 1.0
    if days_to_expiry <= 1:
        return 1.50
    if days_to_expiry <= 7:
        return 1.20
    if days_to_expiry <= 30:
        return 1.00
    if days_to_expiry <= 90:
        return 0.85
    return 0.70

