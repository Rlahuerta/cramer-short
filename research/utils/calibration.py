"""Calibration utilities.

YES-bias correction and other probability calibration helpers.
"""

from __future__ import annotations


YES_BIAS_MULTIPLIER = 0.95


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def adjust_yes_bias(p: float) -> float:
    """Multiplicative YES-bias correction for conditional return estimation.

    Reichenbach & Walther (2025): systematic YES-overtrading across 124M
    Polymarket trades. Applies multiplicative shrinkage toward 0 to
    correct overpricing of YES contracts. Mirrors TypeScript implementation.
    """
    return _clamp(p * YES_BIAS_MULTIPLIER, 0.01, 0.99)
