"""Calibration utilities.

YES-bias correction and other probability calibration helpers.
"""

from __future__ import annotations


YES_BIAS_MULTIPLIER = 0.95


def adjust_yes_bias(p: float, beta: float = 0.035) -> float:
    """Apply additive YES-bias correction.

    Reichenbach & Walther (2025): systematic YES-overtrading across 124M
    Polymarket trades. Applies -β offset when p > 0.5.

    Parameters
    ----------
    p : float
        Raw YES probability [0, 1].
    beta : float
        Additive discount (default 0.035 = 3.5pp).

    Returns
    -------
    float
        Adjusted probability, clamped to [0.01, 0.99].
    """
    if p > 0.5:
        return max(0.01, min(0.99, p - beta))
    return max(0.01, min(0.99, p))
