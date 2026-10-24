"""Entropy-based soft regime blending.

Reference: Blake et al. 2510.03236, matching TS.
"""

from __future__ import annotations

import math

from research.models.markov.core import RegimeState


def compute_regime_entropy(
    mixture: dict[RegimeState, float],
) -> float:
    """Normalized entropy of a regime probability mixture in [0, 1]."""
    denom = math.log(3)
    if denom <= 0:
        return 0.0
    entropy = 0.0
    for prob in mixture.values():
        if prob > 1e-12:
            entropy -= prob * math.log(prob)
    return min(1.0, entropy / denom)


def soft_regime_confidence_multiplier(entropy: float) -> float:
    """Confidence floor scaling: max(0.65, 1 - entropy * 0.35)."""
    return max(0.65, 1.0 - float(entropy) * 0.35)


def soft_regime_ci_scale(entropy: float) -> float:
    """CI width scaling: 1 + entropy * 0.35."""
    return 1.0 + float(entropy) * 0.35


def adjust_hmm_weight(
    hmm_weight: float,
    entropy: float,
) -> float:
    """Attenuate HMM influence under high posterior entropy."""
    return float(hmm_weight) * max(0.5, 1.0 - float(entropy) * 0.4)
