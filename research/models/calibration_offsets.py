"""Domain × horizon Polymarket recalibration.

Mirrors src/tools/finance/calibration-offsets.ts.

Bayesian hierarchical decomposition of prediction-market miscalibration:
    z' = (1 + β · log1p(T_days)) · Φ⁻¹(q) + α
    p' = Φ(z')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm

Domain = Literal["politics", "sports", "crypto", "macro", "unknown"]


@dataclass(frozen=True)
class CalibrationOffset:
    alpha: float
    beta_per_log_t: float


DOMAIN_OFFSETS: dict[str, CalibrationOffset] = {
    "politics": CalibrationOffset(alpha=0.15, beta_per_log_t=0.05),
    "sports":   CalibrationOffset(alpha=0.00, beta_per_log_t=0.00),
    "crypto":   CalibrationOffset(alpha=0.05, beta_per_log_t=0.03),
    "macro":    CalibrationOffset(alpha=0.02, beta_per_log_t=0.04),
    "unknown":  CalibrationOffset(alpha=0.00, beta_per_log_t=0.00),
}


def recalibrate_polymarket_price(q_prob: float, domain: str, days_to_expiry: float) -> float:
    if q_prob <= 0:
        return 0.0
    if q_prob >= 1:
        return 1.0
    offset = DOMAIN_OFFSETS.get(domain, DOMAIN_OFFSETS["unknown"])
    if offset.alpha == 0 and offset.beta_per_log_t == 0:
        return q_prob
    import math
    days = max(days_to_expiry, 1.0)
    slope = 1 + offset.beta_per_log_t * math.log1p(days)
    z = norm.ppf(q_prob)
    z_recal = slope * z + offset.alpha
    return float(norm.cdf(z_recal))
