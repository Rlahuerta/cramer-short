"""Forecast variance and confidence interval computation.

Mirrors TS logic from src/utils/ensemble.ts:
  - Quality-weighted variance from market spreads and probabilities
  - Sentiment variance contribution
  - 95% CI via 1.96 sigma
"""

from __future__ import annotations

import math

from research.utils.calibration import adjust_yes_bias
from research.models.ensemble.quality import compute_market_quality
from research.models.ensemble.types import MarketInput


def compute_variance(
    markets: list[MarketInput],
    pm_weight: float,
    sent_weight: float,
    sent_signal: float | None,
) -> float:
    if not markets:
        return 0.05

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
    return {
        "low": forecast_price * (1 - 1.96 * sigma),
        "high": forecast_price * (1 + 1.96 * sigma),
    }
