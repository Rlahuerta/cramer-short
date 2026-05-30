"""Quality scoring and grade assignment for ensemble forecasts.

Mirrors TS logic from src/utils/ensemble.ts:
  - Five-component score: market count, quality, sigma, signals, whale ratio
  - A/B/C/D grade thresholds at 80/60/40
"""

from __future__ import annotations

from research.models.ensemble.types import MarketInput


def compute_quality_score(
    markets: list[MarketInput],
    avg_quality: float,
    sigma: float,
    signals_with_data: int,
    whale_count: int,
) -> float:
    s1 = 30 * min(len(markets), 5) / 5
    s2 = 25 * avg_quality
    s3 = 20 * max(0, 1 - sigma / 0.20)
    s4 = 15 * (signals_with_data / 4)
    s5 = 10 * (1 - whale_count / len(markets)) if markets else 0
    return round(min(100, max(0, s1 + s2 + s3 + s4 + s5)))


def score_to_grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"
