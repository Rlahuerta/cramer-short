"""Polymarket signal aggregation across markets.

Mirrors TS logic from src/utils/ensemble.ts:
  - YES-bias correction per market
  - Quality-weighted conditional-return averaging
  - Structural warning generation (whale, transitory, ambiguous, longshot)
"""

from __future__ import annotations

from research.utils.calibration import adjust_yes_bias
from research.models.ensemble.quality import (
    compute_market_quality,
    longshot_microstructure_score,
    _MAX_LONGSHOT_MICRO_PENALTY,
    _LONGSHOT_MICRO_PENALTY_WARN_THRESHOLD,
    _LONGSHOT_PROBABILITY_THRESHOLD,
)
from research.models.ensemble.types import MarketInput


def compute_conditional_return(p_adjusted: float, delta_yes: float, delta_no: float) -> float:
    return p_adjusted * delta_yes + (1 - p_adjusted) * delta_no


def compute_polymarket_signal(markets: list[MarketInput]) -> dict:
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
                f'Market "{m.question}" has high YES bias (raw p={m.probability:.3f})'
            )
        if m.price_spike_detected:
            warnings.append(
                f'Market "{m.question}" has a price spike (possible whale activity) — quality discounted 50%'
            )
        if m.transitory_move:
            warnings.append(
                f'Market "{m.question}" shows a transitory 24-48h move — quality discounted 30%'
            )
        if m.market_semantics == "ambiguous":
            warnings.append(
                f'Market "{m.question}" has ambiguous resolution semantics — quality discounted 40%'
            )

        mp = m.probability
        if mp < _LONGSHOT_PROBABILITY_THRESHOLD or mp > 1 - _LONGSHOT_PROBABILITY_THRESHOLD:
            micro_score = longshot_microstructure_score(
                m.age_days, m.volume24h_usd, m.bid_ask_spread, m.signal_tier
            )
            penalty = _MAX_LONGSHOT_MICRO_PENALTY * (1 - micro_score)
            if penalty > _LONGSHOT_MICRO_PENALTY_WARN_THRESHOLD:
                range_label = "longshot" if mp < _LONGSHOT_PROBABILITY_THRESHOLD else "near-certain favourite"
                warnings.append(
                    f'Market "{m.question}" is a {range_label} (p={mp:.3f}) with poor microstructure — quality reduced by additional {penalty:.1%}'
                )

    signal = weighted_sum / total_weight if total_weight > 0 else 0.0
    avg_quality = total_weight / len(markets)

    return {"signal": signal, "avg_quality": avg_quality, "warnings": warnings}
