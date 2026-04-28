"""P3b — VIX-based volatility regime classifier (Python mirror).

Source: docs/polymarket-prediction-improvements-research-2026-07.md §9

Mirrors ``src/utils/vol-regime.ts``.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal

VolRegime = Literal["sticky_strike", "transitional", "sticky_implied_tree"]

_VIX_TRANSITIONAL = 15.0
_VIX_FEAR = 25.0


def get_volatility_regime(vix: float) -> VolRegime:
    if not isfinite(vix) or vix < _VIX_TRANSITIONAL:
        return "sticky_strike"
    if vix < _VIX_FEAR:
        return "transitional"
    return "sticky_implied_tree"


def leverage_vol_multiplier(regime: VolRegime, z: float, asset_class: str) -> float:
    """Per-step vol multiplier; gated on equity/gold (§9.3)."""
    if regime != "sticky_implied_tree":
        return 1.0
    if asset_class not in ("equity", "gold"):
        return 1.0
    if not isfinite(z) or z == 0:
        return 1.0
    return 1.4 if z < 0 else 0.8
