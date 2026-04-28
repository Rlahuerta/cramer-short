"""Polymarket CLOB microstructure helpers (Python mirror).

Mirrors the pure-math portion of `src/tools/finance/polymarket-clob.ts`.
HTTP fetchers are intentionally omitted from the research package — live
network calls are TS-side only. Used by experiments to reason about
quality discounts identically to the production model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ClobPricePoint:
    t_sec: int
    p: float


def parse_clob_price_history(raw: object) -> list[ClobPricePoint]:
    """Parse the `/prices-history` JSON shape into a sorted, valid series."""
    if not isinstance(raw, dict):
        return []
    hist = raw.get("history")
    if not isinstance(hist, list):
        return []
    out: list[ClobPricePoint] = []
    for item in hist:
        if not isinstance(item, dict):
            continue
        try:
            t = float(item.get("t"))
            p = float(item.get("p"))
        except (TypeError, ValueError):
            continue
        if t != t or p != p:  # NaN check
            continue
        if p < 0 or p > 1:
            continue
        out.append(ClobPricePoint(t_sec=int(t), p=p))
    out.sort(key=lambda x: x.t_sec)
    return out


def compute_price_velocity_pp_h(
    history: Sequence[ClobPricePoint], lookback_hours: int = 6
) -> float:
    """OLS slope of price (in pp) vs. time (in hours) over lookback window."""
    if len(history) < 2:
        return 0.0
    newest = history[-1].t_sec
    cutoff = newest - lookback_hours * 3600
    window = [pt for pt in history if pt.t_sec >= cutoff]
    if len(window) < 2:
        return 0.0

    n = len(window)
    sum_x = sum_y = sum_xy = sum_xx = 0.0
    for pt in window:
        x = pt.t_sec / 3600.0
        y = pt.p * 100.0
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_xx += x * x
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def compute_max_hourly_jump(
    history: Sequence[ClobPricePoint], window_hours: int = 24
) -> float:
    """Maximum |Δp| between consecutive points within window_hours of latest."""
    if len(history) < 2:
        return 0.0
    newest = history[-1].t_sec
    cutoff = newest - window_hours * 3600
    max_abs = 0.0
    for i in range(1, len(history)):
        cur = history[i]
        prev = history[i - 1]
        if cur.t_sec < cutoff:
            continue
        d = abs(cur.p - prev.p)
        if d > max_abs:
            max_abs = d
    return max_abs
