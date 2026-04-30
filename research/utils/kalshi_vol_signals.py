"""Kalshi macro volatility signals — Python mirror of the TypeScript helper."""

from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

import requests

KalshiEventType = Literal["fomc", "cpi", "nfp", "other"]
FetchLike = Callable[[str, dict[str, str]], Any | Awaitable[Any]]
MS_PER_DAY = 24 * 60 * 60 * 1000


class KalshiUnconfiguredError(RuntimeError):
    """Raised when KALSHI_API_KEY is not set."""


@dataclass(frozen=True)
class KalshiVolSignal:
    event_at: str
    event_id: str
    probability: float
    intensity_boost: float
    event_type: KalshiEventType
    source_title: str


@dataclass(frozen=True)
class KalshiVolatilityCovariate:
    dates: list[str]
    values: list[float]
    active_signals: int
    peak_value: float


EVENT_PATTERNS: tuple[tuple[KalshiEventType, re.Pattern[str], float], ...] = (
    ("fomc", re.compile(r"\b(fomc|fed|federal reserve|rate decision|rate hike|rate cut)\b", re.I), 1.5),
    ("cpi", re.compile(r"\b(cpi|inflation|consumer price index)\b", re.I), 1.3),
    ("nfp", re.compile(r"\b(nonfarm payroll|non-farm payroll|payrolls|jobs report|nfp)\b", re.I), 1.2),
)


def _parse_date(value: str) -> int:
    iso = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _normalize_probability(market: dict[str, Any]) -> float | None:
    raw = market.get("yes_ask", market.get("last_price"))
    if not isinstance(raw, (int, float)):
        return None
    probability = raw / 100 if raw > 1 else float(raw)
    if 0 <= probability <= 1:
        return probability
    return None


def _classify_event_type(title: str) -> tuple[KalshiEventType, float] | None:
    for event_type, pattern, weight in EVENT_PATTERNS:
        if pattern.search(title):
            return event_type, weight
    return None


def _compute_intensity_boost(probability: float, weight: float, event_at: str, from_date: str) -> float:
    days_until_event = max(0.0, (_parse_date(event_at) - _parse_date(f"{from_date}T00:00:00Z")) / MS_PER_DAY)
    urgency = _clamp(1 - days_until_event / 14, 0.25, 1.0)
    return _clamp(0.2 + probability * weight * urgency, 0.1, 2.5)


def extract_kalshi_vol_signals_from_payload(
    payload: Any,
    *,
    from_date: str,
    to_date: str,
) -> list[KalshiVolSignal]:
    if not isinstance(payload, dict) or not isinstance(payload.get("markets"), list):
        raise ValueError("Kalshi response missing markets array")

    from_ms = _parse_date(f"{from_date}T00:00:00Z")
    to_ms = _parse_date(f"{to_date}T23:59:59Z")
    out: list[KalshiVolSignal] = []

    for market in payload["markets"]:
        if not isinstance(market, dict):
            continue
        title = market.get("title")
        event_at = market.get("expiration_time") or market.get("close_time") or market.get("settlement_time")
        if not isinstance(title, str) or not title.strip() or not isinstance(event_at, str):
            continue
        classified = _classify_event_type(title.strip())
        if classified is None:
            continue
        probability = _normalize_probability(market)
        if probability is None:
            continue
        event_ms = _parse_date(event_at)
        if event_ms < from_ms or event_ms > to_ms:
            continue
        event_type, weight = classified
        out.append(
            KalshiVolSignal(
                event_at=event_at,
                event_id=str(market.get("ticker") or f"{event_type}-{event_at[:10]}"),
                probability=probability,
                intensity_boost=_compute_intensity_boost(probability, weight, event_at, from_date),
                event_type=event_type,
                source_title=title.strip(),
            )
        )
    return out


async def fetch_kalshi_vol_signals(
    *,
    from_date: str,
    to_date: str,
    base_url: str = "https://trading-api.kalshi.com/trade-api/v2",
    api_key: str | None = None,
    fetch_impl: FetchLike | None = None,
) -> list[KalshiVolSignal]:
    key = api_key or os.environ.get("KALSHI_API_KEY")
    if not key:
        raise KalshiUnconfiguredError(
            "fetch_kalshi_vol_signals requires a KALSHI_API_KEY environment variable "
            "or an explicit api_key argument."
        )

    url = f"{base_url.rstrip('/')}/markets?status=open&limit=200"
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    if fetch_impl is None:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
    else:
        payload = fetch_impl(url, headers)
        if inspect.isawaitable(payload):
            payload = await payload

    return extract_kalshi_vol_signals_from_payload(payload, from_date=from_date, to_date=to_date)


def build_kalshi_volatility_covariate(
    dates: list[str],
    signals: list[KalshiVolSignal],
    lookahead_days: int = 5,
) -> KalshiVolatilityCovariate:
    values: list[float] = []
    for date in dates:
        date_ms = _parse_date(f"{date}T00:00:00Z")
        total = 0.0
        for signal in signals:
            days_ahead = (_parse_date(signal.event_at) - date_ms) / MS_PER_DAY
            if days_ahead < 0 or days_ahead > lookahead_days:
                continue
            decay = 1 - days_ahead / max(1, lookahead_days)
            total += signal.intensity_boost * decay
        values.append(round(total, 6))

    return KalshiVolatilityCovariate(
        dates=list(dates),
        values=values,
        active_signals=len(signals),
        peak_value=max(values) if values else 0.0,
    )
