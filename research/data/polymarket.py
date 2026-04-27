"""Polymarket Gamma API client.

Mirrors TS logic from polymarket.ts:
  - Tag-slug inference (keyword param is non-functional)
  - Client-side text filtering
  - 5-minute TTL cache with LRU eviction
  - Exponential backoff retry
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com"
CACHE_TTL_SECONDS = 5 * 60
CACHE_MAX_ENTRIES = 64
_RETRY_DELAYS = [1.0, 2.0, 4.0]

# ---------------------------------------------------------------------------
# Tag-slug map (mirrors TS logic)
# ---------------------------------------------------------------------------

_TAG_SLUG_MAP: dict[str, list[str]] = {
    "bitcoin": ["bitcoin", "crypto-prices", "crypto"],
    "btc": ["bitcoin", "crypto-prices", "crypto"],
    "ethereum": ["ethereum", "crypto-prices", "crypto"],
    "eth": ["ethereum", "crypto-prices", "crypto"],
    "solana": ["solana", "crypto-prices", "crypto"],
    "sol": ["solana", "crypto-prices", "crypto"],
    "crypto": ["crypto-prices", "crypto", "bitcoin"],
    "fed": ["fed-rates", "fed", "economic-policy"],
    "fed rate": ["fed-rates", "fed", "economic-policy"],
    "tariff": ["tariffs", "trade", "economic-policy"],
    "oil": ["oil", "energy", "commodities"],
    "election": ["us-elections", "politics", "us-politics"],
    "trump": ["us-politics", "politics", "us-elections"],
    "recession": ["economy", "macro", "economic-policy"],
    "gdp": ["economy", "macro", "economic-policy"],
    "inflation": ["economy", "macro", "fed"],
    "nvidia": ["nvidia", "tech", "ai"],
    "nvda": ["nvidia", "tech", "ai"],
}


def _infer_tag_slugs(query: str) -> list[str]:
    """Infer tag slugs from query text."""
    q = query.lower().strip()
    # Direct lookup
    if q in _TAG_SLUG_MAP:
        return _TAG_SLUG_MAP[q]
    # Partial match
    for key, slugs in _TAG_SLUG_MAP.items():
        if key in q:
            return slugs
    return []


# ---------------------------------------------------------------------------
# Client-side text filtering
# ---------------------------------------------------------------------------

_STOP_WORDS = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to"}


def _tokenize(text: str) -> set[str]:
    return {
        w.strip(".,!?;:'\"")
        for w in text.lower().split()
        if w.strip(".,!?;:'\"") not in _STOP_WORDS
    }


def _question_matches_query(question: str, query: str) -> bool:
    """Client-side relevance filter (mirrors TS questionMatchesQuery)."""
    q_tokens = _tokenize(query)
    q_tokens.add(query.lower().strip())
    q_text = query.lower()

    q_str = question.lower()
    q_tokens = _tokenize(q_text)

    # Score tokens present in question
    hits = sum(1 for t in q_tokens if t in q_str)
    return hits > 0 or any(t in q_str for t in q_tokens)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    data: pd.DataFrame
    expires_at: float


_search_cache: dict[str, _CacheEntry] = {}


def _cache_get(key: str) -> pd.DataFrame | None:
    entry = _search_cache.get(key)
    if entry is None:
        return None
    if time.time() > entry.expires_at:
        _search_cache.pop(key, None)
        return None
    return entry.data


def _cache_set(key: str, data: pd.DataFrame) -> None:
    if len(_search_cache) >= CACHE_MAX_ENTRIES:
        # Evict oldest (first inserted)
        oldest = next(iter(_search_cache))
        _search_cache.pop(oldest, None)
    _search_cache[key] = _CacheEntry(
        data=data,
        expires_at=time.time() + CACHE_TTL_SECONDS,
    )


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _fetch_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            if 400 <= resp.status_code < 500:
                resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                delay = _RETRY_DELAYS[attempt] if attempt < len(_RETRY_DELAYS) else 4.0
                time.sleep(delay)
    if last_err:
        raise last_err
    return []


# ---------------------------------------------------------------------------
# Market formatting
# ---------------------------------------------------------------------------

def _format_volume(vol: Any) -> float:
    if isinstance(vol, (int, float)):
        return float(vol)
    if isinstance(vol, str):
        return float(vol.replace(",", ""))
    return 0.0


def _parse_markets(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract and flatten markets from Gamma events response."""
    markets: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        for m in event.get("markets", []):
            if not isinstance(m, dict):
                continue
            markets.append({
                "market_id": m.get("id", m.get("marketId", "")),
                "question": m.get("question", ""),
                "probability": float(m.get("probability", 0.0)),
                "volume_24h": _format_volume(m.get("volume24hr", m.get("volume24h", 0))),
                "age_days": None,  # computed later
                "end_date": m.get("endDate"),
                "event_title": event.get("title", ""),
            })
    return markets


def _compute_age_days(created_at: str | None) -> int | None:
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - dt).days
        return max(0, days)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_polymarket_markets(
    query: str,
    limit: int = 10,
    min_volume_24h: float = 1000.0,
) -> pd.DataFrame:
    """Search Polymarket prediction markets.

    Parameters
    ----------
    query : str
        Search query (e.g. 'bitcoin', 'fed rate cut').
    limit : int
        Max markets to return after filtering.
    min_volume_24h : float
        Minimum 24h volume filter.

    Returns
    -------
    pd.DataFrame
        Columns: market_id, question, probability, volume_24h, age_days,
        end_date, event_title.
    """
    cache_key = f"{query.lower().strip()}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached.copy()

    tag_slugs = _infer_tag_slugs(query)
    all_markets: list[dict[str, Any]] = []

    if tag_slugs:
        for tag in tag_slugs:
            url = f"{GAMMA_BASE}/events"
            params = {
                "tag_slug": tag,
                "active": "true",
                "order": "volume24hr",
                "limit": min(limit * 8, 80),
            }
            try:
                events = _fetch_with_retry(url, params)
                markets = _parse_markets(events)
                all_markets.extend(markets)
                if len(all_markets) >= limit * 4:
                    break
            except Exception:
                continue
    else:
        # Broad top-events fetch
        url = f"{GAMMA_BASE}/events"
        params = {
            "active": "true",
            "order": "volume24hr",
            "limit": min(limit * 8, 80),
        }
        try:
            events = _fetch_with_retry(url, params)
            all_markets = _parse_markets(events)
        except Exception:
            all_markets = []

    # Client-side text filtering
    filtered = [
        m for m in all_markets
        if _question_matches_query(m["question"], query)
        and m["volume_24h"] >= min_volume_24h
    ]

    # Deduplicate by market_id
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for m in filtered:
        mid = m["market_id"]
        if mid and mid not in seen:
            seen.add(mid)
            deduped.append(m)

    # Sort by volume desc and limit
    deduped.sort(key=lambda x: x["volume_24h"], reverse=True)
    result = deduped[:limit]

    df = pd.DataFrame(result)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "market_id", "question", "probability", "volume_24h",
                "age_days", "end_date", "event_title",
            ]
        )
    else:
        df["probability"] = df["probability"].astype(float)
        df["volume_24h"] = df["volume_24h"].astype(float)

    _cache_set(cache_key, df)
    return df.copy()
