"""Polymarket Gamma API client.

Mirrors TS logic from polymarket.ts:
  - Tag-slug inference (keyword param is non-functional)
  - Client-side text filtering
  - 5-minute TTL cache with LRU eviction
  - Exponential backoff retry
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from ..models.jump_diffusion import classify_jump_direction

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

_TAG_SLUG_PATTERNS: list[tuple[list[str], list[str]]] = [
    (["bitcoin", "btc"], ["bitcoin", "crypto-prices", "crypto"]),
    (["ethereum", "eth"], ["ethereum", "crypto-prices", "crypto"]),
    (["solana", "sol", "crypto", "defi", "nft", "web3"], ["crypto-prices", "crypto"]),
    (
        ["fed", "fomc", "federal reserve", "rate cut", "rate hike", "interest rate", "basis point"],
        ["fed-rates", "fed", "economic-policy"],
    ),
    (
        ["recession", "gdp", "inflation", "cpi", "unemployment", "economic"],
        ["economy", "business", "economic-policy"],
    ),
    (["tariff", "trade war", "trade deal", "import duty"], ["tariffs", "politics", "world"]),
    (
        ["oil", "opec", "crude", "energy", "wti", "brent", "petroleum"],
        ["commodities", "world", "business"],
    ),
    (
        ["gold", "silver", "copper", "platinum", "palladium", "precious metal", "metal"],
        ["commodities", "business"],
    ),
    (["wheat", "corn", "soybean", "coffee", "sugar", "grain", "natural gas"], ["commodities"]),
    (
        ["fda", "drug approval", "clinical trial", "pharma", "pfizer", "moderna", "eli lilly"],
        ["science", "health"],
    ),
    (
        [
            "nvidia",
            "apple",
            "microsoft",
            "google",
            "amazon",
            "meta",
            "tesla",
            "broadcom",
            "qualcomm",
            "intel",
            "spacex",
        ],
        ["big-tech", "tech", "business"],
    ),
    (["earnings", "revenue", "eps", "quarterly results"], ["business", "finance"]),
    (
        ["ai regulation", "artificial intelligence", "chatgpt", "openai", "antitrust"],
        ["tech", "science"],
    ),
    (
        [
            "middle east",
            "ukraine",
            "russia",
            "china",
            "taiwan",
            "war",
            "conflict",
            "sanctions",
            "geopolitical",
        ],
        ["world", "politics"],
    ),
    (
        ["election", "president", "senate", "congress", "trump", "white house"],
        ["elections", "us-politics", "politics"],
    ),
    (["ipo", "initial public offering"], ["ipos", "ipo", "business"]),
]


def _infer_tag_slugs(query: str) -> list[str]:
    """Infer tag slugs from query text."""
    lower = query.lower()
    for patterns, slugs in _TAG_SLUG_PATTERNS:
        if any(pattern in lower for pattern in patterns):
            return slugs
    return []


# ---------------------------------------------------------------------------
# Client-side text filtering
# ---------------------------------------------------------------------------

_TEXT_FILTER_STOP_WORDS = {
    "the",
    "and",
    "for",
    "are",
    "not",
    "will",
    "can",
    "has",
    "was",
    "how",
    "what",
    "that",
    "this",
    "its",
    "from",
    "with",
}

_WEAK_QUERY_WORDS = {
    "price",
    "prices",
    "market",
    "markets",
    "commodity",
    "commodities",
    "forecast",
    "forecasts",
    "current",
    "target",
    "targets",
}


def _question_matches_query(question: str, query: str) -> bool:
    """Client-side relevance filter (mirrors TS questionMatchesQuery)."""
    words = [
        re.sub(r"[^a-z0-9]", "", word)
        for word in re.split(r"[\s\-_/]+", query.lower())
    ]
    words = [
        word
        for word in words
        if len(word) >= 3 and word not in _TEXT_FILTER_STOP_WORDS
    ]

    if len(words) == 0:
        return True

    anchor_words = [word for word in words if word not in _WEAK_QUERY_WORDS]
    if len(words) > 0 and len(anchor_words) == 0:
        return False

    candidate_words = anchor_words if len(anchor_words) > 0 else words
    lower = question.lower()
    return any(word in lower for word in candidate_words)


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


def _parse_string_array_field(raw: Any) -> list[str]:
    parsed: Any
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
    else:
        parsed = raw

    if not isinstance(parsed, list):
        return []

    return [
        str(item)
        for item in parsed
        if isinstance(item, (str, int, float)) and not isinstance(item, bool)
    ]


def _parse_yes_probability(outcomes: list[str], prices: list[str]) -> float | None:
    if not outcomes or not prices:
        return None
    lower_outcomes = [outcome.lower() for outcome in outcomes]
    price_index = lower_outcomes.index("yes") if "yes" in lower_outcomes else 0
    try:
        probability = float(prices[price_index] if price_index < len(prices) else "0")
    except (TypeError, ValueError):
        probability = 0.0
    if not math.isfinite(probability):
        probability = 0.0
    return min(1.0, max(0.0, probability))


def _first_non_empty_token(tokens: list[str]) -> str | None:
    for token in tokens:
        if token.strip():
            return token
    return None


def _parse_markets(
    events: list[dict[str, Any]],
    *,
    per_event_limit: int = 4,
) -> list[dict[str, Any]]:
    """Extract and flatten markets from Gamma events response."""
    markets: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_markets = event.get("markets", [])
        if not isinstance(event_markets, list):
            continue
        active_markets = [
            m
            for m in event_markets
            if isinstance(m, dict)
            and m.get("active") is True
            and m.get("closed") is not True
        ]
        active_markets.sort(key=lambda m: _format_volume(m.get("volume24hr")), reverse=True)
        for m in active_markets[:per_event_limit]:
            if not isinstance(m, dict):
                continue
            outcomes = _parse_string_array_field(m.get("outcomes"))
            prices = _parse_string_array_field(m.get("outcomePrices"))
            probability = _parse_yes_probability(outcomes, prices)
            if probability is None:
                continue
            clob_token_ids = _parse_string_array_field(m.get("clobTokenIds"))
            markets.append(
                {
                    "market_id": m.get("conditionId") or m.get("id", m.get("marketId", "")),
                    "asset_id": _first_non_empty_token(clob_token_ids),
                    "question": m.get("question", ""),
                    "probability": probability,
                    "volume_24h": _format_volume(m.get("volume24hr", m.get("volume24h", 0))),
                    "age_days": _compute_age_days(m.get("createdAt")),
                    "end_date": m.get("endDateIso"),
                    "event_title": event.get("title", ""),
                    "active": m.get("active"),
                    "closed": m.get("closed"),
                    "enable_order_book": m.get("enableOrderBook"),
                }
            )
    return markets


def _compute_age_days(created_at: str | None) -> int | None:
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - dt).days
        return days if days >= 0 else None
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
                markets = _parse_markets(events, per_event_limit=4)
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
            all_markets = _parse_markets(events, per_event_limit=2)
        except Exception:
            all_markets = []

    # Client-side text filtering
    filtered = [
        m for m in all_markets
        if (
            _question_matches_query(m["event_title"], query)
            or _question_matches_query(m["question"], query)
        )
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
                "age_days", "end_date", "event_title", "asset_id",
                "active", "closed", "enable_order_book",
            ]
        )
    else:
        df["probability"] = df["probability"].astype(float)
        df["volume_24h"] = df["volume_24h"].astype(float)

    _cache_set(cache_key, df)
    return df.copy()


# ---------------------------------------------------------------------------
# Jump-event extraction (mirror of TS extractJumpEventMarkets)
# ---------------------------------------------------------------------------

@dataclass
class JumpEventMarket:
    """Filtered prediction-market event suitable for jump-diffusion injection.

    Probabilities are still in the **Q-measure** — apply a Q→P transformation
    before computing daily hazard rates.
    """

    id: str
    probability: float
    days_to_settlement: int
    question: str
    jump_direction: str = "unknown"  # P2a — 'up' | 'down' | 'unknown'


def extract_jump_event_markets(
    markets: pd.DataFrame,
    horizon_date: datetime,
    *,
    min_volume_24h: float = 5_000.0,
    min_age_days: int = 2,
    now: datetime | None = None,
) -> list[JumpEventMarket]:
    """Filter a market DataFrame down to jump-eligible events.

    Mirrors :func:`extractJumpEventMarkets` in
    ``src/tools/finance/polymarket.ts``.

    Filters applied:
      - probability strictly in (0, 1)
      - 24h volume ≥ ``min_volume_24h``
      - age ≥ ``min_age_days``
      - end_date present, parseable, and ≤ ``horizon_date``
      - **P1c** — settlement at least 24h away (drops sub-1-day markets)
    """
    now_dt = now if now is not None else datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    horizon_ms = horizon_date.timestamp()

    out: list[JumpEventMarket] = []
    if markets is None or len(markets) == 0:
        return out

    for _, row in markets.iterrows():
        try:
            p = float(row.get("probability", 0.0))
        except (TypeError, ValueError):
            continue
        if not (0.0 < p < 1.0):
            continue
        try:
            vol = float(row.get("volume_24h", 0.0))
        except (TypeError, ValueError):
            continue
        if vol < min_volume_24h:
            continue
        age = row.get("age_days")
        if age is None or age < min_age_days:
            continue
        end_date = row.get("end_date")
        if not end_date:
            continue
        try:
            end_dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        end_ms = end_dt.timestamp()
        if end_ms > horizon_ms:
            continue
        raw_days = (end_ms - now_dt.timestamp()) / 86_400.0
        # P1c — drop markets settling in <24h
        if raw_days < 1.0:
            continue
        days_to_settle = max(1, int(-(-raw_days // 1)))  # ceil
        question = str(row.get("question", ""))
        out.append(
            JumpEventMarket(
                id=str(row.get("market_id") or row.get("question", "")),
                probability=p,
                days_to_settlement=days_to_settle,
                question=question,
                jump_direction=classify_jump_direction(question),
            )
        )
    return out
