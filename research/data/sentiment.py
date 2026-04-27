"""Sentiment data fetching.

Mirrors TS social-sentiment and fear-greed tools.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Fear & Greed
# ---------------------------------------------------------------------------

_FNG_URL = "https://api.alternative.me/fng/"


def fetch_fear_greed(limit: int = 30) -> pd.DataFrame:
    """Fetch Crypto Fear & Greed Index.

    Parameters
    ----------
    limit : int
        Number of historical days to fetch.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime64), value (int), classification (str).
    """
    try:
        resp = requests.get(_FNG_URL, params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", [])
        if not isinstance(rows, list):
            return pd.DataFrame(columns=["date", "value", "classification"])

        records = [
            {
                "date": pd.to_datetime(int(r["timestamp"]), unit="s"),
                "value": int(r["value"]),
                "classification": r.get("value_classification", ""),
            }
            for r in rows
            if isinstance(r, dict) and "timestamp" in r and "value" in r
        ]
        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "value", "classification"])


# ---------------------------------------------------------------------------
# Social sentiment
# ---------------------------------------------------------------------------


def fetch_social_sentiment(
    ticker: str,
    limit: int = 25,
) -> dict[str, Any]:
    """Fetch social sentiment for a ticker.

    This is a placeholder that returns structured data.
    In production, integrate with a social sentiment API
    (e.g., LunarCrush, Santiment, or scrape Twitter/Reddit).

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. 'BTC', 'ETH').
    limit : int
        Number of posts/messages to analyze.

    Returns
    -------
    dict
        Structured sentiment result.
    """
    # Placeholder - in a real implementation this would call a sentiment API
    return {
        "ticker": ticker.upper(),
        "note": (
            "Social sentiment is not yet implemented. "
            "Configure a sentiment API key (e.g. LUNARCRUSH_API_KEY) "
            "or use web_search for social analysis."
        ),
        "bullish_score": None,
        "bearish_score": None,
        "posts_analyzed": 0,
    }
