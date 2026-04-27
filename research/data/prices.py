"""Historical price fetching with fallback chain.

Mirrors TS logic:
  1. Financial Datasets API (primary)
  2. Binance daily klines (crypto fallback)
  3. Yahoo Finance chart API (broad fallback)

All functions return oldest-first price arrays or DataFrames.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINANCIAL_DATASETS_BASE = "https://api.financialdatasets.ai"
BINANCE_BASE = "https://api.binance.com"
YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

# ---------------------------------------------------------------------------
# Ticker normalisation
# ---------------------------------------------------------------------------

_TICKER_OVERRIDES: dict[str, str] = {
    "OIL": "USO",
    "WTICOUSD": "USO",
    "CRUDE": "USO",
}


def _normalize_ticker(ticker: str) -> str:
    upper = ticker.strip().upper()
    return _TICKER_OVERRIDES.get(upper, upper)


# ---------------------------------------------------------------------------
# Financial Datasets API
# ---------------------------------------------------------------------------

def _fetch_financial_datasets(
    ticker: str,
    days: int = 120,
) -> list[float]:
    """Fetch daily closes from Financial Datasets API."""
    api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY", "")
    if not api_key:
        return []

    end = datetime.utcnow().date()
    start = end - timedelta(days=days)

    url = f"{FINANCIAL_DATASETS_BASE}/prices/"
    params: dict[str, Any] = {
        "ticker": _normalize_ticker(ticker),
        "interval": "day",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        prices = data.get("prices", data) if isinstance(data, dict) else []
        if not isinstance(prices, list):
            return []
        closes = [
            float(p["close"])
            for p in prices
            if isinstance(p, dict) and isinstance(p.get("close"), (int, float))
        ]
        return closes if len(closes) >= 10 else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Binance
# ---------------------------------------------------------------------------

def _to_binance_symbol(ticker: str) -> str:
    """Map ticker to Binance symbol (e.g. BTC-USD → BTCUSDT)."""
    upper = ticker.strip().upper()
    # Strip common suffixes
    for suffix in ("-USD", ".USD", "-USDT", "/USD", "-USD.P"):
        if upper.endswith(suffix):
            upper = upper[: -len(suffix)]
    # Default to USDT pair
    return f"{upper}USDT"


def _fetch_binance(
    ticker: str,
    days: int = 120,
) -> list[float]:
    """Fetch daily closes from Binance klines."""
    symbol = _to_binance_symbol(ticker)
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = end_time - days * 86_400_000

    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": min(days, 1000),
        "startTime": start_time,
        "endTime": end_time,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            return []
        # kline structure: [openTime, open, high, low, close, volume, ...]
        closes = [
            float(row[4])
            for row in rows
            if isinstance(row, (list, tuple)) and len(row) > 4
        ]
        return closes if len(closes) >= 10 else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

def _fetch_yahoo(
    ticker: str,
    days: int = 120,
) -> list[float]:
    """Fetch daily closes from Yahoo Finance chart API."""
    ticker_clean = _normalize_ticker(ticker)
    # Map days to Yahoo range parameter
    if days <= 30:
        yrange = "1mo"
    elif days <= 90:
        yrange = "3mo"
    elif days <= 180:
        yrange = "6mo"
    else:
        yrange = "1y"

    url = (
        f"{YAHOO_CHART_BASE}/{requests.utils.quote(ticker_clean, safe='')}"
        f"?range={yrange}&interval=1d&includePrePost=false"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return []
        quotes = result[0].get("indicators", {}).get("quote", [{}])
        closes_raw = quotes[0].get("close", []) if quotes else []
        closes = [float(c) for c in closes_raw if isinstance(c, (int, float))]
        return closes if len(closes) >= 10 else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_historical_prices(
    ticker: str,
    days: int = 120,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch daily close prices with automatic fallback.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. 'BTC', 'SPY', 'AAPL').
    days : int
        Number of historical days to fetch.
    sources : list[str] | None
        Ordered list of sources to try. Defaults to
        ['financial_datasets', 'binance', 'yahoo'].

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime64), close (float64).
        Sorted oldest-first.
    """
    if sources is None:
        sources = ["financial_datasets", "binance", "yahoo"]

    closes: list[float] = []
    used_source = ""

    for source in sources:
        if source == "financial_datasets":
            closes = _fetch_financial_datasets(ticker, days)
        elif source == "binance":
            closes = _fetch_binance(ticker, days)
        elif source == "yahoo":
            closes = _fetch_yahoo(ticker, days)
        else:
            continue

        if len(closes) >= 10:
            used_source = source
            break

    if not closes:
        raise RuntimeError(
            f"Could not fetch historical prices for {ticker}. "
            "Tried sources: {sources}"
        )

    # Build date index (oldest first)
    end = datetime.utcnow().date()
    start = end - timedelta(days=len(closes) - 1)
    dates = pd.date_range(start=start, periods=len(closes), freq="D")

    df = pd.DataFrame({"date": dates, "close": closes})
    df.attrs["source"] = used_source
    return df
