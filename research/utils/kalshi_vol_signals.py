"""Stub — Kalshi macro volatility signals (R5 Idea #13).

Python mirror of src/tools/finance/kalshi-vol-signals.ts.

This module reserves the interface for fetching implied volatility
signals from Kalshi prediction markets.

Status: NOT IMPLEMENTED — requires KALSHI_API_KEY environment variable
and REST integration.  Raises KalshiUnconfiguredError until configured.

Source: docs/forecast-improvement-ideas-round5-2026-04-29.md (Idea #13).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class KalshiUnconfiguredError(RuntimeError):
    """Raised when KALSHI_API_KEY is not set."""


@dataclass(frozen=True)
class KalshiVolSignal:
    """A single implied-vol signal from a Kalshi market."""

    market_ticker: str
    """Kalshi market ticker (e.g. 'KXBTCD')."""

    implied_prob: float
    """Market-implied probability of the event, in (0, 1)."""

    volume_24h: float
    """24-hour dollar volume on this market."""

    timestamp_utc: str
    """ISO-8601 UTC timestamp when this signal was fetched."""


async def fetch_kalshi_vol_signals(
    tickers: list[str] | None = None,
    *,
    api_key: str | None = None,
) -> list[KalshiVolSignal]:
    """Fetch implied-vol signals from Kalshi for the given market tickers.

    Args:
        tickers: Kalshi market tickers to query.  None fetches all crypto vol markets.
        api_key: Override KALSHI_API_KEY env var.

    Returns:
        List of KalshiVolSignal objects.

    Raises:
        KalshiUnconfiguredError: When no API key is available.
    """
    key = api_key or os.environ.get("KALSHI_API_KEY")
    if not key:
        raise KalshiUnconfiguredError(
            "fetch_kalshi_vol_signals requires a KALSHI_API_KEY environment variable "
            "or an explicit api_key argument.  Set the key and implement the REST "
            "client in this module."
        )
    raise NotImplementedError(
        "Kalshi REST client integration is not yet implemented.  "
        "See src/tools/finance/kalshi-vol-signals.ts for the intended API contract."
    )
