"""Stub — Crypto-native peer returns loader (R5 Idea #1).

Python mirror of src/tools/finance/crypto-native-peers.ts.

This module reserves the interface for loading peer crypto asset returns
(ETH, SOL, MSTR, COIN) alongside BTC for cross-sectional correlation
analysis.

Status: NOT IMPLEMENTED — requires historical price fixtures for peer
assets.  Raises CryptoPeerLoaderUnavailable until fixtures land.

Source: docs/forecast-improvement-ideas-round5-2026-04-29.md (Idea #1).
"""

from __future__ import annotations


class CryptoPeerLoaderUnavailable(NotImplementedError):
    """Raised when peer-asset fixtures are not yet available."""


def load_crypto_peer_returns(
    peers: list[str],
    anchor: str = "BTC",
) -> dict[str, list[float]]:
    """Load log returns for each peer relative to an anchor asset.

    Args:
        peers: List of tickers e.g. ["ETH", "SOL", "MSTR", "COIN"].
        anchor: Reference anchor asset (default "BTC").

    Returns:
        Mapping of ticker -> list of log returns, aligned to anchor dates.

    Raises:
        CryptoPeerLoaderUnavailable: Until fixtures are installed.
    """
    raise CryptoPeerLoaderUnavailable(
        "load_crypto_peer_returns is not yet implemented. "
        "Install historical fixtures for the following peers and update "
        f"this module: {peers} (anchor: {anchor})."
    )
