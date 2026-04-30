from __future__ import annotations

import asyncio

import pytest

from research.utils.kalshi_vol_signals import (
    KalshiUnconfiguredError,
    build_kalshi_volatility_covariate,
    extract_kalshi_vol_signals_from_payload,
    fetch_kalshi_vol_signals,
)


def test_extract_kalshi_vol_signals_rejects_malformed_payload():
    with pytest.raises(ValueError, match="markets array"):
        extract_kalshi_vol_signals_from_payload({}, from_date="2026-06-01", to_date="2026-06-30")


def test_extract_kalshi_vol_signals_filters_whitelisted_macro_markets():
    payload = {
        "markets": [
            {
                "ticker": "FOMC-25JUN18-HIKE",
                "title": "Will the Fed hike rates at the June FOMC meeting?",
                "expiration_time": "2026-06-18T18:00:00Z",
                "yes_ask": 63,
            },
            {
                "ticker": "CPI-26JUN",
                "title": "Will CPI come in above 0.3% MoM?",
                "expiration_time": "2026-06-11T12:30:00Z",
                "yes_ask": 0.58,
            },
            {
                "ticker": "SPORTS-IGNORE",
                "title": "Will the Knicks win game 7?",
                "expiration_time": "2026-06-20T00:00:00Z",
                "yes_ask": 0.51,
            },
        ]
    }
    signals = extract_kalshi_vol_signals_from_payload(
        payload,
        from_date="2026-06-01",
        to_date="2026-06-30",
    )
    assert [signal.event_type for signal in signals] == ["fomc", "cpi"]
    assert all(signal.intensity_boost > 0 for signal in signals)


def test_build_kalshi_volatility_covariate_peaks_ahead_of_event():
    covariate = build_kalshi_volatility_covariate(
        ["2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14"],
        [
            extract_kalshi_vol_signals_from_payload(
                {
                    "markets": [
                        {
                            "ticker": "CPI-2026-06",
                            "title": "Will CPI come in above 0.3% MoM?",
                            "expiration_time": "2026-06-13T12:30:00Z",
                            "yes_ask": 0.61,
                        }
                    ]
                },
                from_date="2026-06-01",
                to_date="2026-06-30",
            )[0]
        ],
        lookahead_days=3,
    )
    assert covariate.active_signals == 1
    assert covariate.values[0] < covariate.values[2] < covariate.values[3]
    assert covariate.peak_value == covariate.values[3]
    assert covariate.values[4] == 0


def test_fetch_kalshi_vol_signals_requires_api_key():
    with pytest.raises(KalshiUnconfiguredError):
        asyncio.run(fetch_kalshi_vol_signals(from_date="2026-06-01", to_date="2026-06-30", api_key=""))


def test_fetch_kalshi_vol_signals_supports_injected_fetch():
    async def fake_fetch(_url: str, _headers: dict[str, str]):
        return {
            "markets": [
                {
                    "ticker": "FOMC-25JUN18-HIKE",
                    "title": "Will the Fed hike rates at the June FOMC meeting?",
                    "expiration_time": "2026-06-18T18:00:00Z",
                    "yes_ask": 63,
                }
            ]
        }

    signals = asyncio.run(
        fetch_kalshi_vol_signals(
            from_date="2026-06-01",
            to_date="2026-06-30",
            api_key="test-key",
            fetch_impl=fake_fetch,
        )
    )
    assert len(signals) == 1
    assert signals[0].event_type == "fomc"
