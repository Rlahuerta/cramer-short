"""Parity tests for the Polymarket Gamma search helpers."""

from __future__ import annotations

import pytest

from research.data import polymarket
from research.data.polymarket import (
    fetch_polymarket_markets,
    _infer_tag_slugs,
    _parse_markets,
    _parse_string_array_field,
    _question_matches_query,
)


def test_question_matches_query_mirrors_ts_weak_word_filter() -> None:
    assert _question_matches_query("Will the Fed cut rates in 2026?", "Fed rate cut") is True
    assert _question_matches_query("Will Bitcoin reach $100K?", "Bitcoin price") is True
    assert (
        _question_matches_query("Will the Lakers win the NBA championship?", "Bitcoin price")
        is False
    )
    assert (
        _question_matches_query("Will Bitcoin price exceed $100K in 2026?", "gold price")
        is False
    )
    assert (
        _question_matches_query("Will gold reach $3,000 per ounce by June?", "gold price")
        is True
    )
    assert (
        _question_matches_query("Will Bitcoin price exceed $100K in 2026?", "commodity price")
        is False
    )
    assert _question_matches_query("Anything goes here", "") is True
    assert _question_matches_query("Something unrelated", "the and for") is True
    assert _question_matches_query("Sports championship final", "AI") is True


def test_infer_tag_slugs_mirrors_ts_patterns() -> None:
    assert "bitcoin" in _infer_tag_slugs("Bitcoin price")
    assert "fed-rates" in _infer_tag_slugs("Fed rate cut")
    recession_slugs = _infer_tag_slugs("US recession 2026")
    assert "economy" in recession_slugs
    assert "economics" not in recession_slugs
    assert "elections" in _infer_tag_slugs("US presidential election")
    assert "commodities" in _infer_tag_slugs("gold price")
    assert "commodities" in _infer_tag_slugs("natural gas price")
    nvidia_slugs = _infer_tag_slugs("NVIDIA earnings")
    assert any(slug in {"big-tech", "tech", "business"} for slug in nvidia_slugs)
    assert "technology" not in nvidia_slugs
    assert _infer_tag_slugs("completely random unknown topic") == []


def test_parse_string_array_field_accepts_string_or_numeric_arrays() -> None:
    assert _parse_string_array_field('["Yes", "No"]') == ["Yes", "No"]
    assert _parse_string_array_field([0.72, "0.28", True, None]) == ["0.72", "0.28"]
    assert _parse_string_array_field("not json") == []
    assert _parse_string_array_field({"not": "an array"}) == []


def test_parse_string_array_field_only_handles_json_decode_errors(monkeypatch) -> None:
    def raise_unrelated_error(raw: str) -> list[str]:
        raise RuntimeError(f"unexpected parser failure for {raw}")

    monkeypatch.setattr(polymarket.json, "loads", raise_unrelated_error)

    with pytest.raises(RuntimeError, match="unexpected parser failure"):
        _parse_string_array_field('["Yes", "No"]')


def test_parse_markets_mirrors_ts_format_market_and_event_filtering() -> None:
    events = [
        {
            "title": "Fed Rate Decisions 2026",
            "markets": [
                {
                    "id": "inactive",
                    "question": "Inactive market",
                    "outcomes": '["Yes","No"]',
                    "outcomePrices": '["0.90","0.10"]',
                    "volume24hr": 9_000_000,
                    "active": False,
                    "closed": False,
                },
                {
                    "id": "closed",
                    "question": "Closed market",
                    "outcomes": '["Yes","No"]',
                    "outcomePrices": '["0.10","0.90"]',
                    "volume24hr": 8_000_000,
                    "active": True,
                    "closed": True,
                },
                {
                    "id": "1",
                    "conditionId": "condition-1",
                    "question": "Will the Fed cut rates in 2026?",
                    "outcomes": '["Yes","No"]',
                    "outcomePrices": '["0.72","0.28"]',
                    "clobTokenIds": '["asset-yes-1","asset-no-1"]',
                    "endDateIso": "2026-12-31",
                    "volume24hr": 1_500_000,
                    "active": True,
                    "closed": False,
                    "enableOrderBook": True,
                },
                {
                    "id": "2",
                    "question": "Will rates stay flat?",
                    "outcomes": ["No", "Yes"],
                    "outcomePrices": [0.61, 0.39],
                    "volume24hr": 500_000,
                    "active": True,
                    "closed": False,
                },
            ],
        }
    ]

    markets = _parse_markets(events, per_event_limit=4)

    assert [market["question"] for market in markets] == [
        "Will the Fed cut rates in 2026?",
        "Will rates stay flat?",
    ]
    assert markets[0] == {
        "market_id": "condition-1",
        "asset_id": "asset-yes-1",
        "question": "Will the Fed cut rates in 2026?",
        "probability": 0.72,
        "volume_24h": 1_500_000.0,
        "age_days": None,
        "end_date": "2026-12-31",
        "event_title": "Fed Rate Decisions 2026",
        "active": True,
        "closed": False,
        "enable_order_book": True,
    }
    assert markets[1]["probability"] == 0.39


def test_fetch_polymarket_markets_accepts_event_title_or_question_match(monkeypatch) -> None:
    polymarket._search_cache.clear()

    def fake_fetch_with_retry(url, params=None, max_retries=3):
        return [
            {
                "title": "Bitcoin ETF approvals in 2026",
                "markets": [
                    {
                        "id": "event-title-only",
                        "conditionId": "condition-event-title-only",
                        "question": "Will ETF flows exceed $10B?",
                        "outcomes": '["Yes","No"]',
                        "outcomePrices": '["0.64","0.36"]',
                        "volume24hr": 250_000,
                        "active": True,
                        "closed": False,
                    },
                    {
                        "id": "question-match",
                        "conditionId": "condition-question-match",
                        "question": "Will Bitcoin make a new all-time high?",
                        "outcomes": '["Yes","No"]',
                        "outcomePrices": '["0.45","0.55"]',
                        "volume24hr": 125_000,
                        "active": True,
                        "closed": False,
                    },
                ],
            }
        ]

    monkeypatch.setattr(polymarket, "_fetch_with_retry", fake_fetch_with_retry)

    df = fetch_polymarket_markets("Bitcoin price", limit=10, min_volume_24h=1_000)

    assert df["question"].tolist() == [
        "Will ETF flows exceed $10B?",
        "Will Bitcoin make a new all-time high?",
    ]
