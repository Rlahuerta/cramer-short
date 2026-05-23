"""Parity tests for metaforecast.py — mirrors metaforecast.test.ts (P2b)."""

from __future__ import annotations

import pytest

from research.data.metaforecast import (
    MetaforecastEstimate,
    compute_cross_platform_delta,
    find_best_metaforecast_match,
    parse_metaforecast_response,
    should_flag_cross_platform,
)


def test_parse_extracts_valid_record() -> None:
    raw = [
        {
            "title": "Will the Fed cut rates in March 2026?",
            "options": [{"name": "Yes", "probability": 0.42}],
            "platform": "metaculus",
            "qualityindicators": {"stars": 3},
            "url": "https://metaculus.com/q/12345",
        }
    ]
    parsed = parse_metaforecast_response(raw)
    assert len(parsed) == 1
    assert parsed[0].title == "Will the Fed cut rates in March 2026?"
    assert parsed[0].probability == pytest.approx(0.42)
    assert parsed[0].platform == "metaculus"
    assert parsed[0].stars == 3


def test_parse_skips_malformed() -> None:
    raw = [
        {"title": "No probability here", "options": []},
        {"title": "Missing options entirely", "platform": "manifold"},
        None,
        {"title": "Valid", "options": [{"name": "Yes", "probability": 0.5}], "platform": "kalshi"},
    ]
    parsed = parse_metaforecast_response(raw)
    assert len(parsed) == 1
    assert parsed[0].title == "Valid"


def test_parse_clamps_probabilities() -> None:
    raw = [
        {"title": "A", "options": [{"name": "Yes", "probability": 1.5}], "platform": "p"},
        {"title": "B", "options": [{"name": "Yes", "probability": -0.2}], "platform": "p"},
    ]
    parsed = parse_metaforecast_response(raw)
    assert parsed[0].probability == 1.0
    assert parsed[1].probability == 0.0


def test_parse_returns_empty_for_non_list() -> None:
    assert parse_metaforecast_response(None) == []
    assert parse_metaforecast_response({}) == []
    assert parse_metaforecast_response("not list") == []


def test_match_returns_best_keyword_match() -> None:
    candidates = [
        MetaforecastEstimate("Will the Fed cut rates in March 2026?", 0.42, "metaculus", 3),
        MetaforecastEstimate("Will Trump win 2024?", 0.55, "manifold", 2),
        MetaforecastEstimate("Will Bitcoin reach 100k by year end?", 0.30, "kalshi", 4),
    ]
    match = find_best_metaforecast_match("Will Bitcoin hit $100k?", candidates)
    assert match is not None
    assert "Bitcoin" in match.title


def test_match_returns_none_when_no_overlap() -> None:
    candidates = [
        MetaforecastEstimate("Will the Fed cut rates in March 2026?", 0.42, "metaculus", 3),
    ]
    match = find_best_metaforecast_match("Will SpaceX land on Mars?", candidates)
    assert match is None


def test_match_returns_none_for_empty() -> None:
    assert find_best_metaforecast_match("anything", []) is None


def test_match_prefers_higher_stars_on_tie() -> None:
    ties = [
        MetaforecastEstimate("Will rates change soon?", 0.5, "a", 1),
        MetaforecastEstimate("Will rates change soon?", 0.5, "b", 4),
    ]
    match = find_best_metaforecast_match("Will rates change soon?", ties)
    assert match is not None
    assert match.platform == "b"


def test_compute_delta() -> None:
    assert compute_cross_platform_delta(0.40, 0.25) == pytest.approx(0.15)
    assert compute_cross_platform_delta(0.20, 0.50) == pytest.approx(0.30)
    assert compute_cross_platform_delta(0.50, 0.50) == 0.0


def test_should_flag() -> None:
    assert should_flag_cross_platform(0.11) is True
    assert should_flag_cross_platform(0.50) is True
    assert should_flag_cross_platform(0.10) is False
    assert should_flag_cross_platform(0.05) is False
    assert should_flag_cross_platform(0.0) is False
