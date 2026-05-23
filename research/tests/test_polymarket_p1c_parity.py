"""P1c parity tests — `extract_jump_event_markets` Python mirror."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from research.data.polymarket import extract_jump_event_markets


NOW = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
HORIZON = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def _market(**over):
    base = dict(
        market_id="mkt-1",
        question="Will X happen?",
        probability=0.30,
        volume_24h=50_000.0,
        age_days=21,
        end_date="2026-05-15T12:00:00Z",
        event_title="evt",
    )
    base.update(over)
    return base


def _df(*rows):
    return pd.DataFrame(list(rows))


class TestP1cExtraction:
    def test_drops_markets_settling_in_less_than_24h(self):
        end = (NOW + timedelta(hours=12)).isoformat().replace("+00:00", "Z")
        out = extract_jump_event_markets(
            _df(_market(end_date=end)), horizon_date=HORIZON, now=NOW
        )
        assert out == []

    def test_drops_markets_already_settled(self):
        end = (NOW - timedelta(hours=24)).isoformat().replace("+00:00", "Z")
        out = extract_jump_event_markets(
            _df(_market(end_date=end)), horizon_date=HORIZON, now=NOW
        )
        assert out == []

    def test_keeps_markets_settling_in_one_day_or_more(self):
        end = (NOW + timedelta(hours=36)).isoformat().replace("+00:00", "Z")
        out = extract_jump_event_markets(
            _df(_market(end_date=end)), horizon_date=HORIZON, now=NOW
        )
        assert len(out) == 1
        assert out[0].days_to_settlement >= 1

    def test_keeps_mid_horizon_market(self):
        out = extract_jump_event_markets(
            _df(_market()), horizon_date=HORIZON, now=NOW
        )
        assert len(out) == 1
        assert out[0].id == "mkt-1"

    def test_drops_low_volume(self):
        out = extract_jump_event_markets(
            _df(_market(volume_24h=100.0)), horizon_date=HORIZON, now=NOW
        )
        assert out == []

    def test_drops_low_age(self):
        out = extract_jump_event_markets(
            _df(_market(age_days=1)), horizon_date=HORIZON, now=NOW
        )
        assert out == []

    def test_drops_post_horizon(self):
        end = (HORIZON + timedelta(days=10)).isoformat().replace("+00:00", "Z")
        out = extract_jump_event_markets(
            _df(_market(end_date=end)), horizon_date=HORIZON, now=NOW
        )
        assert out == []
