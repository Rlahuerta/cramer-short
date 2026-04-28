"""P1d/P1e parity tests for Polymarket CLOB helpers."""

from __future__ import annotations

import math

import pytest

from research.data.polymarket_clob import (
    ClobPricePoint,
    compute_max_hourly_jump,
    compute_price_velocity_pp_h,
    parse_clob_price_history,
)


class TestParseClobPriceHistory:
    def test_well_formed(self):
        raw = {"history": [{"t": 1000, "p": 0.30}, {"t": 4600, "p": 0.32}]}
        assert parse_clob_price_history(raw) == [
            ClobPricePoint(1000, 0.30),
            ClobPricePoint(4600, 0.32),
        ]

    def test_drops_invalid(self):
        raw = {"history": [{"t": 1, "p": float("nan")}, {"t": 2, "p": 0.5}, {"t": 3, "p": 2.0}]}
        assert parse_clob_price_history(raw) == [ClobPricePoint(2, 0.5)]

    def test_returns_empty_on_malformed(self):
        assert parse_clob_price_history(None) == []
        assert parse_clob_price_history({}) == []
        assert parse_clob_price_history({"history": "oops"}) == []

    def test_sorts_ascending(self):
        raw = {"history": [{"t": 30, "p": 0.4}, {"t": 10, "p": 0.2}, {"t": 20, "p": 0.3}]}
        out = parse_clob_price_history(raw)
        assert [pt.t_sec for pt in out] == [10, 20, 30]


class TestComputePriceVelocity:
    def test_short_series(self):
        assert compute_price_velocity_pp_h([]) == 0
        assert compute_price_velocity_pp_h([ClobPricePoint(0, 0.5)]) == 0

    def test_positive_ramp(self):
        pts = [ClobPricePoint(h * 3600, 0.30 + h * 0.01) for h in range(6)]
        assert compute_price_velocity_pp_h(pts) == pytest.approx(1.0, abs=1e-3)

    def test_negative_ramp(self):
        pts = [ClobPricePoint(h * 3600, 0.50 - h * 0.02) for h in range(6)]
        assert compute_price_velocity_pp_h(pts) == pytest.approx(-2.0, abs=1e-3)

    def test_flat(self):
        pts = [ClobPricePoint(h * 3600, 0.40) for h in range(6)]
        assert compute_price_velocity_pp_h(pts) == pytest.approx(0, abs=1e-6)

    def test_lookback_window(self):
        pts = [ClobPricePoint(h * 3600, 0.30 + h * 0.01) for h in range(24)]
        assert compute_price_velocity_pp_h(pts, lookback_hours=3) == pytest.approx(1.0, abs=1e-3)


class TestComputeMaxHourlyJump:
    def test_short(self):
        assert compute_max_hourly_jump([]) == 0
        assert compute_max_hourly_jump([ClobPricePoint(0, 0.5)]) == 0

    def test_max_abs_delta(self):
        pts = [
            ClobPricePoint(0, 0.30),
            ClobPricePoint(3600, 0.32),
            ClobPricePoint(7200, 0.45),
            ClobPricePoint(10_800, 0.40),
        ]
        assert compute_max_hourly_jump(pts) == pytest.approx(0.13, abs=1e-6)

    def test_excludes_old(self):
        pts = [
            ClobPricePoint(0, 0.10),
            ClobPricePoint(3600, 0.50),
            ClobPricePoint(100 * 3600, 0.50),
            ClobPricePoint(101 * 3600, 0.55),
        ]
        assert compute_max_hourly_jump(pts, window_hours=24) == pytest.approx(0.05, abs=1e-6)
