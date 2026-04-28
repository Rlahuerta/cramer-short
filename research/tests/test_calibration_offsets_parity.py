"""Parity tests for Python calibration-offsets mirror."""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from research.models.calibration_offsets import DOMAIN_OFFSETS, recalibrate_polymarket_price


@pytest.mark.parametrize("q", [0.05, 0.2, 0.5, 0.7, 0.95])
def test_unknown_identity(q):
    assert math.isclose(recalibrate_polymarket_price(q, "unknown", 30), q, rel_tol=1e-12)


def test_sports_identity():
    assert math.isclose(recalibrate_polymarket_price(0.4, "sports", 30), 0.4, rel_tol=1e-12)


def test_boundary():
    assert recalibrate_polymarket_price(0, "politics", 30) == 0
    assert recalibrate_polymarket_price(1, "politics", 30) == 1


@pytest.mark.parametrize("d", list(DOMAIN_OFFSETS.keys()))
@pytest.mark.parametrize("q", [0.001, 0.05, 0.5, 0.95, 0.999])
def test_output_in_unit(d, q):
    p = recalibrate_polymarket_price(q, d, 30)
    assert 0 < p < 1


def test_politics_lifts_sub_half():
    p = recalibrate_polymarket_price(0.30, "politics", 30)
    assert p > 0.30


def test_politics_at_half():
    # T=1, z=0, slope·0+α=0.15 → Φ(0.15)≈0.5596
    p = recalibrate_polymarket_price(0.50, "politics", 1)
    assert 0.55 < p < 0.57


def test_horizon_amplifies_above_half():
    short = recalibrate_polymarket_price(0.70, "politics", 1)
    long_ = recalibrate_polymarket_price(0.70, "politics", 365)
    assert long_ > short


def test_monotone_in_q():
    a = recalibrate_polymarket_price(0.20, "politics", 30)
    b = recalibrate_polymarket_price(0.40, "politics", 30)
    c = recalibrate_polymarket_price(0.60, "politics", 30)
    assert a < b < c


def test_clip_zero_days():
    p = recalibrate_polymarket_price(0.5, "politics", 0)
    assert math.isfinite(p) and 0 < p < 1


def test_macro_smaller_than_politics_at_half():
    pol = recalibrate_polymarket_price(0.30, "politics", 30)
    mac = recalibrate_polymarket_price(0.30, "macro", 30)
    assert pol > mac


def test_parity_known_values():
    # Cross-check against independently computed values
    p = recalibrate_polymarket_price(0.30, "politics", 30)
    z = norm.ppf(0.30)
    slope = 1 + 0.05 * math.log1p(30)
    expected = float(norm.cdf(slope * z + 0.15))
    assert math.isclose(p, expected, rel_tol=1e-12)
