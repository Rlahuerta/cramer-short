"""Parity tests for P1a (`adjust_yes_bias_v2`) and P1b (`compute_expiry_boost`).

Mirrors `src/utils/ensemble-p1.test.ts` to guarantee TS↔Python consistency.
"""

from __future__ import annotations

import math

import pytest

from research.models.ensemble import MarketInput, compute_market_quality
from research.utils.calibration import (
    adjust_yes_bias_v2,
    compute_expiry_boost,
)


# ---------------------------------------------------------------------------
# P1a — adjust_yes_bias_v2
# ---------------------------------------------------------------------------


class TestAdjustYesBiasV2DeepLongshot:
    def test_p_001_strong_30_pct_discount(self):
        assert adjust_yes_bias_v2(0.01) == pytest.approx(0.007, abs=1e-6)

    def test_p_004_just_under_threshold(self):
        assert adjust_yes_bias_v2(0.04) == pytest.approx(0.028, abs=1e-6)


class TestAdjustYesBiasV2ModerateLongshot:
    def test_p_005_boundary(self):
        assert adjust_yes_bias_v2(0.05) == pytest.approx(0.035, abs=1e-6)

    def test_p_010_midpoint(self):
        # mult = 0.825 → 0.10*0.825 = 0.0825
        assert adjust_yes_bias_v2(0.10) == pytest.approx(0.0825, abs=1e-6)

    def test_p_015_boundary(self):
        # mult = 0.95 → 0.15*0.95 = 0.1425
        assert adjust_yes_bias_v2(0.15) == pytest.approx(0.1425, abs=1e-6)


class TestAdjustYesBiasV2MidRange:
    def test_p_030_unchanged(self):
        assert adjust_yes_bias_v2(0.30) == pytest.approx(0.30, abs=1e-6)

    def test_p_050_unchanged(self):
        assert adjust_yes_bias_v2(0.50) == pytest.approx(0.50, abs=1e-6)

    def test_p_070_legacy_minus_35bp(self):
        assert adjust_yes_bias_v2(0.70) == pytest.approx(0.665, abs=1e-6)

    def test_p_085_boundary_legacy_shift(self):
        assert adjust_yes_bias_v2(0.85) == pytest.approx(0.815, abs=1e-6)


class TestAdjustYesBiasV2Favourite:
    def test_p_090_minus_25bp(self):
        assert adjust_yes_bias_v2(0.90) == pytest.approx(0.875, abs=1e-6)

    def test_p_099_clamped(self):
        result = adjust_yes_bias_v2(0.99)
        assert 0.96 <= result <= 0.97


class TestAdjustYesBiasV2EdgeCases:
    def test_zero_returns_min(self):
        assert adjust_yes_bias_v2(0.0) == pytest.approx(0.001)

    def test_one_returns_max(self):
        assert adjust_yes_bias_v2(1.0) == pytest.approx(0.999)


# ---------------------------------------------------------------------------
# P1b — compute_expiry_boost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "days,expected",
    [
        (0.5, 1.50),
        (1, 1.50),
        (3, 1.20),
        (7, 1.20),
        (15, 1.00),
        (30, 1.00),
        (60, 0.85),
        (90, 0.85),
        (180, 0.70),
    ],
)
def test_compute_expiry_boost_schedule(days, expected):
    assert compute_expiry_boost(days) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Quality weight integration
# ---------------------------------------------------------------------------


def _base_input(**overrides) -> MarketInput:
    base = dict(
        question="Q",
        probability=0.5,
        volume24h_usd=10_000,
        age_days=30,
        signal_tier="geopolitical",
        delta_yes=0.05,
        delta_no=-0.03,
    )
    base.update(overrides)
    return MarketInput(**base)


def test_quality_omitting_days_to_expiry_matches_legacy():
    w_legacy = compute_market_quality(_base_input())
    w_explicit = compute_market_quality(_base_input(days_to_expiry=None))
    assert w_legacy == pytest.approx(w_explicit, abs=1e-9)


def test_quality_near_expiry_boosted():
    # days=30: depth_decay_haircut=1.0, expiry_boost=1.0
    w_neutral = compute_market_quality(_base_input(days_to_expiry=30))
    # days=1: depth_decay_haircut=0.5, expiry_boost=1.5 → combined factor 0.75
    w_near = compute_market_quality(_base_input(days_to_expiry=1))
    # The depth-decay liquidity haircut dominates the near-expiry information boost
    assert w_near < w_neutral
    assert w_near / w_neutral == pytest.approx(0.75, abs=1e-4)


def test_quality_far_dated_discounted():
    w_neutral = compute_market_quality(_base_input(days_to_expiry=30))
    w_far = compute_market_quality(_base_input(days_to_expiry=180))
    assert w_far < w_neutral
    assert w_far / w_neutral == pytest.approx(0.7, abs=1e-4)


def test_quality_clamped_when_boost_pushes_above_one():
    huge = _base_input(volume24h_usd=10_000_000, days_to_expiry=1)
    w = compute_market_quality(huge)
    assert 0.0 <= w <= 1.0
