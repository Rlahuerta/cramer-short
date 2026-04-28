"""P3b parity tests — Python vol-regime mirrors TS src/utils/vol-regime.ts."""

from __future__ import annotations

import pytest

from research.models.vol_regime import get_volatility_regime, leverage_vol_multiplier


@pytest.mark.parametrize("vix,expected", [
    (0, "sticky_strike"),
    (-5, "sticky_strike"),
    (10, "sticky_strike"),
    (14.99, "sticky_strike"),
    (15, "transitional"),
    (20, "transitional"),
    (24.99, "transitional"),
    (25, "sticky_implied_tree"),
    (40, "sticky_implied_tree"),
    (80, "sticky_implied_tree"),
])
def test_get_volatility_regime(vix, expected):
    assert get_volatility_regime(vix) == expected


def test_leverage_amplifies_down_in_fear_equity():
    assert leverage_vol_multiplier("sticky_implied_tree", -1.5, "equity") == pytest.approx(1.4)


def test_leverage_mutes_up_in_fear_equity():
    assert leverage_vol_multiplier("sticky_implied_tree", 1.5, "equity") == pytest.approx(0.8)


def test_leverage_neutral_in_sticky_strike():
    assert leverage_vol_multiplier("sticky_strike", -2, "equity") == 1.0
    assert leverage_vol_multiplier("sticky_strike", 2, "equity") == 1.0


def test_leverage_neutral_in_transitional():
    assert leverage_vol_multiplier("transitional", -2, "equity") == 1.0


def test_leverage_gated_for_crypto_and_commodity():
    assert leverage_vol_multiplier("sticky_implied_tree", -2, "crypto") == 1.0
    assert leverage_vol_multiplier("sticky_implied_tree", 2, "crypto") == 1.0
    assert leverage_vol_multiplier("sticky_implied_tree", -2, "commodity") == 1.0


def test_leverage_applies_to_gold():
    assert leverage_vol_multiplier("sticky_implied_tree", -1, "gold") == pytest.approx(1.4)
    assert leverage_vol_multiplier("sticky_implied_tree", 1, "gold") == pytest.approx(0.8)


def test_leverage_z_zero_is_neutral():
    assert leverage_vol_multiplier("sticky_implied_tree", 0, "equity") == 1.0
