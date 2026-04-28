"""Unit tests for research/models/jump_diffusion.py — mirror of the TS suite."""
from __future__ import annotations

import math

import pytest

from research.models.jump_diffusion import (
    JUMP_DEFAULTS,
    JumpEventSpec,
    build_jump_event_spec,
    jump_drift_compensator,
    polymarket_prob_to_hazard,
)


# ---------------------------------------------------------------------------
# polymarket_prob_to_hazard
# ---------------------------------------------------------------------------

def test_hazard_p_zero():
    assert polymarket_prob_to_hazard(0.0, 30) == 0.0


def test_hazard_p_one_saturates():
    assert polymarket_prob_to_hazard(1.0, 30) == 0.95


def test_hazard_p_half_days_ten():
    assert polymarket_prob_to_hazard(0.5, 10) == pytest.approx(-math.log(0.5) / 10, abs=1e-12)


def test_hazard_small_prob():
    assert polymarket_prob_to_hazard(0.1, 30) == pytest.approx(-math.log(0.9) / 30, abs=1e-12)


def test_hazard_capped_at_short_horizon():
    assert polymarket_prob_to_hazard(0.999, 1) <= 0.95


def test_hazard_horizon_floor():
    assert polymarket_prob_to_hazard(0.5, 0) == pytest.approx(-math.log(0.5), abs=1e-12)


# ---------------------------------------------------------------------------
# JUMP_DEFAULTS
# ---------------------------------------------------------------------------

def test_defaults_all_negative_drift():
    for cls in ("etf", "equity", "crypto", "commodity"):
        assert JUMP_DEFAULTS[cls]["mean_log_jump"] < 0
        assert JUMP_DEFAULTS[cls]["std_log_jump"] > 0


def test_crypto_largest_magnitude():
    assert abs(JUMP_DEFAULTS["crypto"]["mean_log_jump"]) > abs(JUMP_DEFAULTS["etf"]["mean_log_jump"])


# ---------------------------------------------------------------------------
# jump_drift_compensator
# ---------------------------------------------------------------------------

def test_compensator_empty():
    assert jump_drift_compensator([]) == 0.0


def test_compensator_single_event():
    e = JumpEventSpec(id="x", daily_intensity=0.01, mean_log_jump=-0.05, std_log_jump=0.03)
    expected = 0.01 * (math.exp(-0.05 + 0.03 * 0.03 / 2) - 1)
    assert jump_drift_compensator([e]) == pytest.approx(expected, abs=1e-14)


def test_compensator_additive():
    e1 = JumpEventSpec(id="a", daily_intensity=0.005, mean_log_jump=-0.05, std_log_jump=0.03)
    e2 = JumpEventSpec(id="b", daily_intensity=0.01, mean_log_jump=-0.08, std_log_jump=0.05)
    assert jump_drift_compensator([e1, e2]) == pytest.approx(
        jump_drift_compensator([e1]) + jump_drift_compensator([e2]), abs=1e-14
    )


# ---------------------------------------------------------------------------
# build_jump_event_spec
# ---------------------------------------------------------------------------

def test_build_spec_basic():
    spec = build_jump_event_spec(
        raw=0.30,
        horizon_days=30,
        historical_drift_annual=0.10,
        risk_free_rate=0.05,
        volatility_annual=0.20,
        prior=JUMP_DEFAULTS["equity"],
        id="mkt-1",
    )
    assert spec.id == "mkt-1"
    assert spec.mean_log_jump == JUMP_DEFAULTS["equity"]["mean_log_jump"]
    assert spec.std_log_jump == JUMP_DEFAULTS["equity"]["std_log_jump"]
    assert 0 < spec.daily_intensity < 0.95


def test_build_spec_zero_prob():
    spec = build_jump_event_spec(
        raw=0.0,
        horizon_days=30,
        historical_drift_annual=0.10,
        risk_free_rate=0.05,
        volatility_annual=0.20,
        prior=JUMP_DEFAULTS["equity"],
        id="mkt-2",
    )
    assert spec.daily_intensity == 0.0
