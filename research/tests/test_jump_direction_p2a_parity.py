"""Parity tests for P2a — jumpDirection classifier + sign-flip.

Mirrors ``src/tools/finance/jump-direction-p2a.test.ts``.
"""

from __future__ import annotations

import pytest

from research.data.polymarket import classify_jump_direction
from research.models.jump_diffusion import (
    JUMP_DEFAULTS,
    build_jump_event_spec,
    effective_jump_mean,
)


# ---------------------------------------------------------------------------
# classify_jump_direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "Will Bitcoin crash below $50k?",
        "Will Russia attack Poland?",
        "Will the US enter a recession in 2026?",
        "Will Tether default by year-end?",
        "Will Trump impose 60% tariffs on China?",
    ],
)
def test_down_keywords_classify_as_down(question: str) -> None:
    assert classify_jump_direction(question) == "down"


@pytest.mark.parametrize(
    "question",
    [
        "Will the Fed cut rates in March?",
        "Will Bitcoin reach $150k?",
        "Will Coinbase get spot ETH ETF approved?",
        "Will OpenAI sign a deal with Anthropic?",
    ],
)
def test_up_keywords_classify_as_up(question: str) -> None:
    assert classify_jump_direction(question) == "up"


def test_ambiguous_questions_return_unknown() -> None:
    assert classify_jump_direction("Will Elon Musk tweet on Tuesday?") == "unknown"
    assert classify_jump_direction("Will the moon be full?") == "unknown"


def test_classification_is_case_insensitive() -> None:
    assert classify_jump_direction("WILL BITCOIN CRASH?") == "down"
    assert classify_jump_direction("Cut Rates In March?") == "up"


def test_conflicting_keywords_return_unknown() -> None:
    # contains both 'crash' (down) and 'rally' (up)
    assert classify_jump_direction("Will rally end in crash?") == "unknown"


# ---------------------------------------------------------------------------
# effective_jump_mean
# ---------------------------------------------------------------------------

def test_effective_jump_mean_up_with_negative_prior() -> None:
    assert effective_jump_mean(-0.10, "up") == pytest.approx(0.10)


def test_effective_jump_mean_up_with_positive_prior() -> None:
    assert effective_jump_mean(0.05, "up") == pytest.approx(0.05)


def test_effective_jump_mean_down_with_negative_prior() -> None:
    assert effective_jump_mean(-0.10, "down") == pytest.approx(-0.10)


def test_effective_jump_mean_down_with_positive_prior() -> None:
    assert effective_jump_mean(0.05, "down") == pytest.approx(-0.05)


def test_effective_jump_mean_unknown_preserves_prior() -> None:
    assert effective_jump_mean(-0.10, "unknown") == pytest.approx(-0.10)
    assert effective_jump_mean(0.05, None) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# build_jump_event_spec — direction propagation
# ---------------------------------------------------------------------------

def test_build_spec_propagates_down_direction() -> None:
    spec = build_jump_event_spec(
        raw=0.20,
        horizon_days=30,
        historical_drift_annual=0.10,
        risk_free_rate=0.05,
        volatility_annual=0.30,
        prior=JUMP_DEFAULTS["geopolitics"],
        id="war-2026",
        jump_direction="down",
    )
    assert spec.jump_direction == "down"
    assert spec.mean_log_jump < 0


def test_build_spec_propagates_up_direction() -> None:
    spec = build_jump_event_spec(
        raw=0.40,
        horizon_days=30,
        historical_drift_annual=0.10,
        risk_free_rate=0.05,
        volatility_annual=0.30,
        prior=JUMP_DEFAULTS["equity"],
        id="rate-cut-march",
        jump_direction="up",
    )
    assert spec.jump_direction == "up"
    assert spec.mean_log_jump > 0


def test_build_spec_default_is_unknown() -> None:
    spec = build_jump_event_spec(
        raw=0.10,
        horizon_days=30,
        historical_drift_annual=0.10,
        risk_free_rate=0.05,
        volatility_annual=0.30,
        prior=JUMP_DEFAULTS["equity"],
        id="ambiguous-event",
    )
    assert spec.jump_direction == "unknown"
    # preserves prior sign (default is negative)
    assert spec.mean_log_jump == pytest.approx(JUMP_DEFAULTS["equity"]["mean_log_jump"])
