"""Parity tests for Markov module — Python outputs vs TypeScript fixtures."""

import numpy as np
import pytest

from research.models.markov import (
    classify_regime,
    classify_regime_series,
    estimate_transition_matrix,
    detect_structural_break,
    compute_markov_forecast,
    _default_matrix,
    NUM_STATES,
    STATE_INDEX,
)


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------

def test_classify_regime_bull():
    assert classify_regime(0.015, 0.01) == "bull"


def test_classify_regime_bear():
    assert classify_regime(-0.015, 0.01) == "bear"


def test_classify_regime_sideways_positive():
    assert classify_regime(0.005, 0.01) == "sideways"


def test_classify_regime_sideways_negative():
    assert classify_regime(-0.005, 0.01) == "sideways"


def test_classify_regime_bull_high_vol():
    # 3-state model: high vol ignored, positive return → bull
    assert classify_regime(0.03, 0.025) == "bull"


def test_classify_regime_bear_high_vol():
    assert classify_regime(-0.04, 0.025) == "bear"


def test_classify_regime_boundary_sideways():
    # exactly 1% return with low vol → sideways (strict > threshold)
    assert classify_regime(0.01, 0.01) == "sideways"


# ---------------------------------------------------------------------------
# classify_regime_series / adaptive thresholds
# ---------------------------------------------------------------------------

def test_classify_regime_series_half_median():
    returns = [0.01, -0.02, 0.03, -0.04, 0.05]
    regimes = classify_regime_series(returns)
    # median(|returns|) = 0.03; threshold = 0.5 * 0.03 = 0.015
    assert regimes == ["sideways", "bear", "bull", "bear", "bull"]


def test_classify_regime_series_custom_multiplier():
    returns = [0.01, -0.02, 0.03, -0.04, 0.05]
    baseline = classify_regime_series(returns)
    widened = classify_regime_series(returns, return_threshold_multiplier=1.0)
    # baseline threshold = 0.015; widened = 0.03
    assert widened.count("bull") <= baseline.count("bull")
    assert widened.count("bear") <= baseline.count("bear")


def test_classify_regime_series_minimum_floor():
    returns = [0.0004, -0.0004, 0.0002]
    regimes = classify_regime_series(returns)
    # threshold would be 0.5 * 0.0004 = 0.0002, but floor is 0.001
    assert all(r == "sideways" for r in regimes)


# ---------------------------------------------------------------------------
# estimate_transition_matrix
# ---------------------------------------------------------------------------

def test_estimate_transition_matrix_short_sequence():
    states = ["bull", "bear"] * 5
    m = estimate_transition_matrix(states, alpha=0.1, min_observations=30)
    np.testing.assert_allclose(m, _default_matrix(), atol=1e-10)


def test_estimate_transition_matrix_rows_sum_to_one():
    states = ["bull", "sideways", "bear"] * 20
    m = estimate_transition_matrix(states)
    for row in m:
        assert sum(row) == pytest.approx(1.0, abs=1e-10)


def test_estimate_transition_matrix_dirichlet_smoothing():
    # Only bull→bear transitions, all other pairs never observed
    states = ["bull", "bear"] * 30
    m = estimate_transition_matrix(states, alpha=0.1)
    # Every cell must be > 0 (Dirichlet prior prevents zeros)
    for row in m:
        for v in row:
            assert v > 0


def test_estimate_transition_matrix_self_persistence():
    states = ["bull"] * 60
    m = estimate_transition_matrix(states, alpha=0.1)
    bull_idx = STATE_INDEX["bull"]
    # bull→bull should be > 0.8
    assert m[bull_idx][bull_idx] > 0.8


# ---------------------------------------------------------------------------
# detect_structural_break
# ---------------------------------------------------------------------------

def test_detect_structural_break_typical():
    # Create a sequence with clear structural change
    first = ["bull", "bull", "bull", "sideways"] * 15
    second = ["bear", "bear", "sideways", "bull"] * 15
    states = first + second
    result = detect_structural_break(states, divergence_threshold=0.05)
    assert result["detected"] is True
    assert result["divergence"] > 0.05
    assert result["first_half_matrix"].shape == (NUM_STATES, NUM_STATES)
    assert result["second_half_matrix"].shape == (NUM_STATES, NUM_STATES)


def test_detect_structural_break_no_break():
    states = ["bull", "sideways", "bear"] * 40
    result = detect_structural_break(states, divergence_threshold=1.0)
    assert result["detected"] is False
    assert result["divergence"] < 1.0


# ---------------------------------------------------------------------------
# compute_markov_forecast
# ---------------------------------------------------------------------------

def test_compute_markov_forecast_default_matrix():
    P = _default_matrix(diagonal=0.6)
    forecast = compute_markov_forecast(P, "bull", 1)
    assert forecast["bull"] > forecast["bear"]
    assert forecast["bull"] > forecast["sideways"]
    assert sum(forecast.values()) == pytest.approx(1.0, abs=1e-10)


def test_compute_markov_forecast_long_horizon():
    P = _default_matrix(diagonal=0.6)
    forecast = compute_markov_forecast(P, "bull", 100)
    # Long horizon → uniform stationary distribution ≈ 1/3 each
    for v in forecast.values():
        assert v == pytest.approx(1 / 3, abs=1e-2)


def test_compute_markov_forecast_from_each_state():
    P = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.2, 0.3, 0.5]])
    for state in ["bull", "bear", "sideways"]:
        forecast = compute_markov_forecast(P, state, 1)
        idx = STATE_INDEX[state]
        assert forecast[state] == pytest.approx(P[idx][idx], abs=1e-10)


# ---------------------------------------------------------------------------
# _default_matrix
# ---------------------------------------------------------------------------

def test_default_matrix_rows_sum_to_one():
    m = _default_matrix()
    for row in m:
        assert sum(row) == pytest.approx(1.0, abs=1e-10)


def test_default_matrix_diagonal_value():
    m = _default_matrix(diagonal=0.6)
    off_diag = (1.0 - 0.6) / (NUM_STATES - 1)
    for i in range(NUM_STATES):
        for j in range(NUM_STATES):
            expected = 0.6 if i == j else off_diag
            assert m[i][j] == pytest.approx(expected, abs=1e-10)


def test_default_matrix_symmetry():
    m = _default_matrix()
    for i in range(NUM_STATES):
        for j in range(NUM_STATES):
            assert m[i][j] == pytest.approx(m[j][i], abs=1e-10)
