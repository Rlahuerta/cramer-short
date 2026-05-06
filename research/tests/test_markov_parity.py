"""Parity tests for Markov module — Python outputs vs TypeScript fixtures."""

import numpy as np
import pytest

from research.models.markov import (
    BtcShortHorizonLivePolicy,
    classify_regime,
    classify_regime_series,
    estimate_transition_matrix,
    detect_structural_break,
    compute_markov_forecast,
    get_btc_short_horizon_live_policy,
    _default_matrix,
    NUM_STATES,
    STATE_INDEX,
)
from research.models.soft_regime import one_hot_regime_mixture


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


def test_get_btc_short_horizon_live_policy_matches_ts_horizons():
    assert get_btc_short_horizon_live_policy("ETH-USD", 1) is None
    assert get_btc_short_horizon_live_policy("BTC-USD", 30) is None
    assert get_btc_short_horizon_live_policy("BTC-USD", 1) == BtcShortHorizonLivePolicy(
        history_days=252,
        break_divergence_threshold=0.10,
        rerun_on_break=True,
        rerun_window_days=60,
    )
    assert get_btc_short_horizon_live_policy("BTC", 3) == BtcShortHorizonLivePolicy(
        history_days=252,
        break_divergence_threshold=0.20,
        rerun_on_break=True,
        rerun_window_days=60,
    )
    assert get_btc_short_horizon_live_policy("BTC-USD", 2) == BtcShortHorizonLivePolicy(
        history_days=252,
        break_divergence_threshold=0.15,
        rerun_on_break=False,
        rerun_window_days=None,
    )
    assert get_btc_short_horizon_live_policy("BTC-USD", 14) == BtcShortHorizonLivePolicy(
        history_days=252,
        break_divergence_threshold=0.15,
        rerun_on_break=False,
        rerun_window_days=None,
    )


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


def test_compute_markov_forecast_supports_soft_start_mixture():
    P = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.25, 0.25, 0.5]])
    hard = compute_markov_forecast(P, "bull", 1)
    soft = compute_markov_forecast(
        P,
        "bull",
        1,
        start_mixture={"bull": 0.5, "bear": 0.0, "sideways": 0.5},
    )
    assert sum(soft.values()) == pytest.approx(1.0, abs=1e-10)
    assert soft["sideways"] > hard["sideways"]


def test_compute_markov_forecast_supports_soft_forecast_blend():
    P = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.25, 0.25, 0.5]])
    base = compute_markov_forecast(P, "bull", 1, start_mixture=one_hot_regime_mixture("bull"))
    blended = compute_markov_forecast(
        P,
        "bull",
        1,
        start_mixture=one_hot_regime_mixture("bull"),
        forecast_mixture={"bull": 0.2, "bear": 0.3, "sideways": 0.5},
        soft_transition_blend_weight=0.6,
    )
    assert sum(blended.values()) == pytest.approx(1.0, abs=1e-10)
    assert blended["sideways"] > base["sideways"]


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


def test_default_matrix_diagonal_matches_ts_default():
    """Default _default_matrix() must use diag=0.7 matching TS markov-distribution."""
    from research.models.markov import _default_matrix, NUM_STATES
    m = _default_matrix()
    off_diag = (1.0 - 0.7) / (NUM_STATES - 1)
    for i in range(NUM_STATES):
        for j in range(NUM_STATES):
            expected = 0.7 if i == j else off_diag
            assert m[i][j] == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Entropy-based soft regime blending (Blake et al. 2510.03236)
# ---------------------------------------------------------------------------


class TestRegimeEntropy:
    def test_uniform_mixture_entropy_is_one(self):
        """Normalized entropy of uniform 3-state mixture = 1.0."""
        from research.models.markov import compute_regime_entropy
        h = compute_regime_entropy({"bull": 1/3, "bear": 1/3, "sideways": 1/3})
        assert h == pytest.approx(1.0, abs=1e-10)

    def test_pure_state_entropy_is_zero(self):
        """Normalized entropy of pure state mixture = 0.0."""
        from research.models.markov import compute_regime_entropy
        h = compute_regime_entropy({"bull": 1.0, "bear": 0.0, "sideways": 0.0})
        assert h == pytest.approx(0.0, abs=1e-10)

    def test_partial_mixture_entropy_between_zero_and_one(self):
        """Normalized entropy of 50-25-25 mixture is between 0 and 1."""
        from research.models.markov import compute_regime_entropy
        h = compute_regime_entropy({"bull": 0.5, "bear": 0.25, "sideways": 0.25})
        assert 0.3 < h < 0.99


class TestSoftRegimeConfidence:
    def test_zero_entropy_returns_full_confidence(self):
        """At zero entropy, confidence multiplier = 1.0."""
        from research.models.markov import soft_regime_confidence_multiplier
        m = soft_regime_confidence_multiplier(0.0)
        assert m == pytest.approx(1.0)

    def test_max_entropy_returns_floor(self):
        """At max entropy, confidence multiplier = 0.65 (floor)."""
        from research.models.markov import soft_regime_confidence_multiplier
        m = soft_regime_confidence_multiplier(1.0)
        assert m == pytest.approx(0.65, abs=1e-10)

    def test_half_entropy_between_floor_and_one(self):
        """At 50% entropy, confidence between 0.65 and 1.0."""
        from research.models.markov import soft_regime_confidence_multiplier
        m = soft_regime_confidence_multiplier(0.5)
        assert 0.65 < m < 1.0


class TestSoftRegimeCiScale:
    def test_zero_entropy_returns_base_scale(self):
        """At zero entropy, CI scale = 1.0."""
        from research.models.markov import soft_regime_ci_scale
        s = soft_regime_ci_scale(0.0)
        assert s == pytest.approx(1.0)

    def test_max_entropy_returns_widest_scale(self):
        """At max entropy, CI scale = 1.35."""
        from research.models.markov import soft_regime_ci_scale
        s = soft_regime_ci_scale(1.0)
        assert s == pytest.approx(1.35, abs=1e-10)

    def test_half_entropy_scales_modestly(self):
        """At 50% entropy, CI scale = 1.175."""
        from research.models.markov import soft_regime_ci_scale
        s = soft_regime_ci_scale(0.5)
        assert s == pytest.approx(1.175, abs=1e-10)


class TestAdjustHmmWeight:
    def test_zero_entropy_preserves_weight(self):
        """At zero entropy, HMM weight unchanged."""
        from research.models.markov import adjust_hmm_weight
        w = adjust_hmm_weight(0.7, 0.0)
        assert w == pytest.approx(0.7)

    def test_max_entropy_halves_weight(self):
        """At max entropy, HMM weight = weight * max(0.5, 1 - entropy*0.4) = 0.48."""
        from research.models.markov import adjust_hmm_weight
        w = adjust_hmm_weight(0.8, 1.0)
        assert w == pytest.approx(0.48, abs=1e-10)

    def test_high_entropy_attenuates_weight(self):
        """Higher entropy reduces HMM influence on forecast."""
        from research.models.markov import adjust_hmm_weight
        w_low = adjust_hmm_weight(0.5, 0.2)
        w_high = adjust_hmm_weight(0.5, 0.8)
        assert w_low > w_high
