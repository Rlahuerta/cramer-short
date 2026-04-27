"""Parity tests for trajectory module — Python outputs vs TypeScript assertions."""

import math

import numpy as np
import pytest

from research.models.trajectory import (
    RegimeStats,
    compute_trajectory,
    compute_horizon_drift_vol,
    compute_scenario_probabilities,
    normal_cdf,
    student_t_cdf,
    student_t_ppf,
    student_t_survival,
    log_normal_survival,
    interpolate_survival,
)


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def test_normal_cdf_symmetry():
    assert normal_cdf(0) == pytest.approx(0.5, abs=1e-6)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert normal_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_student_t_cdf_nu5():
    assert student_t_cdf(0, 5) == pytest.approx(0.5, abs=1e-6)
    # Student-t with nu=5 is wider than normal
    assert student_t_cdf(1.96, 5) < normal_cdf(1.96)


def test_student_t_ppf_inverse():
    for p in [0.01, 0.25, 0.5, 0.75, 0.99]:
        x = student_t_ppf(p, 5)
        assert student_t_cdf(x, 5) == pytest.approx(p, abs=1e-5)


def test_log_normal_survival_monotonic():
    current = 100.0
    targets = [80, 90, 100, 110, 120]
    probs = [log_normal_survival(current, t, 0.05, 0.2) for t in targets]
    assert probs == sorted(probs, reverse=True)  # decreasing in target


def test_student_t_survival_current():
    # P(price > current) when drift=0, vol=0 → undefined; vol > 0 needed
    p = student_t_survival(100, 100, 0.0, 0.01, nu=5)
    assert 0 <= p <= 1


# ---------------------------------------------------------------------------
# Horizon drift/vol
# ---------------------------------------------------------------------------

def test_compute_horizon_drift_vol_default():
    P = np.eye(3) * 0.8 + np.ones((3, 3)) * 0.2 / 3
    regime_stats = {
        "bull": RegimeStats(mean_return=0.01, std_return=0.02),
        "bear": RegimeStats(mean_return=-0.01, std_return=0.025),
        "sideways": RegimeStats(mean_return=0.0, std_return=0.01),
    }
    dv = compute_horizon_drift_vol(7, P, regime_stats, "bull")
    assert dv["mu_n"] > 0  # bull regime has positive drift
    assert dv["sigma_n"] > 0


def test_compute_horizon_drift_vol_horizon_scaling():
    P = np.eye(3) * 0.8 + np.ones((3, 3)) * 0.2 / 3
    regime_stats = {
        "bull": RegimeStats(mean_return=0.01, std_return=0.02),
        "bear": RegimeStats(mean_return=-0.01, std_return=0.025),
        "sideways": RegimeStats(mean_return=0.0, std_return=0.01),
    }
    dv1 = compute_horizon_drift_vol(1, P, regime_stats, "bull")
    dv7 = compute_horizon_drift_vol(7, P, regime_stats, "bull")
    # Sigma should be larger at longer horizon (sqrt scaling dominates)
    assert dv7["sigma_n"] > dv1["sigma_n"]
    # Both should be finite and non-zero
    assert dv1["mu_n"] != 0.0
    assert dv7["mu_n"] != 0.0
    assert math.isfinite(dv1["sigma_n"])
    assert math.isfinite(dv7["sigma_n"])


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def _make_fixture():
    P = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.2, 0.3, 0.5]])
    regime_stats = {
        "bull": RegimeStats(mean_return=0.01, std_return=0.02),
        "bear": RegimeStats(mean_return=-0.01, std_return=0.025),
        "sideways": RegimeStats(mean_return=0.0, std_return=0.015),
    }
    return P, regime_stats


def test_compute_trajectory_finite():
    P, regime_stats = _make_fixture()
    traj = compute_trajectory(100, 7, P, regime_stats, "bull", n_samples=500)
    assert len(traj) == 7
    for pt in traj:
        assert np.isfinite(pt.expected_price)
        assert np.isfinite(pt.lower_bound)
        assert np.isfinite(pt.upper_bound)
        assert np.isfinite(pt.p_up)
        assert pt.lower_bound < pt.expected_price < pt.upper_bound


def test_compute_trajectory_ci_widening():
    P, regime_stats = _make_fixture()
    traj = compute_trajectory(100, 7, P, regime_stats, "bull", n_samples=500)
    for i in range(1, len(traj)):
        prev_w = traj[i - 1].upper_bound - traj[i - 1].lower_bound
        curr_w = traj[i].upper_bound - traj[i].lower_bound
        # CI should generally widen; allow 1% tolerance for MC noise
        assert curr_w >= prev_w - 1.0


def test_compute_trajectory_day1_near_current():
    P, regime_stats = _make_fixture()
    current = 100.0
    traj = compute_trajectory(current, 7, P, regime_stats, "bull", n_samples=500)
    day1 = traj[0]
    pct_diff = abs(day1.expected_price - current) / current
    assert pct_diff < 0.10  # within 10%


def test_compute_trajectory_p_up_in_range():
    P, regime_stats = _make_fixture()
    traj = compute_trajectory(100, 7, P, regime_stats, "bull", n_samples=500)
    for pt in traj:
        assert 0 <= pt.p_up <= 1


def test_compute_trajectory_cumulative_return_format():
    P, regime_stats = _make_fixture()
    traj = compute_trajectory(100, 7, P, regime_stats, "bull", n_samples=500)
    for pt in traj:
        assert pt.cumulative_return.endswith("%")
        assert ("+" in pt.cumulative_return) or ("-" in pt.cumulative_return) or ("0.0%" in pt.cumulative_return)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def test_interpolate_survival_empty():
    assert interpolate_survival([], 100) == 0.5


def test_interpolate_survival_exact_match():
    dist = [
        {"price": 90, "probability": 0.9},
        {"price": 100, "probability": 0.5},
        {"price": 110, "probability": 0.1},
    ]
    assert interpolate_survival(dist, 100) == 0.5


def test_interpolate_survival_below_first():
    dist = [
        {"price": 90, "probability": 0.9},
        {"price": 100, "probability": 0.5},
    ]
    assert interpolate_survival(dist, 80) == 1.0


def test_interpolate_survival_above_last():
    dist = [
        {"price": 90, "probability": 0.9},
        {"price": 100, "probability": 0.5},
    ]
    assert interpolate_survival(dist, 120) == 0.0


def test_interpolate_survival_midpoint():
    dist = [
        {"price": 90, "probability": 0.9},
        {"price": 100, "probability": 0.5},
        {"price": 110, "probability": 0.1},
    ]
    # Halfway between 90 and 100 → halfway between 0.9 and 0.5
    assert interpolate_survival(dist, 95) == pytest.approx(0.7, abs=1e-6)


# ---------------------------------------------------------------------------
# Scenario probabilities
# ---------------------------------------------------------------------------

def test_compute_scenario_probabilities_sum():
    dist = [
        {"price": 80, "probability": 1.0},
        {"price": 90, "probability": 0.9},
        {"price": 95, "probability": 0.7},
        {"price": 100, "probability": 0.5},
        {"price": 105, "probability": 0.3},
        {"price": 110, "probability": 0.1},
        {"price": 120, "probability": 0.0},
    ]
    scenarios = compute_scenario_probabilities(dist, 100)
    total = sum(b["probability"] for b in scenarios["buckets"])
    assert total == pytest.approx(1.0, abs=1e-5)


def test_compute_scenario_probabilities_p_up():
    dist = [
        {"price": 80, "probability": 1.0},
        {"price": 100, "probability": 0.5},
        {"price": 120, "probability": 0.0},
    ]
    scenarios = compute_scenario_probabilities(dist, 100)
    assert scenarios["p_up"] == 0.5


def test_compute_scenario_probabilities_expected_price():
    dist = [
        {"price": 90, "probability": 0.9},
        {"price": 100, "probability": 0.5},
        {"price": 110, "probability": 0.1},
    ]
    scenarios = compute_scenario_probabilities(dist, 100)
    assert scenarios["expected_price"] > 90
    assert scenarios["expected_price"] < 110
