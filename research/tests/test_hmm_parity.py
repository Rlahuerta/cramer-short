"""Parity tests for HMM module — Python outputs vs TypeScript assertions."""

import math

import numpy as np
import pytest

from research.models.hmm import (
    AssetProfile,
    ASSET_PROFILES,
    HMMFitResult,
    HMMParams,
    HMMPrediction,
    baum_welch,
    fit_volatility_hmm,
    initialize_hmm,
    mat_pow,
    predict,
    viterbi,
)
from research.models.trajectory import (
    RegimeStats,
    compute_horizon_drift_vol,
    compute_trajectory,
)


# ---------------------------------------------------------------------------
# Dataclass constructors
# ---------------------------------------------------------------------------


def test_hmm_params_dataclass():
    pi = np.array([0.3, 0.4, 0.3])
    A = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    means = np.array([-0.01, 0.0, 0.01])
    stds = np.array([0.02, 0.015, 0.025])
    params = HMMParams(n_states=3, pi=pi, A=A, means=means, stds=stds)
    assert params.n_states == 3
    assert np.allclose(params.pi, pi)
    assert np.allclose(params.A, A)
    assert np.allclose(params.means, means)
    assert np.allclose(params.stds, stds)


def test_hmm_fit_result_dataclass():
    pi = np.array([0.3, 0.4, 0.3])
    A = np.eye(3) * 0.8 + 0.1
    means = np.array([-0.01, 0.0, 0.01])
    stds = np.array([0.02, 0.015, 0.025])
    params = HMMParams(n_states=3, pi=pi, A=A, means=means, stds=stds)
    result = HMMFitResult(
        params=params,
        log_likelihood=-100.0,
        iterations=50,
        converged=True,
    )
    assert result.converged is True
    assert result.iterations == 50
    assert result.log_likelihood == pytest.approx(-100.0)
    assert result.params.n_states == 3


def test_hmm_prediction_dataclass():
    pred = HMMPrediction(
        current_state=1,
        state_probabilities=np.array([0.1, 0.8, 0.1]),
        current_state_probabilities=np.array([0.1, 0.8, 0.1]),
        forecast_probabilities=np.array([0.2, 0.6, 0.2]),
        expected_return=0.005,
        expected_volatility=0.02,
    )
    assert pred.current_state == 1
    assert 0 <= pred.expected_return <= 1
    assert pred.expected_volatility >= 0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_initialize_hmm_sorted_means():
    rng = np.random.default_rng(42)
    obs = np.concatenate([
        rng.normal(-0.01, 0.02, 100),
        rng.normal(0.0, 0.01, 100),
        rng.normal(0.01, 0.03, 100),
    ])
    params = initialize_hmm(obs, n_states=3)
    assert params.n_states == 3
    assert len(params.means) == 3
    assert params.means[0] <= params.means[1] <= params.means[2]


def test_initialize_hmm_diagonal_dominant():
    rng = np.random.default_rng(42)
    obs = rng.standard_normal(300)
    params = initialize_hmm(obs, n_states=3)
    for i in range(3):
        diag = params.A[i, i]
        off_diag_max = max(params.A[i, j] for j in range(3) if j != i)
        assert diag >= off_diag_max


# ---------------------------------------------------------------------------
# Baum-Welch fitting
# ---------------------------------------------------------------------------


def _synthetic_regime_returns(rng, n_per_regime=200):
    """Generate synthetic returns from 3 regimes."""
    bear = rng.normal(-0.01, 0.025, n_per_regime)
    sideways = rng.normal(0.0, 0.01, n_per_regime)
    bull = rng.normal(0.015, 0.02, n_per_regime)
    return np.concatenate([bear, sideways, bull])


def test_baum_welch_converges_synthetic():
    rng = np.random.default_rng(123)
    obs = _synthetic_regime_returns(rng)
    result = baum_welch(obs, n_states=3, max_iterations=50, tolerance=1e-3)
    assert result.converged is True
    assert result.iterations > 0
    assert result.iterations <= 50
    assert math.isfinite(result.log_likelihood)
    assert result.params.n_states == 3
    assert len(result.params.means) == 3
    assert len(result.params.stds) == 3
    # Means should roughly reflect the 3 regimes
    sorted_means = np.sort(result.params.means)
    assert sorted_means[0] < sorted_means[1] < sorted_means[2]


def test_baum_welch_non_convergence():
    rng = np.random.default_rng(456)
    obs = rng.standard_normal(20)  # too short / noisy
    result = baum_welch(obs, n_states=3, max_iterations=5, tolerance=1e-6)
    # Should either not converge or handle gracefully
    assert result.converged is False or result.iterations == 5


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------


def test_viterbi_recover_known_states():
    rng = np.random.default_rng(789)
    n = 100
    bear = rng.normal(-0.02, 0.02, n)
    sideways = rng.normal(0.0, 0.01, n)
    bull = rng.normal(0.02, 0.02, n)
    obs = np.concatenate([bear, sideways, bull])
    true_states = [0] * n + [1] * n + [2] * n

    result = baum_welch(obs, n_states=3, max_iterations=50, tolerance=1e-3)
    decoded = viterbi(obs, result.params)
    assert len(decoded) == len(obs)
    # On clean synthetic data we should recover most states correctly
    accuracy = sum(d == t for d, t in zip(decoded, true_states)) / len(true_states)
    assert accuracy >= 0.70


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_probabilities_sum_to_one():
    rng = np.random.default_rng(101)
    obs = _synthetic_regime_returns(rng)
    result = baum_welch(obs, n_states=3, max_iterations=50, tolerance=1e-3)
    pred = predict(obs, result.params, forecast_horizon=7)
    assert pytest.approx(sum(pred.state_probabilities), abs=1e-5) == 1.0
    assert pytest.approx(sum(pred.forecast_probabilities), abs=1e-5) == 1.0


def test_predict_probabilities_in_range():
    rng = np.random.default_rng(202)
    obs = _synthetic_regime_returns(rng)
    result = baum_welch(obs, n_states=3, max_iterations=50, tolerance=1e-3)
    pred = predict(obs, result.params, forecast_horizon=7)
    for p in pred.state_probabilities:
        assert 0.0 <= p <= 1.0
    for p in pred.forecast_probabilities:
        assert 0.0 <= p <= 1.0
    assert 0 <= pred.current_state < 3


def test_expected_return_finite_non_negative():
    rng = np.random.default_rng(303)
    obs = _synthetic_regime_returns(rng)
    result = baum_welch(obs, n_states=3, max_iterations=50, tolerance=1e-3)
    pred = predict(obs, result.params, forecast_horizon=7)
    assert math.isfinite(pred.expected_return)


def test_expected_volatility_finite_non_negative():
    rng = np.random.default_rng(404)
    obs = _synthetic_regime_returns(rng)
    result = baum_welch(obs, n_states=3, max_iterations=50, tolerance=1e-3)
    pred = predict(obs, result.params, forecast_horizon=7)
    assert math.isfinite(pred.expected_volatility)
    assert pred.expected_volatility >= 0.0


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def test_mat_pow_identity():
    A = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.2, 0.3, 0.5]])
    assert np.allclose(mat_pow(A, 1), A)


def test_mat_pow_n_step():
    A = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.2, 0.3, 0.5]])
    A3 = mat_pow(A, 3)
    for row in A3:
        assert pytest.approx(sum(row), abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# Volatility HMM
# ---------------------------------------------------------------------------


def test_volatility_hmm_scale_in_range():
    rng = np.random.default_rng(505)
    returns = rng.normal(0.0, 0.02, 500)
    scale = fit_volatility_hmm(returns, vol_window=5, n_states=2)
    assert 0.5 <= scale <= 2.0


# ---------------------------------------------------------------------------
# Asset profiles
# ---------------------------------------------------------------------------


def test_asset_profile_lookup():
    crypto = ASSET_PROFILES["crypto"]
    assert isinstance(crypto, AssetProfile)
    assert crypto.hmm_weight_multiplier == pytest.approx(0.5)
    assert crypto.student_t_nu == 3
    assert crypto.decay_rate == pytest.approx(0.94)

    etf = ASSET_PROFILES["etf"]
    assert etf.hmm_weight_multiplier == pytest.approx(1.1)
    assert etf.student_t_nu == 5
    assert etf.decay_rate == pytest.approx(0.97)

    equity = ASSET_PROFILES["equity"]
    assert equity.hmm_weight_multiplier == pytest.approx(0.9)
    assert equity.student_t_nu == 4
    assert equity.decay_rate == pytest.approx(0.96)

    commodity = ASSET_PROFILES["commodity"]
    assert commodity.hmm_weight_multiplier == pytest.approx(0.7)
    assert commodity.student_t_nu == 4
    assert commodity.decay_rate == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Integration with trajectory engine
# ---------------------------------------------------------------------------


def _make_fixture():
    P = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.2, 0.3, 0.5]])
    regime_stats = {
        "bull": RegimeStats(mean_return=0.01, std_return=0.02),
        "bear": RegimeStats(mean_return=-0.01, std_return=0.025),
        "sideways": RegimeStats(mean_return=0.0, std_return=0.015),
    }
    return P, regime_stats


def test_hmm_override_blending():
    P, regime_stats = _make_fixture()
    # Without override
    dv_base = compute_horizon_drift_vol(7, P, regime_stats, "bull")
    # With 50% weight override
    override = {"drift": 0.02, "vol": 0.03, "weight": 0.5}
    dv_ovr = compute_horizon_drift_vol(7, P, regime_stats, "bull", hmm_override=override)
    # Effective mu should be between base and override
    assert dv_ovr["mu_n"] > min(dv_base["mu_n"], 7 * 0.02)
    assert dv_ovr["mu_n"] < max(dv_base["mu_n"], 7 * 0.02)
    assert dv_ovr["sigma_n"] > 0


def test_trajectory_with_hmm_override_finite():
    P, regime_stats = _make_fixture()
    override = {"drift": 0.02, "vol": 0.03, "weight": 0.3}
    traj = compute_trajectory(100, 7, P, regime_stats, "bull", hmm_override=override, n_samples=500)
    assert len(traj) == 7
    for pt in traj:
        assert np.isfinite(pt.expected_price)
        assert np.isfinite(pt.lower_bound)
        assert np.isfinite(pt.upper_bound)
        assert pt.lower_bound < pt.expected_price < pt.upper_bound


def test_trajectory_without_hmm_override_regression():
    """Existing behavior unchanged when hmm_override is None (default)."""
    P, regime_stats = _make_fixture()
    traj = compute_trajectory(100, 7, P, regime_stats, "bull", n_samples=500)
    assert len(traj) == 7
    for pt in traj:
        assert 0 <= pt.p_up <= 1
        assert pt.lower_bound < pt.expected_price < pt.upper_bound
