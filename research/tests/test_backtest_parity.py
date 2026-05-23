"""Backtest parity tests for walk-forward with optional HMM enhancement."""

import math

import numpy as np
import pytest

from research.backtest.walk_forward import (
    BacktestStep,
    WalkForwardResult,
    walk_forward,
)


# ---------------------------------------------------------------------------
# Regression: walk_forward without HMM
# ---------------------------------------------------------------------------


def test_walk_forward_basic():
    rng = np.random.default_rng(42)
    # Generate 200 days of trending random-walk prices
    returns = rng.normal(0.001, 0.02, 200)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    result = walk_forward(prices, horizon=7, warmup=60, stride=20)
    assert len(result.steps) > 0
    assert len(result.errors) == 0
    for step in result.steps:
        assert 0 <= step.predicted_prob <= 1
        assert not np.isnan(step.predicted_return)
        assert step.ci_lower < step.ci_upper


def test_walk_forward_insufficient_data():
    result = walk_forward([100.0] * 50, horizon=7, warmup=60)
    assert len(result.steps) == 0
    assert len(result.errors) == 1
    assert "Insufficient data" in result.errors[0]


def test_walk_forward_directional_accuracy_reasonable():
    """On random-walk data, directional accuracy should be near 50%."""
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0, 0.015, 300)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    result = walk_forward(prices, horizon=7, warmup=120, stride=15)
    if result.steps:
        correct = sum(s.direction_correct for s in result.steps)
        acc = correct / len(result.steps)
        # Should be near 50% on random data
        assert 0.30 <= acc <= 0.70


def test_walk_forward_ci_coverage_reasonable():
    rng = np.random.default_rng(456)
    returns = rng.normal(0.0, 0.02, 300)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    result = walk_forward(prices, horizon=7, warmup=120, stride=15)
    if result.steps:
        covered = sum(s.in_ci for s in result.steps) / len(result.steps)
        # Random walk: CI coverage should be reasonable (not near 0 or 1)
        assert 0.2 <= covered <= 1.0


def _synthetic_vol_burst_prices() -> list[float]:
    rng = np.random.default_rng(123)
    prices = [100.0]
    for _ in range(180):
        prices.append(prices[-1] * float(np.exp(0.0002 + rng.normal(0.0, 0.006))))
    for _ in range(70):
        prices.append(prices[-1] * float(np.exp(-0.0005 + rng.normal(0.0, 0.055))))
    return prices


def test_walk_forward_garch_changes_high_vol_ci_but_off_path_is_stable():
    prices = _synthetic_vol_burst_prices()

    np.random.seed(77)
    baseline = walk_forward(prices, horizon=30, warmup=120, stride=10)
    np.random.seed(77)
    explicit_off = walk_forward(
        prices,
        horizon=30,
        warmup=120,
        stride=10,
        enable_garch_vol=False,
        garch_horizon_cap=7,
        garch_regime_ceiling=(1.1, 1.2),
    )
    np.random.seed(77)
    enabled = walk_forward(
        prices,
        horizon=30,
        warmup=120,
        stride=10,
        enable_garch_vol=True,
        garch_horizon_cap=7,
        garch_regime_ceiling=(1.1, 1.2),
    )

    assert not baseline.errors
    assert not enabled.errors
    assert [
        (s.predicted_prob, s.predicted_return, s.ci_lower, s.ci_upper)
        for s in explicit_off.steps
    ] == [
        (s.predicted_prob, s.predicted_return, s.ci_lower, s.ci_upper)
        for s in baseline.steps
    ]
    assert any(s.garch_vol_applied for s in enabled.steps)
    assert any(
        abs((e.ci_upper - e.ci_lower) - (b.ci_upper - b.ci_lower)) > 1e-6
        for e, b in zip(enabled.steps, baseline.steps)
    )


def test_walk_forward_entropy_records_metadata_and_modulates_ci():
    prices = _synthetic_vol_burst_prices()
    np.random.seed(91)
    result = walk_forward(
        prices,
        horizon=7,
        warmup=80,
        stride=5,
        enable_entropy_ci_modulation=True,
        entropy_window_size=5,
        entropy_kappa=0.5,
    )

    assert not result.errors
    assert len(result.steps) > 5
    assert all(s.transition_entropy_norm is not None for s in result.steps)
    assert any(s.transition_entropy_z is not None for s in result.steps)
    assert any(s.entropy_ci_scale is not None and abs(s.entropy_ci_scale - 1.0) > 1e-6 for s in result.steps)


# ---------------------------------------------------------------------------
# HMM integration
# ---------------------------------------------------------------------------


def test_walk_forward_with_hmm_runs():
    """walk_forward with use_hmm=True should complete without errors."""
    rng = np.random.default_rng(789)
    returns = rng.normal(0.001, 0.02, 250)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    result = walk_forward(prices, horizon=7, warmup=120, stride=20, use_hmm=True)
    assert len(result.steps) > 0
    assert len(result.errors) == 0
    for step in result.steps:
        assert 0 <= step.predicted_prob <= 1
        assert not np.isnan(step.predicted_return)
        assert step.ci_lower < step.ci_upper


def test_walk_forward_hmm_vs_base_same_count():
    """HMM and base variants should produce same number of steps."""
    rng = np.random.default_rng(101)
    returns = rng.normal(0.0, 0.02, 250)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    result_base = walk_forward(prices, horizon=7, warmup=120, stride=20, use_hmm=False)
    result_hmm = walk_forward(prices, horizon=7, warmup=120, stride=20, use_hmm=True)
    assert len(result_base.steps) == len(result_hmm.steps)


def test_walk_forward_hmm_different_predictions():
    """HMM variant should produce different predictions than base."""
    rng = np.random.default_rng(202)
    returns = rng.normal(0.0, 0.02, 250)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    result_base = walk_forward(prices, horizon=7, warmup=120, stride=20, use_hmm=False)
    result_hmm = walk_forward(prices, horizon=7, warmup=120, stride=20, use_hmm=True)

    diffs = [abs(b.predicted_return - h.predicted_return) for b, h in zip(result_base.steps, result_hmm.steps)]
    assert any(d > 1e-6 for d in diffs), "HMM predictions are identical to base"


def test_walk_forward_with_asset_profile():
    """walk_forward with asset_profile should accept known profiles."""
    rng = np.random.default_rng(303)
    returns = rng.normal(0.0, 0.02, 250)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    for profile in ["etf", "equity", "crypto", "commodity"]:
        result = walk_forward(
            prices, horizon=7, warmup=120, stride=20,
            use_hmm=True, asset_profile=profile,
        )
        assert len(result.steps) > 0
        assert len(result.errors) == 0


def test_walk_forward_step_fields():
    """Verify all BacktestStep fields are populated."""
    rng = np.random.default_rng(404)
    returns = rng.normal(0.0, 0.02, 200)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    result = walk_forward(prices, horizon=7, warmup=60, stride=20)
    assert result.steps
    step = result.steps[0]
    assert step.start_idx >= 0
    assert isinstance(step.direction_correct, bool)
    assert isinstance(step.in_ci, bool)
    assert step.realised_price > 0


def test_walk_forward_result_dataclass():
    """Verify WalkForwardResult dataclass."""
    result = WalkForwardResult()
    assert result.steps == []
    assert result.errors == []
    result.steps.append(BacktestStep(
        start_idx=0, predicted_prob=0.5, predicted_return=0.0,
        ci_lower=90, ci_upper=110, realised_return=0.0,
        realised_price=100, direction_correct=True, in_ci=True,
    ))
    assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# CRPS / scaled CRPS / Murphy-Winkler parity tests
# ---------------------------------------------------------------------------


class TestCrpsParity:
    """CRPS computed as normal approximation from step interval, matching TS."""

    def test_crps_empty_returns_zero(self):
        from research.backtest.metrics import crps
        assert crps([]) == 0.0

    def test_crps_single_step(self):
        from research.backtest.metrics import crps
        step = BacktestStep(
            start_idx=0, predicted_prob=0.55, predicted_return=0.03,
            ci_lower=90, ci_upper=110, realised_return=0.05,
            realised_price=105, direction_correct=True, in_ci=True,
        )
        result = crps([step])
        # fc = 105/(1.05)*1.03 = 103, sigma = 20/(2*1.64485) = 6.07957
        # CRPS(105 | N(103, 6.07957))
        assert result == pytest.approx(1.6809034392, rel=1e-9)

    def test_crps_multiple_steps(self):
        from research.backtest.metrics import crps
        steps = [
            BacktestStep(start_idx=0, predicted_prob=0.55, predicted_return=0.03,
                         ci_lower=90, ci_upper=110, realised_return=0.05,
                         realised_price=105, direction_correct=True, in_ci=True),
            BacktestStep(start_idx=1, predicted_prob=0.60, predicted_return=0.01,
                         ci_lower=95, ci_upper=105, realised_return=0.0,
                         realised_price=100, direction_correct=False, in_ci=True),
            BacktestStep(start_idx=2, predicted_prob=0.45, predicted_return=0.0,
                         ci_lower=190, ci_upper=210, realised_return=-0.02,
                         realised_price=195, direction_correct=False, in_ci=True),
        ]
        result = crps(steps)
        assert result == pytest.approx(1.6485932011, rel=1e-9)

    def test_crps_forecast_center_fallback(self):
        """When realised_return ≈ -1, denominator → 0, uses CI midpoint fallback."""
        from research.backtest.metrics import crps
        step = BacktestStep(
            start_idx=0, predicted_prob=0.5, predicted_return=0.0,
            ci_lower=90, ci_upper=110, realised_return=-0.999999,
            realised_price=100, direction_correct=False, in_ci=True,
        )
        # denominator = 1 + (-0.999999) ≈ 0 → fallback to (90+110)/2 = 100
        # fc = 100 * (1+0) = 100, sigma = 20/(2*1.64485) = 6.07957
        result = crps([step])
        assert result > 0


class TestScaledCrpsParity:
    """Scaled CRPS divides each step CRPS by interval-implied sigma, matching TS."""

    def test_scaled_crps_empty_returns_zero(self):
        from research.backtest.metrics import scaled_crps
        assert scaled_crps([]) == 0.0

    def test_scaled_crps_multiple_steps(self):
        from research.backtest.metrics import scaled_crps
        steps = [
            BacktestStep(start_idx=0, predicted_prob=0.55, predicted_return=0.03,
                         ci_lower=90, ci_upper=110, realised_return=0.05,
                         realised_price=105, direction_correct=True, in_ci=True),
            BacktestStep(start_idx=1, predicted_prob=0.60, predicted_return=0.01,
                         ci_lower=95, ci_upper=105, realised_return=0.0,
                         realised_price=100, direction_correct=False, in_ci=True),
            BacktestStep(start_idx=2, predicted_prob=0.45, predicted_return=0.0,
                         ci_lower=190, ci_upper=210, realised_return=-0.02,
                         realised_price=195, direction_correct=False, in_ci=True),
        ]
        result = scaled_crps(steps)
        assert result == pytest.approx(0.3172501193, rel=1e-9)


class TestMurphyWinklerParity:
    """Murphy-Winkler interval score decomposition, matching TS."""

    def test_empty_returns_zeros(self):
        from research.backtest.metrics import murphy_winkler_decomposition
        d = murphy_winkler_decomposition([])
        assert d == {
            "mean_width": 0.0,
            "lower_miss_penalty": 0.0,
            "upper_miss_penalty": 0.0,
            "total_score": 0.0,
            "coverage": 0.0,
        }

    def test_decomposition_known_values(self):
        from research.backtest.metrics import murphy_winkler_decomposition
        steps = [
            BacktestStep(start_idx=0, predicted_prob=0.5, predicted_return=0.0,
                         ci_lower=90, ci_upper=110, realised_return=0.0,
                         realised_price=95, direction_correct=False, in_ci=True),
            BacktestStep(start_idx=1, predicted_prob=0.5, predicted_return=0.0,
                         ci_lower=90, ci_upper=110, realised_return=0.0,
                         realised_price=85, direction_correct=False, in_ci=False),
            BacktestStep(start_idx=2, predicted_prob=0.5, predicted_return=0.0,
                         ci_lower=90, ci_upper=110, realised_return=0.0,
                         realised_price=115, direction_correct=False, in_ci=False),
            BacktestStep(start_idx=3, predicted_prob=0.5, predicted_return=0.0,
                         ci_lower=90, ci_upper=110, realised_return=0.0,
                         realised_price=105, direction_correct=False, in_ci=True),
        ]
        d = murphy_winkler_decomposition(steps)
        assert d["mean_width"] == pytest.approx(20.0)
        assert d["lower_miss_penalty"] == pytest.approx(25.0)
        assert d["upper_miss_penalty"] == pytest.approx(25.0)
        assert d["total_score"] == pytest.approx(70.0)
        assert d["coverage"] == pytest.approx(0.5)

    def test_decomposition_custom_alpha(self):
        from research.backtest.metrics import murphy_winkler_decomposition
        step = BacktestStep(
            start_idx=0, predicted_prob=0.5, predicted_return=0.0,
            ci_lower=90, ci_upper=110, realised_return=0.0,
            realised_price=80, direction_correct=False, in_ci=False,
        )
        # alpha=0.05 → penalty_scale = 2/0.05 = 40
        d = murphy_winkler_decomposition([step], alpha=0.05)
        assert d["lower_miss_penalty"] == pytest.approx(40 * (90 - 80))

    def test_score_is_total_from_decomposition(self):
        from research.backtest.metrics import murphy_winkler_score, murphy_winkler_decomposition
        steps = [
            BacktestStep(start_idx=0, predicted_prob=0.5, predicted_return=0.0,
                         ci_lower=90, ci_upper=110, realised_return=0.0,
                         realised_price=85, direction_correct=False, in_ci=False),
        ]
        assert murphy_winkler_score(steps) == pytest.approx(
            murphy_winkler_decomposition(steps)["total_score"]
        )
