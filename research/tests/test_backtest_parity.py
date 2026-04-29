"""Backtest parity tests for walk-forward with optional HMM enhancement."""

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
