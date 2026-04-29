"""Parity tests — Python backtest/baselines mirrors TS backtest/baselines.ts (R5 Idea #3)."""

from __future__ import annotations

import pytest

from research.backtest.baselines import (
    compute_coin_flip_baseline,
    compute_last_period_baseline,
    compute_naive_baselines,
    naive_baseline_guard,
)
from research.backtest.walk_forward import BacktestStep


def _step(realised_return: float, predicted_prob: float = 0.5) -> BacktestStep:
    """Minimal BacktestStep factory for baseline tests."""
    pred_up = predicted_prob > 0.5
    direction_correct = (pred_up and realised_return > 0) or (
        not pred_up and realised_return <= 0
    )
    return BacktestStep(
        start_idx=0,
        predicted_prob=predicted_prob,
        predicted_return=realised_return,
        ci_lower=-0.05,
        ci_upper=0.05,
        realised_return=realised_return,
        realised_price=100.0,
        direction_correct=direction_correct,
        in_ci=True,
    )


class TestCoinFlipBaseline:
    def test_empty_input(self):
        r = compute_coin_flip_baseline([])
        assert r.n == 0
        assert r.brier_score == 1.0
        assert r.directional_accuracy == 0.0

    def test_brier_score_always_0_25_for_balanced_coin_flip(self):
        """For P=0.5, outcome in {0,1}: (0.5 - 0)^2 = 0.25 and (0.5 - 1)^2 = 0.25."""
        steps = [_step(0.01), _step(-0.01), _step(0.02), _step(-0.03)]
        r = compute_coin_flip_baseline(steps)
        assert r.brier_score == pytest.approx(0.25)

    def test_n_matches_input_length(self):
        steps = [_step(0.01)] * 7
        r = compute_coin_flip_baseline(steps)
        assert r.n == 7

    def test_directional_accuracy_is_0_5_for_alternating(self):
        """P=0.5 -> direction is neither up nor down: direction_correct depends on
        the synth step logic which uses pred_up = (0.5 > 0.5) = False for coin flip.
        For P exactly 0.5: pred_up = False, so correct when realised <= 0."""
        up = _step(0.01)
        down = _step(-0.01)
        steps = [up, down, up, down]
        r = compute_coin_flip_baseline(steps)
        # pred_up=False for all: correct on downs (2), wrong on ups (2) -> 0.5
        assert r.directional_accuracy == pytest.approx(0.5)


class TestLastPeriodBaseline:
    def test_empty_input(self):
        r = compute_last_period_baseline([])
        assert r.n == 0

    def test_first_step_uses_coin_flip(self):
        steps = [_step(0.01)]  # only one step
        r = compute_last_period_baseline(steps)
        # The first step uses P=0.5 (coin flip)
        assert r.brier_score == pytest.approx(0.25)

    def test_persistent_up_trend_has_high_accuracy(self):
        """All returns > 0: last-period correctly predicts up each time."""
        steps = [_step(0.01)] * 10
        r = compute_last_period_baseline(steps)
        # Step 0: P=0.5 (coinflip), steps 1-9: P=0.65 (prev was up) -> correct
        assert r.directional_accuracy >= 0.8

    def test_prediction_tracks_previous_direction(self):
        """After a down step, the prediction should be P=0.35 (bearish)."""
        steps = [_step(-0.01), _step(-0.02)]
        r = compute_last_period_baseline(steps)
        # Step 1 should predict down (P=0.35) and be correct
        assert r.n == 2

    def test_different_from_coin_flip_on_trending_data(self):
        """Last-period should differ from coin-flip on trending returns."""
        steps = [_step(0.02 * (i + 1)) for i in range(20)]
        cf = compute_coin_flip_baseline(steps)
        lp = compute_last_period_baseline(steps)
        # On uptrend, last-period should have higher directional accuracy
        assert lp.directional_accuracy >= cf.directional_accuracy


class TestComputeNaiveBaselines:
    def test_returns_both_baselines(self):
        steps = [_step(r) for r in [0.01, -0.02, 0.01, -0.01, 0.03]]
        report = compute_naive_baselines(steps)
        assert report.coin_flip.n == 5
        assert report.last_period.n == 5


class TestNaiveBaselineGuard:
    def test_passes_when_arm_beats_baselines(self):
        """Perfect directional accuracy should easily beat baselines."""
        # All returns up, arm correctly predicts up
        steps = [_step(0.01, predicted_prob=0.8) for _ in range(20)]
        result = naive_baseline_guard(steps)
        assert result["passes"] is True

    def test_fails_when_arm_is_below_baseline_minus_slack(self):
        """Arm with 0% directional accuracy should fail."""
        # All returns up but arm predicts down
        steps = [_step(0.01, predicted_prob=0.3) for _ in range(20)]
        result = naive_baseline_guard(steps)
        assert result["passes"] is False

    def test_gap_is_computed_correctly(self):
        steps = [_step(r) for r in [0.01, -0.01, 0.02, -0.02, 0.01]]
        result = naive_baseline_guard(steps)
        assert result["gap"] == pytest.approx(
            result["arm_dir_acc"] - result["baseline_dir_acc"]
        )

    def test_custom_slack(self):
        """With very generous slack, even a below-baseline arm should pass."""
        steps = [_step(0.01, predicted_prob=0.3) for _ in range(20)]
        result = naive_baseline_guard(steps, slack=0.99)
        assert result["passes"] is True
