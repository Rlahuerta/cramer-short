"""R5 Idea #3 — Naive baseline guards for forecast quality.

Python mirror of src/tools/finance/backtest/baselines.ts.

Source: docs/forecast-improvement-ideas-round5-2026-04-29.md (Idea #3),
arXiv:2502.09079 Puoti et al. 2025.

Computes simple model-free baselines that any non-trivial forecast head
must beat:
  - Coin-flip:   always P(up)=0.5
  - Last-period: P(up)=0.65 if previous step was up, else 0.35

The gate: deployed arm directional accuracy must be >= better baseline
accuracy minus `slack` (default 0.02).
"""

from __future__ import annotations
from dataclasses import dataclass
from research.backtest.walk_forward import BacktestStep


@dataclass(frozen=True)
class BaselineMetricBlock:
    """Brier score + directional accuracy for a synthetic baseline."""

    n: int
    brier_score: float
    directional_accuracy: float


@dataclass(frozen=True)
class NaiveBaselineReport:
    """Coin-flip and last-period baseline metrics."""

    coin_flip: BaselineMetricBlock
    last_period: BaselineMetricBlock


def _synth(step: BacktestStep, predicted_prob: float) -> BacktestStep:
    """Return a copy of `step` with an overridden predicted_prob and
    direction_correct recomputed from the new probability."""
    pred_up = predicted_prob > 0.5
    direction_correct = (pred_up and step.realised_return > 0) or (
        not pred_up and step.realised_return <= 0
    )
    return BacktestStep(
        start_idx=step.start_idx,
        predicted_prob=predicted_prob,
        predicted_return=step.predicted_return,
        ci_lower=step.ci_lower,
        ci_upper=step.ci_upper,
        realised_return=step.realised_return,
        realised_price=step.realised_price,
        direction_correct=direction_correct,
        in_ci=step.in_ci,
    )


def compute_coin_flip_baseline(steps: list[BacktestStep]) -> BaselineMetricBlock:
    """Baseline: always predict P(up)=0.5 (strict random null)."""
    if not steps:
        return BaselineMetricBlock(n=0, brier_score=1.0, directional_accuracy=0.0)

    synthetic = [_synth(s, 0.5) for s in steps]
    outcomes = [1.0 if s.realised_return > 0 else 0.0 for s in synthetic]

    bs = sum((s.predicted_prob - o) ** 2 for s, o in zip(synthetic, outcomes)) / len(synthetic)
    da = sum(1 for s in synthetic if s.direction_correct) / len(synthetic)

    return BaselineMetricBlock(n=len(steps), brier_score=bs, directional_accuracy=da)


def compute_last_period_baseline(steps: list[BacktestStep]) -> BaselineMetricBlock:
    """Baseline: predict last realized direction continues.

    For step 0 (no prior): falls back to coin-flip (P=0.5).
    Probability assignment: 0.65 if last-up, 0.35 if last-down (avoids
    degenerate 0/1 Brier extremes on random walks).
    """
    if not steps:
        return BaselineMetricBlock(n=0, brier_score=1.0, directional_accuracy=0.0)

    synthetic: list[BacktestStep] = []
    prev_return = 0.0

    for i, step in enumerate(steps):
        if i == 0:
            synth_step = _synth(step, 0.5)
        else:
            prob = 0.65 if prev_return > 0 else 0.35
            synth_step = _synth(step, prob)
        prev_return = step.realised_return
        synthetic.append(synth_step)

    outcomes = [1.0 if s.realised_return > 0 else 0.0 for s in synthetic]

    bs = sum((s.predicted_prob - o) ** 2 for s, o in zip(synthetic, outcomes)) / len(synthetic)
    da = sum(1 for s in synthetic if s.direction_correct) / len(synthetic)
    return BaselineMetricBlock(n=len(steps), brier_score=bs, directional_accuracy=da)


def compute_naive_baselines(steps: list[BacktestStep]) -> NaiveBaselineReport:
    """Compute both coin-flip and last-period baselines in one call."""
    return NaiveBaselineReport(
        coin_flip=compute_coin_flip_baseline(steps),
        last_period=compute_last_period_baseline(steps),
    )


def naive_baseline_guard(
    arm_steps: list[BacktestStep],
    slack: float = 0.02,
) -> dict:
    """Strict quality gate: arm directional accuracy must beat baselines.

    Returns {passes, arm_dir_acc, baseline_dir_acc, gap}.
    The arm must satisfy: arm_dir_acc >= baseline_dir_acc - slack.
    """
    from research.backtest.metrics import directional_accuracy

    arm_dir_acc = directional_accuracy(arm_steps)
    baselines = compute_naive_baselines(arm_steps)
    baseline_dir_acc = max(
        baselines.coin_flip.directional_accuracy,
        baselines.last_period.directional_accuracy,
    )
    gap = arm_dir_acc - baseline_dir_acc
    return {
        "passes": gap >= -slack,
        "arm_dir_acc": arm_dir_acc,
        "baseline_dir_acc": baseline_dir_acc,
        "gap": gap,
    }
