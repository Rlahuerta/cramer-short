"""Backtest metrics: Brier score, directional accuracy, CI coverage, bootstrap CIs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from research.backtest.walk_forward import BacktestStep


def brier_score(steps: list[BacktestStep]) -> float:
    """Mean squared error of probabilistic forecasts.

    For binary outcomes: Brier = mean((p - outcome)^2) where outcome = 1 if return > 0.
    """
    if not steps:
        return 0.0
    outcomes = [1.0 if s.realised_return > 0 else 0.0 for s in steps]
    probs = [s.predicted_prob for s in steps]
    return float(np.mean([(p - o) ** 2 for p, o in zip(probs, outcomes)]))


def directional_accuracy(steps: list[BacktestStep]) -> float:
    """Fraction of steps where predicted direction matches realised direction."""
    if not steps:
        return 0.0
    correct = sum(1 for s in steps if s.direction_correct)
    return correct / len(steps)


def ci_coverage(steps: list[BacktestStep]) -> float:
    """Fraction of realised prices that fell inside the predicted CI."""
    if not steps:
        return 0.0
    covered = sum(1 for s in steps if s.in_ci)
    return covered / len(steps)


def mean_absolute_error(steps: list[BacktestStep]) -> float:
    """Mean absolute error of predicted returns vs realised returns."""
    if not steps:
        return 0.0
    return float(np.mean([abs(s.predicted_return - s.realised_return) for s in steps]))


def sharpe_of_returns(steps: list[BacktestStep]) -> float:
    """Sharpe ratio of realised returns (informational only)."""
    if not steps:
        return 0.0
    returns = [s.realised_return for s in steps]
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    return float(mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0


def bootstrap_directional_ci(
    steps: list[BacktestStep],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap confidence interval for directional accuracy.

    Returns lower and upper bounds of the bootstrap distribution.
    """
    if not steps:
        return {"lower": 0.0, "upper": 0.0, "mean": 0.0}

    accuracies = []
    n = len(steps)
    for _ in range(n_bootstrap):
        sample = np.random.choice(steps, size=n, replace=True)
        acc = directional_accuracy(sample.tolist())
        accuracies.append(acc)

    alpha = 1 - confidence
    lower = float(np.percentile(accuracies, 100 * alpha / 2))
    upper = float(np.percentile(accuracies, 100 * (1 - alpha / 2)))
    mean = float(np.mean(accuracies))
    return {"lower": lower, "upper": upper, "mean": mean}


def bootstrap_brier_ci(
    steps: list[BacktestStep],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap confidence interval for Brier score."""
    if not steps:
        return {"lower": 0.0, "upper": 0.0, "mean": 0.0}

    scores = []
    n = len(steps)
    for _ in range(n_bootstrap):
        sample = np.random.choice(steps, size=n, replace=True)
        scores.append(brier_score(sample.tolist()))

    alpha = 1 - confidence
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    mean = float(np.mean(scores))
    return {"lower": lower, "upper": upper, "mean": mean}


def calibration_table(steps: list[BacktestStep], bins: int = 5) -> list[dict]:
    """Reliability diagram: observed frequency vs predicted probability bins.

    Returns list of {bin_mid, predicted_mean, observed_freq, count}.
    """
    if not steps:
        return []

    bin_edges = np.linspace(0, 1, bins + 1)
    results = []
    for i in range(bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        subset = [s for s in steps if lo <= s.predicted_prob < hi]
        if subset:
            observed = float(np.mean([1.0 if s.realised_return > 0 else 0.0 for s in subset]))
            results.append(
                {
                    "bin_mid": (lo + hi) / 2,
                    "predicted_mean": float(np.mean([s.predicted_prob for s in subset])),
                    "observed_freq": observed,
                    "count": len(subset),
                }
            )
    return results
