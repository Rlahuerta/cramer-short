"""Backtest metrics: Brier score, directional accuracy, CI coverage, bootstrap CIs."""

from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict

import math
import numpy as np

if TYPE_CHECKING:
    from research.backtest.walk_forward import BacktestStep

CENTRAL_90_Z = 1.6448536269514722
SQRT_PI = math.sqrt(math.pi)
MIN_SCALE = 1e-9


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _derive_forecast_center(step: BacktestStep) -> float:
    denominator = 1 + step.realised_return
    if math.isfinite(denominator) and abs(denominator) > MIN_SCALE:
        current_price = step.realised_price / denominator
    else:
        current_price = (step.ci_lower + step.ci_upper) / 2
    forecast_center = current_price * (1 + step.predicted_return)
    if math.isfinite(forecast_center):
        return forecast_center
    return (step.ci_lower + step.ci_upper) / 2


def _derive_interval_scale(step: BacktestStep) -> float:
    width = abs(step.ci_upper - step.ci_lower)
    sigma = width / (2 * CENTRAL_90_Z)
    if math.isfinite(sigma) and sigma > MIN_SCALE:
        return sigma
    return MIN_SCALE


def _normal_crps(observed: float, mean: float, sigma: float) -> float:
    z = (observed - mean) / sigma
    return sigma * (z * (2 * _normal_cdf(z) - 1) + 2 * _normal_pdf(z) - 1 / SQRT_PI)


def _step_crps(step: BacktestStep) -> float:
    score = _normal_crps(step.realised_price, _derive_forecast_center(step), _derive_interval_scale(step))
    if math.isfinite(score):
        return max(0.0, score)
    return 0.0


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


# ---------------------------------------------------------------------------
# CRPS / scaled CRPS / Murphy-Winkler interval score
# ---------------------------------------------------------------------------


class MurphyWinklerDecomposition(TypedDict):
    mean_width: float
    lower_miss_penalty: float
    upper_miss_penalty: float
    total_score: float
    coverage: float


def crps(steps: list[BacktestStep]) -> float:
    """Continuous Ranked Probability Score using the backtest interval as a
    central-normal approximation. Lower is better."""
    if not steps:
        return 0.0
    return sum(_step_crps(s) for s in steps) / len(steps)


def scaled_crps(steps: list[BacktestStep]) -> float:
    """Locally scaled CRPS: step CRPS divided by interval-implied scale.
    Lower is better."""
    if not steps:
        return 0.0
    return sum(_step_crps(s) / _derive_interval_scale(s) for s in steps) / len(steps)


def murphy_winkler_decomposition(
    steps: list[BacktestStep],
    alpha: float = 0.10,
) -> MurphyWinklerDecomposition:
    """Murphy-Winkler interval score decomposition for the primary step interval."""
    if not steps:
        return MurphyWinklerDecomposition(
            mean_width=0.0,
            lower_miss_penalty=0.0,
            upper_miss_penalty=0.0,
            total_score=0.0,
            coverage=0.0,
        )

    penalty_scale = 2.0 / alpha
    width_sum = 0.0
    lower_penalty_sum = 0.0
    upper_penalty_sum = 0.0
    covered = 0

    for step in steps:
        lower = min(step.ci_lower, step.ci_upper)
        upper = max(step.ci_lower, step.ci_upper)
        width_sum += upper - lower
        if step.realised_price < lower:
            lower_penalty_sum += penalty_scale * (lower - step.realised_price)
        elif step.realised_price > upper:
            upper_penalty_sum += penalty_scale * (step.realised_price - upper)
        else:
            covered += 1

    n = len(steps)
    return MurphyWinklerDecomposition(
        mean_width=width_sum / n,
        lower_miss_penalty=lower_penalty_sum / n,
        upper_miss_penalty=upper_penalty_sum / n,
        total_score=(width_sum + lower_penalty_sum + upper_penalty_sum) / n,
        coverage=covered / n,
    )


def murphy_winkler_score(steps: list[BacktestStep], alpha: float = 0.10) -> float:
    """Total Murphy-Winkler interval score."""
    return murphy_winkler_decomposition(steps, alpha)["total_score"]
