"""Probability calibration with Student-t drift.

Mirrors TS logic:
  - Kappa-based calibration
  - Drift-based mode using Student-t survival
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from research.models.markov.core import DEFAULT_DIVERGENCE_PENALTY_SCHEDULE
from research.models.markov.gates import compute_divergence_penalty

if TYPE_CHECKING:
    from research.models.trajectory import student_t_ppf, student_t_survival


def _read_distribution_bound(
    point: dict,
    camel_key: str,
    snake_key: str,
    fallback: float,
) -> float:
    value = point.get(camel_key, point.get(snake_key, fallback))
    if value is None or not math.isfinite(float(value)):
        return fallback
    return float(value)


def _with_distribution_bounds(point: dict, lower: float, upper: float) -> dict:
    updated = {
        **point,
        "lowerBound": lower,
        "upperBound": upper,
    }
    if "lower_bound" in point:
        updated["lower_bound"] = lower
    if "upper_bound" in point:
        updated["upper_bound"] = upper
    return updated


def calibrate_probabilities(
    distribution: list[dict],
    *,
    ensemble_consensus: int = 0,
    historical_days: int = 60,
    hmm_converged: bool = False,
    base_rate: float = 0.5,
    kappa_multiplier: float = 1.0,
    current_regime: str | None = None,
    mature_bull_calibration_active: bool = False,
    current_price: float | None = None,
    drift_n: float | None = None,
    vol_n: float | None = None,
    nu: int = 5,
) -> list[dict]:
    """Calibrate probabilities with kappa blending and optional drift-based mode."""
    from research.models.trajectory import student_t_ppf, student_t_survival

    consensus = ensemble_consensus
    n_days = historical_days
    center = max(0.25, min(0.80, base_rate))

    kappa = 0.45
    kappa -= consensus * 0.07
    if n_days > 60:
        kappa -= min(0.08, 0.04 * math.log2(n_days / 60))
    if hmm_converged:
        kappa -= 0.03
    kappa *= kappa_multiplier

    if mature_bull_calibration_active:
        kappa += 0.10
    elif current_regime in {"bull", "bear"}:
        kappa -= 0.03
    elif current_regime == "sideways":
        kappa += 0.03

    kappa = max(0.15, min(0.55, kappa))

    if (
        current_price is not None
        and drift_n is not None
        and vol_n is not None
        and vol_n > 0
    ):
        raw_p_up = student_t_survival(current_price, current_price, drift_n, vol_n, nu)
        target_p_up = max(0.01, min(0.99, kappa * center + (1 - kappa) * raw_p_up))
        scaled_vol = vol_n * math.sqrt((nu - 2) / nu) if nu > 2 else vol_n
        z_target = student_t_ppf(1 - target_p_up, nu)
        calibrated_drift = -z_target * scaled_vol

        calibrated: list[dict] = []
        for point in distribution:
            probability = float(point["probability"])
            source = point.get("source", "markov")
            if source == "markov":
                new_prob = student_t_survival(
                    float(point["price"]),
                    current_price,
                    calibrated_drift,
                    vol_n,
                    nu,
                )
            else:
                old_markov = student_t_survival(
                    float(point["price"]),
                    current_price,
                    drift_n,
                    vol_n,
                    nu,
                )
                new_markov = student_t_survival(
                    float(point["price"]),
                    current_price,
                    calibrated_drift,
                    vol_n,
                    nu,
                )
                new_prob = max(0.0, min(1.0, probability + (new_markov - old_markov)))

            delta = new_prob - probability
            new_lower = max(
                0.0,
                min(
                    1.0,
                    _read_distribution_bound(point, "lowerBound", "lower_bound", new_prob)
                    + delta,
                ),
            )
            new_upper = max(
                0.0,
                min(
                    1.0,
                    _read_distribution_bound(point, "upperBound", "upper_bound", new_prob)
                    + delta,
                ),
            )
            calibrated.append(
                _with_distribution_bounds(
                    {
                        **point,
                        "probability": new_prob,
                    },
                    min(new_lower, new_prob),
                    max(new_upper, new_prob),
                )
            )

        for index in range(len(calibrated) - 2, -1, -1):
            if calibrated[index]["probability"] < calibrated[index + 1]["probability"]:
                calibrated[index]["probability"] = calibrated[index + 1]["probability"]
                if calibrated[index]["upperBound"] < calibrated[index]["probability"]:
                    calibrated[index]["upperBound"] = calibrated[index]["probability"]
                if calibrated[index]["lowerBound"] > calibrated[index]["probability"]:
                    calibrated[index]["lowerBound"] = calibrated[index]["probability"]
                if "upper_bound" in calibrated[index]:
                    calibrated[index]["upper_bound"] = calibrated[index]["upperBound"]
                if "lower_bound" in calibrated[index]:
                    calibrated[index]["lower_bound"] = calibrated[index]["lowerBound"]
        return calibrated

    calibrated = []
    for point in distribution:
        probability = float(point["probability"])
        new_prob = kappa * center + (1 - kappa) * probability
        delta = new_prob - probability
        new_lower = max(
            0.0,
            min(
                1.0,
                _read_distribution_bound(point, "lowerBound", "lower_bound", new_prob)
                + delta,
            ),
        )
        new_upper = max(
            0.0,
            min(
                1.0,
                _read_distribution_bound(point, "upperBound", "upper_bound", new_prob)
                + delta,
            ),
        )
        calibrated.append(
            _with_distribution_bounds(
                {
                    **point,
                    "probability": new_prob,
                },
                min(new_lower, new_prob),
                max(new_upper, new_prob),
            )
        )

    for index in range(len(calibrated) - 2, -1, -1):
        if calibrated[index]["probability"] < calibrated[index + 1]["probability"]:
            calibrated[index]["probability"] = calibrated[index + 1]["probability"]

    return calibrated
