"""Scenario probability bucketing and expected return computation.

Mirrors TS logic from src/tools/finance/markov-distribution.ts:
  - Down >5%, Down 3-5%, Flat +/-3%, Up 3-5%, Up >5% buckets
  - Trapezoidal expected price via survival integral
  - TS alias keys for backward compatibility
"""

from __future__ import annotations

from research.models.trajectory.interpolation import interpolate_survival


def compute_scenario_probabilities(
    distribution: list[dict],
    current_price: float,
) -> dict:
    down5 = current_price * 0.95
    down3 = current_price * 0.97
    up3 = current_price * 1.03
    up5 = current_price * 1.05

    p_above_down5 = interpolate_survival(distribution, down5)
    p_above_down3 = interpolate_survival(distribution, down3)
    p_above_up3 = interpolate_survival(distribution, up3)
    p_above_up5 = interpolate_survival(distribution, up5)

    p_down_over5 = 1.0 - p_above_down5
    p_down_3to5 = p_above_down5 - p_above_down3
    p_flat = p_above_down3 - p_above_up3
    p_up_3to5 = p_above_up3 - p_above_up5
    p_up_over5 = p_above_up5

    expected_price = current_price
    if len(distribution) >= 2:
        integral = 0.0
        for i in range(len(distribution) - 1):
            dx = distribution[i + 1]["price"] - distribution[i]["price"]
            avg_p = (distribution[i]["probability"] + distribution[i + 1]["probability"]) / 2
            integral += avg_p * dx
        expected_price = distribution[0]["price"] + integral

    expected_return = (expected_price - current_price) / current_price

    def _bucket(label: str, probability: float, lo: float | None, hi: float | None) -> dict:
        price_range = [round(lo, 2) if lo is not None else None, round(hi, 2) if hi is not None else None]
        return {
            "label": label,
            "probability": max(0.0, probability),
            "range": price_range,
            "priceRange": price_range,
        }

    result = {
        "buckets": [
            _bucket("Down >5%", p_down_over5, None, down5),
            _bucket("Down 3–5%", p_down_3to5, down5, down3),
            _bucket("Flat ±3%", p_flat, down3, up3),
            _bucket("Up 3–5%", p_up_3to5, up3, up5),
            _bucket("Up >5%", p_up_over5, up5, None),
        ],
        "expected_price": round(expected_price, 2),
        "expected_return": round(expected_return, 4),
        "p_up": round(interpolate_survival(distribution, current_price), 3),
    }
    result["expectedPrice"] = result["expected_price"]
    result["expectedReturn"] = result["expected_return"]
    result["pUp"] = result["p_up"]
    return result
