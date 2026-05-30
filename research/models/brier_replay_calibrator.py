"""Replay-only Brier calibrator for sequential probability forecasts.

Maintains a running Brier score from a historical sequence of forecasts
and outcomes.  Applies Platt-style recalibration: fits a logistic mapping
from raw model probabilities to outcome frequencies so that forecasts
become better calibrated over time.

This is a replay/lab module — it feeds on already-realised outcomes
and is not suitable for real-time usage.
"""

from __future__ import annotations

import math


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _logit(probability: float) -> float:
    p = _clamp(probability, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


class BrierReplayCalibrator:
    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        mid_confidence_weight: float = 2.0,
        mid_confidence_min: float = 0.4,
        mid_confidence_max: float = 0.6,
        max_slope: float = 3.0,
        max_bias: float = 1.5,
    ) -> None:
        self.learning_rate = learning_rate
        self.mid_confidence_weight = mid_confidence_weight
        self.mid_confidence_min = mid_confidence_min
        self.mid_confidence_max = mid_confidence_max
        self.max_slope = max(1.0, max_slope)
        self.max_bias = max(0.25, max_bias)
        self.bias = 0.0
        self.slope = 1.0

    def predict(self, raw_probability: float) -> float:
        return _sigmoid(self.slope * _logit(raw_probability) + self.bias)

    def record(self, raw_probability: float, actual_binary: int) -> dict[str, float]:
        prediction = self.predict(raw_probability)
        z = _logit(raw_probability)
        mid_weight = (
            self.mid_confidence_weight
            if self.mid_confidence_min <= raw_probability <= self.mid_confidence_max
            else 1.0
        )
        gradient = 2.0 * (prediction - actual_binary) * prediction * (1.0 - prediction) * mid_weight
        self.bias = _clamp(
            self.bias - self.learning_rate * gradient,
            -self.max_bias,
            self.max_bias,
        )
        self.slope = _clamp(
            self.slope - self.learning_rate * gradient * z,
            1.0 / self.max_slope,
            self.max_slope,
        )
        return self.state()

    def state(self) -> dict[str, float]:
        return {"bias": self.bias, "slope": self.slope}
