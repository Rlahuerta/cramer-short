"""Online Conformal PID wrapper (W3 Idea 2).

Mirrors ``src/tools/finance/conformal.ts``. Pure Python, no I/O.

Reference:
    Angelopoulos, Candès & Tibshirani (2023)
    "Conformal PID Control for Time Series Prediction"
    arXiv:2307.16895

The wrapper sits *outside* any forecasting model. Given a stream of
``(forecast_center, actual)`` pairs, it adapts a single radius ``q`` such that
the long-run miscoverage of the symmetric interval ``[center − q, center + q]``
approaches the target ``α``.  No assumptions on the underlying model — finite
sample coverage holds under arbitrary distribution shift.

The PID update is::

    bias_t  = err_t − α
    I_t     = γ · I_{t−1} + bias_t
    D_t     = bias_t − bias_{t−1}
    q_{t+1} = max(0, q_t + lr · (Kp·b + Ki·I + Kd·D))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ConformalInterval:
    low: float
    high: float


class ConformalPID:
    """Online conformal prediction with a PID quantile controller."""

    def __init__(
        self,
        alpha: float = 0.1,
        initial_radius: float = 1.0,
        learning_rate: float = 0.05,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.1,
        integral_decay: float = 1.0,
    ) -> None:
        self.alpha = float(alpha)
        self.target_coverage = 1.0 - self.alpha
        self.lr = float(learning_rate)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.gamma = float(integral_decay)
        self._q = max(0.0, float(initial_radius))
        self._integral = 0.0
        self._prev_bias = 0.0
        self._samples = 0
        self._hits = 0

    # ------------------------------------------------------------------ #
    def _step(
        self, forecast_center: float, actual: float, learning_rate: float,
    ) -> dict | None:
        if not (math.isfinite(forecast_center) and math.isfinite(actual)):
            return None
        residual = abs(actual - forecast_center)
        covered = 1 if residual <= self._q else 0
        err = 1 - covered
        bias = err - self.alpha

        self._integral = self.gamma * self._integral + bias
        derivative = bias - self._prev_bias
        self._prev_bias = bias

        update = learning_rate * (
            self.kp * bias + self.ki * self._integral + self.kd * derivative
        )
        self._q = max(0.0, self._q + update)

        self._samples += 1
        self._hits += covered
        return {"residual": residual, "covered": covered == 1}

    def record(self, forecast_center: float, actual: float) -> None:
        self._step(forecast_center, actual, self.lr)

    # ------------------------------------------------------------------ #
    def current_radius(self) -> float:
        return self._q

    def wrap(self, forecast_center: float) -> ConformalInterval:
        return ConformalInterval(
            low=forecast_center - self._q, high=forecast_center + self._q
        )

    def sample_count(self) -> int:
        return self._samples

    def empirical_coverage(self) -> Optional[float]:
        if self._samples == 0:
            return None
        return self._hits / self._samples
