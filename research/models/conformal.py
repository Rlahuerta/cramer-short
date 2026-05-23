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
from typing import Optional

from research.utils.forecast_lab_runtime_defaults import (
    create_forecast_lab_asset_scoped_runtime_defaults,
    ForecastLabRuntimeAssetScope,
)

FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS: dict[str, float | int | bool] = {
    "pidLearningRate": 0.05,
    "integralDecay": 1.0,
    "adaptiveBreakEnabled": False,
    "adaptiveBreakLearningRateMultiplier": 1.5,
    "adaptiveBreakCooloffWindow": 0,
    "scoreAggregationMinSamples": 12,
    "scoreAggregationCalibrationWindow": 72,
}

PROMOTED_SOL_CONFORMAL_RUNTIME_DEFAULTS: dict[str, float | int | bool] = {
    "scoreAggregationMinSamples": 10,
    "scoreAggregationCalibrationWindow": 60,
}

_forecast_lab_conformal_runtime_defaults = create_forecast_lab_asset_scoped_runtime_defaults(
    FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS
)
_forecast_lab_conformal_runtime_defaults.set("sol", PROMOTED_SOL_CONFORMAL_RUNTIME_DEFAULTS)


def resolve_forecast_lab_conformal_parameter_defaults(
    asset_scope: ForecastLabRuntimeAssetScope | None = None,
) -> dict[str, float | int | bool]:
    return _forecast_lab_conformal_runtime_defaults.resolve(asset_scope)


def get_forecast_lab_conformal_runtime_defaults(
    asset_scope: ForecastLabRuntimeAssetScope,
) -> dict[str, float | int | bool] | None:
    return _forecast_lab_conformal_runtime_defaults.get(asset_scope)


def set_forecast_lab_conformal_runtime_defaults(
    asset_scope: ForecastLabRuntimeAssetScope,
    overrides: dict[str, float | int | bool] | None = None,
) -> None:
    _forecast_lab_conformal_runtime_defaults.set(asset_scope, overrides)


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
        learning_rate: float | None = None,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.1,
        integral_decay: float | None = None,
    ) -> None:
        defaults = resolve_forecast_lab_conformal_parameter_defaults()
        self.alpha = float(alpha)
        self.target_coverage = 1.0 - self.alpha
        self.lr = float(
            learning_rate
            if learning_rate is not None
            else defaults["pidLearningRate"]
        )
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.gamma = float(
            integral_decay
            if integral_decay is not None
            else defaults["integralDecay"]
        )
        self._initial_radius = max(0.0, float(initial_radius))
        self._q = self._initial_radius
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

    def reset(self, radius: Optional[float] = None) -> None:
        self._q = self._initial_radius if radius is None else max(0.0, float(radius))
        self._integral = 0.0
        self._prev_bias = 0.0
        self._samples = 0
        self._hits = 0

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


class AdaptiveConformalPID(ConformalPID):
    """Break-aware conformal PID wrapper matching the TypeScript behavior."""

    def __init__(
        self,
        alpha: float = 0.1,
        initial_radius: float = 1.0,
        learning_rate: float | None = None,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.1,
        integral_decay: float | None = None,
        enabled: bool | None = None,
        break_learning_rate_multiplier: float | None = None,
        cooloff_window: int | None = None,
    ) -> None:
        defaults = resolve_forecast_lab_conformal_parameter_defaults()
        super().__init__(
            alpha=alpha,
            initial_radius=initial_radius,
            learning_rate=learning_rate,
            kp=kp,
            ki=ki,
            kd=kd,
            integral_decay=integral_decay,
        )
        self._enabled = bool(
            defaults["adaptiveBreakEnabled"] if enabled is None else enabled
        )
        self._break_learning_rate_multiplier = max(
            1.0,
            float(
                defaults["adaptiveBreakLearningRateMultiplier"]
                if break_learning_rate_multiplier is None
                else break_learning_rate_multiplier
            ),
        )
        self._cooloff_window = max(
            0,
            int(
                round(
                    defaults["adaptiveBreakCooloffWindow"]
                    if cooloff_window is None
                    else cooloff_window
                )
            ),
        )
        self._cooloff_remaining = 0
        self._residual_ema: Optional[float] = None
        self._volatility_ema: Optional[float] = None
        self._mode = "normal"
        self._last_applied_radius = self._q

    def wrap(
        self,
        forecast_center: float,
        diagnostics: Optional[dict] = None,
    ) -> ConformalInterval:
        mode = self._resolve_mode(diagnostics)
        radius = self._applied_radius(mode)
        self._last_applied_radius = radius
        return ConformalInterval(
            low=forecast_center - radius,
            high=forecast_center + radius,
        )

    def record(
        self,
        forecast_center: float,
        actual: float,
        diagnostics: Optional[dict] = None,
    ) -> None:
        mode = self._resolve_mode(diagnostics, consume_cooloff=True)
        learning_rate = (
            self.lr * self._break_learning_rate_multiplier
            if mode == "break"
            else self.lr
        )
        stepped = self._step(forecast_center, actual, learning_rate)
        if stepped is None:
            return
        self._update_residual_ema(float(stepped["residual"]))
        self._update_volatility_ema(self._read_positive_float(diagnostics, "realized_vol"))

    def current_mode(self) -> str:
        return self._mode

    def diagnostics(self) -> dict:
        return {
            "applied": True,
            "radius": self._last_applied_radius,
            "coverage_estimate": self.empirical_coverage(),
            "mode": self._mode,
        }

    def reset(self, radius: Optional[float] = None) -> None:
        super().reset(radius=radius)
        self._cooloff_remaining = 0
        self._residual_ema = None
        self._volatility_ema = None
        self._mode = "normal"
        self._last_applied_radius = self._q

    def _resolve_mode(
        self,
        diagnostics: Optional[dict] = None,
        consume_cooloff: bool = False,
    ) -> str:
        if not self._enabled:
            if consume_cooloff:
                self._cooloff_remaining = 0
            self._mode = "normal"
            return self._mode

        triggered = (
            self._read_bool(diagnostics, "structural_break")
            or self._is_volatility_shock(
                self._read_positive_float(diagnostics, "realized_vol")
            )
        )
        in_cooloff = self._cooloff_remaining > 0
        mode = "break" if triggered or in_cooloff else "normal"

        if consume_cooloff:
            if triggered:
                self._cooloff_remaining = self._cooloff_window
            elif in_cooloff:
                self._cooloff_remaining -= 1

        self._mode = mode
        return self._mode

    def _applied_radius(self, mode: str) -> float:
        if mode != "break":
            return self._q
        return self._q * max(1.0, math.sqrt(self._break_learning_rate_multiplier))

    def _is_volatility_shock(self, realized_vol: Optional[float]) -> bool:
        if realized_vol is None or realized_vol <= 0:
            return False
        baseline = self._volatility_ema
        if baseline is None:
            baseline = self._residual_ema
        if baseline is None:
            return False
        return realized_vol >= baseline * self._break_learning_rate_multiplier

    def _update_residual_ema(self, residual: float) -> None:
        if not math.isfinite(residual) or residual < 0:
            return
        if self._residual_ema is None:
            self._residual_ema = residual
            return
        self._residual_ema = (self._residual_ema * 0.95) + (residual * 0.05)

    def _update_volatility_ema(self, realized_vol: Optional[float]) -> None:
        if realized_vol is None or not math.isfinite(realized_vol) or realized_vol <= 0:
            return
        if self._volatility_ema is None:
            self._volatility_ema = realized_vol
            return
        self._volatility_ema = (self._volatility_ema * 0.9) + (realized_vol * 0.1)

    @staticmethod
    def _read_bool(diagnostics: Optional[dict], key: str) -> bool:
        return bool(diagnostics and diagnostics.get(key) is True)

    @staticmethod
    def _read_positive_float(diagnostics: Optional[dict], key: str) -> Optional[float]:
        if not diagnostics:
            return None
        value = diagnostics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value)
        return None


class ScoreAggregatedConformal:
    """Score-aggregation conformal wrapper matching TS ``ScoreAggregatedConformal``.

    Maintains a rolling window of nonconformity scores computed as the max
    of (residual / radius) across source radii, then calibrates a score-based
    quantile multiplier. Used in walk-forward backtesting when
    ``enableConformalScoreAggregation`` is true.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        min_samples: int | None = None,
        calibration_window: int | None = None,
    ) -> None:
        defaults = resolve_forecast_lab_conformal_parameter_defaults()
        self._alpha = float(alpha)
        self._min_samples = max(
            1,
            int(
                round(
                    defaults["scoreAggregationMinSamples"]
                    if min_samples is None
                    else min_samples
                )
            ),
        )
        self._calibration_window = max(
            self._min_samples,
            int(
                round(
                    defaults["scoreAggregationCalibrationWindow"]
                    if calibration_window is None
                    else calibration_window
                )
            ),
        )
        self._scores: list[float] = []

    def wrap(
        self,
        forecast_center: float,
        source_radii: list[float],
    ) -> ConformalInterval:
        radii = _normalize_source_radii(source_radii)
        base_radius = min(radii) if radii else 0.0
        multiplier = self._score_multiplier()
        radius = base_radius * (multiplier if multiplier is not None else 1.0)
        return ConformalInterval(
            low=forecast_center - radius,
            high=forecast_center + radius,
        )

    def record(
        self,
        forecast_center: float,
        actual: float,
        source_radii: list[float],
    ) -> None:
        if not (math.isfinite(forecast_center) and math.isfinite(actual)):
            return
        radii = _normalize_source_radii(source_radii)
        if not radii:
            return

        residual = abs(actual - forecast_center)
        aggregated_score = max(
            (residual / radius for radius in radii),
            default=0.0,
        )
        if not math.isfinite(aggregated_score):
            return

        self._scores.append(aggregated_score)
        if len(self._scores) > self._calibration_window:
            self._scores = self._scores[-self._calibration_window:]

    def sample_count(self) -> int:
        return len(self._scores)

    def reset(self) -> None:
        self._scores.clear()

    def _score_multiplier(self) -> Optional[float]:
        if len(self._scores) < self._min_samples:
            return None
        return max(1.0, _upper_quantile(self._scores, 1.0 - self._alpha))


def _normalize_source_radii(source_radii: list[float]) -> list[float]:
    return [r for r in source_radii if math.isfinite(r) and r > 0]


def _upper_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    clamped = min(1.0, max(0.0, quantile))
    index = min(
        len(sorted_vals) - 1,
        max(0, int(math.ceil(len(sorted_vals) * clamped)) - 1),
    )
    return sorted_vals[index]
