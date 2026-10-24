"""Meta-regime detection for AH-HMM — mirrors ``src/tools/finance/markov-distribution/meta-regime.ts``."""

from __future__ import annotations

from typing import Literal, Sequence, TypedDict

import numpy as np

Environment = Literal["low_uncertainty", "high_uncertainty"]


class MetaRegimeFitResult(TypedDict):
    environment_sequence: list[Environment]
    current_environment: Environment | None
    threshold_used: float


class MetaRegimeDetector:
    """Classifies market environment as low_uncertainty or high_uncertainty."""

    def __init__(self, vol_window: int = 20, high_vol_threshold: float = 0.75) -> None:
        self.vol_window = vol_window
        self.high_vol_threshold = high_vol_threshold
        self._threshold: float | None = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def threshold(self) -> float | None:
        return self._threshold

    def fit(self, returns: Sequence[float]) -> MetaRegimeFitResult:
        arr = np.asarray(returns, dtype=float)
        n = len(arr)

        rolling_vol = np.full(n, np.nan)
        for i in range(self.vol_window - 1, n):
            window = arr[i - self.vol_window + 1 : i + 1]
            rolling_vol[i] = float(np.std(window, ddof=1))

        valid = rolling_vol[~np.isnan(rolling_vol)]
        if len(valid) == 0:
            self._threshold = 0.02
        else:
            self._threshold = float(np.percentile(valid, self.high_vol_threshold * 100))

        env_seq: list[Environment] = []
        for i in range(n):
            if np.isnan(rolling_vol[i]) or rolling_vol[i] <= self._threshold:
                env_seq.append("low_uncertainty")
            else:
                env_seq.append("high_uncertainty")

        self._fitted = True
        return {
            "environment_sequence": env_seq,
            "current_environment": env_seq[-1] if env_seq else None,
            "threshold_used": self._threshold,
        }
