"""Volume regime detection mirroring ``src/tools/finance/markov-distribution/volume-regime.ts``."""

from __future__ import annotations

from typing import Literal

import numpy as np

VolumeRegime = Literal["low", "normal", "high"]
VolumeEnvironment = Literal["low_volume", "normal_volume", "high_volume"]


def classify_volume_regime(
    volumes: np.ndarray | list[float],
    lookback: int = 20,
    threshold_multiplier: float = 1.5,
) -> list[VolumeRegime]:
    """Classify volume as high, normal, or low relative to the previous rolling median."""
    if threshold_multiplier <= 0:
        raise ValueError(f"threshold_multiplier must be positive, got {threshold_multiplier}")

    arr = np.asarray(volumes, dtype=float)
    n = len(arr)
    if n == 0:
        return []
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")

    result: list[VolumeRegime] = ["normal"] * n
    if n <= lookback:
        return result

    for i in range(lookback, n):
        window = arr[i - lookback : i]
        median = float(np.median(window))
        if median <= 0 or np.isnan(median):
            continue
        value = float(arr[i])
        if np.isnan(value):
            continue
        if value > threshold_multiplier * median:
            result[i] = "high"
        elif value < median / threshold_multiplier:
            result[i] = "low"

    return result


class VolumeRegimeDetector:
    """Map rolling-median volume labels to volume environment names."""

    def __init__(self, lookback: int = 20, threshold_multiplier: float = 1.5) -> None:
        self.lookback = lookback
        self.threshold_multiplier = threshold_multiplier
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, volumes: np.ndarray | list[float]) -> dict[str, list[VolumeEnvironment] | VolumeEnvironment]:
        labels = classify_volume_regime(volumes, lookback=self.lookback, threshold_multiplier=self.threshold_multiplier)
        env_map: dict[VolumeRegime, VolumeEnvironment] = {
            "high": "high_volume",
            "normal": "normal_volume",
            "low": "low_volume",
        }
        env_seq = [env_map[label] for label in labels]

        self._fitted = True
        return {
            "environment_sequence": env_seq,
            "current_environment": env_seq[-1] if env_seq else "normal_volume",
        }
