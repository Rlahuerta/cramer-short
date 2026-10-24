from __future__ import annotations

import math

import pytest

from research.models.markov.volume_regime import VolumeRegimeDetector, classify_volume_regime


def test_classify_volume_regime_returns_empty_list_for_empty_volumes() -> None:
    assert classify_volume_regime([]) == []


def test_classify_volume_regime_defaults_to_normal_without_enough_lookback_history() -> None:
    assert classify_volume_regime([100, 200, 50], lookback=3) == ["normal", "normal", "normal"]


def test_classify_volume_regime_uses_previous_rolling_median() -> None:
    assert classify_volume_regime([100, 100, 100, 200, 50, 100], lookback=3, threshold_multiplier=1.5) == [
        "normal",
        "normal",
        "normal",
        "high",
        "low",
        "normal",
    ]


def test_classify_volume_regime_treats_nan_current_values_as_normal() -> None:
    assert classify_volume_regime([100, 100, 100, math.nan, 200], lookback=3, threshold_multiplier=1.5) == [
        "normal",
        "normal",
        "normal",
        "normal",
        "normal",
    ]


def test_classify_volume_regime_keeps_non_positive_or_nan_median_points_normal() -> None:
    assert classify_volume_regime([0, 0, 0, 100, math.nan, 100, 100], lookback=3, threshold_multiplier=1.5) == [
        "normal",
        "normal",
        "normal",
        "normal",
        "normal",
        "normal",
        "normal",
    ]


def test_classify_volume_regime_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="lookback must be >= 2"):
        classify_volume_regime([100, 200], lookback=1)

    with pytest.raises(ValueError, match="threshold_multiplier must be positive"):
        classify_volume_regime([100, 200], threshold_multiplier=0)


def test_volume_regime_detector_maps_short_labels_to_environment_names() -> None:
    detector = VolumeRegimeDetector(lookback=3, threshold_multiplier=1.5)

    result = detector.fit([100, 100, 100, 200, 50])

    assert detector.fitted is True
    assert result == {
        "environment_sequence": [
            "normal_volume",
            "normal_volume",
            "normal_volume",
            "high_volume",
            "low_volume",
        ],
        "current_environment": "low_volume",
    }


def test_volume_regime_detector_returns_normal_volume_for_empty_input() -> None:
    detector = VolumeRegimeDetector()

    assert detector.fit([]) == {"environment_sequence": [], "current_environment": "normal_volume"}
