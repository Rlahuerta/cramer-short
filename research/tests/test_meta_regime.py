"""Tests for meta-regime detection."""

from __future__ import annotations


from research.models.meta_regime import MetaRegimeDetector


def test_classifies_rolling_volatility_at_threshold_as_low_uncertainty():
    detector = MetaRegimeDetector(vol_window=2, high_vol_threshold=0)

    result = detector.fit([0, 0, 0.1])

    assert detector.fitted is True
    assert result["threshold_used"] == 0
    assert result["environment_sequence"][1] == "low_uncertainty"


def test_classifies_rolling_volatility_above_threshold_as_high_uncertainty():
    detector = MetaRegimeDetector(vol_window=2, high_vol_threshold=0)

    result = detector.fit([0, 0, 0.1])

    assert result["environment_sequence"][2] == "high_uncertainty"
    assert result["current_environment"] == "high_uncertainty"


def test_uses_fallback_threshold_when_data_is_insufficient_for_rolling_volatility():
    detector = MetaRegimeDetector(vol_window=20, high_vol_threshold=0.75)

    result = detector.fit([0.03])

    assert detector.threshold == 0.02
    assert result["threshold_used"] == 0.02
    assert result["environment_sequence"] == ["low_uncertainty"]
    assert result["current_environment"] == "low_uncertainty"


def test_uses_fallback_threshold_for_empty_data():
    detector = MetaRegimeDetector(vol_window=20, high_vol_threshold=0.75)

    result = detector.fit([])

    assert detector.threshold == 0.02
    assert result["threshold_used"] == 0.02
    assert result["environment_sequence"] == []
    assert result["current_environment"] is None
