"""Parity tests — Python regime_calibrator_two_pass mirrors TS two-pass (R5 Idea #6)."""

from __future__ import annotations

import math

import pytest

from research.utils.regime_calibrator import RegimeCalibrationSample
from research.utils.regime_calibrator_two_pass import (
    apply_two_pass_regime_platt,
    deserialize_two_pass_regime_platt,
    fit_two_pass_regime_platt,
    serialize_two_pass_regime_platt,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _make_samples(
    regime: str,
    n: int,
    seed: int = 42,
    shift: float = 0.0,
) -> list[RegimeCalibrationSample]:
    """Generate synthetic samples optionally shifted to simulate over-confidence."""
    rng = seed
    out = []
    for _ in range(n):
        rng = (1103515245 * rng + 12345) % (2**31)
        p_raw = 0.1 + 0.8 * (rng / 2**31)
        rng = (1103515245 * rng + 12345) % (2**31)
        true_p = min(max(p_raw + shift, 0.01), 0.99)
        outcome = 1 if (rng / 2**31) < true_p else 0
        out.append(
            RegimeCalibrationSample(regime=regime, p_raw=p_raw, outcome=outcome)  # type: ignore[arg-type]
        )
    return out


def _log_loss(preds: list[float], outcomes: list[int]) -> float:
    EPS = 1e-9
    return -sum(
        y * math.log(max(p, EPS)) + (1 - y) * math.log(max(1 - p, EPS))
        for p, y in zip(preds, outcomes)
    ) / len(preds)


class TestFitTwoPassRegimePlatt:
    def test_returns_empty_on_insufficient_samples(self):
        samples = _make_samples("bull", 10)
        fits = fit_two_pass_regime_platt(samples)
        assert "bull" not in fits

    def test_fits_regime_with_enough_samples(self):
        samples = _make_samples("bull", 80)
        fits = fit_two_pass_regime_platt(samples)
        assert "bull" in fits
        assert fits["bull"].pass1 is not None
        assert fits["bull"].pass2 is not None

    def test_deterministic(self):
        samples = _make_samples("bear", 60)
        f1 = fit_two_pass_regime_platt(samples)
        f2 = fit_two_pass_regime_platt(samples)
        assert f1["bear"].pass1.a == f2["bear"].pass1.a
        assert f1["bear"].pass2.a == f2["bear"].pass2.a

    def test_log_loss_non_regression_vs_single_pass(self):
        """Two-pass should not be worse than single-pass on over-confident data."""
        from research.utils.regime_calibrator import apply_regime_platt, fit_regime_platt

        samples = _make_samples("bull", 120, shift=0.2)
        test_s = _make_samples("bull", 60, seed=99, shift=0.2)

        sp_fits = fit_regime_platt(samples)
        tp_fits = fit_two_pass_regime_platt(samples)

        sp_preds = [apply_regime_platt(s.p_raw, "bull", sp_fits) for s in test_s]
        tp_preds = [apply_two_pass_regime_platt(s.p_raw, "bull", tp_fits) for s in test_s]
        outcomes = [s.outcome for s in test_s]

        ll_sp = _log_loss(sp_preds, outcomes)
        ll_tp = _log_loss(tp_preds, outcomes)
        # Two-pass should not increase log-loss by more than 3%
        assert ll_tp <= ll_sp * 1.03


class TestApplyTwoPassRegimePlatt:
    def test_passthrough_on_none_regime(self):
        p = apply_two_pass_regime_platt(0.6, None, {})
        assert p == pytest.approx(0.6, abs=1e-5)

    def test_passthrough_on_missing_fit(self):
        p = apply_two_pass_regime_platt(0.4, "bull", {})
        assert p == pytest.approx(0.4, abs=1e-5)

    def test_output_in_unit_interval(self):
        samples = _make_samples("bull", 80)
        fits = fit_two_pass_regime_platt(samples)
        for raw_p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            out = apply_two_pass_regime_platt(raw_p, "bull", fits)
            assert 0.0 < out < 1.0

    def test_composition_differs_from_single_pass(self):
        """Two-pass is NOT equivalent to a single Platt on over-confident data."""
        from research.utils.regime_calibrator import apply_regime_platt, fit_regime_platt

        samples = _make_samples("bull", 80, shift=0.3)
        sp_fits = fit_regime_platt(samples)
        tp_fits = fit_two_pass_regime_platt(samples)

        diffs = [
            abs(
                apply_regime_platt(0.9, "bull", sp_fits)
                - apply_two_pass_regime_platt(0.9, "bull", tp_fits)
            )
        ]
        # At least the two transforms should produce non-identical results
        # for extreme probabilities on this over-confident dataset.
        assert any(d > 0 for d in diffs)


class TestSerializeDeserializeTwoPass:
    def test_round_trip(self):
        samples = _make_samples("bull", 80) + _make_samples("bear", 80)
        fits = fit_two_pass_regime_platt(samples)
        raw = serialize_two_pass_regime_platt(fits)
        recovered = deserialize_two_pass_regime_platt(raw)
        for regime in fits:
            assert recovered[regime].pass1.a == pytest.approx(fits[regime].pass1.a, abs=1e-12)
            assert recovered[regime].pass2.b == pytest.approx(fits[regime].pass2.b, abs=1e-12)
            assert recovered[regime].pass2.n == fits[regime].pass2.n

    def test_malformed_json_returns_empty(self):
        assert deserialize_two_pass_regime_platt("bogus") == {}

    def test_partial_json_loads_valid_regimes_only(self):
        import json

        raw = json.dumps(
            {
                "bull": {
                    "pass1": {"a": 1.0, "b": 0.0, "n": 40},
                    "pass2": {"a": 0.9, "b": 0.1, "n": 40},
                },
                "bear": "garbage",
            }
        )
        recovered = deserialize_two_pass_regime_platt(raw)
        assert "bull" in recovered
        assert "bear" not in recovered
