"""Parity tests — Python regime_calibrator mirrors TS regime-calibrator.ts (R4)."""

from __future__ import annotations

import json
import math

import pytest

from research.utils.regime_calibrator import (
    PlattFit,
    RegimeCalibrationSample,
    apply_regime_platt,
    deserialize_regime_platt,
    fit_regime_platt,
    serialize_regime_platt,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _make_samples(
    regime: str,
    n: int,
    seed: int = 1,
    over_confident: bool = False,
) -> list[RegimeCalibrationSample]:
    """Generate synthetic (regime, p_raw, outcome) pairs."""
    rng_state = seed
    samples = []
    for _ in range(n):
        rng_state = (1103515245 * rng_state + 12345) % (2**31)
        p = 0.1 + 0.8 * (rng_state / 2**31)
        rng_state = (1103515245 * rng_state + 12345) % (2**31)
        true_p = p if not over_confident else (0.9 if p > 0.5 else 0.1)
        outcome = 1 if (rng_state / 2**31) < true_p else 0
        samples.append(
            RegimeCalibrationSample(regime=regime, p_raw=p, outcome=outcome)  # type: ignore[arg-type]
        )
    return samples


class TestFitRegimePlatt:
    def test_skips_regimes_with_too_few_samples(self):
        samples = _make_samples("bull", 10)
        fits = fit_regime_platt(samples)
        assert "bull" not in fits

    def test_fits_regime_with_enough_samples(self):
        samples = _make_samples("bull", 50)
        fits = fit_regime_platt(samples)
        assert "bull" in fits

    def test_deterministic(self):
        samples = _make_samples("bear", 50)
        f1 = fit_regime_platt(samples)
        f2 = fit_regime_platt(samples)
        assert f1["bear"].a == f2["bear"].a
        assert f1["bear"].b == f2["bear"].b

    def test_all_same_outcome_is_skipped(self):
        samples = [
            RegimeCalibrationSample(regime="bull", p_raw=0.7, outcome=1)  # type: ignore[arg-type]
            for _ in range(50)
        ]
        fits = fit_regime_platt(samples)
        assert "bull" not in fits

    def test_sample_count_recorded(self):
        n = 60
        samples = _make_samples("sideways", n)
        fits = fit_regime_platt(samples)
        assert fits["sideways"].n == n


class TestApplyRegimePlatt:
    def test_passthrough_when_no_fit(self):
        p = apply_regime_platt(0.6, "bull", {})
        assert p == pytest.approx(0.6, abs=1e-5)

    def test_passthrough_when_regime_none(self):
        p = apply_regime_platt(0.3, None, {})
        assert p == pytest.approx(0.3, abs=1e-5)

    def test_identity_fit(self):
        """a=1, b=0 should leave probability unchanged."""
        fit = PlattFit(a=1.0, b=0.0, n=50)
        p_in = 0.7
        p_out = apply_regime_platt(p_in, "bull", {"bull": fit})
        assert p_out == pytest.approx(p_in, abs=1e-5)

    def test_shrinkage_fit(self):
        """a=0.5, b=0 should shrink toward 0.5."""
        fit = PlattFit(a=0.5, b=0.0, n=50)
        p_out = apply_regime_platt(0.9, "bear", {"bear": fit})
        assert p_out < 0.9

    def test_output_in_unit_interval(self):
        fit = PlattFit(a=2.0, b=1.0, n=50)
        p = apply_regime_platt(0.99, "bull", {"bull": fit})
        assert 0.0 < p < 1.0


class TestSerializeDeserialize:
    def test_round_trip(self):
        samples = _make_samples("bull", 50) + _make_samples("bear", 50)
        fits = fit_regime_platt(samples)
        json_str = serialize_regime_platt(fits)
        recovered = deserialize_regime_platt(json_str)
        for regime in fits:
            assert recovered[regime].a == pytest.approx(fits[regime].a, abs=1e-12)
            assert recovered[regime].b == pytest.approx(fits[regime].b, abs=1e-12)
            assert recovered[regime].n == fits[regime].n

    def test_malformed_json_returns_empty(self):
        assert deserialize_regime_platt("not json") == {}

    def test_empty_json_object_returns_empty(self):
        assert deserialize_regime_platt("{}") == {}

    def test_partial_json_only_loads_valid_regimes(self):
        raw = json.dumps({"bull": {"a": 1.0, "b": 0.5, "n": 40}, "bear": "bad"})
        recovered = deserialize_regime_platt(raw)
        assert "bull" in recovered
        assert "bear" not in recovered
