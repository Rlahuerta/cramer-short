"""Parity tests — Python vs TypeScript ConformalPID.

Pinned through the shared deterministic Box-Muller PRNG used in the TS test.
"""

from __future__ import annotations

import math

import pytest

from research.models.conformal import ConformalPID


# ------------------------------------------------------------------ #
# Deterministic Gaussian sampler — matches the TS test bit-for-bit.   #
# ------------------------------------------------------------------ #
def make_gaussian(seed: int):
    state = [seed & 0xFFFFFFFF]

    def rand() -> float:
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] / 0x100000000

    def gauss() -> float:
        u1 = max(rand(), 1e-12)
        u2 = rand()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    return gauss


# ------------------------------------------------------------------ #
class TestConstruction:
    def test_default_alpha_target_coverage(self):
        c = ConformalPID()
        assert c.alpha == pytest.approx(0.1)
        assert c.target_coverage == pytest.approx(0.9)

    def test_custom_alpha(self):
        c = ConformalPID(alpha=0.05)
        assert c.alpha == pytest.approx(0.05)
        assert c.target_coverage == pytest.approx(0.95)

    def test_initial_radius(self):
        c = ConformalPID(initial_radius=2.5)
        assert c.current_radius() == pytest.approx(2.5)

    def test_wrap_api(self):
        c = ConformalPID(initial_radius=1.0)
        iv = c.wrap(100.0)
        assert iv.low == pytest.approx(99.0)
        assert iv.high == pytest.approx(101.0)


class TestQuantileConvergence:
    def test_radius_converges_to_one_sided_90pct(self):
        gauss = make_gaussian(42)
        c = ConformalPID(alpha=0.1, initial_radius=0.5, learning_rate=0.05)
        for _ in range(5_000):
            c.record(0.0, gauss())
        # True 90% absolute-residual quantile of N(0,1) ≈ 1.645.
        assert 1.4 < c.current_radius() < 1.9

    def test_empirical_coverage_within_2pp_of_target(self):
        gauss = make_gaussian(123)
        c = ConformalPID(alpha=0.1, initial_radius=1.0, learning_rate=0.05)
        covered = 0
        total = 0
        for i in range(5_000):
            actual = gauss()
            if i >= 1_000:
                iv = c.wrap(0.0)
                if iv.low <= actual <= iv.high:
                    covered += 1
                total += 1
            c.record(0.0, actual)
        empirical = covered / total
        assert 0.88 < empirical < 0.92

    def test_adapts_to_volatility_regime_shift(self):
        lo = make_gaussian(7)
        hi = make_gaussian(8)
        c = ConformalPID(alpha=0.1, initial_radius=1.0, learning_rate=0.05)
        for _ in range(2_000):
            c.record(0.0, lo())
        radius_low = c.current_radius()
        for _ in range(2_000):
            c.record(0.0, 3.0 * hi())
        radius_high = c.current_radius()
        assert radius_high > radius_low * 1.5


class TestStability:
    def test_radius_never_negative(self):
        c = ConformalPID(alpha=0.1, initial_radius=0.01, learning_rate=0.5)
        for _ in range(1_000):
            c.record(0.0, 0.0)
        assert c.current_radius() >= 0.0

    def test_integral_decay_prevents_runaway(self):
        c = ConformalPID(
            alpha=0.1, initial_radius=1.0, learning_rate=0.05, integral_decay=0.99
        )
        for _ in range(200):
            c.record(0.0, 5.0)  # permanent miscoverage shock
        radius = c.current_radius()
        assert 0.5 < radius < 50


class TestCoverageHelpers:
    def test_running_coverage_reports_target_after_calibration(self):
        c = ConformalPID(alpha=0.1, initial_radius=1.0, learning_rate=0.05)
        for i in range(200):
            c.record(0.0, 0.5 if i % 2 == 0 else -0.5)
        cov = c.empirical_coverage()
        assert cov is not None
        assert 0.85 < cov < 1.0
        assert c.sample_count() == 200

    def test_no_data_returns_none(self):
        c = ConformalPID()
        assert c.sample_count() == 0
        assert c.empirical_coverage() is None
