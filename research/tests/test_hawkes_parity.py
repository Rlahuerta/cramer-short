"""Parity tests for Python Hawkes mirror — mirrors TS hawkes.test.ts."""

from __future__ import annotations

import math

import pytest

from research.models.hawkes import (
    HawkesIntensity,
    HawkesParams,
    fit_hawkes_mle,
    simulate_hawkes,
)


def make_rng(seed: int):
    state = [seed & 0xFFFFFFFF]

    def rng() -> float:
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] / 0x100000000

    return rng


class TestIntensity:
    def test_baseline_only(self) -> None:
        h = HawkesIntensity(mu=0.5, alpha=0.4, beta=1.0)
        assert h.intensity(10, []) == pytest.approx(0.5, abs=1e-10)

    def test_single_jump(self) -> None:
        h = HawkesIntensity(mu=0.5, alpha=0.4, beta=1.0)
        assert h.intensity(0, [0]) == pytest.approx(0.9, abs=1e-10)

    def test_decay(self) -> None:
        h = HawkesIntensity(mu=0.5, alpha=0.4, beta=1.0)
        assert h.intensity(1, [0]) == pytest.approx(0.5 + 0.4 * math.exp(-1), abs=1e-10)
        assert h.intensity(1000, [0]) == pytest.approx(0.5, abs=1e-8)

    def test_additive(self) -> None:
        h = HawkesIntensity(mu=0.0, alpha=1.0, beta=1.0)
        assert h.intensity(2, [0, 1]) == pytest.approx(math.exp(-2) + math.exp(-1), abs=1e-10)

    def test_ignores_future(self) -> None:
        h = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)
        assert h.intensity(1, [0.5, 2.0]) == pytest.approx(
            0.1 + 0.5 * math.exp(-0.5), abs=1e-10
        )


class TestStability:
    def test_branching_ratio(self) -> None:
        assert HawkesIntensity(1, 0.5, 1).branching_ratio() == pytest.approx(0.5)
        assert HawkesIntensity(1, 0.9, 1).branching_ratio() == pytest.approx(0.9)

    def test_is_stable(self) -> None:
        assert HawkesIntensity(1, 0.5, 1).is_stable() is True
        assert HawkesIntensity(1, 1.0, 1).is_stable() is False
        assert HawkesIntensity(1, 1.5, 1).is_stable() is False

    def test_constructor_rejects_invalid(self) -> None:
        with pytest.raises(ValueError):
            HawkesIntensity(float("nan"), 0.1, 1)
        with pytest.raises(ValueError):
            HawkesIntensity(1, -0.1, 1)
        with pytest.raises(ValueError):
            HawkesIntensity(1, 0.1, 0)


class TestLogLikelihood:
    def test_empty(self) -> None:
        h = HawkesIntensity(mu=0.5, alpha=0.3, beta=1.0)
        assert h.log_likelihood([], 10) == pytest.approx(-5.0, abs=1e-10)

    def test_single_event(self) -> None:
        h = HawkesIntensity(mu=1.0, alpha=0.0, beta=1.0)
        assert h.log_likelihood([0], 5) == pytest.approx(-5.0, abs=1e-8)

    def test_compensator(self) -> None:
        h = HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)
        ll = h.log_likelihood([0, 1], 2)
        expected = (
            math.log(0.1)
            + math.log(0.1 + 0.5 * math.exp(-1))
            - (0.2 + 0.5 * (1 - math.exp(-2)) + 0.5 * (1 - math.exp(-1)))
        )
        assert ll == pytest.approx(expected, abs=1e-8)


class TestSimulation:
    def test_poisson_recovery(self) -> None:
        rng = make_rng(42)
        events = simulate_hawkes(HawkesParams(mu=5, alpha=0, beta=1), 100, rng)
        assert 425 < len(events) < 575
        gaps = [events[i + 1] - events[i] for i in range(len(events) - 1)]
        mean_gap = sum(gaps) / len(gaps)
        assert 0.16 < mean_gap < 0.24

    def test_clustering(self) -> None:
        rng_h = make_rng(7)
        rng_p = make_rng(7)
        T = 200
        baseline = simulate_hawkes(HawkesParams(mu=1, alpha=0, beta=1), T, rng_p)
        clustered = simulate_hawkes(HawkesParams(mu=1, alpha=0.7, beta=1), T, rng_h)
        assert len(clustered) > len(baseline) * 1.8
        short_frac = lambda ev: (
            sum(1 for g in (ev[i + 1] - ev[i] for i in range(len(ev) - 1)) if g < 0.1)
            / max(1, len(ev) - 1)
        )
        assert short_frac(clustered) > short_frac(baseline)

    def test_strictly_increasing(self) -> None:
        events = simulate_hawkes(HawkesParams(mu=2, alpha=0.3, beta=1.5), 50, make_rng(123))
        for i in range(1, len(events)):
            assert events[i] > events[i - 1]
        if events:
            assert events[-1] <= 50
            assert events[0] > 0


class TestFit:
    def test_baseline_only(self) -> None:
        events = simulate_hawkes(HawkesParams(mu=2, alpha=0, beta=1), 500, make_rng(2024))
        fit = fit_hawkes_mle(events, 500, initial_mu=1, initial_alpha=0.01, initial_beta=1)
        assert abs(fit.mu - 2) < 0.5
        assert fit.alpha < 0.5
        assert fit.is_stable

    def test_stability_constraint(self) -> None:
        events = simulate_hawkes(HawkesParams(mu=1, alpha=0.5, beta=1), 1000, make_rng(99))
        fit = fit_hawkes_mle(events, 1000)
        assert fit.alpha / fit.beta < 1.0
        assert fit.is_stable

    def test_finite_params(self) -> None:
        events = simulate_hawkes(HawkesParams(mu=1, alpha=0.4, beta=1), 200, make_rng(5))
        fit = fit_hawkes_mle(events, 200)
        assert math.isfinite(fit.mu)
        assert math.isfinite(fit.alpha)
        assert math.isfinite(fit.beta)
        assert math.isfinite(fit.log_likelihood)
