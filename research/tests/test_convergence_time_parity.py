"""Parity tests for convergence-time mirror."""

from __future__ import annotations

import math

import pytest

from research.models.convergence_time import (
    ConvergenceResult,
    convergence_time,
    convergence_time_factor,
)


def test_no_convergence():
    r = convergence_time([0.5, 0.48, 0.52, 0.49, 0.51], 0.05)
    assert r.converged is False
    assert r.days_to_converge is None


def test_yes_convergence():
    r = convergence_time([0.5, 0.7, 0.85, 0.96, 0.97], 0.05)
    assert r.converged and r.direction == "yes" and r.days_to_converge == 3


def test_no_direction_convergence():
    r = convergence_time([0.5, 0.3, 0.1, 0.04, 0.02], 0.05)
    assert r.converged and r.direction == "no" and r.days_to_converge == 3


def test_empty():
    r = convergence_time([], 0.05)
    assert r.converged is False


def test_custom_epsilon():
    assert convergence_time([0.5, 0.85, 0.92], 0.10).converged is True
    assert convergence_time([0.5, 0.85, 0.92], 0.05).converged is False


def test_factor_unconverged():
    assert math.isclose(
        convergence_time_factor(ConvergenceResult(False, None, None)), 1.0
    )


def test_factor_fast_window_linear():
    day1 = convergence_time_factor(ConvergenceResult(True, 1, "yes"))
    day5 = convergence_time_factor(ConvergenceResult(True, 5, "yes"))
    day7 = convergence_time_factor(ConvergenceResult(True, 7, "yes"))
    assert math.isclose(day1, 1.15)
    assert day1 > day5 > day7 > 1.0


def test_factor_five_day_moderate_boost():
    f = convergence_time_factor(ConvergenceResult(True, 5, "yes"))
    assert 1.05 < f < 1.08


def test_factor_slow_damp():
    f = convergence_time_factor(ConvergenceResult(True, 35, "yes"))
    assert 0.85 < f < 0.95


def test_factor_intermediate():
    f = convergence_time_factor(ConvergenceResult(True, 14, "yes"))
    assert 0.95 < f < 1.10


def test_factor_monotone():
    a = convergence_time_factor(ConvergenceResult(True, 3, "yes"))
    b = convergence_time_factor(ConvergenceResult(True, 14, "yes"))
    c = convergence_time_factor(ConvergenceResult(True, 35, "yes"))
    assert a > b > c


def test_no_asymmetric_speedup():
    yes_f = convergence_time_factor(ConvergenceResult(True, 14, "yes"))
    no_f = convergence_time_factor(ConvergenceResult(True, 14, "no"))
    assert no_f > yes_f


@pytest.mark.parametrize("d", [1, 3, 7, 14, 21, 30, 60, 120])
@pytest.mark.parametrize("dir_", ["yes", "no"])
def test_factor_bounded(d, dir_):
    f = convergence_time_factor(ConvergenceResult(True, d, dir_))
    assert 0.85 <= f <= 1.20
