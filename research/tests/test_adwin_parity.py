"""Parity tests for Python ADWIN mirror — mirrors TS adwin.test.ts."""

from __future__ import annotations

import math

import pytest

from research.models.adwin import Adwin


def make_rng(seed: int):
    state = [seed & 0xFFFFFFFF]

    def rng() -> float:
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] / 0x100000000

    return rng


def make_gauss(rng):
    cache = [None]

    def g(mean: float = 0.0, std: float = 1.0) -> float:
        if cache[0] is not None:
            v = cache[0]
            cache[0] = None
            return mean + std * v
        u = 0.0
        v = 0.0
        while u == 0:
            u = rng()
        while v == 0:
            v = rng()
        mag = math.sqrt(-2 * math.log(u))
        cache[0] = mag * math.sin(2 * math.pi * v)
        return mean + std * (mag * math.cos(2 * math.pi * v))

    return g


class TestBasicApi:
    def test_empty_init(self) -> None:
        a = Adwin()
        assert a.size() == 0
        assert math.isnan(a.mean())
        assert a.drift_detected() is False

    def test_stationary_no_drift(self) -> None:
        a = Adwin(delta=0.002)
        g = make_gauss(make_rng(1))
        for _ in range(200):
            a.add(g(0, 1))
        assert a.drift_detected() is False
        assert a.size() > 0
        assert abs(a.mean()) < 0.3

    def test_drift_drops_old(self) -> None:
        a = Adwin(delta=0.002)
        g = make_gauss(make_rng(2))
        for _ in range(100):
            a.add(g(0, 0.1))
        size_before = a.size()
        drifted = False
        for _ in range(100):
            if a.add(g(5, 0.1)):
                drifted = True
        assert drifted
        assert a.size() < size_before + 100


class TestDriftDetection:
    def test_sudden_shift_detected(self) -> None:
        a = Adwin(delta=0.002)
        g = make_gauss(make_rng(7))
        for _ in range(100):
            a.add(g(0, 0.1))
        assert a.drift_detected() is False
        steps = -1
        for i in range(50):
            a.add(g(2, 0.1))
            if a.drift_detected():
                steps = i + 1
                break
        assert steps > 0
        assert steps <= 15

    def test_no_false_positive_on_stationary(self) -> None:
        a = Adwin(delta=0.002)
        g = make_gauss(make_rng(123))
        false_positives = 0
        for _ in range(500):
            a.add(g(0, 1))
            if a.drift_detected():
                false_positives += 1
        assert false_positives < 20

    def test_no_drift_when_only_variance_unchanged(self) -> None:
        a = Adwin(delta=0.002)
        g = make_gauss(make_rng(11))
        for _ in range(80):
            a.add(g(0, 1))
        detected = False
        for _ in range(80):
            a.add(g(0, 1))
            if a.drift_detected():
                detected = True
                break
        assert detected is False


class TestMeanTracking:
    def test_mean_converges_post_drift(self) -> None:
        a = Adwin(delta=0.002)
        g = make_gauss(make_rng(33))
        for _ in range(100):
            a.add(g(0, 0.1))
        for _ in range(100):
            a.add(g(5, 0.1))
        assert 3 < a.mean() < 6

    def test_window_grows_during_stationarity(self) -> None:
        a = Adwin(delta=0.002)
        g = make_gauss(make_rng(44))
        for _ in range(50):
            a.add(g(1.0, 0.05))
        m1 = a.mean()
        for _ in range(200):
            a.add(g(1.0, 0.05))
        m2 = a.mean()
        assert abs(m1 - 1.0) < 0.1
        assert abs(m2 - 1.0) < 0.05
        assert a.size() > 50


class TestCompression:
    def test_sublinear_buckets(self) -> None:
        a = Adwin(delta=0.002, max_buckets=5)
        g = make_gauss(make_rng(55))
        for _ in range(10_000):
            a.add(g(0, 1))
        assert a.bucket_count() < 500
        assert a.size() > 1000
