"""Parity tests — Python transition_entropy mirrors TS transition-entropy.ts (R5 Idea #14)."""

from __future__ import annotations

import math

import pytest

from research.models.transition_entropy import (
    EntropyZScoreTracker,
    TransitionEntropyResult,
    approximate_stationary,
    compute_transition_entropy,
    entropy_z_to_ci_scale,
)


class TestApproximateStationary:
    def test_empty_returns_empty(self):
        assert approximate_stationary([]) == []

    def test_uniform_matrix_gives_uniform_stationary(self):
        P = [[0.5, 0.5], [0.5, 0.5]]
        pi = approximate_stationary(P)
        assert pi == pytest.approx([0.5, 0.5], abs=1e-6)

    def test_absorbing_state(self):
        """State 0 always stays in state 0."""
        P = [[1.0, 0.0], [0.5, 0.5]]
        pi = approximate_stationary(P)
        # State 0 is absorbing -> pi[0] == 1.0 in the limit
        assert pi[0] > pi[1]

    def test_normalised(self):
        P = [[0.7, 0.3], [0.4, 0.6]]
        pi = approximate_stationary(P)
        assert sum(pi) == pytest.approx(1.0, abs=1e-6)


class TestComputeTransitionEntropy:
    def test_empty_returns_zero(self):
        r = compute_transition_entropy([])
        assert r == TransitionEntropyResult(entropy_nats=0.0, entropy_norm=0.0, K=0)

    def test_deterministic_matrix_has_zero_entropy(self):
        """Each row is a point mass -> H_row = 0."""
        P = [[1.0, 0.0], [0.0, 1.0]]
        r = compute_transition_entropy(P)
        assert r.entropy_nats == pytest.approx(0.0, abs=1e-9)
        assert r.entropy_norm == pytest.approx(0.0, abs=1e-9)

    def test_uniform_matrix_has_max_entropy(self):
        P = [[0.5, 0.5], [0.5, 0.5]]
        r = compute_transition_entropy(P)
        assert r.entropy_norm == pytest.approx(1.0, abs=1e-6)

    def test_k_equals_number_of_states(self):
        P = [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
        r = compute_transition_entropy(P)
        assert r.K == 3

    def test_norm_in_unit_interval(self):
        for seed in range(5):
            import random
            rng = random.Random(seed)
            K = 4
            P = []
            for _ in range(K):
                row = [rng.random() for _ in range(K)]
                s = sum(row)
                P.append([x / s for x in row])
            r = compute_transition_entropy(P)
            assert 0.0 <= r.entropy_norm <= 1.0 + 1e-9

    def test_entropy_nats_positive_for_nondeterministic(self):
        P = [[0.7, 0.3], [0.4, 0.6]]
        r = compute_transition_entropy(P)
        assert r.entropy_nats > 0


class TestEntropyZScoreTracker:
    def test_raises_on_small_window(self):
        with pytest.raises(ValueError):
            EntropyZScoreTracker(window_size=4)

    def test_returns_none_before_5_pushes(self):
        t = EntropyZScoreTracker(10)
        for i in range(4):
            t.push(float(i) * 0.1)
            assert t.z_score(0.2) is None

    def test_returns_float_after_5_pushes(self):
        t = EntropyZScoreTracker(10)
        for i in range(5):
            t.push(float(i) * 0.1)
        assert isinstance(t.z_score(0.2), float)

    def test_zero_std_returns_zero(self):
        t = EntropyZScoreTracker(10)
        for _ in range(10):
            t.push(0.5)
        assert t.z_score(0.5) == pytest.approx(0.0)

    def test_window_bounded(self):
        t = EntropyZScoreTracker(5)
        for i in range(20):
            t.push(float(i))
        assert t.size() == 5

    def test_z_score_positive_above_mean(self):
        t = EntropyZScoreTracker(10)
        for _ in range(10):
            t.push(0.5)
        # Push many high values to set mean ~0.5, then query with higher value
        t2 = EntropyZScoreTracker(10)
        for v in [0.3, 0.4, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.3, 0.4]:
            t2.push(v)
        z = t2.z_score(0.9)
        assert z is not None and z > 0


class TestEntropyZToCiScale:
    def test_zero_z_returns_1(self):
        assert entropy_z_to_ci_scale(0.0) == pytest.approx(1.0)

    def test_positive_z_widens_ci(self):
        assert entropy_z_to_ci_scale(2.0) > 1.0

    def test_negative_z_tightens_ci(self):
        assert entropy_z_to_ci_scale(-2.0) < 1.0

    def test_clamped_at_lower_bound(self):
        assert entropy_z_to_ci_scale(-100.0) == pytest.approx(0.7)

    def test_clamped_at_upper_bound(self):
        assert entropy_z_to_ci_scale(100.0) == pytest.approx(1.4)

    def test_custom_kappa(self):
        # z=2, kappa=0.1 -> raw = 1.2
        assert entropy_z_to_ci_scale(2.0, kappa=0.10) == pytest.approx(1.2)

    def test_custom_bounds(self):
        assert entropy_z_to_ci_scale(-100.0, bounds=(0.5, 2.0)) == pytest.approx(0.5)
        assert entropy_z_to_ci_scale(100.0, bounds=(0.5, 2.0)) == pytest.approx(2.0)
