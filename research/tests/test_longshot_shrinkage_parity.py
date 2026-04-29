"""Parity tests — Python longshot_shrinkage mirrors TS rnd-integration.ts (R5 Idea #11)."""

from __future__ import annotations

import pytest

from research.models.longshot_shrinkage import apply_longshot_shrinkage


class TestApplyLongshotShrinkage:
    def test_no_shrinkage_in_central_region(self):
        res = apply_longshot_shrinkage(0.4)
        assert res.p == pytest.approx(0.4)
        assert res.applied is False

    def test_no_shrinkage_at_exact_boundary(self):
        # p=0.05 and p=0.95 are NOT in the tail (condition is strictly p>lo and p<hi)
        res_lo = apply_longshot_shrinkage(0.05)
        assert res_lo.applied is True  # 0.05 is NOT > 0.05

        res_hi = apply_longshot_shrinkage(0.95)
        assert res_hi.applied is True  # 0.95 is NOT < 0.95

    def test_shrinkage_applied_below_threshold(self):
        res = apply_longshot_shrinkage(0.02)
        assert res.applied is True
        # With weight=0.5: shrunk = 0.5*0.5 + 0.5*0.02 = 0.26
        assert res.p == pytest.approx(0.26)

    def test_shrinkage_applied_above_threshold(self):
        res = apply_longshot_shrinkage(0.98)
        assert res.applied is True
        # With weight=0.5: shrunk = 0.5*0.5 + 0.5*0.98 = 0.74
        assert res.p == pytest.approx(0.74)

    def test_tail_distance_correct(self):
        res = apply_longshot_shrinkage(0.9)
        assert res.tail_distance == pytest.approx(0.4)

    def test_custom_thresholds(self):
        res = apply_longshot_shrinkage(0.15, low_threshold=0.10, high_threshold=0.90)
        assert res.applied is False  # 0.15 is between 0.10 and 0.90

        res2 = apply_longshot_shrinkage(0.08, low_threshold=0.10, high_threshold=0.90)
        assert res2.applied is True

    def test_custom_weight(self):
        res = apply_longshot_shrinkage(0.02, weight=0.8)
        # shrunk = 0.8*0.5 + 0.2*0.02 = 0.404
        assert res.p == pytest.approx(0.404)

    def test_weight_clamped_to_0_1(self):
        res_over = apply_longshot_shrinkage(0.02, weight=1.5)
        # weight clamped to 1.0 -> pure 0.5
        assert res_over.p == pytest.approx(0.5)

        res_under = apply_longshot_shrinkage(0.02, weight=-0.5)
        # weight clamped to 0.0 -> no shrinkage, but still applied
        assert res_under.p == pytest.approx(0.02)

    def test_result_clamped_to_unit_interval(self):
        """Even with extreme weights the result stays in [0, 1]."""
        res = apply_longshot_shrinkage(0.0)
        assert 0.0 <= res.p <= 1.0

        res2 = apply_longshot_shrinkage(1.0)
        assert 0.0 <= res2.p <= 1.0
