"""Parity tests — Python garch_scales mirrors TS garch-scales.ts (R5 Idea #5)."""

from __future__ import annotations

import math

import pytest

from research.models.garch_scales import (
    GarchClampOptions,
    compute_garch_scales,
    detect_recent_regime,
)


def _make_returns(n: int, vol: float, seed: int = 7) -> list[float]:
    """Deterministic LCG normal approximation (Box-Muller)."""
    s = seed
    out = []
    i = 0
    while i < n:
        s = (1103515245 * s + 12345) % (2**31)
        u1 = max(1e-12, s / 2**31)
        s = (1103515245 * s + 12345) % (2**31)
        u2 = s / 2**31
        r = math.sqrt(-2 * math.log(u1))
        out.append(r * math.cos(2 * math.pi * u2) * vol)
        if i + 1 < n:
            out.append(r * math.sin(2 * math.pi * u2) * vol)
        i += 2
    return out[:n]


class TestDetectRecentRegime:
    def test_short_series_is_turbulent(self):
        rets = _make_returns(10, 0.01)
        assert detect_recent_regime(rets, 20) == "turbulent"

    def test_calm_when_recent_lower_vol(self):
        # Calm tail: low vol at the end
        calm_tail = [0.001 * (i % 3) for i in range(50)]
        turbulent_head = _make_returns(50, 0.03, seed=42)
        rets = turbulent_head + calm_tail
        assert detect_recent_regime(rets, 20) == "calm"

    def test_turbulent_when_recent_higher_vol(self):
        calm_head = [0.001 * (i % 3) for i in range(50)]
        turbulent_tail = _make_returns(50, 0.03, seed=42)
        rets = calm_head + turbulent_tail
        assert detect_recent_regime(rets, 20) == "turbulent"


class TestComputeGarchScales:
    def test_returns_empty_on_short_input(self):
        assert compute_garch_scales([0.01, 0.02, 0.03], 10) == []

    def test_returns_empty_on_zero_horizon(self):
        assert compute_garch_scales(_make_returns(50, 0.01), 0) == []

    def test_length_matches_horizon(self):
        rets = _make_returns(100, 0.02)
        for h in [1, 5, 30]:
            assert len(compute_garch_scales(rets, h)) == h

    def test_all_positive_and_finite(self):
        rets = _make_returns(100, 0.02)
        scales = compute_garch_scales(rets, 30)
        assert all(math.isfinite(s) and s > 0 for s in scales)

    def test_legacy_defaults_clamp_within_bounds(self):
        """Without opts, clamp is [0.33, 3.0]."""
        rets = _make_returns(200, 0.02)
        scales = compute_garch_scales(rets, 30)
        assert all(0.33 <= s <= 3.0 for s in scales)

    def test_horizon_decay_flattens_at_3x_cap(self):
        """Past 3*horizonCap the scalar should be exactly 1.0."""
        rets = _make_returns(200, 0.02)
        cap = 5
        opts = GarchClampOptions(horizon_cap=cap)
        scales = compute_garch_scales(rets, cap * 3 + 5, opts)
        tail = scales[cap * 3:]
        assert all(s == pytest.approx(1.0, abs=1e-9) for s in tail)

    def test_regime_override_calm_applies_lower_ceiling(self):
        """Calm ceiling (1.5) < turbulent ceiling (3.0)."""
        rets = _make_returns(200, 0.02)
        opts_calm = GarchClampOptions(
            ceiling=(1.5, 3.0), regime_override="calm"
        )
        opts_turbulent = GarchClampOptions(
            ceiling=(1.5, 3.0), regime_override="turbulent"
        )
        scales_calm = compute_garch_scales(rets, 30, opts_calm)
        scales_turbulent = compute_garch_scales(rets, 30, opts_turbulent)
        assert all(s <= 1.5 for s in scales_calm)
        assert all(s <= 3.0 for s in scales_turbulent)

    def test_no_opts_same_as_default_opts(self):
        """opts=None must be byte-identical to opts=GarchClampOptions()."""
        rets = _make_returns(100, 0.015)
        s1 = compute_garch_scales(rets, 20, None)
        s2 = compute_garch_scales(rets, 20, GarchClampOptions())
        assert s1 == s2

    def test_empty_returns_when_zero_variance(self):
        rets = [0.0] * 50
        assert compute_garch_scales(rets, 10) == []
