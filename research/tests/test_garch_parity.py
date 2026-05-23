"""P3a parity tests — Python GARCH(1,1) mirrors TS src/utils/garch.ts."""

from __future__ import annotations

import math

import pytest

from research.models.garch import (
    GARCH_DEFAULT_ALPHA,
    GARCH_DEFAULT_BETA,
    fit_garch11,
    garch_forecast,
    garch_step,
)


def _make_returns(n: int, vol: float, seed: int = 1):
    s = seed
    out = []
    i = 0
    while i < n:
        s = (1103515245 * s + 12345) % (2 ** 31)
        u1 = max(1e-12, s / 2 ** 31)
        s = (1103515245 * s + 12345) % (2 ** 31)
        u2 = s / 2 ** 31
        r = math.sqrt(-2 * math.log(u1))
        out.append(r * math.cos(2 * math.pi * u2) * vol)
        if i + 1 < n:
            out.append(r * math.sin(2 * math.pi * u2) * vol)
        i += 2
    return out[:n]


def test_defaults_are_stationary():
    assert GARCH_DEFAULT_ALPHA + GARCH_DEFAULT_BETA < 1


def test_fit_matches_sample_variance():
    rets = _make_returns(252, 0.01)
    p = fit_garch11(rets)
    sample_var = sum(r * r for r in rets) / len(rets)
    uncond = p.omega / (1 - p.alpha - p.beta)
    assert uncond == pytest.approx(sample_var, rel=1e-9)


def test_fit_h0_equals_sample_variance():
    rets = _make_returns(100, 0.015)
    p = fit_garch11(rets)
    sample_var = sum(r * r for r in rets) / len(rets)
    assert p.h0 == pytest.approx(sample_var, rel=1e-9)


def test_fit_rejects_short_input():
    with pytest.raises(ValueError):
        fit_garch11([0.01, 0.02])


def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        fit_garch11([])


def test_garch_step_recursion():
    from research.models.garch import Garch11Params

    p = Garch11Params(omega=1e-6, alpha=0.10, beta=0.85, h0=1e-4)
    expected = p.omega + p.alpha * 1.5 * 1.5 * 1e-4 + p.beta * 1e-4
    assert garch_step(1e-4, 1.5, p) == pytest.approx(expected, rel=1e-12)


def test_garch_step_reverts_to_beta_fixed_point_with_z_zero():
    from research.models.garch import Garch11Params

    p = Garch11Params(omega=1e-6, alpha=0.10, beta=0.85, h0=1e-4)
    fp = p.omega / (1 - p.beta)
    h = 5 * fp
    for _ in range(200):
        h = garch_step(h, 0.0, p)
    assert h == pytest.approx(fp, abs=1e-9)


def test_forecast_length_and_positivity():
    from research.models.garch import Garch11Params

    p = Garch11Params(omega=1e-6, alpha=0.10, beta=0.85, h0=1e-4)
    sigmas = garch_forecast(p, 30)
    assert len(sigmas) == 30
    assert all(s > 0 and math.isfinite(s) for s in sigmas)


def test_forecast_converges_to_unconditional():
    from research.models.garch import Garch11Params

    p = Garch11Params(omega=1e-6, alpha=0.10, beta=0.85, h0=1e-2)
    uncond = math.sqrt(p.omega / (1 - p.alpha - p.beta))
    sigmas = garch_forecast(p, 500)
    assert sigmas[-1] == pytest.approx(uncond, abs=1e-5)
