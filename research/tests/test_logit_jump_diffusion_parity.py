"""Parity tests for Python logit-jump-diffusion mirror."""

from __future__ import annotations

import math

import numpy as np
import pytest

from research.models.logit_jump_diffusion import (
    inv_logit,
    ito_martingale_drift,
    logit,
    simulate_logit_jump_diffusion,
)


@pytest.mark.parametrize("p", [0.05, 0.1, 0.3, 0.5, 0.7, 0.95])
def test_logit_round_trip(p):
    assert math.isclose(inv_logit(logit(p)), p, rel_tol=1e-12)


def test_logit_clips_boundary():
    assert math.isfinite(logit(0.0))
    assert math.isfinite(logit(1.0))


@pytest.mark.parametrize("z", [-50, -1, 0, 1, 50])
def test_inv_logit_in_open_unit(z):
    p = inv_logit(z)
    assert 0 < p < 1


def test_drift_zero_at_half():
    assert math.isclose(ito_martingale_drift(0.5, 0.1, 0, 0, 0), 0.0, abs_tol=1e-12)


def test_drift_negative_below_half():
    # sigmoid is convex for p<0.5 → need negative drift in x
    assert ito_martingale_drift(0.2, 0.1, 0, 0, 0) < 0


def test_drift_positive_above_half():
    assert ito_martingale_drift(0.8, 0.1, 0, 0, 0) > 0


def test_drift_grows_with_sigma():
    a = abs(ito_martingale_drift(0.2, 0.1, 0, 0, 0))
    b = abs(ito_martingale_drift(0.2, 0.5, 0, 0, 0))
    assert b > a


def test_simulator_prices_strictly_in_unit():
    rng = np.random.default_rng(7)
    out = simulate_logit_jump_diffusion(
        initial_price=0.30, days=30, sigma_per_day=0.20,
        n_paths=2000, rng=rng.random,
    )
    assert (out.terminal > 0).all() and (out.terminal < 1).all()


def test_simulator_zero_vol_zero_jumps_constant():
    rng = np.random.default_rng(7)
    out = simulate_logit_jump_diffusion(
        initial_price=0.30, days=30, sigma_per_day=0,
        n_paths=500, rng=rng.random,
    )
    assert np.allclose(out.terminal, 0.30, atol=1e-8)


def test_simulator_martingale_diffusion_only():
    rng = np.random.default_rng(7)
    out = simulate_logit_jump_diffusion(
        initial_price=0.30, days=30, sigma_per_day=0.10,
        n_paths=5000, rng=rng.random,
    )
    assert abs(out.terminal.mean() - 0.30) < 0.01


def test_simulator_martingale_with_jumps():
    rng = np.random.default_rng(7)
    out = simulate_logit_jump_diffusion(
        initial_price=0.50, days=30, sigma_per_day=0.05,
        jump_intensity_per_day=0.02,
        jump_logit_mean=0.5, jump_logit_std=0.3,
        n_paths=8000, rng=rng.random,
    )
    assert abs(out.terminal.mean() - 0.50) < 0.02


def test_polymarket_lambda_scaling():
    rng = np.random.default_rng(11)
    out = simulate_logit_jump_diffusion(
        initial_price=0.5, days=30, sigma_per_day=0,
        polymarket_jump_prob=0.30,
        jump_logit_mean=1.0, jump_logit_std=0.1,
        n_paths=4000, rng=rng.random,
    )
    rate = out.total_jumps / (len(out.terminal) * 30)
    assert 0.005 < rate < 0.02
