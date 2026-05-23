"""Parity tests for Beta-HMM Python mirror against TS module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from research.models.beta_hmm import (
    BetaEmission,
    BetaHMMParams,
    baum_welch_beta,
    beta_pdf,
    fit_beta_mom,
    forward_beta,
    initialize_beta_hmm,
    viterbi_beta,
)


def test_beta_pdf_outside_support_is_zero():
    e = BetaEmission(2.0, 5.0)
    assert beta_pdf(-0.1, e) == 0.0
    assert beta_pdf(0.0, e) == 0.0
    assert beta_pdf(1.0, e) == 0.0
    assert beta_pdf(1.1, e) == 0.0


def test_beta_pdf_uniform():
    e = BetaEmission(1.0, 1.0)
    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert math.isclose(beta_pdf(x, e), 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("a,b", [(2, 5), (5, 2), (3, 3), (0.7, 0.7)])
def test_beta_pdf_integrates_to_one(a, b):
    N = 10_000
    xs = np.linspace(1 / N, 1 - 1 / N, N - 1)
    integral = sum(beta_pdf(float(x), BetaEmission(a, b)) for x in xs) / N
    assert 0.99 < integral < 1.01


def test_fit_beta_mom_recovers_known_alpha_beta():
    rng = np.random.default_rng(42)
    samples = rng.beta(2.0, 5.0, size=5000)
    fit = fit_beta_mom([1.0] * len(samples), samples.tolist())
    assert 1.6 < fit.alpha < 2.5
    assert 4.0 < fit.beta < 6.5


def test_fit_beta_mom_zero_variance_returns_uniform():
    fit = fit_beta_mom([1.0, 1.0, 1.0], [0.4, 0.4, 0.4])
    assert fit.alpha == 1.0 and fit.beta == 1.0


def test_fit_beta_mom_handles_boundary_values():
    fit = fit_beta_mom([1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.5, 0.5])
    assert math.isfinite(fit.alpha) and math.isfinite(fit.beta)
    assert fit.alpha > 0 and fit.beta > 0


def test_forward_beta_normalised():
    p = BetaHMMParams(
        n_states=2,
        pi=np.array([0.5, 0.5]),
        A=np.array([[0.9, 0.1], [0.1, 0.9]]),
        emissions=[BetaEmission(2, 8), BetaEmission(8, 2)],
    )
    obs = [0.1, 0.15, 0.2, 0.85, 0.9]
    alpha, _, _ = forward_beta(obs, p)
    for t in range(len(obs)):
        assert math.isclose(alpha[t].sum(), 1.0, abs_tol=1e-9)


def test_baum_welch_beta_recovers_two_state_regimes():
    rng = np.random.default_rng(123)
    n = 400
    true_states = []
    obs = []
    state = 0
    for _ in range(n):
        if rng.random() < 0.1:
            state = 1 - state
        true_states.append(state)
        if state == 0:
            obs.append(float(rng.beta(2.0, 8.0)))
        else:
            obs.append(float(rng.beta(8.0, 2.0)))
    init = initialize_beta_hmm(obs, 2)
    fit = baum_welch_beta(obs, init, max_iter=30, tol=1e-4)
    path = viterbi_beta(obs, fit.params)
    agree = sum(1 for a, b in zip(path, true_states) if a == b)
    acc = agree / n
    if acc < 0.5:
        acc = 1 - acc
    assert acc > 0.85
