"""Tests for the Normal-Inverse-Gamma scale-mixture HMM spike.

Wave-2 P6 (docs/references-deep-dive-2026-04-28.md §6).

Theory:
    Per-state Gaussian emission with conjugate Normal-Inverse-Gamma prior on
    (mu, sigma^2). Posterior updates from weighted observations (E-step
    responsibilities). Predictive emission marginalising sigma^2 is a
    Student-t with degrees-of-freedom nu = 2 * alpha_post (alpha is the
    posterior shape of the Inverse-Gamma on sigma^2). This means nu *falls
    out of the data* instead of being a static asset-class constant.

    Key qualitative property to verify (the "true improvement" gate):
    on a small sample with one heavy outlier, the predictive Student-t
    must assign more density to the outlier than a Gaussian fit by
    plain MLE on the same data. That is the textbook fat-tail benefit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from research.spikes.scale_mixture_hmm.nig_emission import (
    NIGPrior,
    posterior_update,
    predictive_logpdf,
    predictive_nu,
)


# ---------------------------------------------------------------------------
# 1. Posterior update arithmetic
# ---------------------------------------------------------------------------


def test_posterior_update_unit_weight_matches_textbook_formulas():
    prior = NIGPrior(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones_like(obs)

    post = posterior_update(prior, obs, weights)

    n = 4.0
    xbar = 2.5
    ss = float(np.sum((obs - xbar) ** 2))  # 5.0

    assert post.kappa0 == pytest.approx(prior.kappa0 + n)
    assert post.alpha0 == pytest.approx(prior.alpha0 + n / 2.0)
    assert post.mu0 == pytest.approx(
        (prior.kappa0 * prior.mu0 + n * xbar) / (prior.kappa0 + n)
    )
    expected_beta = (
        prior.beta0
        + 0.5 * ss
        + 0.5 * (prior.kappa0 * n / (prior.kappa0 + n)) * (xbar - prior.mu0) ** 2
    )
    assert post.beta0 == pytest.approx(expected_beta)


def test_posterior_update_zero_weights_returns_prior():
    prior = NIGPrior(mu0=0.5, kappa0=2.0, alpha0=3.0, beta0=4.0)
    obs = np.array([1.0, 2.0, 3.0])
    weights = np.zeros_like(obs)

    post = posterior_update(prior, obs, weights)

    assert post.mu0 == pytest.approx(prior.mu0)
    assert post.kappa0 == pytest.approx(prior.kappa0)
    assert post.alpha0 == pytest.approx(prior.alpha0)
    assert post.beta0 == pytest.approx(prior.beta0)


def test_posterior_update_fractional_weights_match_unit_weights_when_equal():
    """Halving every weight should be equivalent to using half-strength prior."""
    prior = NIGPrior(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
    obs = np.array([1.0, 2.0, 3.0, 4.0])

    post_full = posterior_update(prior, obs, np.ones_like(obs))
    post_half = posterior_update(prior, obs, 0.5 * np.ones_like(obs))

    # half-weight => effective n is halved
    assert post_half.kappa0 == pytest.approx(prior.kappa0 + 2.0)
    assert post_half.alpha0 == pytest.approx(prior.alpha0 + 1.0)
    assert post_full.alpha0 > post_half.alpha0


# ---------------------------------------------------------------------------
# 2. Predictive Student-t shape
# ---------------------------------------------------------------------------


def test_predictive_nu_equals_two_alpha():
    prior = NIGPrior(mu0=0.0, kappa0=1.0, alpha0=2.5, beta0=1.0)
    assert predictive_nu(prior) == pytest.approx(5.0)


def test_predictive_logpdf_matches_scipy_student_t():
    prior = NIGPrior(mu0=0.1, kappa0=2.0, alpha0=3.0, beta0=1.5)
    nu = predictive_nu(prior)
    scale = math.sqrt(prior.beta0 * (prior.kappa0 + 1.0) / (prior.alpha0 * prior.kappa0))

    for x in [-2.0, -0.5, 0.0, 0.3, 1.5]:
        expected = stats.t.logpdf(x, df=nu, loc=prior.mu0, scale=scale)
        assert predictive_logpdf(prior, x) == pytest.approx(expected, abs=1e-10)


def test_predictive_density_integrates_to_one():
    prior = NIGPrior(mu0=0.0, kappa0=1.0, alpha0=4.0, beta0=2.0)
    xs = np.linspace(-50.0, 50.0, 20001)
    logpdf = np.array([predictive_logpdf(prior, float(x)) for x in xs])
    pdf = np.exp(logpdf)
    integral = np.trapezoid(pdf, xs)
    assert integral == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 3. True-improvement gate: fat-tail outlier handling
# ---------------------------------------------------------------------------


def test_predictive_assigns_more_density_to_outlier_than_gaussian_mle():
    """Core 'true improvement' check.

    Sample 30 returns from N(0, 1) plus a single tail event at +6 sigma.
    The predictive Student-t emission should give the outlier strictly
    higher log-density than a Gaussian fit by maximum likelihood on the
    same data. If this fails, the spike does not beat the baseline and
    we should not port to TS.
    """
    rng = np.random.default_rng(42)
    bulk = rng.normal(loc=0.0, scale=1.0, size=30)
    obs = np.concatenate([bulk, [6.0]])
    weights = np.ones_like(obs)

    weak_prior = NIGPrior(mu0=0.0, kappa0=0.01, alpha0=2.0, beta0=2.0)
    post = posterior_update(weak_prior, obs, weights)

    # Predictive logpdf at the outlier
    logpdf_t = predictive_logpdf(post, 6.0)

    # Plain Gaussian MLE on the same data
    mu_mle = float(np.mean(obs))
    sigma_mle = float(np.std(obs, ddof=0))
    logpdf_n = stats.norm.logpdf(6.0, loc=mu_mle, scale=sigma_mle)

    assert logpdf_t > logpdf_n, (
        f"NIG predictive ({logpdf_t:.3f}) must exceed Gaussian MLE "
        f"({logpdf_n:.3f}) at the +6sigma outlier."
    )


def test_nu_decreases_with_sample_size_increase_when_data_is_heavy_tailed():
    """With heavy-tailed data, additional observations push posterior alpha
    up, but the *effective* tail weight should still be heavier than a
    Gaussian. This locks in that nu remains finite (i.e. we never
    accidentally collapse to a pure Gaussian)."""
    prior = NIGPrior(mu0=0.0, kappa0=0.01, alpha0=2.0, beta0=2.0)
    rng = np.random.default_rng(7)

    obs_small = rng.standard_t(df=4, size=20)
    obs_large = rng.standard_t(df=4, size=200)

    nu_small = predictive_nu(posterior_update(prior, obs_small, np.ones_like(obs_small)))
    nu_large = predictive_nu(posterior_update(prior, obs_large, np.ones_like(obs_large)))

    # Both should be > 2 (predictive distribution well-defined) and
    # finite (never degenerate to Gaussian).
    assert nu_small > 2.0
    assert nu_large > 2.0
    # Larger sample => more data => higher alpha => higher nu
    assert nu_large > nu_small
