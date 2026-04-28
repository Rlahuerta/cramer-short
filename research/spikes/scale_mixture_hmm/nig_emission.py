"""Normal-Inverse-Gamma conjugate emission for a scale-mixture HMM.

Theory
------
Per-state emission y_t | state=k ~ Normal(mu_k, sigma_k^2). Conjugate prior:

    sigma_k^2 ~ InvGamma(alpha0, beta0)
    mu_k | sigma_k^2 ~ Normal(mu0, sigma_k^2 / kappa0)

Posterior update from weighted observations (E-step responsibilities w_i for
data x_i, sum w_i = n_eff):

    kappa_n = kappa0 + n_eff
    mu_n    = (kappa0 * mu0 + sum(w_i * x_i)) / kappa_n
    alpha_n = alpha0 + n_eff / 2
    beta_n  = beta0
              + 0.5 * sum(w_i * (x_i - xbar_w)^2)
              + 0.5 * (kappa0 * n_eff / kappa_n) * (xbar_w - mu0)^2

where xbar_w = sum(w_i * x_i) / n_eff.

Marginalising sigma^2, the predictive distribution for a new observation is
Student-t with:

    nu       = 2 * alpha_n
    location = mu_n
    scale    = sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))

This means the degrees-of-freedom *fall out of the data* rather than being a
static asset-class constant (the current TS HMM uses 3-5 hardcoded by asset
class). With a weak prior, small samples or heavy-tailed data yield small
alpha_n, hence small nu and visibly fatter tails than a Gaussian fit by MLE.

Reference
---------
docs/references-deep-dive-2026-04-28.md §6 — Taleb & Cirillo, Risks 13(12):247.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


@dataclass
class NIGPrior:
    """Normal-Inverse-Gamma hyper-parameters.

    The same dataclass represents both prior (alpha0, beta0, kappa0, mu0) and
    posterior (alpha_n, beta_n, kappa_n, mu_n) — only the values change.
    """

    mu0: float
    kappa0: float
    alpha0: float
    beta0: float


def posterior_update(
    prior: NIGPrior, observations: np.ndarray, weights: np.ndarray
) -> NIGPrior:
    """Conjugate Normal-Inverse-Gamma posterior from weighted observations.

    See module docstring for derivation. ``weights`` are the E-step
    responsibilities for one HMM state; pass an all-ones array for MAP
    estimation on a single Gaussian cluster.
    """
    obs = np.asarray(observations, dtype=float)
    w = np.asarray(weights, dtype=float)
    if obs.shape != w.shape:
        raise ValueError("observations and weights must have the same shape")

    n_eff = float(np.sum(w))
    if n_eff <= 0.0:
        return NIGPrior(prior.mu0, prior.kappa0, prior.alpha0, prior.beta0)

    sum_wx = float(np.sum(w * obs))
    xbar_w = sum_wx / n_eff

    kappa_n = prior.kappa0 + n_eff
    mu_n = (prior.kappa0 * prior.mu0 + sum_wx) / kappa_n
    alpha_n = prior.alpha0 + n_eff / 2.0

    ss_w = float(np.sum(w * (obs - xbar_w) ** 2))
    correction = (prior.kappa0 * n_eff / kappa_n) * (xbar_w - prior.mu0) ** 2
    beta_n = prior.beta0 + 0.5 * ss_w + 0.5 * correction

    return NIGPrior(mu0=mu_n, kappa0=kappa_n, alpha0=alpha_n, beta0=beta_n)


def predictive_nu(params: NIGPrior) -> float:
    """Degrees-of-freedom of the predictive Student-t."""
    return 2.0 * params.alpha0


def predictive_logpdf(params: NIGPrior, x: float) -> float:
    """Log-density of the predictive Student-t at x.

    Equivalent to ``scipy.stats.t.logpdf(x, df=nu, loc=mu_n, scale=scale)``
    but written out to keep the spike free of scipy.stats at runtime.
    """
    nu = predictive_nu(params)
    scale_sq = params.beta0 * (params.kappa0 + 1.0) / (params.alpha0 * params.kappa0)
    scale = math.sqrt(scale_sq)
    z = (x - params.mu0) / scale

    log_norm = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
        - math.log(scale)
    )
    return float(log_norm - ((nu + 1.0) / 2.0) * math.log1p(z * z / nu))
