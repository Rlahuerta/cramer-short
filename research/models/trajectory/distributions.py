"""Statistical distribution helpers for trajectory computation.

Mirrors TS logic from src/tools/finance/markov-distribution.ts:
  - Student-t CDF, PPF and survival functions
  - Log-normal survival for benchmarking
"""

from __future__ import annotations

import math

from scipy import stats


def normal_cdf(x: float) -> float:
    return float(stats.norm.cdf(x))


def student_t_cdf(x: float, nu: int = 5) -> float:
    return float(stats.t.cdf(x, df=nu))


def student_t_ppf(p: float, nu: int = 5) -> float:
    return float(stats.t.ppf(p, df=nu))


def student_t_survival(
    target_price: float,
    current_price: float,
    mu_n: float,
    sigma_n: float,
    nu: int = 5,
) -> float:
    if target_price <= 0:
        return 1.0  # price can't go below 0
    if sigma_n <= 0:
        return 1.0 if mu_n > math.log(target_price / current_price) else 0.0
    # sigma_n is the t's standard deviation: Var(t_nu) = nu/(nu-2) for nu > 2.
    scaled_sigma = sigma_n * math.sqrt((nu - 2) / nu) if nu > 2 else sigma_n
    z = (math.log(target_price / current_price) - mu_n) / scaled_sigma
    return 1.0 - student_t_cdf(z, nu)


def log_normal_survival(
    current_price: float,
    target_price: float,
    drift: float,
    vol: float,
) -> float:
    if vol <= 0:
        return 1.0 if drift > math.log(target_price / current_price) else 0.0
    d = (math.log(target_price / current_price) - drift) / vol
    return 1.0 - normal_cdf(d)
