"""Beta-HMM — Hidden Markov Model with Beta emissions for bounded [0,1] data.

Mirrors src/tools/finance/beta-hmm.ts. Designed for Polymarket prices.

Reference: Voigt (2025), *Predicting Prediction Markets: A Beta-Hidden Markov
Modeling Approach* — see references/prediction-markets/BetaHMMpolymarket-18-1.pdf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import math

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class BetaEmission:
    alpha: float
    beta: float


@dataclass
class BetaHMMParams:
    n_states: int
    pi: np.ndarray  # shape (K,)
    A: np.ndarray   # shape (K, K)
    emissions: List[BetaEmission] = field(default_factory=list)


@dataclass
class BetaHMMFitResult:
    params: BetaHMMParams
    log_likelihood: float
    iterations: int
    converged: bool


# ---------------------------------------------------------------------------
# Beta PDF
# ---------------------------------------------------------------------------


def beta_pdf(x: float, e: BetaEmission) -> float:
    """Beta probability density. Returns 0 outside (0,1)."""
    if x <= 0 or x >= 1:
        return 0.0
    if e.alpha <= 0 or e.beta <= 0:
        return 0.0
    ln_b = gammaln(e.alpha) + gammaln(e.beta) - gammaln(e.alpha + e.beta)
    log_pdf = (e.alpha - 1) * math.log(x) + (e.beta - 1) * math.log(1 - x) - ln_b
    return math.exp(max(-700.0, min(700.0, log_pdf)))


# ---------------------------------------------------------------------------
# Method-of-moments fit (weighted)
# ---------------------------------------------------------------------------


def fit_beta_mom(weights: Sequence[float], samples: Sequence[float]) -> BetaEmission:
    """Weighted method-of-moments Beta(α,β) fit.

    See TS docstring for equations. Falls back to Beta(1,1) if variance ≤ 0.
    """
    if len(weights) != len(samples):
        raise ValueError("fit_beta_mom: weights and samples must have equal length")
    eps = 1e-6
    w = np.asarray(weights, dtype=float)
    x = np.clip(np.asarray(samples, dtype=float), eps, 1 - eps)
    valid = np.isfinite(w) & (w > 0)
    if not valid.any():
        return BetaEmission(1.0, 1.0)
    w = w[valid]
    x = x[valid]
    wsum = float(w.sum())
    if wsum <= 0:
        return BetaEmission(1.0, 1.0)
    m = float((w * x).sum() / wsum)
    v = float((w * (x - m) ** 2).sum() / wsum)
    if v <= 1e-10 or m <= 0 or m >= 1:
        return BetaEmission(1.0, 1.0)
    nu = m * (1 - m) / v - 1
    if nu <= 0:
        return BetaEmission(1.0, 1.0)
    alpha = nu * m
    beta = nu * (1 - m)
    if not (math.isfinite(alpha) and math.isfinite(beta)) or alpha <= 0 or beta <= 0:
        return BetaEmission(1.0, 1.0)
    return BetaEmission(float(alpha), float(beta))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def initialize_beta_hmm(observations: Sequence[float], n_states: int) -> BetaHMMParams:
    if n_states < 1:
        raise ValueError("initialize_beta_hmm: n_states must be >= 1")
    sorted_obs = np.sort(np.asarray(observations, dtype=float))
    n = len(sorted_obs)
    emissions: List[BetaEmission] = []
    for i in range(n_states):
        lo = (i * n) // n_states
        hi = max(((i + 1) * n) // n_states, lo + 1)
        sl = sorted_obs[lo:hi]
        emissions.append(fit_beta_mom([1.0] * len(sl), sl.tolist()))
    stay = 0.85
    off = (1 - stay) / max(n_states - 1, 1)
    A = np.full((n_states, n_states), off)
    np.fill_diagonal(A, stay)
    pi = np.full(n_states, 1.0 / n_states)
    return BetaHMMParams(n_states=n_states, pi=pi, A=A, emissions=emissions)


# ---------------------------------------------------------------------------
# Forward / Backward
# ---------------------------------------------------------------------------


def _emit_vec(obs_t: float, p: BetaHMMParams) -> np.ndarray:
    return np.array([max(beta_pdf(obs_t, e), 1e-300) for e in p.emissions])


def forward_beta(obs: Sequence[float], p: BetaHMMParams):
    T = len(obs)
    K = p.n_states
    alpha = np.zeros((T, K))
    scales = np.zeros(T)
    e0 = _emit_vec(obs[0], p)
    alpha[0] = p.pi * e0
    scales[0] = alpha[0].sum() or 1.0
    alpha[0] /= scales[0]
    for t in range(1, T):
        e_t = _emit_vec(obs[t], p)
        alpha[t] = (alpha[t - 1] @ p.A) * e_t
        scales[t] = alpha[t].sum() or 1.0
        alpha[t] /= scales[t]
    log_lik = float(np.log(scales).sum())
    return alpha, scales, log_lik


def backward_beta(obs: Sequence[float], p: BetaHMMParams, scales: np.ndarray) -> np.ndarray:
    T = len(obs)
    K = p.n_states
    beta = np.zeros((T, K))
    beta[-1] = 1.0 / scales[-1]
    for t in range(T - 2, -1, -1):
        e_next = _emit_vec(obs[t + 1], p)
        beta[t] = (p.A @ (e_next * beta[t + 1])) / scales[t]
    return beta


# ---------------------------------------------------------------------------
# Baum-Welch
# ---------------------------------------------------------------------------


def baum_welch_beta(
    obs: Sequence[float],
    init: BetaHMMParams,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> BetaHMMFitResult:
    p = BetaHMMParams(
        n_states=init.n_states,
        pi=init.pi.copy(),
        A=init.A.copy(),
        emissions=[BetaEmission(e.alpha, e.beta) for e in init.emissions],
    )
    obs_arr = np.asarray(obs, dtype=float)
    T = len(obs_arr)
    K = p.n_states
    prev_ll = -math.inf
    converged = False
    last_ll = -math.inf
    iter_count = 0
    for iter_count in range(max_iter):
        alpha, scales, log_lik = forward_beta(obs_arr, p)
        beta = backward_beta(obs_arr, p, scales)
        last_ll = log_lik

        gamma = alpha * beta * scales[:, None]
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        gamma = gamma / row_sums

        xi_sum = np.zeros((K, K))
        for t in range(T - 1):
            e_next = _emit_vec(obs_arr[t + 1], p)
            xi_t = (alpha[t][:, None] * p.A) * (e_next * beta[t + 1])[None, :]
            denom = xi_t.sum()
            if denom > 0:
                xi_sum += xi_t / denom

        pi_new = gamma[0].copy()
        s = pi_new.sum() or 1.0
        pi_new /= s

        A_new = np.zeros((K, K))
        for i in range(K):
            denom = xi_sum[i].sum()
            if denom > 0:
                A_new[i] = xi_sum[i] / denom
            else:
                A_new[i, i] = 1.0

        emissions_new: List[BetaEmission] = []
        for i in range(K):
            emissions_new.append(fit_beta_mom(gamma[:, i].tolist(), obs_arr.tolist()))

        p = BetaHMMParams(n_states=K, pi=pi_new, A=A_new, emissions=emissions_new)

        if abs(log_lik - prev_ll) < tol:
            converged = True
            break
        prev_ll = log_lik

    return BetaHMMFitResult(params=p, log_likelihood=last_ll, iterations=iter_count + 1, converged=converged)


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------


def viterbi_beta(obs: Sequence[float], p: BetaHMMParams) -> List[int]:
    T = len(obs)
    K = p.n_states
    delta = np.full((T, K), -math.inf)
    psi = np.zeros((T, K), dtype=int)
    e0 = _emit_vec(obs[0], p)
    delta[0] = np.log(np.maximum(p.pi, 1e-300)) + np.log(np.maximum(e0, 1e-300))
    log_A = np.log(np.maximum(p.A, 1e-300))
    for t in range(1, T):
        e_t = _emit_vec(obs[t], p)
        for j in range(K):
            vals = delta[t - 1] + log_A[:, j]
            best_i = int(np.argmax(vals))
            delta[t, j] = vals[best_i] + math.log(max(e_t[j], 1e-300))
            psi[t, j] = best_i
    path = [0] * T
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = int(psi[t + 1, path[t + 1]])
    return path
