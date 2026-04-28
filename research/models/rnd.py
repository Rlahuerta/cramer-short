"""Risk-Neutral Density extraction from Polymarket strike markets.

Extracts forward-looking regime probabilities from a chain of Polymarket
"Will asset be above $K?" contracts, transforms Q-measure to P-measure,
and maps the fitted distribution to Markov regime buckets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from research.models.markov import STATE_INDEX


# Default cap on |Market Price of Risk| to prevent runaway shifts when
# historical drift estimates are noisy (e.g. crypto bull-run windows
# producing mu ≈ 200% / sigma ≈ 60% -> MPR ≈ 3 -> silly P-prob shifts).
DEFAULT_MPR_CAP: float = 1.5


def transform_q_to_p(
    q_prob: float,
    historical_drift: float,
    risk_free_rate: float,
    volatility: float,
    days_to_expiry: int,
    mpr_cap: float = DEFAULT_MPR_CAP,
) -> float:
    """Convert risk-neutral probability to physical probability via Girsanov shift.

    Prob^P(S_T > K) = Phi(Phi^{-1}(Prob^Q(S_T > K)) + lambda * sqrt(T))

    Where the Market Price of Risk is lambda = (mu - r_f) / sigma.

    Inputs are **annualised**:
      - ``historical_drift``  mu      (e.g. 0.40 for 40 % annual)
      - ``risk_free_rate``    r_f     (e.g. 0.05 for 5 % annual)
      - ``volatility``        sigma   (e.g. 0.50 for 50 % annual)
      - ``days_to_expiry``    T       (calendar days; converted to years via /365)

    ``mpr_cap`` clamps ``|lambda|`` to a finite range (default
    :data:`DEFAULT_MPR_CAP`).  The cap is necessary because crypto / momentum
    windows can produce pathological MPR estimates that map every Q-prob to
    ~0 or ~1.

    Returns a probability in ``[0, 1]``.  Boundary inputs (0 / 1) pass through.
    """
    if q_prob <= 0.0 or q_prob >= 1.0:
        return float(np.clip(q_prob, 0.0, 1.0))

    # Guardrails for probit stability
    q_clipped = float(np.clip(q_prob, 0.001, 0.999))

    T_years = max(days_to_expiry, 1) / 365.0
    raw_mpr = (historical_drift - risk_free_rate) / max(volatility, 1e-6)
    cap = max(mpr_cap, 0.0)
    lambda_mpr = float(np.clip(raw_mpr, -cap, cap))

    z_q = stats.norm.ppf(q_clipped)
    z_p = z_q + lambda_mpr * math.sqrt(T_years)

    return float(stats.norm.cdf(z_p))


def transform_q_to_p_with_shift(
    q_prob: float,
    historical_drift: float,
    risk_free_rate: float,
    volatility: float,
    days_to_expiry: int,
    mpr_cap: float = DEFAULT_MPR_CAP,
) -> dict[str, float]:
    """Diagnostic variant returning the applied Z-shift and MPR provenance."""
    T_years = max(days_to_expiry, 1) / 365.0
    raw_mpr = (historical_drift - risk_free_rate) / max(volatility, 1e-6)
    cap = max(mpr_cap, 0.0)
    mpr_used = float(np.clip(raw_mpr, -cap, cap))
    z_shift = mpr_used * math.sqrt(T_years)
    return {
        "p_prob": transform_q_to_p(
            q_prob, historical_drift, risk_free_rate, volatility, days_to_expiry, mpr_cap
        ),
        "z_shift": z_shift,
        "mpr_used": mpr_used,
        "mpr_raw": float(raw_mpr),
    }


# ---------------------------------------------------------------------------
# Log-Normal fitting
# ---------------------------------------------------------------------------


def _lognormal_survival_cdf(
    K: np.ndarray, mu_ln: float, sigma_ln: float
) -> np.ndarray:
    """P(S_T > K) under Log-Normal: 1 - Phi((ln K - mu_ln) / sigma_ln)."""
    if sigma_ln <= 0:
        return np.ones_like(K) if mu_ln > np.log(K).mean() else np.zeros_like(K)
    d = (np.log(K) - mu_ln) / sigma_ln
    return 1.0 - stats.norm.cdf(d)


def fit_lognormal_from_strikes(
    strikes: list[float],
    yes_prices: list[float],
    current_price: float,
) -> tuple[float, float]:
    """Fit a Log-Normal distribution to physical survival probabilities.

    Uses Least Squares on P(S_T > K) vs observed (transformed) prices.
    Returns (mu_ln, sigma_ln).

    Falls back to a single-point drift estimate if < 2 strikes.
    """
    if len(strikes) < 2:
        # Fallback: assume log-price drift = 0, vol from historical
        return math.log(current_price), 0.3

    K = np.array(strikes, dtype=float)
    p_obs = np.array(yes_prices, dtype=float)

    # Objective: minimise squared error on survival probabilities
    def _objective(params: np.ndarray) -> float:
        mu_ln, sigma_ln = float(params[0]), float(params[1])
        if sigma_ln <= 0:
            return 1e6
        pred = _lognormal_survival_cdf(K, mu_ln, sigma_ln)
        return float(np.sum((pred - p_obs) ** 2))

    # Warm start: method-of-moments from observed quantiles
    log_strikes = np.log(K)
    # Use median and IQR for robust initial estimate
    med = float(np.median(log_strikes))
    iqr = float(np.percentile(log_strikes, 75) - np.percentile(log_strikes, 25))
    sigma0 = max(iqr / 1.349, 0.05)
    x0 = np.array([med, sigma0])

    result = optimize.minimize(_objective, x0, method="Nelder-Mead")
    mu_ln = float(result.x[0])
    sigma_ln = max(float(result.x[1]), 1e-4)

    return mu_ln, sigma_ln


# ---------------------------------------------------------------------------
# Regime probability mapping
# ---------------------------------------------------------------------------


def lognormal_to_regime_probabilities(
    mu_ln: float,
    sigma_ln: float,
    current_price: float,
    bull_threshold: float = 0.01,
    bear_threshold: float = -0.01,
) -> dict[str, float]:
    """Map fitted Log-Normal to bull/bear/sideways probabilities.

    Integrates the Log-Normal PDF over return buckets defined by
    thresholds relative to current_price.
    """
    bull_price = current_price * (1 + bull_threshold)
    bear_price = current_price * (1 + bear_threshold)

    def _cdf(price: float) -> float:
        if sigma_ln <= 0 or price <= 0:
            return 1.0 if math.log(price) >= mu_ln else 0.0
        d = (math.log(price) - mu_ln) / sigma_ln
        return float(stats.norm.cdf(d))

    # P(return <= bear_threshold) = P(S_T <= bear_price)
    prob_bear = _cdf(bear_price)

    # P(return >= bull_threshold) = P(S_T >= bull_price) = 1 - CDF(bull_price)
    prob_bull = 1.0 - _cdf(bull_price)

    # Sideways is the remainder
    prob_sideways = max(0.0, 1.0 - prob_bear - prob_bull)

    # Ensure positivity
    return {
        "bull": max(0.01, prob_bull),
        "bear": max(0.01, prob_bear),
        "sideways": max(0.01, prob_sideways),
    }


# ---------------------------------------------------------------------------
# Transition matrix nudge
# ---------------------------------------------------------------------------


def nudge_transition_matrix(
    P: np.ndarray,
    current_regime: str,
    target_terminal_dist: dict[str, float],
    horizon: int,
    quality_score: float,
) -> np.ndarray:
    """Nudge transition matrix toward a target terminal distribution.

    Finds P' close to P such that (P')^horizon[current_regime]
    approximates target_terminal_dist.  Nudge strength scales with
    market quality: strength = 0.5 * (q / 100), capped at 0.5.

    Returns a new row-stochastic matrix.
    """
    nudge_strength = min(0.5 * (quality_score / 100.0), 0.5)
    if nudge_strength <= 0 or horizon <= 0:
        return P.copy()

    n_states = P.shape[0]
    P_nudged = P.copy().astype(float)
    current_idx = STATE_INDEX[current_regime]

    # Compute current terminal distribution from P^horizon
    Ph = np.linalg.matrix_power(P, horizon)
    current_terminal = Ph[current_idx]

    # Target terminal distribution as array
    target = np.zeros(n_states)
    for state, w in target_terminal_dist.items():
        target[STATE_INDEX[state]] = w
    target = target / max(target.sum(), 1e-12)

    # Per-state delta: how much each terminal prob needs to shift
    delta = target - current_terminal

    # We want to nudge the CURRENT REGIME row of P toward a row that,
    # when raised to horizon, produces the target.
    # Approximation: adjust the current row proportionally to delta.
    # This is a heuristic nudge, not a full constrained optimisation.
    row = P_nudged[current_idx].copy()

    # Spread the nudge across columns proportionally to delta signs
    for j in range(n_states):
        if delta[j] > 0:
            # Need more mass in state j: increase P[current, j]
            row[j] += nudge_strength * delta[j]
        elif delta[j] < 0:
            # Need less mass: decrease
            row[j] += nudge_strength * delta[j]

    # Ensure non-negative and row-stochastic
    row = np.maximum(row, 0.0)
    row_sum = row.sum()
    if row_sum > 0:
        P_nudged[current_idx] = row / row_sum
    else:
        # Degenerate case: reset to uniform
        P_nudged[current_idx] = np.ones(n_states) / n_states

    return P_nudged
