"""Price trajectory and scenario probability computation.

Mirrors TS logic from src/tools/finance/markov-distribution.ts:
  - Monte Carlo day-by-day paths with Student-t innovations
  - Regime-weighted drift/vol via matrix powers
  - Survival interpolation and scenario bucketing
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from research.models.markov import (
    NUM_STATES,
    STATE_INDEX,
    RegimeState,
    REGIME_STATES,
)
from research.models.jump_diffusion import JumpEventSpec, jump_drift_compensator


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    mean_return: float
    std_return: float


@dataclass
class TrajectoryPoint:
    day: int
    expected_price: float
    lower_bound: float
    upper_bound: float
    p_up: float
    cumulative_return: str
    regime: RegimeState


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return float(stats.norm.cdf(x))


def student_t_cdf(x: float, nu: int = 5) -> float:
    """Student-t CDF."""
    return float(stats.t.cdf(x, df=nu))


def student_t_ppf(p: float, nu: int = 5) -> float:
    """Student-t quantile (inverse CDF)."""
    return float(stats.t.ppf(p, df=nu))


def student_t_survival(
    target_price: float,
    current_price: float,
    mu_n: float,
    sigma_n: float,
    nu: int = 5,
) -> float:
    """P(price > target) under log-Student-t model.

    Uses the approximation: log(price/current) ~ t(μ, σ, ν).
    """
    if sigma_n <= 0:
        return 1.0 if mu_n > math.log(target_price / current_price) else 0.0
    z = (math.log(target_price / current_price) - mu_n) / sigma_n
    return 1.0 - student_t_cdf(z, nu)


def log_normal_survival(
    current_price: float,
    target_price: float,
    drift: float,
    vol: float,
) -> float:
    """P(price > target) under log-normal model."""
    if vol <= 0:
        return 1.0 if drift > math.log(target_price / current_price) else 0.0
    d = (math.log(target_price / current_price) - drift) / vol
    return 1.0 - normal_cdf(d)


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def _mat_pow(P: np.ndarray, n: int) -> np.ndarray:
    """Matrix power P^n."""
    if n <= 0:
        return np.eye(P.shape[0])
    return np.linalg.matrix_power(P, n)


# ---------------------------------------------------------------------------
# Horizon drift/vol
# ---------------------------------------------------------------------------

def compute_horizon_drift_vol(
    horizon: int,
    P: np.ndarray,
    regime_stats: dict[RegimeState, RegimeStats],
    initial_state: RegimeState,
    momentum_adjustment: float = 0.0,
    start_mixture: dict[RegimeState, float] | None = None,
    hmm_override: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute regime-weighted drift and vol at a given horizon.

    mu_n = horizon * (mu_eff + momentum_adjustment)
    sigma_n = sigma_eff * sqrt(horizon)
    """
    Pn = _mat_pow(P, horizon)

    if start_mixture:
        state_weights = np.zeros(NUM_STATES)
        for state, w in start_mixture.items():
            idx = STATE_INDEX[state]
            state_weights += w * Pn[idx]
    else:
        state_weights = Pn[STATE_INDEX[initial_state]]

    mu_obs = sum(
        state_weights[i] * regime_stats[state].mean_return
        for i, state in enumerate(REGIME_STATES)
    )

    # Mixture variance: E[sigma^2] + Var(mu)
    var_of_means = sum(
        state_weights[i] * (regime_stats[state].mean_return - mu_obs) ** 2
        for i, state in enumerate(REGIME_STATES)
    )
    mixture_sigma = math.sqrt(
        sum(
            state_weights[i] * regime_stats[state].std_return ** 2
            for i, state in enumerate(REGIME_STATES)
        )
        + var_of_means
    )

    mu_eff = mu_obs
    sigma_eff = mixture_sigma

    mu_n = horizon * (mu_eff + momentum_adjustment)
    sigma_n = sigma_eff * math.sqrt(horizon)

    if hmm_override:
        w = hmm_override.get("weight", 0.0)
        hmm_drift = hmm_override.get("drift", mu_eff)
        hmm_vol = hmm_override.get("vol", sigma_eff)
        mu_n = w * (horizon * hmm_drift) + (1 - w) * mu_n
        sigma_n = w * (hmm_vol * math.sqrt(horizon)) + (1 - w) * sigma_n

    return {
        "mu_n": mu_n,
        "sigma_n": sigma_n,
    }


# ---------------------------------------------------------------------------
# Trajectory computation
# ---------------------------------------------------------------------------

def compute_trajectory(
    current_price: float,
    days: int,
    P: np.ndarray,
    regime_stats: dict[RegimeState, RegimeStats],
    initial_state: RegimeState,
    momentum_adjustment: float = 0.0,
    n_samples: int = 1000,
    nu: int = 5,
    empirical_daily_vol: float | None = None,
    start_mixture: dict[RegimeState, float] | None = None,
    hmm_override: dict[str, float] | None = None,
    jump_spec: list[JumpEventSpec] | None = None,
    garch_scales: list[float] | None = None,
) -> list[TrajectoryPoint]:
    """Compute day-by-day price trajectory via Monte Carlo.

    Uses a SINGLE shared set of MC paths sampled at each day, ensuring
    monotonically widening CIs and ~7x speedup over independent simulations.

    When ``jump_spec`` is None or empty the inner MC loop is byte-identical
    to the pre-Idea-2 implementation — no extra RNG draws are consumed.
    """
    initial_idx = STATE_INDEX[initial_state]
    trajectory: list[TrajectoryPoint] = []

    # Pre-compute regime weights per day via matrix powers
    regime_weights_per_day: list[np.ndarray] = []
    for d in range(1, days + 1):
        Pd = _mat_pow(P, d)
        if start_mixture:
            weights = np.zeros(NUM_STATES)
            for state, w in start_mixture.items():
                idx = STATE_INDEX[state]
                weights += w * Pd[idx]
            regime_weights_per_day.append(weights)
        else:
            regime_weights_per_day.append(Pd[initial_idx])

    # Compute per-day mixture drift and vol from regime weights
    daily_drifts = np.zeros(days)
    daily_vols = np.zeros(days)

    for d in range(days):
        weights = regime_weights_per_day[d]

        mu_obs = sum(
            weights[i] * regime_stats[state].mean_return
            for i, state in enumerate(REGIME_STATES)
        )

        var_of_means = sum(
            weights[i] * (regime_stats[state].mean_return - mu_obs) ** 2
            for i, state in enumerate(REGIME_STATES)
        )
        expected_var = sum(
            weights[i] * regime_stats[state].std_return ** 2
            for i, state in enumerate(REGIME_STATES)
        )
        sigma_obs = math.sqrt(expected_var + var_of_means)

        # Apply momentum adjustment per-day
        mu_obs += momentum_adjustment

        # Apply HMM override per-day (HMM drift/vol are daily quantities)
        if hmm_override:
            w = hmm_override.get("weight", 0.0)
            hmm_drift = hmm_override.get("drift", mu_obs)
            hmm_vol = hmm_override.get("vol", sigma_obs)
            mu_obs = w * hmm_drift + (1 - w) * mu_obs
            sigma_obs = w * hmm_vol + (1 - w) * sigma_obs

        # Use empirical vol as floor when provided
        if empirical_daily_vol:
            sigma_obs = max(sigma_obs, empirical_daily_vol)

        daily_drifts[d] = mu_obs
        daily_vols[d] = sigma_obs

    if garch_scales:
        for d in range(min(days, len(garch_scales))):
            k = garch_scales[d]
            if math.isfinite(k) and k > 0:
                daily_vols[d] *= k

    # Idea 2 — Merton drift compensator applied once.  Empty/None ⇒ 0 ⇒ no-op.
    has_jumps = bool(jump_spec)
    if has_jumps:
        compensator = jump_drift_compensator(jump_spec)
        daily_drifts -= compensator

    # Run shared Monte Carlo with per-day mixture drift/vol
    paths = np.zeros((n_samples, days))
    for s in range(n_samples):
        cum_log_return = 0.0
        for d in range(days):
            u = np.random.random()
            z = student_t_ppf(u, nu)
            scaled_vol = daily_vols[d] * math.sqrt((nu - 2) / nu) if nu > 2 else daily_vols[d]
            cum_log_return += daily_drifts[d] + z * scaled_vol

            # Jump term — only consume RNG when at least one event exists
            if has_jumps:
                for e in jump_spec:
                    if np.random.random() < e.daily_intensity:
                        u1 = max(1e-12, np.random.random())
                        u2 = np.random.random()
                        z_j = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                        cum_log_return += e.mean_log_jump + z_j * e.std_log_jump

            paths[s, d] = cum_log_return

    for d in range(1, days + 1):
        day_idx = d - 1
        state_weights = regime_weights_per_day[day_idx]

        # Cumulative drift and vol from daily arrays
        mu_n = float(np.sum(daily_drifts[:d]))
        sigma_n = math.sqrt(float(np.sum(daily_vols[:d] ** 2)))

        # Prices from MC paths
        prices = current_price * np.exp(paths[:, day_idx])
        prices_sorted = np.sort(prices)

        p5_idx = max(0, int(n_samples * 0.05) - 1)
        p50_idx = int(n_samples * 0.5)
        p95_idx = min(n_samples - 1, int(np.ceil(n_samples * 0.95)))

        lower_bound = float(prices_sorted[p5_idx])
        upper_bound = float(prices_sorted[p95_idx])

        # Expected price: MC median when empirical vol used, else analytical
        if empirical_daily_vol:
            expected_price = float(prices_sorted[p50_idx])
        else:
            expected_price = current_price * math.exp(mu_n)

        # P(up) from Student-t survival
        p_up = student_t_survival(current_price, current_price, mu_n, sigma_n, nu)

        # Cumulative return
        ret = (expected_price - current_price) / current_price
        cumulative_return = f"{(ret * 100):+.1f}%"

        # Most likely regime at this horizon
        max_weight = -1.0
        regime: RegimeState = initial_state
        for i, state in enumerate(REGIME_STATES):
            if state_weights[i] > max_weight:
                max_weight = float(state_weights[i])
                regime = state

        trajectory.append(
            TrajectoryPoint(
                day=d,
                expected_price=round(expected_price, 2),
                lower_bound=round(lower_bound, 2),
                upper_bound=round(upper_bound, 2),
                p_up=round(p_up, 3),
                cumulative_return=cumulative_return,
                regime=regime,
            )
        )

    return trajectory


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_survival(
    distribution: list[dict],
    target_price: float,
) -> float:
    """Linearly interpolate P(price > target_price) from a survival table.

    Parameters
    ----------
    distribution : list[dict]
        List of {price: float, probability: float} sorted ascending by price.
        probability must be non-increasing.
    target_price : float
        Target price level.
    """
    if not distribution:
        return 0.5
    prices = [d["price"] for d in distribution]
    probs = [d["probability"] for d in distribution]

    if target_price <= prices[0]:
        return 1.0
    if target_price >= prices[-1]:
        return 0.0

    for i in range(len(prices) - 1):
        lo_price = prices[i]
        hi_price = prices[i + 1]
        if lo_price <= target_price <= hi_price:
            t = (target_price - lo_price) / (hi_price - lo_price)
            return probs[i] + t * (probs[i + 1] - probs[i])

    return 0.0


# ---------------------------------------------------------------------------
# Scenario probabilities
# ---------------------------------------------------------------------------

def compute_scenario_probabilities(
    distribution: list[dict],
    current_price: float,
) -> dict:
    """Compute scenario probability buckets from a calibrated CDF.

    Buckets: Down >5%, Down 3-5%, Flat +/-3%, Up 3-5%, Up >5%.
    """
    down5 = current_price * 0.95
    down3 = current_price * 0.97
    up3 = current_price * 1.03
    up5 = current_price * 1.05

    p_above_down5 = interpolate_survival(distribution, down5)
    p_above_down3 = interpolate_survival(distribution, down3)
    p_above_up3 = interpolate_survival(distribution, up3)
    p_above_up5 = interpolate_survival(distribution, up5)

    p_down_over5 = 1.0 - p_above_down5
    p_down_3to5 = p_above_down5 - p_above_down3
    p_flat = p_above_down3 - p_above_up3
    p_up_3to5 = p_above_up3 - p_above_up5
    p_up_over5 = p_above_up5

    # Expected return via trapezoidal integration of survival
    expected_price = current_price
    if len(distribution) >= 2:
        integral = 0.0
        for i in range(len(distribution) - 1):
            dx = distribution[i + 1]["price"] - distribution[i]["price"]
            avg_p = (distribution[i]["probability"] + distribution[i + 1]["probability"]) / 2
            integral += avg_p * dx
        expected_price = distribution[0]["price"] + integral

    expected_return = (expected_price - current_price) / current_price

    return {
        "buckets": [
            {"label": "Down >5%", "probability": max(0.0, p_down_over5), "range": [None, round(down5, 2)]},
            {"label": "Down 3-5%", "probability": max(0.0, p_down_3to5), "range": [round(down5, 2), round(down3, 2)]},
            {"label": "Flat +/-3%", "probability": max(0.0, p_flat), "range": [round(down3, 2), round(up3, 2)]},
            {"label": "Up 3-5%", "probability": max(0.0, p_up_3to5), "range": [round(up3, 2), round(up5, 2)]},
            {"label": "Up >5%", "probability": max(0.0, p_up_over5), "range": [round(up5, 2), None]},
        ],
        "expected_price": round(expected_price, 2),
        "expected_return": round(expected_return, 4),
        "p_up": round(interpolate_survival(distribution, current_price), 3),
    }
