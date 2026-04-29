"""Walk-forward backtest harness.

Slides a window over historical prices, estimates the Markov model at each
step, records predictions vs realised outcomes, and aggregates results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from research.data.prices import fetch_historical_prices
from research.models.markov import (
    classify_regime_series,
    estimate_transition_matrix,
    compute_markov_forecast,
)
from research.models.garch_scales import GarchClampOptions, compute_garch_scales
from research.models.hmm import ASSET_PROFILES, baum_welch, fit_volatility_hmm, predict
from research.models.transition_entropy import (
    EntropyZScoreTracker,
    compute_transition_entropy,
    entropy_z_to_ci_scale,
)
from research.models.trajectory import compute_trajectory, RegimeStats


@dataclass
class BacktestStep:
    start_idx: int
    predicted_prob: float
    predicted_return: float
    ci_lower: float
    ci_upper: float
    realised_return: float
    realised_price: float
    direction_correct: bool
    in_ci: bool
    garch_vol_applied: bool | None = None
    transition_entropy: float | None = None
    transition_entropy_norm: float | None = None
    transition_entropy_z: float | None = None
    entropy_ci_scale: float | None = None
    entropy_ci_modulation_applied: bool | None = None


@dataclass
class WalkForwardResult:
    steps: list[BacktestStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def walk_forward(
    prices: list[float],
    horizon: int = 7,
    warmup: int = 120,
    stride: int = 10,
    return_threshold_multiplier: float = 0.5,
    decay_rate: float = 0.97,
    use_hmm: bool = False,
    asset_profile: str = "crypto",
    enable_garch_vol: bool = False,
    garch_horizon_cap: int | None = None,
    garch_regime_ceiling: tuple[float, float] | None = None,
    enable_entropy_ci_modulation: bool = False,
    entropy_window_size: int = 60,
    entropy_kappa: float = 0.15,
) -> WalkForwardResult:
    """Run a walk-forward backtest on a price series.

    Parameters
    ----------
    prices : list[float]
        Historical close prices (oldest first).
    horizon : int
        Forecast horizon in days.
    warmup : int
        Minimum history for regime estimation.
    stride : int
        Days between consecutive test windows.
    return_threshold_multiplier : float
        Adaptive threshold multiplier for regime classification.
    decay_rate : float
        Exponential decay for transition matrix weighting.
    use_hmm : bool
        Whether to blend HMM predictions into the forecast.
    asset_profile : str
        Asset profile key (etf, equity, crypto, commodity) for HMM weight tuning.

    Returns
    -------
    WalkForwardResult
        Steps and any errors encountered.
    """
    result = WalkForwardResult()
    entropy_tracker = EntropyZScoreTracker(max(5, entropy_window_size))

    if len(prices) < warmup + horizon + 10:
        result.errors.append(
            f"Insufficient data: {len(prices)} prices, need {warmup + horizon + 10}"
        )
        return result

    returns = [
        (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
    ]

    for start in range(warmup, len(prices) - horizon, stride):
        try:
            window_returns = np.array(returns[start - warmup : start])
            current_price = prices[start]
            realised_price = prices[start + horizon]
            realised_return = (realised_price - current_price) / current_price

            regimes = classify_regime_series(
                window_returns, return_threshold_multiplier=return_threshold_multiplier
            )
            P = estimate_transition_matrix(regimes, decay_rate=decay_rate)
            current_regime = regimes[-1] if regimes else "sideways"

            forecast = compute_markov_forecast(P, current_regime, horizon)
            entropy = compute_transition_entropy(P)
            entropy_z: float | None = None
            entropy_ci_scale = 1.0
            if enable_entropy_ci_modulation:
                entropy_z = entropy_tracker.z_score(entropy.entropy_norm)
                if entropy_z is not None:
                    entropy_ci_scale = entropy_z_to_ci_scale(entropy_z, entropy_kappa)

            # Empirical regime stats from the window
            regime_stats: dict[str, RegimeStats] = {}
            for state in ["bull", "bear", "sideways"]:
                mask = [r == state for r in regimes]
                if any(mask):
                    state_returns = window_returns[mask]
                    regime_stats[state] = RegimeStats(
                        mean_return=float(np.mean(state_returns)),
                        std_return=float(np.std(state_returns, ddof=1)) if len(state_returns) > 1 else 0.01,
                    )
                else:
                    regime_stats[state] = RegimeStats(mean_return=0.0, std_return=0.01)

            # Combined P(up) from regime probabilities × up-rates
            # Simple up-rate: fraction of positive returns in each regime
            up_rates: dict[str, float] = {}
            for state in ["bull", "bear", "sideways"]:
                mask = [r == state for r in regimes]
                if any(mask):
                    state_returns = window_returns[mask]
                    up_rates[state] = float(np.mean(state_returns > 0))
                else:
                    up_rates[state] = 0.5

            p_up = sum(forecast[s] * up_rates[s] for s in ["bull", "bear", "sideways"])

            # Optional HMM enhancement
            hmm_override: dict[str, float] | None = None
            if use_hmm:
                hmm_result = baum_welch(
                    window_returns,
                    n_states=3,
                    max_iterations=50,
                    tolerance=1e-3,
                )
                if hmm_result.converged:
                    hmm_pred = predict(window_returns, hmm_result.params, forecast_horizon=horizon)
                    vol_scale = fit_volatility_hmm(window_returns, vol_window=5, n_states=2)
                    profile = ASSET_PROFILES.get(asset_profile, ASSET_PROFILES["crypto"])
                    hmm_weight = np.clip(profile.hmm_weight_multiplier * 0.5, 0.0, 1.0)
                    hmm_override = {
                        "drift": hmm_pred.expected_return,
                        "vol": hmm_pred.expected_volatility * vol_scale,
                        "weight": float(hmm_weight),
                    }
                else:
                    result.errors.append(f"Step {start}: HMM did not converge; using pure Markov forecast")

            garch_scales: list[float] | None = None
            garch_vol_applied = False
            if enable_garch_vol:
                log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, start + 1)]
                opts: GarchClampOptions | None = None
                if garch_horizon_cap is not None or garch_regime_ceiling is not None:
                    opts = GarchClampOptions(
                        horizon_cap=garch_horizon_cap,
                        ceiling=garch_regime_ceiling or (1.5, 3.0),
                    )
                scales = compute_garch_scales(log_returns, horizon, opts)
                if scales:
                    garch_scales = scales
                    garch_vol_applied = True

            # Generate trajectory with MC-based CIs (Student-t, not Gaussian)
            traj = compute_trajectory(
                current_price,
                horizon,
                P,
                regime_stats,
                current_regime,
                hmm_override=hmm_override,
                n_samples=500,
                garch_scales=garch_scales,
            )
            horizon_point = traj[-1]
            predicted_return = (horizon_point.expected_price - current_price) / current_price
            ci_lower = horizon_point.lower_bound
            ci_upper = horizon_point.upper_bound
            if entropy_ci_scale != 1.0:
                center = (ci_lower + ci_upper) / 2.0
                half_width = (ci_upper - ci_lower) / 2.0
                ci_lower = center - half_width * entropy_ci_scale
                ci_upper = center + half_width * entropy_ci_scale

            direction_correct = (p_up > 0.5 and realised_return > 0) or (
                p_up <= 0.5 and realised_return <= 0
            )
            in_ci = ci_lower <= realised_price <= ci_upper

            result.steps.append(
                BacktestStep(
                    start_idx=start,
                    predicted_prob=float(p_up),
                    predicted_return=float(predicted_return),
                    ci_lower=float(ci_lower),
                    ci_upper=float(ci_upper),
                    realised_return=float(realised_return),
                    realised_price=float(realised_price),
                    direction_correct=bool(direction_correct),
                    in_ci=bool(in_ci),
                    garch_vol_applied=bool(garch_vol_applied),
                    transition_entropy=float(entropy.entropy_nats),
                    transition_entropy_norm=float(entropy.entropy_norm),
                    transition_entropy_z=None if entropy_z is None else float(entropy_z),
                    entropy_ci_scale=float(entropy_ci_scale),
                    entropy_ci_modulation_applied=bool(abs(entropy_ci_scale - 1.0) > 1e-12),
                )
            )
            entropy_tracker.push(entropy.entropy_norm)
        except Exception as e:
            result.errors.append(f"Step {start}: {e}")

    return result
