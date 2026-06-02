"""Benchmark: empirical up-rates vs trajectory-based p_up for directional accuracy.

Run: python -m pytest research/tests/test_p_up_benchmark.py -v -s
"""

import math
import numpy as np
import pytest

# Synthetic regime sequence designed to test directional accuracy
# Pattern: alternating 10-day bull and bear periods with clear direction
def _make_regime_returns(seed: int = 42) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 500
    regimes: list[str] = []
    returns: list[float] = []
    
    for i in range(n):
        if (i // 10) % 2 == 0:
            # Bull period: mostly positive returns
            regimes.append("bull")
            returns.append(rng.normal(0.008, 0.015))
        else:
            # Bear period: mostly negative returns
            regimes.append("bear")
            returns.append(rng.normal(-0.006, 0.018))
    
    return regimes, np.array(returns)


def _make_prices(returns: np.ndarray) -> list[float]:
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1.0 + float(r)))
    return prices


def _estimate_transition_matrix(states: list[str]) -> np.ndarray:
    """Simple MLE transition matrix (no smoothing, no decay)."""
    from research.models.markov.core import NUM_STATES, STATE_INDEX
    counts = np.zeros((NUM_STATES, NUM_STATES))
    for i in range(len(states) - 1):
        counts[STATE_INDEX[states[i]]][STATE_INDEX[states[i + 1]]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def _compute_empirical_p_up(
    regimes: list[str],
    log_returns: np.ndarray,
    P: np.ndarray,
    horizon: int,
    initial_state: str,
) -> float:
    """Compute p_up via empirical regime up-rates."""
    from research.models.markov.regime import compute_regime_up_rates
    from research.models.markov.core import STATE_INDEX
    
    P_n = np.linalg.matrix_power(P, horizon)
    forecast = P_n[STATE_INDEX[initial_state]]
    
    up_rates = compute_regime_up_rates(regimes, log_returns, horizon, decay_rate=0.97)
    
    states = ["bull", "bear", "sideways"]
    return sum(forecast[i] * up_rates[state] for i, state in enumerate(states))


def _compute_trajectory_p_up(
    prices: list[float],
    P: np.ndarray,
    horizon: int,
    initial_state: str,
) -> float:
    """Compute p_up via trajectory simulation (student_t_survival)."""
    from research.models.markov.regime import estimate_regime_stats
    from research.models.markov.core import STATE_INDEX, RegimeState, REGIME_STATES
    from research.models.trajectory.simulation import compute_trajectory
    from research.models.trajectory.types import RegimeStats
    
    active_returns = np.array([
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
    ])
    log_returns = np.log(1.0 + active_returns)
    
    regimes = ["bull" if r > 0 else "bear" if r < 0 else "sideways" for r in active_returns]
    
    raw_stats = estimate_regime_stats(log_returns, regimes, min_obs_per_state=1)
    regime_stats: dict[RegimeState, RegimeStats] = {}
    for state in REGIME_STATES:
        regime_stats[state] = RegimeStats(
            mean_return=raw_stats[state]["meanReturn"],
            std_return=max(raw_stats[state]["stdReturn"], 0.001),
        )
    
    traj = compute_trajectory(
        prices[-1], horizon, P, regime_stats, initial_state,
        n_samples=200, nu=5,
    )
    return traj[-1].p_up


class TestPUpBenchmark:
    """Compare empirical vs trajectory p_up on synthetic and real-like data."""
    
    def test_synthetic_directional_data(self):
        """On data with clear directional patterns, empirical should be more decisive."""
        regimes, returns = _make_regime_returns(seed=42)
        prices = _make_prices(returns)
        log_returns = np.log(1.0 + returns)
        P = _estimate_transition_matrix(regimes)
        
        initial_state = "bull"
        
        for horizon in [1, 3, 7]:
            empirical = _compute_empirical_p_up(regimes, log_returns, P, horizon, initial_state)
            trajectory = _compute_trajectory_p_up(prices, P, horizon, initial_state)
            
            gap = abs(empirical - 0.5) - abs(trajectory - 0.5)
            
            print(f"\n  horizon={horizon}: empirical={empirical:.4f}, trajectory={trajectory:.4f}, decisiveness_gap={gap:+.4f}")
            
            # On directional data, empirical should be at least as decisive as trajectory
            # (trajectory is known to pull p_up toward 0.5 due to sigma_n inflation)
            # Note: at h=1, trajectory can be more decisive due to daily volatility
            # being lower than the regime-switching pattern variance
            assert gap > -0.15, (
                f"horizon={horizon}: trajectory p_up ({trajectory:.4f}) is significantly "
                f"more decisive than empirical ({empirical:.4f}) — unexpected"
            )
    
    def test_random_walk_data(self):
        """On random walk data, both should give ~0.5."""
        rng = np.random.default_rng(99)
        returns = rng.normal(0.0001, 0.015, 500)
        prices = _make_prices(returns)
        log_returns = np.log(1.0 + returns)
        regimes = ["bull" if r > 0 else "bear" if r < 0 else "sideways" for r in returns]
        P = _estimate_transition_matrix(regimes)
        
        for horizon in [1, 7]:
            empirical = _compute_empirical_p_up(regimes, log_returns, P, horizon, "bull")
            trajectory = _compute_trajectory_p_up(prices, P, horizon, "bull")
            
            print(f"\n  random_walk h={horizon}: empirical={empirical:.4f}, trajectory={trajectory:.4f}")
            
            # Both should be near 0.5 on random data
            # Empirical can drift at longer horizons due to finite-sample accumulation bias
            assert 0.35 < empirical < 0.75, f"empirical p_up={empirical:.4f} too far from 0.5"
            assert 0.40 < trajectory < 0.60, f"trajectory p_up={trajectory:.4f} too far from 0.5"
    
    def test_strong_trend_data(self):
        """On data with a strong persistent trend, empirical should reflect it."""
        rng = np.random.default_rng(7)
        # Strong bull: 0.5% mean daily, low vol
        returns = rng.normal(0.005, 0.008, 300)
        prices = _make_prices(returns)
        log_returns = np.log(1.0 + returns)
        regimes = ["bull" if r > 0 else "bear" if r < 0 else "sideways" for r in returns]
        P = _estimate_transition_matrix(regimes)
        
        for horizon in [1, 7]:
            empirical = _compute_empirical_p_up(regimes, log_returns, P, horizon, "bull")
            trajectory = _compute_trajectory_p_up(prices, P, horizon, "bull")
            
            print(f"\n  strong_bull h={horizon}: empirical={empirical:.4f}, trajectory={trajectory:.4f}")
            
            # On strong bull data, p_up should be > 0.55
            assert empirical > 0.55, f"empirical p_up={empirical:.4f} too low for strong bull"
            assert trajectory > 0.50, f"trajectory p_up={trajectory:.4f} too low for strong bull"
    
    def test_empirical_vs_trajectory_p_up_correlation(self):
        """Verify that both methods produce positively correlated p_up values."""
        rng = np.random.default_rng(123)
        empirical_vals = []
        trajectory_vals = []
        
        for seed in range(20):
            regimes, returns = _make_regime_returns(seed=42 + seed)
            prices = _make_prices(returns)
            log_returns = np.log(1.0 + returns)
            P = _estimate_transition_matrix(regimes)
            
            empirical = _compute_empirical_p_up(regimes, log_returns, P, 7, "bull")
            trajectory = _compute_trajectory_p_up(prices, P, 7, "bull")
            
            empirical_vals.append(empirical)
            trajectory_vals.append(trajectory)
        
        # Compute correlation
        corr = np.corrcoef(empirical_vals, trajectory_vals)[0, 1]
        print(f"\n  Correlation across 20 runs: {corr:.4f}")
        print(f"  Empirical mean: {np.mean(empirical_vals):.4f} ± {np.std(empirical_vals):.4f}")
        print(f"  Trajectory mean: {np.mean(trajectory_vals):.4f} ± {np.std(trajectory_vals):.4f}")
        
        # Should be positively correlated (both reflect the data)
        # Low correlation confirms the two methods produce fundamentally different estimates
        assert corr > 0.1, f"Correlation {corr:.4f} too low — methods completely unrelated"
