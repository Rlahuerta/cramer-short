"""BTC/USD Markov parameters ported from the xiphos optimizer.

Mirrors ``src/tools/finance/markov-distribution/xiphos-btc-profile.ts``.

Source: ``proopsios/trading/xiphos/markov/backtest/_parameter_sources.py``.
``vbparam_1d`` is the optimizer-precise 1-day profile (cramer-short's finest daily
horizon), so it is the profile used. ``vbparam_6h`` is optimizer-precise but sub-daily,
recorded only for provenance. ``break_ci_slope`` / ``break_ci_max`` / ``target_steps``
have no direct cramer-short equivalent and are not wired into the forecast.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XiphosMarkovProfile:
    """BTC/USD Markov design levers + run-config flags from the xiphos optimizer."""

    warmup_days: int
    decay_rate: float
    return_threshold_multiplier: float
    break_divergence_threshold: float
    abstention_threshold: float
    student_t_nu: int
    garch_horizon_cap: int
    garch_calm_ceiling: float
    garch_turbulent_ceiling: float
    entropy_window_size: int
    hmm_num_states: int
    meta_vol_window: int
    meta_high_vol_threshold: float
    volume_lookback: int
    volume_threshold_multiplier: float
    enable_garch_vol: bool
    enable_entropy_ci_modulation: bool
    use_hmm_regime: bool
    use_meta_regime: bool
    trajectory_n_samples: int


# xiphos vbparam_1d (BTC/USD, optimizer-precise 1-day profile).
XIPHOS_BTC_MARKOV_PROFILE_1D = XiphosMarkovProfile(
    warmup_days=54,
    decay_rate=0.9494,
    return_threshold_multiplier=0.6771,
    break_divergence_threshold=0.1165,
    abstention_threshold=0.002,
    student_t_nu=16,
    garch_horizon_cap=19,
    garch_calm_ceiling=1.3603,
    garch_turbulent_ceiling=3.097,
    entropy_window_size=38,
    hmm_num_states=2,
    meta_vol_window=40,
    meta_high_vol_threshold=0.7824,
    volume_lookback=8,
    volume_threshold_multiplier=1.4051,
    enable_garch_vol=True,
    enable_entropy_ci_modulation=True,
    use_hmm_regime=True,
    use_meta_regime=True,
    trajectory_n_samples=200,
)

# xiphos vbparam_6h (BTC/USD, optimizer-precise 6-hour profile). Provenance only.
XIPHOS_BTC_MARKOV_PROFILE_6H = XiphosMarkovProfile(
    warmup_days=50,
    decay_rate=0.93621,
    return_threshold_multiplier=0.43254,
    break_divergence_threshold=0.23088,
    abstention_threshold=0.00366,
    student_t_nu=12,
    garch_horizon_cap=10,
    garch_calm_ceiling=1.4054,
    garch_turbulent_ceiling=3.61353,
    entropy_window_size=87,
    hmm_num_states=4,
    meta_vol_window=11,
    meta_high_vol_threshold=0.7989,
    volume_lookback=45,
    volume_threshold_multiplier=1.8659,
    enable_garch_vol=True,
    enable_entropy_ci_modulation=True,
    use_hmm_regime=True,
    use_meta_regime=True,
    trajectory_n_samples=200,
)

# The BTC profile cramer-short uses (the optimizer-precise 1-day profile).
XIPHOS_BTC_MARKOV_PROFILE = XIPHOS_BTC_MARKOV_PROFILE_1D
