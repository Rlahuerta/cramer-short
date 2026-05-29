from __future__ import annotations

from dataclasses import dataclass, field


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
    structural_break_detected: bool | None = None
    structural_break_rerun_triggered: bool | None = None
    original_structural_break_detected: bool | None = None
    original_structural_break_divergence: float | None = None


@dataclass
class WalkForwardResult:
    steps: list[BacktestStep] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class BacktestConfig:
    horizon: int = 7
    warmup: int = 120
    stride: int = 10
    ticker: str | None = None
    return_threshold_multiplier: float = 0.5
    decay_rate: float = 0.97
    break_divergence_threshold: float = 0.05
    btc_break_divergence_threshold: float | None = None
    post_break_short_window: bool | None = None
    post_break_window_size: int | None = None
    use_live_btc_short_horizon_policy: bool = False
    use_hmm: bool = False
    asset_profile: str = "crypto"
    enable_garch_vol: bool = False
    garch_horizon_cap: int | None = None
    garch_regime_ceiling: tuple[float, float] | None = None
    enable_entropy_ci_modulation: bool = False
    entropy_window_size: int = 60
    entropy_kappa: float = 0.15
