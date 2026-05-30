"""Forecasting engines for the Cramer-Short research package.

Models are grouped by domain:

**Markov core** (``markov/``)
    3-state regime classification, transition matrix estimation, structural
    break detection, horizon forecasting, probability calibration, action
    signal generation.  Mirrors ``src/tools/finance/markov-distribution/``.

**Trajectory & scenarios** (``trajectory.py``)
    Monte Carlo day-by-day price paths with Student-t innovations,
    survival interpolation, and scenario probability bucketing.
    Mirrors ``src/tools/finance/markov-distribution.ts``.

**Ensemble blending** (``ensemble.py``)
    Polymarket weighted ensemble: market quality scoring, YES-bias
    correction, variance/CI computation.  Mirrors ``src/utils/ensemble.ts``.

**Hidden Markov Models** (``hmm.py``)
    Gaussian HMM via hmmlearn: Baum-Welch, Viterbi, volatility HMM,
    Student-t volatility scaling.

**Volatility** (``garch.py``, ``garch_scales.py``, ``vol_regime.py``)
    GARCH(1,1) fitting and horizon-aware scaling; VIX-based regime
    classifier.

**Calibration & corrections** (``conformal.py``, ``soft_regime.py``,
``calibration_offsets.py``, ``brier_replay_calibrator.py``)
    Online conformal PID, soft-regime blending, Polymarket recalibration,
    Brier replay calibration.

**Signal extraction** (``rnd.py``, ``jump_diffusion.py``,
``logit_jump_diffusion.py``, ``hawkes.py``, ``convergence_time.py``,
``transition_entropy.py``, ``adwin.py``, ``longshot_shrinkage.py``,
``beta_hmm.py``, ``crypto_native_peers.py``)
    Risk-neutral density from Polymarket strikes, Merton jump diffusion,
    Hawkes self-exciting intensity, convergence-time signals, entropy CI
    modulation, ADWIN drift detection, longshot shrinkage, Beta-HMM.

**Policy & trust** (``forecast_policy.py``)
    Forecast trust-policy logic: when to trust/reject model predictions.
"""

from research.models.markov import (
    BtcShortHorizonLivePolicy,
    GoldShortHorizonLivePolicy,
    adjust_hmm_weight,
    calibrate_probabilities,
    cap_btc_short_horizon_confidence,
    classify_regime,
    classify_regime_series,
    compute_action_levels,
    compute_action_signal,
    compute_divergence_penalty,
    compute_markov_forecast,
    compute_prediction_confidence,
    compute_prediction_confidence_breakdown,
    compute_regime_entropy,
    DEFAULT_DIVERGENCE_PENALTY_SCHEDULE,
    detect_structural_break,
    estimate_transition_matrix,
    get_btc_short_horizon_live_policy,
    get_gold_short_horizon_live_policy,
    is_btc_ticker_symbol,
    is_gold_ticker_symbol,
    should_apply_btc14d_bearish_break_sell_gate,
    should_emit_context_only_canonical,
    soft_regime_ci_scale,
    soft_regime_confidence_multiplier,
)
from research.models.ensemble import (
    MarketInput,
    OtherSignals,
    EnsembleResult,
    adjust_yes_bias,
    compute_market_quality,
    compute_conditional_return,
    compute_polymarket_signal,
    compute_ensemble,
    compute_variance,
    compute_ci,
    compute_quality_score,
    score_to_grade,
    run_ensemble,
)
from research.models.hmm import (
    AssetProfile,
    ASSET_PROFILES,
    HMMFitResult,
    HMMParams,
    HMMPrediction,
    baum_welch,
    fit_2state_return_hmm,
    fit_volatility_hmm,
    initialize_hmm,
    mat_pow,
    predict,
    student_t_is_finite_variance,
    student_t_log_pdf,
    student_t_volatility_scale,
    viterbi,
)
from research.models.soft_regime import (
    blend_regime_mixtures,
    map_hmm_probabilities_to_regime_mixture,
    one_hot_regime_mixture,
)
from research.models.forecast_policy import (
    ForecastTrustPolicy,
    ForecastTrustPolicyInput,
    compute_forecast_trust_policy,
)
from research.models.trajectory import (
    RegimeStats,
    TrajectoryPoint,
    compute_trajectory,
    compute_horizon_drift_vol,
    compute_mixing_weight,
    compute_scenario_probabilities,
    interpolate_distribution,
    normal_cdf,
    student_t_cdf,
    student_t_ppf,
    student_t_survival,
    log_normal_survival,
    interpolate_survival,
)

__all__ = [
    "adjust_hmm_weight",
    "calibrate_probabilities",
    "cap_btc_short_horizon_confidence",
    "classify_regime",
    "classify_regime_series",
    "compute_action_levels",
    "compute_action_signal",
    "compute_divergence_penalty",
    "compute_markov_forecast",
    "compute_prediction_confidence",
    "compute_prediction_confidence_breakdown",
    "compute_regime_entropy",
    "DEFAULT_DIVERGENCE_PENALTY_SCHEDULE",
    "detect_structural_break",
    "estimate_transition_matrix",
    "soft_regime_ci_scale",
    "soft_regime_confidence_multiplier",
    "BtcShortHorizonLivePolicy",
    "GoldShortHorizonLivePolicy",
    "get_btc_short_horizon_live_policy",
    "get_gold_short_horizon_live_policy",
    "is_btc_ticker_symbol",
    "is_gold_ticker_symbol",
    "should_apply_btc14d_bearish_break_sell_gate",
    "should_emit_context_only_canonical",
    "MarketInput",
    "OtherSignals",
    "EnsembleResult",
    "adjust_yes_bias",
    "compute_market_quality",
    "compute_conditional_return",
    "compute_polymarket_signal",
    "compute_ensemble",
    "compute_variance",
    "compute_ci",
    "compute_quality_score",
    "score_to_grade",
    "run_ensemble",
    "RegimeStats",
    "TrajectoryPoint",
    "compute_trajectory",
    "compute_horizon_drift_vol",
    "compute_mixing_weight",
    "compute_scenario_probabilities",
    "interpolate_distribution",
    "normal_cdf",
    "student_t_cdf",
    "student_t_ppf",
    "student_t_survival",
    "log_normal_survival",
    "interpolate_survival",
    "AssetProfile",
    "ASSET_PROFILES",
    "HMMFitResult",
    "HMMParams",
    "HMMPrediction",
    "baum_welch",
    "fit_2state_return_hmm",
    "fit_volatility_hmm",
    "initialize_hmm",
    "mat_pow",
    "predict",
    "student_t_is_finite_variance",
    "student_t_log_pdf",
    "student_t_volatility_scale",
    "viterbi",
    "blend_regime_mixtures",
    "map_hmm_probabilities_to_regime_mixture",
    "one_hot_regime_mixture",
    "ForecastTrustPolicy",
    "ForecastTrustPolicyInput",
    "compute_forecast_trust_policy",
]
