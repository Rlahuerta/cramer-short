"""Markov regime model — decomposed package.

Backwards-compatible re-export barrel for all sub-modules.
"""

from research.models.markov.action import (
    compute_action_levels,
    compute_action_signal,
)
from research.models.markov.calibration import calibrate_probabilities
from research.models.markov.confidence import (
    compute_prediction_confidence,
    compute_prediction_confidence_breakdown,
)
from research.models.markov.core import (
    ActionConfidence,
    ActionRecommendation,
    BreakConfidencePolicy,
    BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
    BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
    BTC_SHORT_HORIZON_LIVE_RERUN_WINDOW_DAYS,
    COMMODITY_MODEL_ONLY_MIN_CONFIDENCE,
    COMMODITY_MODEL_ONLY_MIN_R2,
    CRYPTO_MODEL_ONLY_MIN_CONFIDENCE,
    CRYPTO_MODEL_ONLY_MIN_R2,
    DEFAULT_DIVERGENCE_PENALTY_SCHEDULE,
    FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS,
    GOLD_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
    GOLD_SHORT_HORIZON_LIVE_HISTORY_DAYS,
    GOLD_ULTRA_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
    NUM_STATES,
    PredictionConfidenceMode,
    PROMOTED_HYPE_MARKOV_RUNTIME_DEFAULTS,
    PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS,
    REGIME_STATES,
    RegimeState,
    STATE_INDEX,
)
from research.models.markov.entropy import (
    adjust_hmm_weight,
    compute_regime_entropy,
    soft_regime_ci_scale,
    soft_regime_confidence_multiplier,
)
from research.models.markov.forecast import compute_markov_forecast
from research.models.markov.gates import (
    cap_btc_short_horizon_confidence,
    compute_divergence_penalty,
    evaluate_can_emit_canonical,
    should_apply_btc14d_bearish_break_sell_gate,
    should_emit_context_only_canonical,
)
from research.models.markov.validation import (
    is_composite_validation_acceptable,
)
from research.models.markov.policies import (
    BtcShortHorizonLivePolicy,
    GoldShortHorizonLivePolicy,
    get_btc_short_horizon_live_policy,
    get_forecast_lab_markov_runtime_defaults,
    get_gold_short_horizon_live_policy,
    is_btc_ticker_symbol,
    is_gold_ticker_symbol,
    resolve_forecast_lab_markov_parameter_defaults,
    set_forecast_lab_markov_runtime_defaults,
)
from research.models.markov.regime import (
    classify_regime,
    classify_regime_series,
    estimate_regime_stats,
)
from research.models.markov.transition import (
    _default_matrix,
    detect_structural_break,
    estimate_transition_matrix,
)
from research.models.markov.validation import (
    compute_r2_os,
    compute_validation_r2_os,
    evaluate_model_only_bypass,
    evaluate_validation_acceptability,
)

__all__ = [
    # core
    "RegimeState",
    "PredictionConfidenceMode",
    "BreakConfidencePolicy",
    "ActionRecommendation",
    "ActionConfidence",
    "REGIME_STATES",
    "STATE_INDEX",
    "NUM_STATES",
    "BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS",
    "BTC_SHORT_HORIZON_LIVE_RERUN_WINDOW_DAYS",
    "BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT",
    "GOLD_SHORT_HORIZON_LIVE_HISTORY_DAYS",
    "GOLD_ULTRA_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT",
    "GOLD_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT",
    "FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS",
    "PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS",
    "PROMOTED_HYPE_MARKOV_RUNTIME_DEFAULTS",
    "DEFAULT_DIVERGENCE_PENALTY_SCHEDULE",
    "COMMODITY_MODEL_ONLY_MIN_R2",
    "COMMODITY_MODEL_ONLY_MIN_CONFIDENCE",
    "CRYPTO_MODEL_ONLY_MIN_R2",
    "CRYPTO_MODEL_ONLY_MIN_CONFIDENCE",
    # policies
    "BtcShortHorizonLivePolicy",
    "GoldShortHorizonLivePolicy",
    "is_btc_ticker_symbol",
    "is_gold_ticker_symbol",
    "get_btc_short_horizon_live_policy",
    "get_gold_short_horizon_live_policy",
    "resolve_forecast_lab_markov_parameter_defaults",
    "get_forecast_lab_markov_runtime_defaults",
    "set_forecast_lab_markov_runtime_defaults",
    # regime
    "classify_regime",
    "classify_regime_series",
    "estimate_regime_stats",
    # transition
    "estimate_transition_matrix",
    "detect_structural_break",
    # forecast
    "compute_markov_forecast",
    # entropy
    "compute_regime_entropy",
    "soft_regime_confidence_multiplier",
    "soft_regime_ci_scale",
    "adjust_hmm_weight",
    # validation
    "compute_r2_os",
    "compute_validation_r2_os",
    "evaluate_validation_acceptability",
    "is_composite_validation_acceptable",
    "evaluate_model_only_bypass",
    # confidence
    "compute_prediction_confidence_breakdown",
    "compute_prediction_confidence",
    # calibration
    "calibrate_probabilities",
    # action
    "compute_action_levels",
    "compute_action_signal",
    # gates
    "compute_divergence_penalty",
    "evaluate_can_emit_canonical",
    "should_emit_context_only_canonical",
    "cap_btc_short_horizon_confidence",
    "should_apply_btc14d_bearish_break_sell_gate",
]
