"""Core types, constants, and helpers for the Markov regime model.

Mirrors TS logic:
  - 3-state regime classification (bull/bear/sideways)
  - State indexing and constants
"""

from __future__ import annotations

from typing import Literal

RegimeState = Literal["bull", "bear", "sideways"]
PredictionConfidenceMode = Literal["legacy", "rebalanced"]
BreakConfidencePolicy = Literal["default", "trend_penalty_only", "divergence_weighted"]
ActionRecommendation = Literal["BUY", "HOLD", "SELL"]
ActionConfidence = Literal["HIGH", "MEDIUM", "LOW"]

REGIME_STATES: list[RegimeState] = ["bull", "bear", "sideways"]
STATE_INDEX: dict[RegimeState, int] = {"bull": 0, "bear": 1, "sideways": 2}
NUM_STATES = len(REGIME_STATES)

BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS = 252
BTC_SHORT_HORIZON_LIVE_RERUN_WINDOW_DAYS = 60
BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT = 0.15

GOLD_SHORT_HORIZON_LIVE_HISTORY_DAYS = 252
GOLD_ULTRA_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT = 0.12
GOLD_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT = 0.15

FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS: dict[str, float | int | bool] = {
    "recommendedConfidenceThreshold": 0.22,
    "transitionMinObservations": 30,
    "transitionDecay": 0.97,
    "structuralBreakMinLength": 36,
    "momentumLookback": 10,
    "momentumAdjustmentScale": 0.25,
    "momentumAdjustmentClamp": 0.003,
    "trendPenaltyOnlyBreakConfidence": True,
    "divergenceWeightedBreakConfidence": False,
}

PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS: dict[str, float | int | bool] = {
    "transitionMinObservations": 31,
    "structuralBreakMinLength": 28,
    "momentumLookback": 9,
    "momentumAdjustmentScale": 0.252,
    "momentumAdjustmentClamp": 0.00305,
}

PROMOTED_HYPE_MARKOV_RUNTIME_DEFAULTS: dict[str, float | int | bool] = {
    "recommendedConfidenceThreshold": 0.15,
    "momentumAdjustmentScale": 0.48,
    "momentumAdjustmentClamp": 0.0058,
}

DEFAULT_DIVERGENCE_PENALTY_SCHEDULE: dict[str, float] = {
    "mild": 0.80,
    "medium": 0.70,
    "high": 0.60,
}

COMMODITY_MODEL_ONLY_MIN_R2 = -0.02
COMMODITY_MODEL_ONLY_MIN_CONFIDENCE = 0.15
CRYPTO_MODEL_ONLY_MIN_R2 = -0.03
CRYPTO_MODEL_ONLY_MIN_CONFIDENCE = 0.18
