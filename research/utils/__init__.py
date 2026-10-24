"""Calibration helpers and runtime configuration for the research package.

Modules
-------
``calibration.py``
    Polymarket YES-bias correction and probability calibration.

``anchor_trust.py``
    Anchor-to-Polymarket trust policy: decides when to trust market
    signals vs. model predictions.  Mirrors TS anchor-trust logic.

``kalshi_vol_signals.py``
    Kalshi macro volatility signals: extracts event-based volatility
    covariates from Kalshi prediction-market data.

``forecast_lab_runtime_defaults.py``
    Asset-scoped runtime defaults for the Forecast Lab system.  Mirrors
    ``src/tools/finance/forecast-lab-runtime-defaults.ts``.

``regime_calibrator.py``
    Single-pass regime-conditional Platt recalibrator.

``regime_calibrator_two_pass.py``
    Two-pass regime-conditional Platt recalibrator with improved fit.
"""

from research.utils.calibration import adjust_yes_bias, YES_BIAS_MULTIPLIER
from research.utils.anchor_trust import AnchorTrustEvaluation, evaluate_anchor_trust
from research.utils.kalshi_vol_signals import (
    KalshiUnconfiguredError,
    KalshiVolSignal,
    KalshiVolatilityCovariate,
    build_kalshi_volatility_covariate,
    extract_kalshi_vol_signals_from_payload,
    fetch_kalshi_vol_signals,
)

__all__ = [
    "adjust_yes_bias",
    "YES_BIAS_MULTIPLIER",
    "AnchorTrustEvaluation",
    "evaluate_anchor_trust",
    "KalshiUnconfiguredError",
    "KalshiVolSignal",
    "KalshiVolatilityCovariate",
    "build_kalshi_volatility_covariate",
    "extract_kalshi_vol_signals_from_payload",
    "fetch_kalshi_vol_signals",
]
