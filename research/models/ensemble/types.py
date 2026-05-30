from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Tier = Literal["macro", "geopolitical", "electoral"]


@dataclass
class MarketInput:
    question: str
    probability: float
    volume24h_usd: float
    age_days: int | None = None
    price_spike_detected: bool = False
    transitory_move: bool = False
    signal_tier: Tier = "geopolitical"
    delta_yes: float = 0.06
    delta_no: float = -0.04
    days_to_expiry: float | None = None
    requested_horizon_days: int | None = None
    bid_ask_spread: float | None = None
    price_velocity_pp_h: float | None = None
    price_velocity_logit_per_hour: float | None = None
    max_hourly_jump: float | None = None
    max_hourly_logit_jump: float | None = None
    market_semantics: str | None = None
    stable_path: bool = False


@dataclass
class OtherSignals:
    sentiment_score: float | None = None
    fundamental_return: float | None = None
    options_skew: float | None = None
    markov_return: float | None = None
    horizon_days: int = 7


@dataclass
class EnsembleResult:
    forecast_return: float
    forecast_price: float
    ci_low95: float
    ci_high95: float
    sigma: float
    quality_score: float
    quality_grade: str
    pm_signal: float
    pm_effective_weight: float
    pm_normalized_weight: float
    avg_market_quality: float
    warnings: list[str]
    weights: dict[str, float]
