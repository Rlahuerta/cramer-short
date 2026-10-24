from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Tier = Literal["macro", "geopolitical", "electoral"]


@dataclass
class MarketInput:
    """A single Polymarket market with probability, microstructure, and signal metadata.

    Fields
    ------
    question : str
        Market question text.
    probability : float
        Current YES-share probability in [0, 1].
    volume24h_usd : float
        24-hour trading volume in USD.
    age_days : int or None
        Days since market creation (None if unknown).
    price_spike_detected : bool
        Whether a whale-driven price spike was detected.
    transitory_move : bool
        Whether a short-lived (24–48h) price move was detected.
    signal_tier : Tier
        Market category: ``"macro"``, ``"geopolitical"``, or ``"electoral"``.
    delta_yes : float
        Expected return if the event resolves YES (default 0.06).
    delta_no : float
        Expected return if the event resolves NO (default -0.04).
    days_to_expiry : float or None
        Days until market resolution.
    requested_horizon_days : int or None
        Forecast horizon for anchor/forecast gap computation.
    bid_ask_spread : float or None
        Current bid-ask spread as a fraction of price.
    price_velocity_pp_h : float or None
        Legacy hourly price velocity in percentage points.
    price_velocity_logit_per_hour : float or None
        Hourly price velocity in logit space (preferred over legacy).
    max_hourly_jump : float or None
        Legacy max hourly price jump in percentage points.
    max_hourly_logit_jump : float or None
        Max hourly price jump in logit space (preferred over legacy).
    market_semantics : str or None
        Resolution semantics label (``"ambiguous"`` triggers quality discount).
    stable_path : bool
        Whether the market shows a stable price path (triggers quality boost).
    """
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
    """Auxiliary signal sources available for ensemble blending.

    Fields
    ------
    sentiment_score : float or None
        Sentiment signal (missing if None or NaN).
    fundamental_return : float or None
        DCF-based fundamental expected return (missing if None or NaN).
    options_skew : float or None
        Options-market skew signal (missing if None or NaN).
    markov_return : float or None
        Markov regime forecast return (missing if None or NaN).
    horizon_days : int
        Forecast horizon in days (default 7).
    """
    sentiment_score: float | None = None
    fundamental_return: float | None = None
    options_skew: float | None = None
    markov_return: float | None = None
    horizon_days: int = 7


@dataclass
class EnsembleResult:
    """Full ensemble forecast result with all diagnostic metadata.

    Fields
    ------
    forecast_return : float
        Blended expected return across all signal sources.
    forecast_price : float
        Expected price at the forecast horizon.
    ci_low95 : float
        Lower bound of the 95% confidence interval.
    ci_high95 : float
        Upper bound of the 95% confidence interval.
    sigma : float
        Forecast standard deviation (fraction of price).
    quality_score : float
        Composite quality score in [0, 100].
    quality_grade : str
        Letter grade: ``"A"``, ``"B"``, ``"C"``, or ``"D"``.
    pm_signal : float
        Raw Polymarket quality-weighted conditional return.
    pm_effective_weight : float
        Polymarket weight before normalization (0.40 × avg_quality).
    pm_normalized_weight : float
        Polymarket weight after normalization across all sources.
    avg_market_quality : float
        Mean quality score across all Polymarket markets.
    warnings : list[str]
        Structural warning messages from signal aggregation.
    weights : dict[str, float]
        Normalized weight per signal source (pm, sentiment, fundamental, options, markov).
    """
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
