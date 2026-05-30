"""Runtime defaults and asset-scoped parameter policies.

Mirrors TS logic:
  - Asset-specific parameter overrides (SOL, HYPE)
  - Forecast Lab default resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from research.models.markov.core import (
    BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
    BTC_SHORT_HORIZON_LIVE_RERUN_WINDOW_DAYS,
    BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
    FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS,
    GOLD_SHORT_HORIZON_LIVE_HISTORY_DAYS,
    GOLD_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
    GOLD_ULTRA_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
    PROMOTED_HYPE_MARKOV_RUNTIME_DEFAULTS,
    PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS,
)
from research.utils.forecast_lab_runtime_defaults import (
    create_forecast_lab_asset_scoped_runtime_defaults,
    ForecastLabRuntimeAssetScope,
    resolve_forecast_lab_runtime_asset_scope_for_ticker,
)


@dataclass(frozen=True)
class BtcShortHorizonLivePolicy:
    """Parameter defaults for BTC short-horizon (h <= 14) live backtests."""

    history_days: int
    break_divergence_threshold: float
    rerun_on_break: bool
    rerun_window_days: int | None = None


@dataclass(frozen=True)
class GoldShortHorizonLivePolicy:
    """Parameter defaults for GLD short-horizon live backtests."""

    history_days: int
    break_divergence_threshold: float
    rerun_on_break: Literal[False]


_forecast_lab_markov_runtime_defaults = create_forecast_lab_asset_scoped_runtime_defaults(
    FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS
)
_forecast_lab_markov_runtime_defaults.set("sol", PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS)
_forecast_lab_markov_runtime_defaults.set("hype", PROMOTED_HYPE_MARKOV_RUNTIME_DEFAULTS)


def resolve_forecast_lab_markov_parameter_defaults(
    asset_scope: ForecastLabRuntimeAssetScope | None = None,
) -> dict[str, float | int | bool]:
    return _forecast_lab_markov_runtime_defaults.resolve(asset_scope)


def get_forecast_lab_markov_runtime_defaults(
    asset_scope: ForecastLabRuntimeAssetScope,
) -> dict[str, float | int | bool] | None:
    return _forecast_lab_markov_runtime_defaults.get(asset_scope)


def set_forecast_lab_markov_runtime_defaults(
    asset_scope: ForecastLabRuntimeAssetScope,
    overrides: dict[str, float | int | bool] | None = None,
) -> None:
    _forecast_lab_markov_runtime_defaults.set(asset_scope, overrides)


def is_btc_ticker_symbol(ticker: str) -> bool:
    upper = ticker.strip().upper()
    return upper in {"BTC", "BTC-USD"}


def is_gold_ticker_symbol(ticker: str) -> bool:
    upper = ticker.strip().upper()
    return upper == "GLD"


def get_btc_short_horizon_live_policy(
    ticker: str,
    horizon: int,
) -> BtcShortHorizonLivePolicy | None:
    if not is_btc_ticker_symbol(ticker) or horizon < 1 or horizon > 14:
        return None

    if horizon == 1:
        return BtcShortHorizonLivePolicy(
            history_days=BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
            break_divergence_threshold=0.10,
            rerun_on_break=True,
            rerun_window_days=BTC_SHORT_HORIZON_LIVE_RERUN_WINDOW_DAYS,
        )

    if horizon == 2:
        return BtcShortHorizonLivePolicy(
            history_days=BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
            break_divergence_threshold=BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
            rerun_on_break=True,
            rerun_window_days=120,
        )

    if horizon == 3:
        return BtcShortHorizonLivePolicy(
            history_days=BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
            break_divergence_threshold=BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
            rerun_on_break=True,
            rerun_window_days=45,
        )

    return BtcShortHorizonLivePolicy(
        history_days=BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
        break_divergence_threshold=0.08 if horizon == 14 else BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
        rerun_on_break=False,
    )


def get_gold_short_horizon_live_policy(
    ticker: str,
    horizon: int,
) -> GoldShortHorizonLivePolicy | None:
    if not is_gold_ticker_symbol(ticker) or horizon < 1 or horizon > 14:
        return None
    return GoldShortHorizonLivePolicy(
        history_days=GOLD_SHORT_HORIZON_LIVE_HISTORY_DAYS,
        break_divergence_threshold=(
            GOLD_ULTRA_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT
            if horizon <= 3
            else GOLD_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT
        ),
        rerun_on_break=False,
    )
