"""Forecast-lab asset-scoped runtime defaults.

Python mirror of ``src/tools/finance/forecast-lab-runtime-defaults.ts``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Literal, TypeVar

ForecastLabRuntimeAssetScope = Literal["shared", "btc", "gold", "sol", "hype"]
ForecastLabRuntimeScalarValue = bool | int | float

_runtime_asset_scope: ContextVar[ForecastLabRuntimeAssetScope | None] = ContextVar(
    "forecast_lab_runtime_asset_scope",
    default=None,
)


def resolve_forecast_lab_runtime_asset_scope_for_ticker(
    ticker: str,
) -> ForecastLabRuntimeAssetScope | None:
    normalized_ticker = ticker.strip().upper()
    if normalized_ticker in {"BTC", "BTC-USD"}:
        return "btc"

    if normalized_ticker in {"SOL", "SOL-USD", "SOLUSD", "SOLUSDT"}:
        return "sol"

    if normalized_ticker in {"HYPE", "HYPE-USD", "HYPEUSD", "HYPEUSDT"}:
        return "hype"

    return "gold" if normalized_ticker in {"GLD", "GOLD"} else "shared"


def _get_runtime_scope_resolution_order(
    asset_scope: ForecastLabRuntimeAssetScope | None,
) -> tuple[ForecastLabRuntimeAssetScope, ...]:
    if asset_scope is None:
        return ()
    if asset_scope == "gold":
        return ("shared", "gold")
    return (asset_scope,)


@contextmanager
def forecast_lab_runtime_asset_scope(
    asset_scope: ForecastLabRuntimeAssetScope | None,
) -> Iterator[None]:
    token = _runtime_asset_scope.set(asset_scope)
    try:
        yield
    finally:
        _runtime_asset_scope.reset(token)


def with_forecast_lab_runtime_asset_scope(
    asset_scope: ForecastLabRuntimeAssetScope | None,
    callback: Callable[[], T],
) -> T:
    with forecast_lab_runtime_asset_scope(asset_scope):
        return callback()


def get_forecast_lab_runtime_asset_scope() -> ForecastLabRuntimeAssetScope | None:
    return _runtime_asset_scope.get()


class ForecastLabAssetScopedRuntimeDefaults:
    def __init__(
        self,
        shipped_defaults: dict[str, ForecastLabRuntimeScalarValue],
    ) -> None:
        self._shipped_defaults = dict(shipped_defaults)
        self._active_overrides: dict[
            ForecastLabRuntimeAssetScope,
            dict[str, ForecastLabRuntimeScalarValue],
        ] = {}

    def resolve(
        self,
        asset_scope: ForecastLabRuntimeAssetScope | None = None,
        explicit_overrides: dict[str, ForecastLabRuntimeScalarValue] | None = None,
    ) -> dict[str, ForecastLabRuntimeScalarValue]:
        active_scope = (
            get_forecast_lab_runtime_asset_scope() if asset_scope is None else asset_scope
        )
        layered_overrides: dict[str, ForecastLabRuntimeScalarValue] = {}
        for scope in _get_runtime_scope_resolution_order(active_scope):
            layered_overrides.update(self._active_overrides.get(scope, {}))

        resolved = {
            **self._shipped_defaults,
            **layered_overrides,
        }
        if explicit_overrides:
            resolved.update(explicit_overrides)
        return resolved

    def get(
        self,
        asset_scope: ForecastLabRuntimeAssetScope,
    ) -> dict[str, ForecastLabRuntimeScalarValue] | None:
        overrides = self._active_overrides.get(asset_scope)
        return dict(overrides) if overrides else None

    def set(
        self,
        asset_scope: ForecastLabRuntimeAssetScope,
        overrides: dict[str, ForecastLabRuntimeScalarValue] | None = None,
    ) -> None:
        if not overrides:
            self._active_overrides.pop(asset_scope, None)
            return
        self._active_overrides[asset_scope] = dict(overrides)


def create_forecast_lab_asset_scoped_runtime_defaults(
    shipped_defaults: dict[str, ForecastLabRuntimeScalarValue],
) -> ForecastLabAssetScopedRuntimeDefaults:
    return ForecastLabAssetScopedRuntimeDefaults(shipped_defaults)
T = TypeVar("T")

