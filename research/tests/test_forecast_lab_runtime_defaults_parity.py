from __future__ import annotations

import importlib

from research.backtest.walk_forward import walk_forward
from research.models.conformal import (
    resolve_forecast_lab_conformal_parameter_defaults,
)
from research.models.markov import (
    FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS,
    resolve_forecast_lab_markov_parameter_defaults,
)
from research.utils.forecast_lab_runtime_defaults import (
    create_forecast_lab_asset_scoped_runtime_defaults,
    resolve_forecast_lab_runtime_asset_scope_for_ticker,
)
from research.utils.regime_calibrator import (
    resolve_forecast_lab_regime_calibrator_defaults,
)

walk_forward_module = importlib.import_module("research.backtest.walk_forward")


def _make_prices(length: int = 220) -> list[float]:
    prices = [100.0]
    for idx in range(1, length):
        shock = 0.03 if idx % 11 == 0 else (-0.018 if idx % 7 == 0 else 0.012)
        prices.append(round(prices[-1] * (1 + shock), 4))
    return prices


def test_runtime_scope_resolution_matches_ts():
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("BTC") == "btc"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("BTC-USD") == "btc"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("GLD") == "gold"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("SOL") == "sol"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("SOL-USD") == "sol"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("SOLUSD") == "sol"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("SOLUSDT") == "sol"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("HYPE") == "hype"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("HYPE-USD") == "hype"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("HYPEUSD") == "hype"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("HYPEUSDT") == "hype"
    assert resolve_forecast_lab_runtime_asset_scope_for_ticker("QQQ") == "shared"


def test_asset_scoped_runtime_defaults_isolation_matches_ts():
    defaults = create_forecast_lab_asset_scoped_runtime_defaults(
        {
            "momentumLookback": 10,
            "transitionDecay": 0.97,
        }
    )

    defaults.set("shared", {"momentumLookback": 20})
    defaults.set("gold", {"transitionDecay": 0.95})
    defaults.set("sol", {"momentumLookback": 7})
    defaults.set("hype", {"transitionDecay": 0.94})

    assert defaults.resolve("shared") == {
        "momentumLookback": 20,
        "transitionDecay": 0.97,
    }
    assert defaults.resolve("gold") == {
        "momentumLookback": 20,
        "transitionDecay": 0.95,
    }
    assert defaults.resolve("sol") == {
        "momentumLookback": 7,
        "transitionDecay": 0.97,
    }
    assert defaults.resolve("hype") == {
        "momentumLookback": 10,
        "transitionDecay": 0.94,
    }


def test_python_markov_conformal_and_regime_defaults_match_ts_promotions():
    shared = resolve_forecast_lab_markov_parameter_defaults("shared")
    sol = resolve_forecast_lab_markov_parameter_defaults("sol")
    hype = resolve_forecast_lab_markov_parameter_defaults("hype")
    sol_conformal = resolve_forecast_lab_conformal_parameter_defaults("sol")
    sol_regime = resolve_forecast_lab_regime_calibrator_defaults("sol")

    assert shared["recommendedConfidenceThreshold"] == FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS[
        "recommendedConfidenceThreshold"
    ]
    assert sol["transitionMinObservations"] == 31
    assert sol["structuralBreakMinLength"] == 28
    assert sol["momentumLookback"] == 9
    assert sol["momentumAdjustmentScale"] == 0.252
    assert sol["momentumAdjustmentClamp"] == 0.00305
    assert hype["recommendedConfidenceThreshold"] == 0.15
    assert hype["momentumAdjustmentScale"] == 0.48
    assert hype["momentumAdjustmentClamp"] == 0.0058
    assert sol_conformal["scoreAggregationMinSamples"] == 10
    assert sol_conformal["scoreAggregationCalibrationWindow"] == 60
    assert sol_regime["minSamplesPerRegime"] == 14


def test_walk_forward_uses_sol_scoped_markov_defaults(monkeypatch):
    captured: dict[str, float | int | None] = {}

    real_estimate_transition_matrix = walk_forward_module.estimate_transition_matrix
    real_detect_structural_break = walk_forward_module.detect_structural_break
    sol_defaults = resolve_forecast_lab_markov_parameter_defaults("sol")

    def wrapped_estimate_transition_matrix(
        states,
        alpha=None,
        min_observations=None,
        decay_rate=None,
    ):
        captured["min_observations"] = min_observations
        captured["decay_rate"] = decay_rate
        captured["resolved_transition_min_observations"] = (
            resolve_forecast_lab_markov_parameter_defaults()["transitionMinObservations"]
        )
        captured["resolved_transition_decay"] = (
            resolve_forecast_lab_markov_parameter_defaults()["transitionDecay"]
        )
        return real_estimate_transition_matrix(
            states,
            alpha=alpha,
            min_observations=min_observations,
            decay_rate=decay_rate,
        )

    def wrapped_detect_structural_break(
        states,
        divergence_threshold=0.05,
        alpha=0.1,
        decay_rate=None,
        min_length=None,
    ):
        captured["min_length"] = min_length
        captured["resolved_structural_break_min_length"] = (
            resolve_forecast_lab_markov_parameter_defaults()["structuralBreakMinLength"]
        )
        return real_detect_structural_break(
            states,
            divergence_threshold=divergence_threshold,
            alpha=alpha,
            decay_rate=decay_rate,
            min_length=min_length,
        )

    monkeypatch.setattr(
        walk_forward_module,
        "estimate_transition_matrix",
        wrapped_estimate_transition_matrix,
    )
    monkeypatch.setattr(
        walk_forward_module,
        "detect_structural_break",
        wrapped_detect_structural_break,
    )

    result = walk_forward(
        _make_prices(),
        ticker="SOLUSD",
        horizon=2,
        warmup=120,
        stride=20,
    )

    assert not result.errors
    assert result.steps
    assert captured["min_observations"] is None
    assert captured["decay_rate"] == 0.97
    assert captured["min_length"] is None
    assert (
        captured["resolved_transition_min_observations"]
        == sol_defaults["transitionMinObservations"]
    )
    assert captured["resolved_transition_decay"] == sol_defaults["transitionDecay"]
    assert (
        captured["resolved_structural_break_min_length"]
        == sol_defaults["structuralBreakMinLength"]
    )


def test_walk_forward_sol_and_hype_runtime_scopes_produce_steps():
    prices = _make_prices(365)

    sol = walk_forward(prices, ticker="SOLUSD", horizon=1, warmup=120, stride=10)
    hype = walk_forward(prices, ticker="HYPEUSD", horizon=1, warmup=120, stride=10)

    assert not sol.errors
    assert not hype.errors
    assert sol.steps
    assert hype.steps
