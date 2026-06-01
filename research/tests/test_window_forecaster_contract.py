"""Regression contracts for the rolling-window Markov forecaster."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import research.backtest._window_forecaster as wf


def _prices_from_returns(returns: list[float]) -> list[float]:
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1.0 + ret))
    return prices


def _patch_base_markov(monkeypatch: pytest.MonkeyPatch, regimes: list[str]) -> None:
    monkeypatch.setattr(wf, "classify_regime_series", lambda *_args, **_kwargs: regimes)
    monkeypatch.setattr(wf, "estimate_transition_matrix", lambda *_args, **_kwargs: np.eye(3))
    monkeypatch.setattr(
        wf,
        "detect_structural_break",
        lambda *_args, **_kwargs: {"detected": False, "divergence": 0.0},
    )
    monkeypatch.setattr(
        wf,
        "compute_markov_forecast",
        lambda *_args, **_kwargs: {"bull": 0.5, "bear": 0.25, "sideways": 0.25},
    )


def _point(price: float, p_up: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(
        expected_price=price,
        lower_bound=price * 0.95,
        upper_bound=price * 1.05,
        p_up=p_up,
    )


def test_window_forecast_passes_log_return_regime_stats_to_trajectory(
    monkeypatch: pytest.MonkeyPatch,
):
    simple_returns = [0.20, 0.10, -0.05, 0.02, 0.03, -0.01]
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear"]
    _patch_base_markov(monkeypatch, regimes)

    captured: dict[str, object] = {}

    def fake_compute_trajectory(
        current_price,
        _horizon,
        _P,
        regime_stats,
        *_args,
        **_kwargs,
    ):
        captured["regime_stats"] = regime_stats
        return [_point(current_price * 1.01, p_up=0.55)]

    monkeypatch.setattr(wf, "compute_trajectory", fake_compute_trajectory)

    wf.compute_window_forecast(
        _prices_from_returns(simple_returns),
        horizon=1,
        return_threshold_multiplier=0.5,
        decay_rate=0.95,
        break_divergence_threshold=0.2,
        use_hmm=False,
        asset_profile="crypto",
        enable_garch_vol=False,
        garch_horizon=None,
        garch_ceiling=None,
    )

    stats = captured["regime_stats"]
    expected_bull_mu = float(np.mean(np.log1p([0.20, 0.10, 0.03])))
    assert stats["bull"].mean_return == pytest.approx(expected_bull_mu)


def test_window_forecast_applies_positive_vol_floor_for_sparse_regimes(
    monkeypatch: pytest.MonkeyPatch,
):
    simple_returns = [0.08, -0.03, 0.01]
    regimes = ["bull", "bear", "sideways"]
    _patch_base_markov(monkeypatch, regimes)

    captured: dict[str, object] = {}

    def fake_compute_trajectory(
        current_price,
        _horizon,
        _P,
        regime_stats,
        *_args,
        **_kwargs,
    ):
        captured["regime_stats"] = regime_stats
        return [_point(current_price * 1.01, p_up=0.55)]

    monkeypatch.setattr(wf, "compute_trajectory", fake_compute_trajectory)

    wf.compute_window_forecast(
        _prices_from_returns(simple_returns),
        horizon=1,
        return_threshold_multiplier=0.5,
        decay_rate=0.95,
        break_divergence_threshold=0.2,
        use_hmm=False,
        asset_profile="crypto",
        enable_garch_vol=False,
        garch_horizon=None,
        garch_ceiling=None,
    )

    stats = captured["regime_stats"]
    expected_floor = float(np.std(np.log1p(simple_returns)))
    assert expected_floor > 0
    for state in ["bull", "bear", "sideways"]:
        assert np.isfinite(stats[state].std_return)
        assert stats[state].std_return == pytest.approx(expected_floor)
        assert stats[state].std_return > 0
    assert stats["bear"].mean_return == pytest.approx(float(np.log1p(-0.03)))


def test_hmm_non_convergence_preserves_base_markov_forecast(monkeypatch: pytest.MonkeyPatch):
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear"]
    _patch_base_markov(monkeypatch, regimes)
    monkeypatch.setattr(
        wf,
        "baum_welch",
        lambda *_args, **_kwargs: SimpleNamespace(converged=False),
    )
    monkeypatch.setattr(
        wf,
        "compute_markov_forecast",
        lambda *_args, **_kwargs: {"bull": 0.7, "bear": 0.2, "sideways": 0.1},
    )

    captured: dict[str, object] = {}

    def fake_compute_trajectory(current_price, *_args, hmm_override=None, **_kwargs):
        captured["hmm_override"] = hmm_override
        return [_point(current_price * 1.08, p_up=0.7)]

    monkeypatch.setattr(wf, "compute_trajectory", fake_compute_trajectory)

    result = wf.compute_window_forecast(
        _prices_from_returns([0.02, 0.01, -0.03, 0.01, 0.02, -0.01]),
        horizon=3,
        return_threshold_multiplier=0.5,
        decay_rate=0.95,
        break_divergence_threshold=0.2,
        use_hmm=True,
        asset_profile="crypto",
        enable_garch_vol=False,
        garch_horizon=None,
        garch_ceiling=None,
    )

    assert captured["hmm_override"] is None
    assert result["hmm_status"] == "non_converged"
    assert result["p_up"] == pytest.approx(0.70)
    assert result["predicted_return"] == pytest.approx(0.08)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("degenerate HMM covariance"),
        RuntimeError("Baum-Welch HMM fit failed"),
    ],
)
def test_hmm_fit_error_preserves_base_markov_forecast(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
):
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear"]
    _patch_base_markov(monkeypatch, regimes)
    monkeypatch.setattr(
        wf,
        "baum_welch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(exc),
    )

    captured: dict[str, object] = {}

    def fake_compute_trajectory(current_price, *_args, hmm_override=None, **_kwargs):
        captured["hmm_override"] = hmm_override
        return [_point(current_price * 1.06, p_up=0.65)]

    monkeypatch.setattr(wf, "compute_trajectory", fake_compute_trajectory)

    result = wf.compute_window_forecast(
        _prices_from_returns([0.02, 0.01, -0.03, 0.01, 0.02, -0.01]),
        horizon=3,
        return_threshold_multiplier=0.5,
        decay_rate=0.95,
        break_divergence_threshold=0.2,
        use_hmm=True,
        asset_profile="crypto",
        enable_garch_vol=False,
        garch_horizon=None,
        garch_ceiling=None,
    )

    assert captured["hmm_override"] is None
    assert result["hmm_status"] == f"error:{type(exc).__name__}"
    assert result["p_up"] == pytest.approx(0.65)
    assert result["predicted_return"] == pytest.approx(0.06)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("invalid unrelated input"),
        RuntimeError("database unavailable"),
    ],
)
def test_unrelated_hmm_fit_error_propagates(monkeypatch: pytest.MonkeyPatch, exc: Exception):
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear"]
    _patch_base_markov(monkeypatch, regimes)
    monkeypatch.setattr(
        wf,
        "baum_welch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(exc),
    )

    with pytest.raises(type(exc), match=str(exc)):
        wf.compute_window_forecast(
            _prices_from_returns([0.02, 0.01, -0.03, 0.01, 0.02, -0.01]),
            horizon=3,
            return_threshold_multiplier=0.5,
            decay_rate=0.95,
            break_divergence_threshold=0.2,
            use_hmm=True,
            asset_profile="crypto",
            enable_garch_vol=False,
            garch_horizon=None,
            garch_ceiling=None,
        )


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("hidden markov covariance failure"),
        RuntimeError("transition matrix is singular"),
    ],
)
def test_hmm_overlay_error_preserves_base_markov_forecast(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
):
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear"]
    _patch_base_markov(monkeypatch, regimes)
    monkeypatch.setattr(
        wf,
        "baum_welch",
        lambda *_args, **_kwargs: SimpleNamespace(converged=True, params=object()),
    )
    monkeypatch.setattr(
        wf,
        "predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(exc),
    )

    captured: dict[str, object] = {}

    def fake_compute_trajectory(current_price, *_args, hmm_override=None, **_kwargs):
        captured["hmm_override"] = hmm_override
        return [_point(current_price * 1.06, p_up=0.65)]

    monkeypatch.setattr(wf, "compute_trajectory", fake_compute_trajectory)

    result = wf.compute_window_forecast(
        _prices_from_returns([0.02, 0.01, -0.03, 0.01, 0.02, -0.01]),
        horizon=3,
        return_threshold_multiplier=0.5,
        decay_rate=0.95,
        break_divergence_threshold=0.2,
        use_hmm=True,
        asset_profile="crypto",
        enable_garch_vol=False,
        garch_horizon=None,
        garch_ceiling=None,
    )

    assert captured["hmm_override"] is None
    assert result["hmm_status"] == f"error:{type(exc).__name__}"
    assert result["p_up"] == pytest.approx(0.65)
    assert result["predicted_return"] == pytest.approx(0.06)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("invalid unrelated prediction input"),
        RuntimeError("cache unavailable"),
    ],
)
def test_unrelated_hmm_overlay_error_propagates(monkeypatch: pytest.MonkeyPatch, exc: Exception):
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear"]
    _patch_base_markov(monkeypatch, regimes)
    monkeypatch.setattr(
        wf,
        "baum_welch",
        lambda *_args, **_kwargs: SimpleNamespace(converged=True, params=object()),
    )
    monkeypatch.setattr(
        wf,
        "predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(exc),
    )

    with pytest.raises(type(exc), match=str(exc)):
        wf.compute_window_forecast(
            _prices_from_returns([0.02, 0.01, -0.03, 0.01, 0.02, -0.01]),
            horizon=3,
            return_threshold_multiplier=0.5,
            decay_rate=0.95,
            break_divergence_threshold=0.2,
            use_hmm=True,
            asset_profile="crypto",
            enable_garch_vol=False,
            garch_horizon=None,
            garch_ceiling=None,
        )


@pytest.mark.parametrize(
    ("vol_scale", "asset_profile", "weight_multiplier"),
    [
        (float("nan"), "crypto", None),
        (1.0, "invalid-weight", float("nan")),
    ],
)
def test_invalid_hmm_overlay_components_preserve_base_markov_forecast(
    monkeypatch: pytest.MonkeyPatch,
    vol_scale: float,
    asset_profile: str,
    weight_multiplier: float | None,
):
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear"]
    _patch_base_markov(monkeypatch, regimes)
    monkeypatch.setattr(
        wf,
        "baum_welch",
        lambda *_args, **_kwargs: SimpleNamespace(converged=True, params=object()),
    )
    monkeypatch.setattr(
        wf,
        "predict",
        lambda *_args, **_kwargs: SimpleNamespace(expected_return=0.02, expected_volatility=0.03),
    )
    monkeypatch.setattr(wf, "fit_volatility_hmm", lambda *_args, **_kwargs: vol_scale)
    if weight_multiplier is not None:
        monkeypatch.setitem(
            wf.ASSET_PROFILES,
            asset_profile,
            SimpleNamespace(hmm_weight_multiplier=weight_multiplier),
        )

    captured: dict[str, object] = {}

    def fake_compute_trajectory(current_price, *_args, hmm_override=None, **_kwargs):
        captured["hmm_override"] = hmm_override
        return [_point(current_price * 1.04, p_up=0.62)]

    monkeypatch.setattr(wf, "compute_trajectory", fake_compute_trajectory)

    result = wf.compute_window_forecast(
        _prices_from_returns([0.02, 0.01, -0.03, 0.01, 0.02, -0.01]),
        horizon=3,
        return_threshold_multiplier=0.5,
        decay_rate=0.95,
        break_divergence_threshold=0.2,
        use_hmm=True,
        asset_profile=asset_profile,
        enable_garch_vol=False,
        garch_horizon=None,
        garch_ceiling=None,
    )

    assert captured["hmm_override"] is None
    assert result["hmm_status"] == "non_finite"
    assert result["p_up"] == pytest.approx(0.62)
    assert result["predicted_return"] == pytest.approx(0.04)


def test_window_forecast_p_up_matches_final_trajectory_without_calibration(
    monkeypatch: pytest.MonkeyPatch,
):
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear"]
    _patch_base_markov(monkeypatch, regimes)
    monkeypatch.setattr(
        wf,
        "compute_markov_forecast",
        lambda *_args, **_kwargs: {"bull": 1.0, "bear": 0.0, "sideways": 0.0},
    )

    monkeypatch.setattr(
        wf,
        "compute_trajectory",
        lambda current_price, *_args, **_kwargs: [_point(current_price * 1.02, p_up=0.82)],
    )

    result = wf.compute_window_forecast(
        _prices_from_returns([0.02, 0.01, -0.03, 0.01, 0.02, -0.01]),
        horizon=3,
        return_threshold_multiplier=0.5,
        decay_rate=0.95,
        break_divergence_threshold=0.2,
        use_hmm=False,
        asset_profile="crypto",
        enable_garch_vol=False,
        garch_horizon=None,
        garch_ceiling=None,
    )

    assert result["p_up"] == pytest.approx(0.82)


def test_hmm_disabled_does_not_attempt_hmm_or_change_base_forecast(
    monkeypatch: pytest.MonkeyPatch,
):
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear"]
    _patch_base_markov(monkeypatch, regimes)
    monkeypatch.setattr(
        wf,
        "baum_welch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("HMM disabled")),
    )

    captured: dict[str, object] = {}

    def fake_compute_trajectory(current_price, *_args, hmm_override=None, **_kwargs):
        captured["hmm_override"] = hmm_override
        return [_point(current_price * 1.03, p_up=0.58)]

    monkeypatch.setattr(wf, "compute_trajectory", fake_compute_trajectory)

    result = wf.compute_window_forecast(
        _prices_from_returns([0.02, 0.01, -0.03, 0.01, 0.02, -0.01]),
        horizon=3,
        return_threshold_multiplier=0.5,
        decay_rate=0.95,
        break_divergence_threshold=0.2,
        use_hmm=False,
        asset_profile="crypto",
        enable_garch_vol=False,
        garch_horizon=None,
        garch_ceiling=None,
    )

    assert captured["hmm_override"] is None
    assert result["hmm_status"] == "disabled"
    assert result["p_up"] == pytest.approx(0.58)
    assert result["predicted_return"] == pytest.approx(0.03)


def test_combined_uncertainty_ci_scale_preserves_structural_break_floor():
    assert wf._combine_uncertainty_ci_scale(
        structural_scale=1.5,
        entropy_scale=0.7,
    ) == pytest.approx(1.5)
    assert wf._combine_uncertainty_ci_scale(
        structural_scale=1.0,
        entropy_scale=0.7,
    ) == pytest.approx(1.0)


def test_window_forecast_passes_hmm_daily_emission_payload_as_per_day_override(
    monkeypatch: pytest.MonkeyPatch,
):
    horizon = 7
    regimes = ["bull", "bull", "bear", "sideways", "bull", "bear", "sideways", "bull"]
    _patch_base_markov(monkeypatch, regimes)
    monkeypatch.setattr(
        wf,
        "baum_welch",
        lambda *_args, **_kwargs: SimpleNamespace(converged=True, params=object()),
    )
    monkeypatch.setattr(
        wf,
        "predict",
        lambda *_args, **_kwargs: SimpleNamespace(expected_return=0.14, expected_volatility=0.07),
    )
    monkeypatch.setattr(wf, "fit_volatility_hmm", lambda *_args, **_kwargs: 1.0)

    captured: dict[str, object] = {}

    def fake_compute_trajectory(current_price, *_args, hmm_override=None, **_kwargs):
        captured["hmm_override"] = hmm_override
        return [_point(current_price * 1.02, p_up=0.55)]

    monkeypatch.setattr(wf, "compute_trajectory", fake_compute_trajectory)

    wf.compute_window_forecast(
        _prices_from_returns([0.02, 0.01, -0.03, 0.01, 0.02, -0.01, 0.01, 0.02]),
        horizon=horizon,
        return_threshold_multiplier=0.5,
        decay_rate=0.95,
        break_divergence_threshold=0.2,
        use_hmm=True,
        asset_profile="crypto",
        enable_garch_vol=False,
        garch_horizon=None,
        garch_ceiling=None,
    )

    override = captured["hmm_override"]
    assert override["drift"] == pytest.approx(0.14)
    assert override["vol"] == pytest.approx(0.07)
