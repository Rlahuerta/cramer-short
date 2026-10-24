import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.backtest.metrics import brier_score, ci_coverage, directional_accuracy
from research.backtest.walk_forward import walk_forward

walk_forward_module = importlib.import_module("research.backtest.walk_forward")


def _load_btc_fixture_prices() -> list[float]:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_path = repo_root / "src" / "tools" / "finance" / "fixtures" / "backtest-prices.json"
    payload = json.loads(fixture_path.read_text())
    return payload["tickers"]["BTC-USD"]["closes"]


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


@pytest.mark.parametrize(
    ("explicit_use_empirical_up_rates", "expected_use_empirical_up_rates"),
    [
        (None, False),
        (True, True),
        (False, False),
    ],
    ids=["live-policy-default", "explicit-true", "explicit-false"],
)
def test_walk_forward_live_btc_policy_uses_trajectory_p_up_by_default_but_preserves_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch,
    explicit_use_empirical_up_rates: bool | None,
    expected_use_empirical_up_rates: bool,
):
    prices = [100.0 + i for i in range(270)]
    observed_use_empirical_up_rates: list[bool] = []

    def fake_compute_window_forecast(window_prices, **kwargs):
        observed_use_empirical_up_rates.append(kwargs["use_empirical_up_rates"])
        current_price = float(window_prices[-1])
        return {
            "p_up": 0.55,
            "predicted_return": 0.01,
            "ci_lower": current_price * 0.95,
            "ci_upper": current_price * 1.05,
            "garch_vol_applied": False,
            "entropy": SimpleNamespace(entropy_nats=0.0, entropy_norm=0.0),
            "break_result": {"detected": False, "divergence": 0.0},
            "original_break_result": {"detected": False, "divergence": 0.0},
        }

    monkeypatch.setattr(walk_forward_module, "compute_window_forecast", fake_compute_window_forecast)

    kwargs = {
        "ticker": "BTC-USD",
        "horizon": 1,
        "warmup": 120,
        "stride": 20,
        "use_live_btc_short_horizon_policy": True,
    }
    if explicit_use_empirical_up_rates is not None:
        kwargs["use_empirical_up_rates"] = explicit_use_empirical_up_rates

    result = walk_forward(prices, **kwargs)

    assert not result.errors
    assert observed_use_empirical_up_rates == [expected_use_empirical_up_rates]


def test_btc_live_short_horizon_policy_wires_ts_knobs_and_improves_shortest_horizons():
    prices = _load_btc_fixture_prices()
    assert len(prices) > 300

    lines = ["", "═══ PYTHON BTC LIVE SHORT-HORIZON POLICY ═══"]

    for horizon in (1, 2, 3, 14):
        baseline = walk_forward(
            prices,
            ticker="BTC-USD",
            horizon=horizon,
            warmup=120,
            stride=5,
        )
        tuned = walk_forward(
            prices,
            ticker="BTC-USD",
            horizon=horizon,
            warmup=120,
            stride=5,
            use_live_btc_short_horizon_policy=True,
        )

        baseline_dir = directional_accuracy(baseline.steps)
        tuned_dir = directional_accuracy(tuned.steps)
        baseline_brier = brier_score(baseline.steps)
        tuned_brier = brier_score(tuned.steps)
        baseline_ci = ci_coverage(baseline.steps)
        tuned_ci = ci_coverage(tuned.steps)
        tuned_rerun_rate = (
            sum(1 for step in tuned.steps if step.structural_break_rerun_triggered) / len(tuned.steps)
            if tuned.steps
            else 0.0
        )
        tuned_first_start = tuned.steps[0].start_idx if tuned.steps else None

        lines.append(
            f"{horizon}d | baseline-120: dir={_format_pct(baseline_dir)} brier={baseline_brier:.4f} ci={_format_pct(baseline_ci)} rerun=0.0%"
        )
        lines.append(
            f"{horizon}d | btc-live:     dir={_format_pct(tuned_dir)} brier={tuned_brier:.4f} ci={_format_pct(tuned_ci)} rerun={_format_pct(tuned_rerun_rate)}"
        )

        assert not baseline.errors
        assert not tuned.errors
        assert tuned_first_start == 252

        if horizon == 1:
            assert tuned_dir > baseline_dir
            assert tuned_dir > 0.50
            assert tuned_rerun_rate > 0.50
        elif horizon == 2:
            assert tuned_dir > baseline_dir
            assert 0.20 < tuned_rerun_rate < 0.50
        elif horizon == 3:
            assert 0.20 < tuned_rerun_rate < 0.50
        elif horizon == 14:
            assert tuned_brier < baseline_brier
            assert tuned_rerun_rate == 0.0

    lines.append("════════════════════════════════════════════")
    print("\n".join(lines))


def test_btc_live_short_horizon_policy_only_reruns_for_1d_and_3d():
    prices = _load_btc_fixture_prices()
    rerun_rates: dict[int, float] = {}

    for horizon in (1, 2, 3, 14):
        tuned = walk_forward(
            prices,
            ticker="BTC-USD",
            horizon=horizon,
            warmup=120,
            stride=5,
            use_live_btc_short_horizon_policy=True,
        )
        rerun_rates[horizon] = (
            sum(1 for step in tuned.steps if step.structural_break_rerun_triggered) / len(tuned.steps)
            if tuned.steps
            else 0.0
        )

    assert rerun_rates[1] > 0.50
    assert 0.20 < rerun_rates[2] < 0.50
    assert 0.20 < rerun_rates[3] < 0.50
    assert rerun_rates[14] == 0.0


def test_walk_forward_btc_live_policy_overrides_warmup_and_records_reruns():
    prices = [65000.0]
    for i in range(1, 340):
        shock = 0.02 if i >= 260 else (0.02 if i % 2 == 0 else -0.02)
        prices.append(round(prices[-1] * (1 + shock), 2))

    result = walk_forward(
        prices,
        ticker="BTC-USD",
        horizon=1,
        warmup=120,
        stride=10,
        use_live_btc_short_horizon_policy=True,
    )

    assert not result.errors
    assert result.steps
    assert result.steps[0].start_idx == 252
    assert any(step.structural_break_rerun_triggered for step in result.steps)
