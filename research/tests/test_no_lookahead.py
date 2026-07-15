"""No-look-ahead regression lock for the Markov walk-forward forecast."""

from __future__ import annotations

import numpy as np

from research.backtest.walk_forward import walk_forward


def _prices(n: int = 400, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    return list(100.0 * np.exp(np.cumsum(rets)))


def test_future_corruption_does_not_affect_earlier_forecasts_or_scores():
    horizon = 7
    warmup = 120
    stride = 10
    prices = _prices()
    corrupt_from = 320

    corrupted = list(prices)
    for i in range(corrupt_from, len(corrupted)):
        corrupted[i] *= 100.0

    np.random.seed(2024)
    base = walk_forward(prices, horizon=horizon, warmup=warmup, stride=stride)
    np.random.seed(2024)
    tainted = walk_forward(corrupted, horizon=horizon, warmup=warmup, stride=stride)

    assert not base.errors
    assert not tainted.errors

    tainted_by_idx = {s.start_idx: s for s in tainted.steps}
    safe_steps = [
        s
        for s in base.steps
        if s.start_idx + horizon < corrupt_from and s.start_idx in tainted_by_idx
    ]
    assert safe_steps, "expected at least one window fully before the corruption"

    for b in safe_steps:
        t = tainted_by_idx[b.start_idx]
        assert t.predicted_prob == b.predicted_prob, f"p_up leaked future at idx {b.start_idx}"
        assert t.predicted_return == b.predicted_return, f"return leaked future at idx {b.start_idx}"
        assert t.ci_lower == b.ci_lower, f"CI lower leaked future at idx {b.start_idx}"
        assert t.ci_upper == b.ci_upper, f"CI upper leaked future at idx {b.start_idx}"
        assert t.realised_return == b.realised_return, f"realised return changed at idx {b.start_idx}"
        assert t.realised_price == b.realised_price, f"realised price changed at idx {b.start_idx}"
        assert t.direction_correct == b.direction_correct, f"score leaked future at idx {b.start_idx}"
        assert t.in_ci == b.in_ci, f"CI score leaked future at idx {b.start_idx}"
