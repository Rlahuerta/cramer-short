"""Feature-flag integration tests for Python walk-forward Markov features."""

from __future__ import annotations

import numpy as np

from research.backtest.walk_forward import walk_forward


def _prices_from_rng(seed: int = 2718, n: int = 260) -> list[float]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.018, n)
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * float(np.exp(ret)))
    return prices


def _step_fingerprint(result):
    return [
        (
            step.start_idx,
            step.predicted_prob,
            step.predicted_return,
            step.ci_lower,
            step.ci_upper,
            step.realised_return,
            step.realised_price,
            step.direction_correct,
            step.in_ci,
        )
        for step in result.steps
    ]


def test_walk_forward_random_state_makes_monte_carlo_reproducible():
    prices = _prices_from_rng()

    same_a = walk_forward(prices, horizon=9, warmup=120, stride=15, random_state=123)
    same_b = walk_forward(prices, horizon=9, warmup=120, stride=15, random_state=123)
    different = walk_forward(prices, horizon=9, warmup=120, stride=15, random_state=456)

    assert not same_a.errors
    assert not same_b.errors
    assert not different.errors
    assert _step_fingerprint(same_a) == _step_fingerprint(same_b)
    assert _step_fingerprint(same_a) != _step_fingerprint(different)


def test_walk_forward_meta_conditional_transitions_runs_and_can_change_forecast():
    prices = _prices_from_rng(seed=31415, n=320)

    default = walk_forward(prices, horizon=11, warmup=160, stride=20, random_state=99)
    conditional = walk_forward(
        prices,
        horizon=11,
        warmup=160,
        stride=20,
        random_state=99,
        use_meta_regime=True,
        use_conditional_transitions=True,
    )

    assert not default.errors
    assert not conditional.errors
    assert len(default.steps) == len(conditional.steps) > 0
    assert _step_fingerprint(default) != _step_fingerprint(conditional)


def test_walk_forward_all_new_flags_off_equals_default_path():
    prices = _prices_from_rng(seed=1618, n=240)

    np.random.seed(88)
    default = walk_forward(prices, horizon=7, warmup=120, stride=15)
    np.random.seed(88)
    explicit_off = walk_forward(
        prices,
        horizon=7,
        warmup=120,
        stride=15,
        random_state=None,
        use_meta_regime=False,
        use_conditional_transitions=False,
        use_path_integrated_drift=False,
    )

    assert not default.errors
    assert not explicit_off.errors
    assert _step_fingerprint(explicit_off) == _step_fingerprint(default)
