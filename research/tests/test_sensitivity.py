from __future__ import annotations

import numpy as np

from research.models.sensitivity import (
    INTEGER_MARKOV_DESIGN_VARIABLES,
    MARKOV_DESIGN_VARIABLES,
    sobol_sensitivity,
)


def test_sobol_sensitivity_estimates_additive_linear_indices() -> None:
    bounds = {
        "x1": (0.0, 1.0),
        "x2": (0.0, 1.0),
        "x3": (0.0, 1.0),
    }
    coefficients = np.array([1.0, 2.0, 3.0])

    def objective(row: np.ndarray) -> float:
        return float(np.dot(coefficients, row))

    result = sobol_sensitivity(objective, bounds, n_base=2048, seed=123)

    expected = coefficients**2 / np.sum(coefficients**2)
    np.testing.assert_allclose(result["S1"], expected, atol=0.06)
    np.testing.assert_allclose(result["ST"], expected, atol=0.06)
    assert np.all(result["S1"] >= -1e-9)
    assert float(np.sum(result["S1"])) <= 1.0 + 0.06


def test_sobol_sensitivity_returns_zero_indices_for_zero_variance_too_few_samples() -> None:
    result = sobol_sensitivity(lambda row: 1.0, {"x": (0.0, 1.0)}, n_base=0, seed=123)

    np.testing.assert_array_equal(result["S1"], np.array([0.0]))
    np.testing.assert_array_equal(result["ST"], np.array([0.0]))


def test_markov_design_variables_mirror_xiphos_order_and_integer_mask() -> None:
    assert MARKOV_DESIGN_VARIABLES == (
        "warmup",
        "decay_rate",
        "return_threshold_multiplier",
        "break_divergence_threshold",
        "abstention_threshold",
        "student_t_nu",
        "break_ci_slope",
        "break_ci_max",
        "garch_horizon_cap",
        "garch_calm_ceiling",
        "garch_turbulent_ceiling",
        "entropy_window_size",
        "hmm_num_states",
        "meta_vol_window",
        "meta_high_vol_threshold",
        "volume_lookback",
        "volume_threshold_multiplier",
    )
    assert INTEGER_MARKOV_DESIGN_VARIABLES == frozenset(
        {
            "warmup",
            "student_t_nu",
            "garch_horizon_cap",
            "entropy_window_size",
            "hmm_num_states",
            "meta_vol_window",
            "volume_lookback",
        }
    )
