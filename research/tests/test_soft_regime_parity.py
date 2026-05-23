from __future__ import annotations

import math

from research.models.soft_regime import (
    blend_regime_mixtures,
    map_hmm_probabilities_to_regime_mixture,
    one_hot_regime_mixture,
)


def test_one_hot_regime_mixture():
    assert one_hot_regime_mixture("bull") == {"bull": 1.0, "bear": 0.0, "sideways": 0.0}


def test_blend_regime_mixtures_normalizes_and_clamps():
    blended = blend_regime_mixtures(
        {"bull": 1.0, "bear": 0.0, "sideways": 0.0},
        {"bull": 0.2, "bear": 0.3, "sideways": 0.5},
        0.5,
    )
    assert math.isclose(sum(blended.values()), 1.0)
    assert blended["bull"] > blended["bear"]
    assert blended["sideways"] > 0


def test_map_hmm_probabilities_to_regime_mixture_orders_by_mean():
    mixture = map_hmm_probabilities_to_regime_mixture(
        [0.2, 0.5, 0.3],
        [-0.02, 0.0, 0.03],
    )
    assert mixture is not None
    assert math.isclose(sum(mixture.values()), 1.0)
    assert math.isclose(mixture["bear"], 0.2)
    assert math.isclose(mixture["sideways"], 0.5)
    assert math.isclose(mixture["bull"], 0.3)
