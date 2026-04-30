"""Helpers mirroring soft-regime mixture wiring from TypeScript."""

from __future__ import annotations

from typing import Literal

RegimeState = Literal["bull", "bear", "sideways"]


def one_hot_regime_mixture(state: RegimeState) -> dict[RegimeState, float]:
    return {
        "bull": 1.0 if state == "bull" else 0.0,
        "bear": 1.0 if state == "bear" else 0.0,
        "sideways": 1.0 if state == "sideways" else 0.0,
    }


def blend_regime_mixtures(
    base: dict[RegimeState, float],
    overlay: dict[RegimeState, float],
    weight: float,
) -> dict[RegimeState, float]:
    w = min(1.0, max(0.0, weight))
    mixed = {
        "bull": (1 - w) * base["bull"] + w * overlay["bull"],
        "bear": (1 - w) * base["bear"] + w * overlay["bear"],
        "sideways": (1 - w) * base["sideways"] + w * overlay["sideways"],
    }
    total = mixed["bull"] + mixed["bear"] + mixed["sideways"]
    if total <= 0:
        return dict(base)
    return {key: value / total for key, value in mixed.items()}


def map_hmm_probabilities_to_regime_mixture(
    probabilities: list[float],
    means: list[float],
) -> dict[RegimeState, float] | None:
    if len(probabilities) != len(means) or not probabilities:
        return None
    ranked = sorted(enumerate(means), key=lambda item: item[1])
    bear_index = ranked[0][0]
    bull_index = ranked[-1][0]
    bull = 0.0
    bear = 0.0
    sideways = 0.0
    for index, probability in enumerate(probabilities):
        if index == bear_index:
            bear += probability
        elif index == bull_index:
            bull += probability
        else:
            sideways += probability
    total = bull + bear + sideways
    if total <= 0:
        return None
    return {
        "bull": bull / total,
        "bear": bear / total,
        "sideways": sideways / total,
    }
