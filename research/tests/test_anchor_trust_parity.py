from __future__ import annotations

from research.utils.anchor_trust import evaluate_anchor_trust


def test_anchor_trust_decision_table():
    assert evaluate_anchor_trust(
        has_volume=True,
        is_young=False,
        is_short_horizon_crypto=False,
        is_long_horizon_crypto=False,
        is_near_target_resolution=False,
    ).trust_score == "high"

    young_short = evaluate_anchor_trust(
        has_volume=True,
        is_young=True,
        is_short_horizon_crypto=True,
        is_long_horizon_crypto=False,
        is_near_target_resolution=True,
    )
    assert young_short.trust_score == "high"
    assert young_short.low_trust_reasons == ["young_market"]

    young_short_off_window = evaluate_anchor_trust(
        has_volume=True,
        is_young=True,
        is_short_horizon_crypto=True,
        is_long_horizon_crypto=False,
        is_near_target_resolution=False,
    )
    assert young_short_off_window.trust_score == "low"
    assert young_short_off_window.low_trust_reasons == ["young_market", "resolution_mismatch"]

    mature_long = evaluate_anchor_trust(
        has_volume=True,
        is_young=False,
        is_short_horizon_crypto=False,
        is_long_horizon_crypto=True,
        is_near_target_resolution=True,
    )
    assert mature_long.trust_score == "high"
    assert mature_long.low_trust_reasons == []

    mature_long_off_window = evaluate_anchor_trust(
        has_volume=True,
        is_young=False,
        is_short_horizon_crypto=False,
        is_long_horizon_crypto=True,
        is_near_target_resolution=False,
    )
    assert mature_long_off_window.trust_score == "low"
    assert mature_long_off_window.low_trust_reasons == ["resolution_mismatch"]
