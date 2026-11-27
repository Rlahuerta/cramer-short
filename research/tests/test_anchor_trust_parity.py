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
    assert young_short.trust_weight == 0.7
    assert young_short.low_trust_reasons == ["young_market"]

    young_short_off_window = evaluate_anchor_trust(
        has_volume=True,
        is_young=True,
        is_short_horizon_crypto=True,
        is_long_horizon_crypto=False,
        is_near_target_resolution=False,
    )
    assert young_short_off_window.trust_score == "low"
    assert young_short_off_window.trust_weight == 0.35
    assert young_short_off_window.low_trust_reasons == ["young_market", "resolution_mismatch"]

    mature_long = evaluate_anchor_trust(
        has_volume=True,
        is_young=False,
        is_short_horizon_crypto=False,
        is_long_horizon_crypto=True,
        is_near_target_resolution=True,
    )
    assert mature_long.trust_score == "high"
    assert mature_long.trust_weight == 0.9
    assert mature_long.low_trust_reasons == []

    mature_long_off_window = evaluate_anchor_trust(
        has_volume=True,
        is_young=False,
        is_short_horizon_crypto=False,
        is_long_horizon_crypto=True,
        is_near_target_resolution=False,
    )
    assert mature_long_off_window.trust_score == "low"
    assert mature_long_off_window.trust_weight == 0.35
    assert mature_long_off_window.low_trust_reasons == ["resolution_mismatch"]


def test_rescale_anchor_to_horizon_matches_typescript():
    from research.utils.anchor_trust import rescale_anchor_to_horizon

    # Values from confidence-intervals.ts rescaleAnchorToHorizon; its normal CDF is a ~1e-7 approximation.
    cases = [
        ((0.2, 7, 14), 0.2758829351419396),
        ((0.8, 0.67, 1), 0.7545558752893825),
        ((0.35, 6.5, 2), 0.243638326456473),
        ((0.05, 3, 1), 0.002193116201492873),
        ((0.5, 7, 14), 0.5),
        ((0.3, 0, 14), 0.3),
        ((1.0, 2, 14), 1.0),
    ]
    for (probability, market_days, horizon_days), expected in cases:
        assert abs(rescale_anchor_to_horizon(probability, market_days, horizon_days) - expected) < 1e-6
