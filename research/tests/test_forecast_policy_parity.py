from __future__ import annotations

from research.models.forecast_policy import (
    ForecastTrustPolicyInput,
    compute_forecast_trust_policy,
)


def test_structural_break_with_strong_terminal_support_can_stay_full():
    policy = compute_forecast_trust_policy(
        ForecastTrustPolicyInput(
            leverage=2,
            semantic_primary="terminal",
            is_divergent=False,
            markov_confidence=0.78,
            structural_break=True,
            flat_probability=0.28,
            has_terminal_support=True,
            conformal_applied=True,
            conformal_radius=0.039,
            conformal_coverage=0.91,
            conformal_mode="normal",
        )
    )
    assert policy.level == "full"
    assert policy.trade_eligible is True


def test_structural_break_with_stressed_conformal_and_no_terminal_support_abstains():
    policy = compute_forecast_trust_policy(
        ForecastTrustPolicyInput(
            leverage=5,
            semantic_primary="terminal",
            is_divergent=False,
            markov_confidence=0.58,
            structural_break=True,
            flat_probability=0.74,
            has_terminal_support=False,
            conformal_applied=True,
            conformal_radius=0.088,
            conformal_coverage=0.61,
            conformal_mode="break",
        )
    )
    assert policy.level in {"context-only", "abstain"}
    assert policy.trade_eligible is False
