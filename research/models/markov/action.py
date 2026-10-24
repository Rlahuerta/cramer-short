"""Action signals and price levels.

Mirrors TS logic:
  - BUY/HOLD/SELL recommendation with confidence
  - Scenario overrides for short-horizon crypto
"""

from __future__ import annotations

import math
from typing import Literal

from research.models.markov.core import ActionConfidence, ActionRecommendation


def compute_action_levels(
    distribution: list[dict],
    current_price: float,
) -> dict[str, float]:
    """Extract price levels from the survival distribution for trade sizing."""
    def find_price_at_prob(target_prob: float) -> float:
        if not distribution:
            return current_price
        for index in range(len(distribution) - 1):
            hi = distribution[index]
            lo = distribution[index + 1]
            hi_prob = float(hi["probability"])
            lo_prob = float(lo["probability"])
            if hi_prob >= target_prob >= lo_prob:
                if abs(hi_prob - lo_prob) < 1e-10:
                    return float(hi["price"])
                t = (hi_prob - target_prob) / (hi_prob - lo_prob)
                return float(hi["price"]) + t * (float(lo["price"]) - float(hi["price"]))
        if target_prob >= float(distribution[0]["probability"]):
            return float(distribution[0]["price"])
        return float(distribution[-1]["price"])

    return {
        "medianPrice": find_price_at_prob(0.50),
        "targetPrice": find_price_at_prob(0.30),
        "stopLoss": find_price_at_prob(0.90),
        "bullCase": find_price_at_prob(0.20),
        "bearCase": find_price_at_prob(0.80),
    }


def compute_action_signal(
    distribution: list[dict],
    current_price: float,
    buy_threshold: float = 0.05,
    sell_threshold: float = 0.03,
    horizon: int = 30,
    recent_vol: float | None = None,
    scenarios: dict | None = None,
    asset_type: Literal["etf", "equity", "crypto", "commodity"] | None = None,
) -> dict:
    """Compute action signal with recommendation and confidence."""
    from research.models.trajectory import interpolate_survival

    if not distribution:
        action_levels = compute_action_levels([], current_price)
        return {
            "buyProbability": 0.0,
            "holdProbability": 1.0,
            "sellProbability": 0.0,
            "recommendation": "HOLD",
            "baseRecommendation": "HOLD",
            "recommendationSource": "expected_return",
            "confidence": "LOW",
            "expectedReturn": 0.0,
            "riskRewardRatio": 1.0,
            "buyThreshold": buy_threshold,
            "sellThreshold": sell_threshold,
            "actionLevels": action_levels,
        }

    p_above_buy = interpolate_survival(distribution, current_price * (1 + buy_threshold))
    p_above_sell = interpolate_survival(distribution, current_price * (1 - sell_threshold))
    p_below_sell = 1 - p_above_sell
    p_hold = max(0.0, 1 - p_above_buy - p_below_sell)

    expected_price = 0.0
    for index in range(len(distribution) - 1):
        mass = float(distribution[index]["probability"]) - float(distribution[index + 1]["probability"])
        mid = (float(distribution[index]["price"]) + float(distribution[index + 1]["price"])) / 2
        expected_price += mass * mid
    expected_price += (1 - float(distribution[0]["probability"])) * float(distribution[0]["price"])
    expected_price += float(distribution[-1]["probability"]) * float(distribution[-1]["price"])

    expected_return = (
        (expected_price - current_price) / current_price
        if current_price > 0 and math.isfinite(expected_price)
        else 0.0
    )

    expected_upside = 0.0
    expected_downside = 0.0
    for index in range(len(distribution) - 1):
        mass = float(distribution[index]["probability"]) - float(distribution[index + 1]["probability"])
        mid = (float(distribution[index]["price"]) + float(distribution[index + 1]["price"])) / 2
        expected_upside += mass * max(0.0, mid - current_price)
        expected_downside += mass * max(0.0, current_price - mid)
    expected_downside += (1 - float(distribution[0]["probability"])) * max(
        0.0, current_price - float(distribution[0]["price"])
    )
    expected_upside += float(distribution[-1]["probability"]) * max(
        0.0, float(distribution[-1]["price"]) - current_price
    )

    raw_rrr = expected_upside / expected_downside if expected_downside > 0 else 1.0
    risk_reward_ratio = raw_rrr if math.isfinite(raw_rrr) else 1.0

    if recent_vol is not None and recent_vol > 0:
        vol_scaled = recent_vol * math.sqrt(horizon)
        action_buy_thr = max(0.001, 0.08 * vol_scaled)
        action_sell_thr = max(0.001, 0.06 * vol_scaled)
    else:
        action_buy_thr = 0.003 if horizon <= 7 else 0.005 if horizon <= 30 else 0.008
        action_sell_thr = 0.002 if horizon <= 7 else 0.003 if horizon <= 30 else 0.005

    if expected_return > action_buy_thr:
        recommendation: ActionRecommendation = "BUY"
    elif expected_return < -action_sell_thr:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    base_recommendation = recommendation
    recommendation_source = "expected_return"

    short_horizon_crypto = asset_type == "crypto" and horizon <= 14
    if short_horizon_crypto and recommendation == "HOLD" and scenarios:
        scenarios_p_up = float(scenarios.get("pUp", scenarios.get("p_up", 0.5)))
        if scenarios_p_up >= 0.55 and expected_return >= 0 and risk_reward_ratio >= 1:
            scenario_recommendation: ActionRecommendation = "BUY"
        elif scenarios_p_up <= 0.45 and expected_return <= 0 and risk_reward_ratio <= 1:
            scenario_recommendation = "SELL"
        else:
            scenario_recommendation = "HOLD"
        recommendation = scenario_recommendation
        if recommendation != base_recommendation:
            recommendation_source = "short_horizon_scenario"

    if scenarios:
        scenario_p_up = float(scenarios.get("pUp", scenarios.get("p_up", 0.5)))
        buckets = scenarios.get("buckets", [])
        up_scenarios = sum(float(bucket.get("probability", 0.0)) for bucket in buckets[3:5])
        down_scenarios = sum(float(bucket.get("probability", 0.0)) for bucket in buckets[0:2])

        if recommendation == "BUY":
            if scenario_p_up < 0.45 and risk_reward_ratio < 1:
                recommendation = "HOLD"
            elif down_scenarios > up_scenarios + 0.05:
                recommendation = "HOLD"
        elif recommendation == "SELL":
            if scenario_p_up > 0.55 and risk_reward_ratio > 1:
                recommendation = "HOLD"
            elif up_scenarios > down_scenarios + 0.05:
                recommendation = "HOLD"

    active_thr = action_buy_thr if recommendation == "BUY" else action_sell_thr
    conviction = abs(expected_return)
    confidence: ActionConfidence = (
        "HIGH"
        if conviction >= 2 * active_thr
        else "MEDIUM"
        if conviction >= active_thr
        else "LOW"
    )

    action_levels = compute_action_levels(distribution, current_price)
    median_return = (action_levels["medianPrice"] - current_price) / current_price if current_price > 0 else 0.0
    if (
        (expected_return > 0 and median_return < -0.005)
        or (expected_return < 0 and median_return > 0.005)
    ) and confidence == "HIGH":
        confidence = "MEDIUM"

    return {
        "buyProbability": p_above_buy,
        "holdProbability": p_hold,
        "sellProbability": p_below_sell,
        "recommendation": recommendation,
        "baseRecommendation": base_recommendation,
        "recommendationSource": recommendation_source,
        "confidence": confidence,
        "expectedReturn": expected_return,
        "riskRewardRatio": risk_reward_ratio,
        "buyThreshold": buy_threshold,
        "sellThreshold": sell_threshold,
        "actionLevels": action_levels,
    }
