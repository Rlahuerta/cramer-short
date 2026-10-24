"""Survival interpolation and distribution point generation.

Mirrors TS logic from src/tools/finance/markov-distribution.ts:
  - Linear survival interpolation from anchored tables
  - Anchor blending with trust scores and mixing weights
  - Monte Carlo confidence interval widening
"""

from __future__ import annotations


def interpolate_survival(
    distribution: list[dict],
    target_price: float,
) -> float:
    if not distribution:
        return 0.5
    prices = [d["price"] for d in distribution]
    probs = [d["probability"] for d in distribution]

    if target_price <= prices[0]:
        return 1.0
    if target_price >= prices[-1]:
        return 0.0

    for i in range(len(prices) - 1):
        lo_price = prices[i]
        hi_price = prices[i + 1]
        if lo_price <= target_price <= hi_price:
            t = (target_price - lo_price) / (hi_price - lo_price)
            return probs[i] + t * (probs[i + 1] - probs[i])

    return 0.0


def interpolate_distribution(
    current_price: float,
    horizon: int,
    P,
    regime_stats: dict,
    initial_state,
    anchors: list[dict],
    second_eigenvalue: float,
    num_levels: int = 20,
    monte_carlo_samples: int = 1000,
    ci_width_multiplier: float = 1.0,
    momentum_adjustment: float = 0.0,
    hmm_override: dict[str, float] | None = None,
    daily_vol: float | None = None,
    start_mixture: dict | None = None,
    nu: int = 5,
    regime_specific_sigma: bool = False,
    regime_specific_sigma_threshold: float | None = None,
    sample_size: int | None = None,
    garch_scales: list[float] | None = None,
    terminal_state_weights: list[float] | None = None,
) -> list[dict]:
    import math

    import numpy as np

    from research.models.trajectory.distributions import student_t_survival
    from research.models.trajectory.driftvol import compute_horizon_drift_vol, compute_mixing_weight

    vol = daily_vol or 0.015
    vol_range = 3.5 * vol * math.sqrt(horizon)
    half_range = max(0.15, min(0.90, vol_range))
    min_price = current_price * (1 - half_range)
    max_price = current_price * (1 + half_range)

    for anchor in anchors:
        price = float(anchor["price"])
        if price < min_price:
            min_price = price * 0.95
        if price > max_price:
            max_price = price * 1.05

    prices = [
        min_price * math.pow(max_price / min_price, step / num_levels)
        for step in range(num_levels + 1)
    ]

    for anchor in anchors:
        price = float(anchor["price"])
        closest_dist = min(abs(existing - price) / price for existing in prices)
        if closest_dist > 0.005:
            prices.append(price)
    prices.sort()

    mix_weight = compute_mixing_weight(second_eigenvalue, horizon)
    horizon_stats = compute_horizon_drift_vol(
        horizon,
        P,
        regime_stats,
        initial_state,
        momentum_adjustment,
        start_mixture,
        hmm_override,
        regime_specific_sigma,
        regime_specific_sigma_threshold,
        garch_scales,
        terminal_state_weights,
    )
    mu_n = float(horizon_stats["mu_n"])
    sigma_n = float(horizon_stats["sigma_n"])

    def find_anchor(price: float) -> dict | None:
        tolerance_pct = 0.02
        raw = next(
            (
                anchor
                for anchor in anchors
                if abs(float(anchor["price"]) - price) / price < tolerance_pct
            ),
            None,
        )
        if raw is None:
            return None
        dist_from_current = abs(float(raw["price"]) - current_price) / current_price
        distance_weight = math.exp(-5.0 * dist_from_current * dist_from_current)
        return {**raw, "distanceWeight": distance_weight}

    sample_n = sample_size if sample_size and sample_size > 0 else None
    drift_scale = min(0.20, 1 / math.sqrt(sample_n)) if sample_n else 0.20
    vol_lower_scale = max(0.85, 1 - drift_scale * 0.5) if sample_n else 0.90
    vol_upper_scale = min(1.15, 1 + drift_scale * 0.5) if sample_n else 1.10
    ci_samples: dict[float, list[float]] = {price: [] for price in prices}

    for _ in range(monte_carlo_samples):
        perturbed_mu = mu_n + (float(np.random.random()) - 0.5) * sigma_n * drift_scale
        perturbed_vol = sigma_n * (
            vol_lower_scale + float(np.random.random()) * (vol_upper_scale - vol_lower_scale)
        )
        for price in prices:
            probability = student_t_survival(price, current_price, perturbed_mu, perturbed_vol, nu)
            ci_samples[price].append(probability)

    raw_points: list[dict] = []
    for price in prices:
        anchor = find_anchor(price)
        markov_est = student_t_survival(price, current_price, mu_n, sigma_n, nu)

        if anchor is not None and anchor.get("trustScore") == "high":
            anchor_weight = (1 - mix_weight) * float(anchor["distanceWeight"])
            probability = (1 - anchor_weight) * markov_est + anchor_weight * float(anchor["probability"])
            source = "markov" if anchor_weight < 0.05 else "polymarket" if anchor_weight > 0.5 else "blend"
        elif anchor is not None and anchor.get("trustScore") == "low":
            anchor_weight = (1 - mix_weight) * 0.5 * float(anchor["distanceWeight"])
            probability = (1 - anchor_weight) * markov_est + anchor_weight * float(anchor["probability"])
            source = "blend"
        else:
            probability = markov_est
            source = "markov"

        samples = sorted(ci_samples[price])
        lo = samples[math.floor(0.05 * len(samples))]
        hi = samples[math.floor(0.95 * len(samples))]
        half_width = (hi - lo) / 2
        center = (hi + lo) / 2
        widened_lo = max(0.0, center - half_width * ci_width_multiplier)
        widened_hi = min(1.0, center + half_width * ci_width_multiplier)

        raw_points.append(
            {
                "price": price,
                "probability": probability,
                "lowerBound": widened_lo,
                "upperBound": widened_hi,
                "source": source,
            }
        )

    for index in range(len(raw_points) - 2, -1, -1):
        if raw_points[index]["probability"] < raw_points[index + 1]["probability"]:
            raw_points[index]["probability"] = raw_points[index + 1]["probability"]

    return raw_points
