"""Parity tests for ensemble module — Python outputs vs TypeScript fixtures."""

import math

import pytest

from research.models.ensemble import (
    MarketInput,
    OtherSignals,
    adjust_yes_bias,
    compute_market_quality,
    compute_conditional_return,
    compute_polymarket_signal,
    compute_ensemble,
    compute_variance,
    compute_ci,
    compute_quality_score,
    score_to_grade,
    run_ensemble,
)


# ---------------------------------------------------------------------------
# adjust_yes_bias
# ---------------------------------------------------------------------------

def test_adjust_yes_bias_p07():
    # Multiplicative: 0.7 * 0.95 = 0.665
    assert adjust_yes_bias(0.7) == pytest.approx(0.665, abs=1e-6)


def test_adjust_yes_bias_p04():
    # Additive: p ≤ 0.5 → unchanged
    assert adjust_yes_bias(0.4) == pytest.approx(0.4, abs=1e-6)


def test_adjust_yes_bias_p05():
    # Additive: p ≤ 0.5 → unchanged (boundary, no discontinuity)
    assert adjust_yes_bias(0.5) == pytest.approx(0.5, abs=1e-6)


def test_adjust_yes_bias_p0535():
    # Additive: p > 0.5 → p - 0.035 = 0.500
    assert adjust_yes_bias(0.535) == pytest.approx(0.500, abs=1e-6)


def test_adjust_yes_bias_p099():
    # Additive: p > 0.5 → p - 0.035 = 0.955
    assert adjust_yes_bias(0.99) == pytest.approx(0.955, abs=1e-6)


def test_adjust_yes_bias_p001():
    # p ≤ 0.5 → unchanged, clamped to floor 0.01
    assert adjust_yes_bias(0.01) == pytest.approx(0.01, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_market_quality
# ---------------------------------------------------------------------------

def test_compute_market_quality_mature_high_vol_macro():
    m = MarketInput(
        question="Fed cut",
        probability=0.6,
        volume24h_usd=1_000_000,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.05,
        delta_no=-0.03,
    )
    w = compute_market_quality(m)
    # wAge=1, wLiq≈1, tau=0.90, no whale → w≈0.90
    assert w == pytest.approx(0.9, abs=1e-1)


def test_compute_market_quality_new_low_vol_electoral():
    m = MarketInput(
        question="Election result",
        probability=0.5,
        volume24h_usd=10,
        age_days=3,
        signal_tier="electoral",
        delta_yes=0.04,
        delta_no=-0.02,
    )
    w = compute_market_quality(m)
    assert w < 0.05


def test_compute_market_quality_whale_penalty():
    base = MarketInput(
        question="Rate decision",
        probability=0.7,
        volume24h_usd=1_000_000,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.06,
        delta_no=-0.04,
    )
    no_whale = compute_market_quality(base)
    whale = compute_market_quality(
        MarketInput(**{**base.__dict__, "price_spike_detected": True})
    )
    assert whale == pytest.approx(no_whale * 0.5, abs=1e-5)


def test_compute_market_quality_transitory_penalty():
    base = MarketInput(
        question="Persistence reversal",
        probability=0.7,
        volume24h_usd=1_000_000,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.06,
        delta_no=-0.04,
    )
    baseline = compute_market_quality(base)
    transitory = compute_market_quality(
        MarketInput(**{**base.__dict__, "transitory_move": True})
    )
    assert transitory == pytest.approx(baseline * 0.7, abs=1e-5)


def test_compute_market_quality_whale_dominates_transitory():
    base = MarketInput(
        question="Overheated market",
        probability=0.7,
        volume24h_usd=1_000_000,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.06,
        delta_no=-0.04,
    )
    whale_only = compute_market_quality(
        MarketInput(**{**base.__dict__, "price_spike_detected": True})
    )
    both_flags = compute_market_quality(
        MarketInput(
            **{**base.__dict__, "price_spike_detected": True, "transitory_move": True}
        )
    )
    assert both_flags == pytest.approx(whale_only, abs=1e-5)


def test_compute_market_quality_undefined_age():
    m = MarketInput(
        question="Q",
        probability=0.5,
        volume24h_usd=1_000_000,
        age_days=None,
        signal_tier="geopolitical",
        delta_yes=0.05,
        delta_no=-0.03,
    )
    w = compute_market_quality(m)
    # wAge=1, wLiq≈1, tau=0.75 → ~0.75
    assert w == pytest.approx(0.75, abs=1e-1)


# ---------------------------------------------------------------------------
# compute_conditional_return
# ---------------------------------------------------------------------------

def test_compute_conditional_return_typical():
    assert compute_conditional_return(0.7, 0.06, -0.04) == pytest.approx(0.030, abs=1e-6)


def test_compute_conditional_return_p_zero():
    assert compute_conditional_return(0.0, 0.08, -0.05) == pytest.approx(-0.05, abs=1e-6)


def test_compute_conditional_return_p_one():
    assert compute_conditional_return(1.0, 0.08, -0.05) == pytest.approx(0.08, abs=1e-6)


def test_compute_conditional_return_p_half():
    assert compute_conditional_return(0.5, 0.10, -0.10) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_polymarket_signal
# ---------------------------------------------------------------------------

def test_compute_polymarket_signal_empty():
    result = compute_polymarket_signal([])
    assert result["signal"] == 0.0
    assert result["avg_quality"] == 0.0
    assert len(result["warnings"]) == 1
    assert "No Polymarket markets" in result["warnings"][0]


def test_compute_polymarket_signal_single_market():
    m = MarketInput(
        question="Oil supply cut",
        probability=0.65,
        volume24h_usd=500_000,
        age_days=30,
        signal_tier="macro",
        delta_yes=0.08,
        delta_no=-0.03,
    )
    result = compute_polymarket_signal([m])
    expected = compute_conditional_return(adjust_yes_bias(0.65), 0.08, -0.03)
    assert result["signal"] == pytest.approx(expected, abs=1e-6)
    assert len(result["warnings"]) == 0


def test_compute_polymarket_signal_two_markets():
    m1 = MarketInput(
        question="M1",
        probability=0.7,
        volume24h_usd=1_000_000,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.10,
        delta_no=-0.05,
    )
    m2 = MarketInput(
        question="M2",
        probability=0.3,
        volume24h_usd=1_000_000,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.06,
        delta_no=-0.02,
    )
    result = compute_polymarket_signal([m1, m2])
    r1 = compute_conditional_return(adjust_yes_bias(0.7), 0.10, -0.05)
    r2 = compute_conditional_return(adjust_yes_bias(0.3), 0.06, -0.02)
    assert result["signal"] == pytest.approx((r1 + r2) / 2, abs=1e-5)


def test_compute_polymarket_signal_whale_warning():
    m = MarketInput(
        question="Whale market",
        probability=0.55,
        volume24h_usd=100_000,
        age_days=14,
        price_spike_detected=True,
        signal_tier="geopolitical",
        delta_yes=0.05,
        delta_no=-0.03,
    )
    result = compute_polymarket_signal([m])
    assert any("price spike" in w for w in result["warnings"])


def test_compute_polymarket_signal_transitory_warning():
    m = MarketInput(
        question="Reversed market",
        probability=0.55,
        volume24h_usd=100_000,
        age_days=14,
        transitory_move=True,
        signal_tier="geopolitical",
        delta_yes=0.05,
        delta_no=-0.03,
    )
    result = compute_polymarket_signal([m])
    assert any("transitory 24-48h move" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# compute_ensemble
# ---------------------------------------------------------------------------

def test_compute_ensemble_all_signals():
    others = OtherSignals(
        sentiment_score=0.5,
        fundamental_return=0.12,
        options_skew=1,
        horizon_days=7,
    )
    result = compute_ensemble(0.02, 1.0, others)
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-5)

    r_pm = 0.02
    r_sent = 0.5 * 0.04
    r_fund = 0.12 * (7 / 365)
    r_opt = 1 * 0.03
    expected = 0.40 * r_pm + 0.20 * r_sent + 0.25 * r_fund + 0.15 * r_opt
    assert result["forecast_return"] == pytest.approx(expected, abs=1e-5)


def test_compute_ensemble_missing_sentiment():
    others = OtherSignals(fundamental_return=0.10, options_skew=-1, horizon_days=7)
    result = compute_ensemble(0.01, 0.8, others)
    assert "sentiment" not in result["weights"]
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-5)


def test_compute_ensemble_only_pm():
    result = compute_ensemble(0.03, 0.5, OtherSignals())
    assert result["weights"]["pm"] == pytest.approx(1.0, abs=1e-5)
    assert result["forecast_return"] == pytest.approx(0.03, abs=1e-5)


def test_compute_ensemble_includes_markov():
    others = OtherSignals(markov_return=0.03, horizon_days=7)
    result = compute_ensemble(0.02, 1.0, others)
    assert "markov" in result["weights"]
    assert "pm" in result["weights"]
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["forecast_return"] > 0.02


def test_compute_ensemble_markov_absent():
    others = OtherSignals(
        sentiment_score=0.5,
        fundamental_return=0.12,
        options_skew=1,
        horizon_days=7,
    )
    without_markov = compute_ensemble(0.02, 1.0, others)
    with_undefined = compute_ensemble(
        0.02, 1.0, OtherSignals(**{**others.__dict__, "markov_return": None})
    )
    assert with_undefined["forecast_return"] == pytest.approx(
        without_markov["forecast_return"], abs=1e-8
    )
    assert with_undefined["weights"] == without_markov["weights"]


# ---------------------------------------------------------------------------
# compute_quality_score
# ---------------------------------------------------------------------------

def test_compute_quality_score_high():
    markets = [
        MarketInput(
            question=f"M{i}",
            probability=0.5,
            volume24h_usd=1_000_000,
            age_days=30,
            signal_tier="macro",
            delta_yes=0.05,
            delta_no=-0.03,
        )
        for i in range(5)
    ]
    score = compute_quality_score(markets, 1.0, 0.01, 4, 0)
    assert score >= 80


def test_compute_quality_score_zero_markets():
    score = compute_quality_score([], 0, 0.10, 1, 0)
    assert score < 20


def test_compute_quality_score_integer_range():
    score = compute_quality_score([], 0, 1.0, 0, 0)
    assert score == 0
    assert isinstance(score, int)


# ---------------------------------------------------------------------------
# score_to_grade
# ---------------------------------------------------------------------------

def test_score_to_grade_boundaries():
    assert score_to_grade(80) == "A"
    assert score_to_grade(100) == "A"
    assert score_to_grade(60) == "B"
    assert score_to_grade(79) == "B"
    assert score_to_grade(40) == "C"
    assert score_to_grade(59) == "C"
    assert score_to_grade(39) == "D"
    assert score_to_grade(0) == "D"


# ---------------------------------------------------------------------------
# compute_ci
# ---------------------------------------------------------------------------

def test_compute_ci_typical():
    ci = compute_ci(100, 0.05)
    assert ci["low"] == pytest.approx(100 * (1 - 1.96 * 0.05), abs=1e-5)
    assert ci["high"] == pytest.approx(100 * (1 + 1.96 * 0.05), abs=1e-5)


def test_compute_ci_zero_sigma():
    ci = compute_ci(200, 0)
    assert ci["low"] == 200
    assert ci["high"] == 200


# ---------------------------------------------------------------------------
# compute_variance
# ---------------------------------------------------------------------------

def test_compute_variance_empty_markets():
    assert compute_variance([], 0.4, 0.2, 0.5) == 0.05


def test_compute_variance_single_market():
    m = MarketInput(
        question="Q",
        probability=0.6,
        volume24h_usd=100_000,
        age_days=21,
        signal_tier="geopolitical",
        delta_yes=0.08,
        delta_no=-0.04,
    )
    sigma = compute_variance([m], 0.4, 0.2, 0.3)
    assert sigma > 0
    assert math.isfinite(sigma)


def test_compute_variance_zero_volume():
    m = MarketInput(
        question="Q",
        probability=0.5,
        volume24h_usd=0,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.10,
        delta_no=-0.10,
    )
    # With pmWeight=1, sentWeight=0, optWeight=0
    assert compute_variance([m], 1.0, 0, 0) == 0


def test_compute_variance_spread_monotonic():
    def mk(spread):
        return MarketInput(
            question="Q",
            probability=0.5,
            volume24h_usd=500_000,
            age_days=21,
            signal_tier="macro",
            delta_yes=spread / 2,
            delta_no=-spread / 2,
        )

    s1 = compute_variance([mk(0.04)], 1.0, 0, 0)
    s2 = compute_variance([mk(0.10)], 1.0, 0, 0)
    s3 = compute_variance([mk(0.20)], 1.0, 0, 0)
    assert s1 < s2 < s3


def test_compute_variance_sentiment_only():
    m = MarketInput(
        question="Q",
        probability=0.5,
        volume24h_usd=500_000,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.10,
        delta_no=-0.10,
    )
    sent_weight = 0.20
    sigma = compute_variance([m], 0, sent_weight, None)
    assert sigma == pytest.approx(sent_weight * 0.04 * 1.2, abs=1e-6)


def test_compute_variance_doubles_with_pm_weight():
    m = MarketInput(
        question="Q",
        probability=0.5,
        volume24h_usd=1_000_000,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.10,
        delta_no=-0.10,
    )
    sigma_half = compute_variance([m], 0.5, 0, None)
    sigma_full = compute_variance([m], 1.0, 0, None)
    assert sigma_full == pytest.approx(sigma_half * 2, abs=1e-4)


# ---------------------------------------------------------------------------
# run_ensemble — end-to-end
# ---------------------------------------------------------------------------

MARKETS = [
    MarketInput(
        question="OPEC supply cut",
        probability=0.65,
        volume24h_usd=800_000,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.06,
        delta_no=-0.04,
    ),
    MarketInput(
        question="US sanctions relief",
        probability=0.40,
        volume24h_usd=300_000,
        age_days=14,
        signal_tier="geopolitical",
        delta_yes=0.05,
        delta_no=-0.02,
    ),
]

OTHERS = OtherSignals(
    sentiment_score=0.3,
    fundamental_return=0.08,
    options_skew=1,
    horizon_days=7,
)


def test_run_ensemble_forecast_price():
    result = run_ensemble(100, MARKETS, OTHERS)
    expected = 100 * (1 + result.forecast_return)
    assert result.forecast_price == pytest.approx(expected, abs=1e-6)


def test_run_ensemble_ci_brackets():
    result = run_ensemble(100, MARKETS, OTHERS)
    assert result.ci_low95 < result.forecast_price
    assert result.ci_high95 > result.forecast_price


def test_run_ensemble_sigma_finite_positive():
    result = run_ensemble(100, MARKETS, OTHERS)
    assert result.sigma > 0
    assert math.isfinite(result.sigma)


def test_run_ensemble_valid_grade():
    result = run_ensemble(100, MARKETS, OTHERS)
    assert result.quality_grade in {"A", "B", "C", "D"}


def test_run_ensemble_pm_weight_range():
    result = run_ensemble(100, MARKETS, OTHERS)
    assert 0 <= result.pm_effective_weight <= 0.40


def test_run_ensemble_no_markets_warnings():
    result = run_ensemble(100, [], OTHERS)
    assert len(result.warnings) > 0


def test_run_ensemble_pm_signal_matches():
    result = run_ensemble(100, MARKETS, OTHERS)
    expected_signal = compute_polymarket_signal(MARKETS)["signal"]
    assert result.pm_signal == pytest.approx(expected_signal, abs=1e-5)


def test_run_ensemble_markov_influence():
    without_markov = run_ensemble(100, MARKETS, OTHERS)
    with_markov = run_ensemble(
        100, MARKETS, OtherSignals(**{**OTHERS.__dict__, "markov_return": 0.025})
    )
    assert with_markov.forecast_return != pytest.approx(
        without_markov.forecast_return, abs=1e-8
    )
    assert with_markov.quality_score > without_markov.quality_score


# ---------------------------------------------------------------------------
# run_ensemble — real price scaling
# ---------------------------------------------------------------------------

def test_run_ensemble_price_scaling():
    result = run_ensemble(414.84, MARKETS[:1], OtherSignals(horizon_days=7))
    assert result.forecast_price == pytest.approx(
        414.84 * (1 + result.forecast_return), abs=1e-4
    )


def test_run_ensemble_ci_near_real_price():
    result = run_ensemble(414.84, MARKETS[:1], OtherSignals(horizon_days=7))
    assert result.ci_low95 > 300
    assert result.ci_high95 > 300


def test_run_ensemble_ci_brackets_real_price():
    result = run_ensemble(414.84, MARKETS[:1], OtherSignals(horizon_days=7))
    assert result.ci_low95 < result.forecast_price
    assert result.ci_high95 > result.forecast_price


def test_run_ensemble_relative_width_scale_invariant():
    r100 = run_ensemble(100, MARKETS[:1], OtherSignals(horizon_days=7))
    r414 = run_ensemble(414.84, MARKETS[:1], OtherSignals(horizon_days=7))
    r634 = run_ensemble(634.09, MARKETS[:1], OtherSignals(horizon_days=7))
    rel100 = (r100.ci_high95 - r100.ci_low95) / r100.forecast_price
    rel414 = (r414.ci_high95 - r414.ci_low95) / r414.forecast_price
    rel634 = (r634.ci_high95 - r634.ci_low95) / r634.forecast_price
    assert rel100 == pytest.approx(rel414, abs=1e-4)
    assert rel100 == pytest.approx(rel634, abs=1e-4)


def test_run_ensemble_forecast_return_independent_of_price():
    r100 = run_ensemble(100, MARKETS[:1], OtherSignals(horizon_days=7))
    r634 = run_ensemble(634.09, MARKETS[:1], OtherSignals(horizon_days=7))
    assert r100.forecast_return == pytest.approx(r634.forecast_return, abs=1e-8)


def test_run_ensemble_sigma_independent_of_price():
    s100 = run_ensemble(100, MARKETS[:1], OtherSignals(horizon_days=7)).sigma
    s500 = run_ensemble(500, MARKETS[:1], OtherSignals(horizon_days=7)).sigma
    assert s100 == pytest.approx(s500, abs=1e-8)


# ---------------------------------------------------------------------------
# run_ensemble — sigma floor
# ---------------------------------------------------------------------------

ZERO_VOLUME_MARKETS = [
    MarketInput(
        question="Extreme-probability event",
        probability=0.99,
        volume24h_usd=0,
        age_days=21,
        signal_tier="macro",
        delta_yes=0.05,
        delta_no=-0.02,
    ),
]


def test_run_ensemble_sigma_floor_7d():
    result = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=7))
    expected_floor = 0.10 * math.sqrt(7 / 252)
    assert result.sigma == pytest.approx(expected_floor, abs=1e-4)


def test_run_ensemble_sigma_floor_7d_percent():
    result = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=7))
    assert result.sigma * 100 == pytest.approx(1.667, abs=1e-2)


def test_run_ensemble_sigma_floor_30d():
    result = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=30))
    expected_floor = 0.10 * math.sqrt(30 / 252)
    assert result.sigma == pytest.approx(expected_floor, abs=1e-4)


def test_run_ensemble_sigma_floor_90d():
    result = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=90))
    expected_floor = 0.10 * math.sqrt(90 / 252)
    assert result.sigma == pytest.approx(expected_floor, abs=1e-4)


def test_run_ensemble_sigma_floor_252d():
    result = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=252))
    assert result.sigma == pytest.approx(0.10, abs=1e-4)


def test_run_ensemble_sigma_floor_monotonic():
    s7 = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=7)).sigma
    s30 = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=30)).sigma
    s90 = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=90)).sigma
    s252 = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=252)).sigma
    assert s7 < s30 < s90 < s252


def test_run_ensemble_floor_prevents_zero_ci():
    result = run_ensemble(100, ZERO_VOLUME_MARKETS, OtherSignals(horizon_days=7))
    assert result.ci_high95 - result.ci_low95 > 0
    assert result.ci_high95 > result.forecast_price
    assert result.ci_low95 < result.forecast_price


# ---------------------------------------------------------------------------
# compute_ensemble — degenerate pmAvgQuality = 0
# ---------------------------------------------------------------------------

def test_compute_ensemble_pm_only_zero_quality():
    result = compute_ensemble(0.03, 0, OtherSignals())
    assert result["weights"]["pm"] == pytest.approx(1.0, abs=1e-5)
    assert result["forecast_return"] == pytest.approx(0.03, abs=1e-5)


def test_compute_ensemble_pm_and_sentiment_zero_quality():
    sentiment_score = 0.5
    result = compute_ensemble(0.03, 0, OtherSignals(sentiment_score=sentiment_score))
    assert result["weights"]["pm"] == pytest.approx(0.0, abs=1e-5)
    assert result["weights"]["sentiment"] == pytest.approx(1.0, abs=1e-5)
    assert result["forecast_return"] == pytest.approx(sentiment_score * 0.04, abs=1e-6)


def test_compute_ensemble_full_quality():
    result = compute_ensemble(0.05, 1.0, OtherSignals())
    assert result["weights"]["pm"] == pytest.approx(1.0, abs=1e-5)


def test_compute_ensemble_partial_quality_normalises():
    result = compute_ensemble(0.05, 0.5, OtherSignals(sentiment_score=0.3))
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-5)
