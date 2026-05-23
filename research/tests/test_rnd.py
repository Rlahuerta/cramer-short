import math

import numpy as np
import pytest
from scipy import stats

from research.models.rnd import (
    DEFAULT_MPR_CAP,
    fit_lognormal_from_strikes,
    lognormal_to_regime_probabilities,
    nudge_transition_matrix,
    transform_q_to_p,
    transform_q_to_p_with_shift,
)


class TestTransformQToP:
    """Q→P measure transformation tests."""

    def test_identity_when_lambda_zero(self):
        """When historical drift = risk-free rate, P^P = P^Q."""
        q = 0.3
        p = transform_q_to_p(q, historical_drift=0.05, risk_free_rate=0.05, volatility=0.5, days_to_expiry=30)
        assert pytest.approx(p, abs=1e-6) == q

    def test_increases_for_bullish(self):
        """Positive risk premium increases P(S_T > K) for K > S_0."""
        q = 0.3
        p = transform_q_to_p(q, historical_drift=0.40, risk_free_rate=0.05, volatility=0.5, days_to_expiry=30)
        assert p > q

    def test_increases_for_bearish_survival(self):
        """Positive risk premium shifts distribution right, so P(S_T > K)
        for K << S_0 increases (more likely to stay above a low strike)."""
        q = 0.8
        p = transform_q_to_p(q, historical_drift=0.40, risk_free_rate=0.05, volatility=0.5, days_to_expiry=30)
        assert p > q

    def test_clip_extremes(self):
        """Extreme probabilities are handled safely without infinities."""
        p_low = transform_q_to_p(0.0001, historical_drift=0.40, risk_free_rate=0.05, volatility=0.5, days_to_expiry=30)
        p_high = transform_q_to_p(0.9999, historical_drift=0.40, risk_free_rate=0.05, volatility=0.5, days_to_expiry=30)
        assert 0 <= p_low <= 1
        assert 0 <= p_high <= 1

    def test_boundary_zero_and_one(self):
        """Exactly 0 and 1 are returned unchanged."""
        assert transform_q_to_p(0.0, 0.1, 0.05, 0.3, 7) == 0.0
        assert transform_q_to_p(1.0, 0.1, 0.05, 0.3, 7) == 1.0

    def test_default_mpr_cap_is_15(self):
        assert DEFAULT_MPR_CAP == 1.5

    def test_caps_pathological_mpr(self):
        """Drift well above what historical vol can justify must clamp at the cap."""
        p = transform_q_to_p(0.30, historical_drift=3.0, risk_free_rate=0.05, volatility=0.3, days_to_expiry=30)
        # Without the cap the shift would be ~9.83·sqrt(30/365) ≈ 2.82 → P≈0.999
        # With cap 1.5 the shift is 1.5·sqrt(30/365) ≈ 0.43 → P≈0.61
        assert p < 0.97

    def test_explicit_cap_narrows_shift(self):
        wide = transform_q_to_p(0.30, 0.50, 0.05, 0.30, 30, mpr_cap=5.0)
        tight = transform_q_to_p(0.30, 0.50, 0.05, 0.30, 30, mpr_cap=0.1)
        assert wide > tight

    def test_with_shift_returns_provenance(self):
        out = transform_q_to_p_with_shift(0.30, historical_drift=3.0, risk_free_rate=0.05, volatility=0.3, days_to_expiry=30)
        assert out["mpr_raw"] > 2.0
        assert pytest.approx(out["mpr_used"], abs=1e-6) == 1.5
        assert pytest.approx(out["z_shift"], abs=1e-6) == 1.5 * math.sqrt(30 / 365)
        assert out["longshot_shrinkage_applied"] is False

    def test_with_shift_can_apply_longshot_shrinkage(self):
        out = transform_q_to_p_with_shift(
            0.01,
            historical_drift=0.05,
            risk_free_rate=0.05,
            volatility=0.3,
            days_to_expiry=30,
            apply_longshot=True,
        )
        assert out["longshot_shrinkage_applied"] is True
        assert out["p_prob"] == pytest.approx(0.255, abs=1e-6)
        assert out["longshot_tail_distance"] > 0.45


class TestFitLognormalFromStrikes:
    """Log-Normal parametric fitting tests."""

    def test_recovers_known_params(self):
        """Fit to synthetic log-normal data recovers (mu, sigma)."""
        mu_true, sigma_true = math.log(50000), 0.2
        strikes = [40000, 45000, 50000, 55000, 60000]
        # True survival probs under the known Log-Normal
        d = [(math.log(k) - mu_true) / sigma_true for k in strikes]
        yes_prices = [1.0 - stats.norm.cdf(di) for di in d]

        mu_ln, sigma_ln = fit_lognormal_from_strikes(strikes, yes_prices, current_price=50000)
        assert pytest.approx(mu_ln, rel=0.05) == mu_true
        assert pytest.approx(sigma_ln, rel=0.10) == sigma_true

    def test_handles_arbitrage_violations(self):
        """Non-monotonic quotes still produce a valid CDF."""
        # Higher strike has higher probability (arbitrage violation)
        strikes = [90000, 95000, 100000]
        yes_prices = [0.40, 0.42, 0.30]
        mu_ln, sigma_ln = fit_lognormal_from_strikes(strikes, yes_prices, current_price=95000)
        assert sigma_ln > 0
        # All strikes should map to valid probabilities
        for k in strikes:
            d = (math.log(k) - mu_ln) / sigma_ln
            p = 1.0 - stats.norm.cdf(d)
            assert 0 <= p <= 1

    def test_falls_back_single_point(self):
        """1 strike → returns drift-based estimate."""
        mu_ln, sigma_ln = fit_lognormal_from_strikes([100000], [0.5], current_price=95000)
        assert sigma_ln == pytest.approx(0.3, abs=1e-6)  # fallback vol
        assert math.isfinite(mu_ln)

    def test_produces_positive_survival(self):
        """Fitted model yields valid survival probabilities for all strikes."""
        strikes = [30000, 40000, 50000, 60000, 70000]
        yes_prices = [0.90, 0.70, 0.50, 0.30, 0.10]
        mu_ln, sigma_ln = fit_lognormal_from_strikes(strikes, yes_prices, current_price=50000)
        for k in strikes:
            d = (math.log(k) - mu_ln) / sigma_ln
            p = 1.0 - stats.norm.cdf(d)
            assert 0 <= p <= 1


class TestLognormalToRegimeProbabilities:
    """Regime mapping from fitted Log-Normal."""

    def test_sum_one(self):
        """Output probabilities sum to 1."""
        mu_ln = math.log(50000)
        sigma_ln = 0.2
        probs = lognormal_to_regime_probabilities(mu_ln, sigma_ln, current_price=50000)
        total = sum(probs.values())
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_single_regime_concentrated(self):
        """LN concentrated far above current price → bull ≈ 1."""
        mu_ln = math.log(100000)  # far above current 50000
        sigma_ln = 0.05  # tight
        probs = lognormal_to_regime_probabilities(mu_ln, sigma_ln, current_price=50000)
        assert probs["bull"] > 0.90
        assert probs["bear"] < 0.05

    def test_bearish_distribution(self):
        """LN concentrated far below current price → bear ≈ 1."""
        mu_ln = math.log(30000)  # far below current 50000
        sigma_ln = 0.05
        probs = lognormal_to_regime_probabilities(mu_ln, sigma_ln, current_price=50000)
        assert probs["bear"] > 0.90
        assert probs["bull"] < 0.05

    def test_all_positive(self):
        """All regime probabilities are positive (≥ 0.01 floor)."""
        probs = lognormal_to_regime_probabilities(math.log(50000), 0.2, current_price=50000)
        for v in probs.values():
            assert v >= 0.01


class TestNudgeTransitionMatrix:
    """Transition matrix nudge tests."""

    def test_preserves_rows(self):
        """Row sums still equal 1 after nudge."""
        P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        target = {"bull": 0.5, "bear": 0.3, "sideways": 0.2}
        P_nudged = nudge_transition_matrix(P, "bull", target, horizon=7, quality_score=80)
        for row in P_nudged:
            assert pytest.approx(row.sum(), abs=1e-6) == 1.0

    def test_terminal_match(self):
        """P^nudged^horizon is closer to target than original P^horizon."""
        P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        target = {"bull": 0.60, "bear": 0.25, "sideways": 0.15}
        P_nudged = nudge_transition_matrix(P, "bull", target, horizon=7, quality_score=100)

        Ph_orig = np.linalg.matrix_power(P, 7)
        Ph_nudged = np.linalg.matrix_power(P_nudged, 7)

        target_arr = np.array([0.60, 0.25, 0.15])
        err_orig = np.sum((Ph_orig[0] - target_arr) ** 2)
        err_nudged = np.sum((Ph_nudged[0] - target_arr) ** 2)

        assert err_nudged < err_orig

    def test_strength_scales_with_quality(self):
        """Higher quality score → stronger nudge."""
        P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        target = {"bull": 0.9, "bear": 0.05, "sideways": 0.05}

        P_low = nudge_transition_matrix(P, "bull", target, horizon=7, quality_score=20)
        P_high = nudge_transition_matrix(P, "bull", target, horizon=7, quality_score=100)

        Ph_low = np.linalg.matrix_power(P_low, 7)
        Ph_high = np.linalg.matrix_power(P_high, 7)

        # Higher quality should shift terminal distribution further toward target
        assert Ph_high[0, 0] >= Ph_low[0, 0]

    def test_identity_when_target_matches(self):
        """If target already matches historical, P unchanged."""
        P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        Ph = np.linalg.matrix_power(P, 7)
        target = {"bull": Ph[0, 0], "bear": Ph[0, 1], "sideways": Ph[0, 2]}
        P_nudged = nudge_transition_matrix(P, "bull", target, horizon=7, quality_score=50)
        assert np.allclose(P_nudged, P, atol=1e-6)

    def test_zero_quality_no_nudge(self):
        """Quality score of 0 produces no nudge."""
        P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
        target = {"bull": 0.9, "bear": 0.05, "sideways": 0.05}
        P_nudged = nudge_transition_matrix(P, "bull", target, horizon=7, quality_score=0)
        assert np.allclose(P_nudged, P, atol=1e-12)
