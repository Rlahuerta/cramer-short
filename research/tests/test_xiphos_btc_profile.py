"""Parity + provenance tests for the xiphos BTC Markov profile."""

from __future__ import annotations

from dataclasses import asdict

from research.models.markov.xiphos_btc_profile import (
    XIPHOS_BTC_MARKOV_PROFILE,
    XIPHOS_BTC_MARKOV_PROFILE_1D,
    XIPHOS_BTC_MARKOV_PROFILE_6H,
)


def test_profile_1d_matches_xiphos_vbparam_1d():
    assert asdict(XIPHOS_BTC_MARKOV_PROFILE_1D) == {
        "warmup_days": 54,
        "decay_rate": 0.9494,
        "return_threshold_multiplier": 0.6771,
        "break_divergence_threshold": 0.1165,
        "abstention_threshold": 0.002,
        "student_t_nu": 16,
        "garch_horizon_cap": 19,
        "garch_calm_ceiling": 1.3603,
        "garch_turbulent_ceiling": 3.097,
        "entropy_window_size": 38,
        "hmm_num_states": 2,
        "meta_vol_window": 40,
        "meta_high_vol_threshold": 0.7824,
        "volume_lookback": 8,
        "volume_threshold_multiplier": 1.4051,
        "enable_garch_vol": True,
        "enable_entropy_ci_modulation": True,
        "use_hmm_regime": True,
        "use_meta_regime": True,
        "trajectory_n_samples": 200,
    }


def test_active_profile_is_1d():
    assert XIPHOS_BTC_MARKOV_PROFILE is XIPHOS_BTC_MARKOV_PROFILE_1D


def test_profile_invariants():
    for p in (XIPHOS_BTC_MARKOV_PROFILE_1D, XIPHOS_BTC_MARKOV_PROFILE_6H):
        assert 0 < p.decay_rate < 1
        assert isinstance(p.student_t_nu, int) and p.student_t_nu > 2
        assert isinstance(p.hmm_num_states, int) and p.hmm_num_states >= 2
        assert p.garch_calm_ceiling < p.garch_turbulent_ceiling
        assert 0 < p.meta_high_vol_threshold < 1
        assert p.volume_threshold_multiplier > 1
        assert p.enable_garch_vol is True
        assert p.use_meta_regime is True
