"""Gaussian Hidden Markov Model for financial regime detection.

Wraps hmmlearn.GaussianHMM with a TypeScript-compatible API.

Note on raw-return HMMs:
  Gaussian HMMs maximize likelihood by fitting Gaussian distributions to
  observations. In financial returns, variance differences between regimes
  are typically orders of magnitude larger than mean differences. As a
  result, a 3-state HMM on raw daily returns tends to cluster by
  volatility (e.g. one tight "normal" state + outlier states) rather
  than by return direction. The value of the HMM is in the continuous
  expected_return and expected_volatility forecasts, not in hard state
  labels. For explicit bull/bear/sideways regimes, use the observable
  Markov model in markov.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from hmmlearn import hmm


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class AssetProfile:
    hmm_weight_multiplier: float
    student_t_nu: int
    decay_rate: float


ASSET_PROFILES: dict[str, AssetProfile] = {
    "etf": AssetProfile(hmm_weight_multiplier=1.1, student_t_nu=5, decay_rate=0.97),
    "equity": AssetProfile(hmm_weight_multiplier=0.9, student_t_nu=4, decay_rate=0.96),
    "crypto": AssetProfile(hmm_weight_multiplier=0.5, student_t_nu=3, decay_rate=0.94),
    "commodity": AssetProfile(hmm_weight_multiplier=0.7, student_t_nu=4, decay_rate=0.95),
}


@dataclass
class HMMParams:
    n_states: int
    pi: np.ndarray
    A: np.ndarray
    means: np.ndarray
    stds: np.ndarray


@dataclass
class HMMFitResult:
    params: HMMParams
    log_likelihood: float
    iterations: int
    converged: bool


@dataclass
class HMMPrediction:
    current_state: int
    state_probabilities: np.ndarray
    current_state_probabilities: np.ndarray
    forecast_probabilities: np.ndarray
    expected_return: float
    expected_volatility: float


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_hmm(observations: np.ndarray, n_states: int) -> HMMParams:
    """K-means-style quantile initialization with sorted means.

    State 0 = bearish (lowest mean), state N-1 = bullish (highest mean).
    """
    obs = np.asarray(observations).flatten()
    sorted_obs = np.sort(obs)
    n = len(sorted_obs)

    # Quantile-based means
    quantiles = [int(n * i / n_states) for i in range(n_states + 1)]
    means = np.array([
        float(np.mean(sorted_obs[quantiles[i]:quantiles[i + 1]]))
        for i in range(n_states)
    ])

    # Sort means ascending and track permutation
    order = np.argsort(means)
    means = means[order]

    # Per-state stds from quantile slices
    stds = np.array([
        max(float(np.std(sorted_obs[quantiles[i]:quantiles[i + 1]])), 1e-4)
        for i in range(n_states)
    ])[order]

    # Diagonal-dominant transition matrix
    A = np.eye(n_states) * 0.8 + np.ones((n_states, n_states)) * 0.2 / n_states
    A = A / A.sum(axis=1, keepdims=True)

    # Uniform initial distribution
    pi = np.ones(n_states) / n_states

    return HMMParams(n_states=n_states, pi=pi, A=A, means=means, stds=stds)


# ---------------------------------------------------------------------------
# Baum-Welch fitting
# ---------------------------------------------------------------------------


def baum_welch(
    observations: np.ndarray,
    n_states: int = 3,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    min_std: float = 1e-4,
) -> HMMFitResult:
    """Fit Gaussian HMM via Baum-Welch EM.

    Returns sorted parameters so state 0 = lowest mean, state N-1 = highest.
    Note: on real financial returns the model often collapses to a
    single low-volatility state with a few outlier states. The value is
    in the continuous forecasts (expected_return, expected_volatility),
    not hard regime labels.
    """
    obs = np.asarray(observations).reshape(-1, 1)
    init = initialize_hmm(observations, n_states)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=max_iterations,
        tol=tolerance,
        init_params="",
    )

    model.startprob_ = init.pi.copy()
    model.transmat_ = init.A.copy()
    model.means_ = init.means.reshape(-1, 1).copy()
    model.covars_ = (init.stds ** 2).reshape(-1, 1).copy()

    try:
        model.fit(obs)
        converged = model.monitor_.converged if hasattr(model, "monitor_") else True
        iterations = model.monitor_.iter if hasattr(model, "monitor_") else max_iterations
        log_likelihood = float(model.score(obs))
    except Exception:
        converged = False
        iterations = max_iterations
        log_likelihood = float("-inf")

    # Extract parameters
    means = model.means_.flatten()
    covars = model.covars_.flatten()
    stds = np.sqrt(np.maximum(covars, min_std ** 2))
    A = model.transmat_
    pi = model.startprob_

    # Sort means ascending (bear -> sideways -> bull) and permute everything
    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    pi = pi[order]
    A = A[np.ix_(order, order)]

    params = HMMParams(n_states=n_states, pi=pi, A=A, means=means, stds=stds)
    return HMMFitResult(
        params=params,
        log_likelihood=log_likelihood,
        iterations=iterations,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------


def viterbi(observations: np.ndarray, params: HMMParams) -> list[int]:
    """Decode most likely state sequence via Viterbi."""
    obs = np.asarray(observations).reshape(-1, 1)
    model = _build_model(params)
    states = model.predict(obs)
    return states.tolist()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    observations: np.ndarray,
    params: HMMParams,
    forecast_horizon: int,
) -> HMMPrediction:
    """Compute posterior state probabilities and n-step forecast."""
    obs = np.asarray(observations).reshape(-1, 1)
    model = _build_model(params)

    # Posterior probabilities for the last observation
    probs = model.predict_proba(obs)
    current_state_probs = probs[-1]
    current_state = int(np.argmax(current_state_probs))

    # n-step forecast via matrix power
    A_n = mat_pow(params.A, forecast_horizon)
    forecast_probs = A_n[current_state]

    # Expected return and volatility as weighted averages
    expected_return = float(np.dot(forecast_probs, params.means))
    expected_volatility = float(np.dot(forecast_probs, params.stds))

    return HMMPrediction(
        current_state=current_state,
        state_probabilities=current_state_probs,
        current_state_probabilities=current_state_probs,
        forecast_probabilities=forecast_probs,
        expected_return=expected_return,
        expected_volatility=expected_volatility,
    )


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def mat_pow(A: np.ndarray, n: int) -> np.ndarray:
    """Matrix exponentiation for n-step transition probabilities."""
    if n <= 0:
        return np.eye(A.shape[0])
    return np.linalg.matrix_power(A, n)


# ---------------------------------------------------------------------------
# Volatility HMM helper
# ---------------------------------------------------------------------------


def fit_2state_return_hmm(
    returns: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
) -> dict:
    """Fit a 2-state Gaussian HMM on raw returns.

    Labels states by volatility (low-vol = "calm", high-vol = "volatile").
    Returns stationary probabilities, per-state means/vols, and current
    state information. More balanced than 3-state on volatile assets.

    Returns
    -------
    dict
        converged, state_labels, state_probs, state_means, state_vols,
        expected_return, expected_volatility, current_state, current_state_prob.
    """
    arr = np.asarray(returns)
    if len(arr) < 20:
        return {
            "converged": False,
            "state_labels": ["calm", "volatile"],
            "state_probs": [0.5, 0.5],
            "state_means": [0.0, 0.0],
            "state_vols": [0.01, 0.02],
            "expected_return": 0.0,
            "expected_volatility": 0.01,
            "current_state": 0,
            "current_state_prob": 0.5,
        }

    result = baum_welch(
        arr,
        n_states=2,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    if not result.converged:
        return {
            "converged": False,
            "state_labels": ["calm", "volatile"],
            "state_probs": [0.5, 0.5],
            "state_means": [0.0, 0.0],
            "state_vols": [0.01, 0.02],
            "expected_return": 0.0,
            "expected_volatility": 0.01,
            "current_state": 0,
            "current_state_prob": 0.5,
        }

    params = result.params
    # Sort by volatility (ascending): state 0 = calm, state 1 = volatile
    order = np.argsort(params.stds)
    means = params.means[order]
    vols = params.stds[order]
    A = params.A[np.ix_(order, order)]

    # Stationary distribution via power iteration (more stable than eig)
    stationary = np.ones(2) / 2
    for _ in range(100):
        next_stationary = stationary @ A
        if np.allclose(next_stationary, stationary, atol=1e-10):
            break
        stationary = next_stationary
    stationary = np.maximum(stationary, 0.0)
    stationary = stationary / stationary.sum()

    # Current state probabilities from last observation
    model = _build_model(
        HMMParams(n_states=2, pi=params.pi[order], A=A, means=means, stds=vols)
    )
    obs = arr.reshape(-1, 1)
    probs = model.predict_proba(obs)
    current_state_probs = probs[-1]
    current_state = int(np.argmax(current_state_probs))

    expected_return = float(np.dot(stationary, means))
    expected_volatility = float(np.dot(stationary, vols))

    return {
        "converged": True,
        "state_labels": ["calm", "volatile"],
        "state_probs": stationary.tolist(),
        "state_means": means.tolist(),
        "state_vols": vols.tolist(),
        "expected_return": expected_return,
        "expected_volatility": expected_volatility,
        "current_state": current_state,
        "current_state_prob": float(current_state_probs[current_state]),
    }


def fit_volatility_hmm(
    returns: np.ndarray,
    vol_window: int = 5,
    n_states: int = 2,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
) -> float:
    """Fit 2-state HMM on rolling realised volatility.

    Returns a vol scale factor clamped to [0.5, 2.0].
    """
    arr = np.asarray(returns)
    # Rolling realised volatility (std of returns over window)
    vols = []
    for i in range(vol_window - 1, len(arr)):
        window = arr[i - vol_window + 1:i + 1]
        vols.append(float(np.std(window, ddof=1)))
    vols = np.array(vols)

    if len(vols) < n_states * 10:
        return 1.0

    result = baum_welch(
        vols,
        n_states=n_states,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if not result.converged:
        return 1.0

    # Identify high-vol state
    params = result.params
    high_vol_state = int(np.argmax(params.means))
    low_vol_state = int(np.argmin(params.means))

    # Current vol is the last observation
    current_vol = float(vols[-1])

    # Baseline: midpoint between high and low state means
    baseline = (params.means[high_vol_state] + params.means[low_vol_state]) / 2.0
    if baseline <= 0:
        return 1.0

    scale = current_vol / baseline
    return float(np.clip(scale, 0.5, 2.0))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_model(params: HMMParams) -> hmm.GaussianHMM:
    """Reconstruct a fitted GaussianHMM from parameters."""
    model = hmm.GaussianHMM(
        n_components=params.n_states,
        covariance_type="diag",
        n_iter=1,
        init_params="",
    )
    model.startprob_ = params.pi.copy()
    model.transmat_ = params.A.copy()
    model.means_ = params.means.reshape(-1, 1).copy()
    model.covars_ = (params.stds ** 2).reshape(-1, 1).copy()
    return model
