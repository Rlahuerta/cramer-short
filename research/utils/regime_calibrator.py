"""R4 — Regime-conditional Platt recalibrator (single-pass).

Python mirror of src/tools/finance/regime-calibrator.ts.

Source: docs/forecast-improvement-ideas-round4-2026-04-28.md.

Fits one Platt-style 2-param logistic per regime (bull / bear / sideways)
on (raw probability, binary outcome) training pairs.

Math:
    p_calibrated = sigmoid( a * logit(p_raw) + b )

Fit method: gradient descent on binary cross-entropy.
    L = -sum [ y*log(p_cal) + (1-y)*log(1 - p_cal) ]

Deterministic: identical input -> identical fits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

RegimeState = Literal["bull", "bear", "sideways"]

EPS = 1e-6
_REGIMES: tuple[RegimeState, ...] = ("bull", "bear", "sideways")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    c = max(EPS, min(1.0 - EPS, p))
    return math.log(c / (1.0 - c))


@dataclass(frozen=True)
class RegimeCalibrationSample:
    """A single (regime, p_raw, outcome) training pair."""

    regime: RegimeState
    p_raw: float
    """Raw probability from un-calibrated forecaster, in (0, 1)."""
    outcome: int
    """Realised binary outcome, 0 or 1."""


@dataclass(frozen=True)
class PlattFit:
    """Result of fitting a single Platt logistic."""

    a: float
    """Slope on logit(p_raw); a < 1 shrinks toward 0.5."""
    b: float
    """Bias term; b > 0 shifts predictions upward."""
    n: int
    """Number of training samples used."""


RegimePlattFits = dict[RegimeState, PlattFit]

_DEFAULT_MIN_SAMPLES = 30
_DEFAULT_LR = 0.05
_DEFAULT_MAX_ITER = 500
_DEFAULT_TOL = 1e-6


def _fit_one_regime(
    p_raws: list[float],
    outcomes: list[float],
    lr: float,
    max_iter: int,
    tol: float,
) -> PlattFit | None:
    """Fit a single Platt logistic via gradient descent on BCE loss.

    Returns None if input is degenerate (all outcomes identical).
    """
    if not p_raws:
        return None
    sum_y = sum(outcomes)
    if sum_y == 0 or sum_y == len(outcomes):
        return None
    xs = [_logit(p) for p in p_raws]
    a, b = 1.0, 0.0
    n = len(xs)
    for _ in range(max_iter):
        grad_a = grad_b = 0.0
        for i in range(n):
            z = a * xs[i] + b
            p = _sigmoid(z)
            err = p - outcomes[i]
            grad_a += err * xs[i]
            grad_b += err
        grad_a /= n
        grad_b /= n
        a_new = a - lr * grad_a
        b_new = b - lr * grad_b
        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return PlattFit(a=a, b=b, n=n)


def fit_regime_platt(
    samples: list[RegimeCalibrationSample],
    min_samples_per_regime: int = _DEFAULT_MIN_SAMPLES,
    learning_rate: float = _DEFAULT_LR,
    max_iter: int = _DEFAULT_MAX_ITER,
    tol: float = _DEFAULT_TOL,
) -> RegimePlattFits:
    """Fit one Platt logistic per regime present in `samples`.

    Regimes with fewer than `min_samples_per_regime` are skipped.
    Deterministic: identical input -> identical fits.
    """
    buckets: dict[RegimeState, tuple[list[float], list[float]]] = {
        r: ([], []) for r in _REGIMES
    }
    for s in samples:
        if s.regime in buckets:
            buckets[s.regime][0].append(s.p_raw)
            buckets[s.regime][1].append(float(s.outcome))
    out: RegimePlattFits = {}
    for regime in _REGIMES:
        p_raws, ys = buckets[regime]
        if len(p_raws) < min_samples_per_regime:
            continue
        fit = _fit_one_regime(p_raws, ys, learning_rate, max_iter, tol)
        if fit is not None:
            out[regime] = fit
    return out


def apply_regime_platt(
    p_raw: float,
    regime: RegimeState | None,
    fits: RegimePlattFits,
) -> float:
    """Apply the regime-specific Platt fit to a raw probability.

    Falls back to the clamped raw value if no fit exists or regime is None.
    """
    p_clamped = max(EPS, min(1.0 - EPS, p_raw))
    if regime is None:
        return p_clamped
    fit = fits.get(regime)
    if fit is None:
        return p_clamped
    return _sigmoid(fit.a * _logit(p_clamped) + fit.b)


def serialize_regime_platt(fits: RegimePlattFits) -> str:
    """Serialize fits to JSON for persistence."""
    return json.dumps({r: {"a": f.a, "b": f.b, "n": f.n} for r, f in fits.items()})


def deserialize_regime_platt(raw: str) -> RegimePlattFits:
    """Restore fits from JSON; returns empty dict on malformed input."""
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        out: RegimePlattFits = {}
        for regime in _REGIMES:
            v = parsed.get(regime)
            if isinstance(v, dict) and all(
                isinstance(v.get(k), (int, float)) for k in ("a", "b", "n")
            ):
                out[regime] = PlattFit(a=float(v["a"]), b=float(v["b"]), n=int(v["n"]))
        return out
    except Exception:
        return {}
