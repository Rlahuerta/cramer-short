"""R5 Sprint 2 Idea #6 — Two-pass regime-conditional Platt recalibrator.

Python mirror of src/tools/finance/regime-calibrator-two-pass.ts.

Source: docs/forecast-improvement-ideas-round5-2026-04-29.md,
Niculescu-Mizil & Caruana 2005 (iterative Platt).

Two-pass composition per regime:
    p_pass1 = sigmoid( a1 * logit(p_raw)   + b1 )
    p_pass2 = sigmoid( a2 * logit(p_pass1) + b2 )

The composed transform is NOT equivalent to a single (a, b) because
the intermediate is clamped to (eps, 1-eps) before the second pass.
This lifts log-loss vs single-pass on over-confident data.

Deterministic contract: identical samples -> identical fits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from research.utils.regime_calibrator import (
    EPS,
    PlattFit,
    RegimeCalibrationSample,
    RegimePlattFits,
    _REGIMES,
    apply_regime_platt,
    fit_regime_platt,
)
from research.utils.regime_calibrator import RegimeState


@dataclass(frozen=True)
class TwoPassPlattFit:
    """Per-regime two-pass Platt parameters."""

    pass1: PlattFit
    pass2: PlattFit


TwoPassRegimePlattFits = dict[RegimeState, TwoPassPlattFit]


def fit_two_pass_regime_platt(
    samples: list[RegimeCalibrationSample],
    **fit_opts,
) -> TwoPassRegimePlattFits:
    """Fit a two-pass Platt logistic per regime.

    Pass 1: fit (p_raw, y) -> pass1 per regime.
    Pass 2: apply pass1 to every sample's p_raw, then fit (p_pass1, y).

    Regimes where pass-1 is degenerate are silently dropped.
    Extra keyword arguments are forwarded to fit_regime_platt.
    """
    pass1 = fit_regime_platt(samples, **fit_opts)

    pass2_samples = [
        RegimeCalibrationSample(
            regime=s.regime,
            p_raw=apply_regime_platt(s.p_raw, s.regime, pass1),
            outcome=s.outcome,
        )
        for s in samples
    ]
    pass2 = fit_regime_platt(pass2_samples, **fit_opts)

    out: TwoPassRegimePlattFits = {}
    for regime in _REGIMES:
        a = pass1.get(regime)
        b = pass2.get(regime)
        if a is not None and b is not None:
            out[regime] = TwoPassPlattFit(pass1=a, pass2=b)
    return out


def apply_two_pass_regime_platt(
    p_raw: float,
    regime: RegimeState | None,
    fits: TwoPassRegimePlattFits,
) -> float:
    """Apply the composed two-pass Platt transform.

    Falls back to the clamped raw probability when no fit is available
    or regime is None.
    """
    if regime is None:
        return max(EPS, min(1.0 - EPS, p_raw))
    fit = fits.get(regime)
    if fit is None:
        return max(EPS, min(1.0 - EPS, p_raw))
    p1 = apply_regime_platt(p_raw, regime, {regime: fit.pass1})
    p2 = apply_regime_platt(p1, regime, {regime: fit.pass2})
    return p2


def serialize_two_pass_regime_platt(fits: TwoPassRegimePlattFits) -> str:
    """Serialize two-pass fits to JSON."""
    return json.dumps(
        {
            r: {
                "pass1": {"a": f.pass1.a, "b": f.pass1.b, "n": f.pass1.n},
                "pass2": {"a": f.pass2.a, "b": f.pass2.b, "n": f.pass2.n},
            }
            for r, f in fits.items()
        }
    )


def deserialize_two_pass_regime_platt(raw: str) -> TwoPassRegimePlattFits:
    """Restore two-pass fits from JSON; returns empty dict on malformed input."""
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        out: TwoPassRegimePlattFits = {}
        for regime in _REGIMES:
            v = parsed.get(regime)
            if not isinstance(v, dict):
                continue
            p1 = v.get("pass1")
            p2 = v.get("pass2")
            if not (
                isinstance(p1, dict)
                and isinstance(p2, dict)
                and all(isinstance(p1.get(k), (int, float)) for k in ("a", "b", "n"))
                and all(isinstance(p2.get(k), (int, float)) for k in ("a", "b", "n"))
            ):
                continue
            out[regime] = TwoPassPlattFit(
                pass1=PlattFit(a=float(p1["a"]), b=float(p1["b"]), n=int(p1["n"])),
                pass2=PlattFit(a=float(p2["a"]), b=float(p2["b"]), n=int(p2["n"])),
            )
        return out
    except Exception:
        return {}
