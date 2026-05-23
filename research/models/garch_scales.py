"""R5 Idea #5 — Horizon-aware + regime-conditional GARCH(1,1) scaler.

Python mirror of src/tools/finance/garch-scales.ts.

Source: docs/forecast-improvement-ideas-round5-2026-04-29.md (Idea #5),
arXiv:2603.10299 Asaad et al. 2026 (regime-aware vol forecasting).

Given a series of log returns and a horizon in days, produces a
length-`horizon` list of multiplicative scalars s[t] such that

    sigma_garch(t) = s[t] * sigma_unconditional

Two optional R5 modulations on top of the base GARCH scalar:

1. Horizon decay.  Beyond `horizon_cap` (default 7d), soft-blend the
   GARCH-vs-1.0 displacement toward 0:
       blend  = max(0, 1 - (d - cap) / (2 * cap))
       s_eff  = 1 + blend * (s_base - 1)
   For d >= 3*cap the scalar is exactly 1.0.

2. Regime ceiling.  Two ceilings instead of the static 3.0 cap:
       calm     -> max scalar = ceiling.calm     (default 1.5)
       turbulent -> max scalar = ceiling.turbulent (default 3.0)
   Regime detected by comparing recent sigma (last `regime_window`) vs
   full-series sigma.

Both modulations require the respective option to be set. When omitted
the function preserves byte-identical pre-R5 behaviour (clamp [0.33, 3.0],
no horizon decay).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from research.models.garch import fit_garch11


@dataclass(frozen=True)
class GarchClampOptions:
    """Optional R5 modulation parameters for compute_garch_scales."""

    horizon_cap: int | None = None
    """Soft-blend GARCH scalar toward 1.0 beyond this day count.
    Past 3*horizon_cap the scalar is exactly 1.0.  None = pre-R5 behaviour."""

    ceiling: tuple[float, float] | None = None
    """(calm_ceiling, turbulent_ceiling).  None = pre-R5 static 3.0 cap."""

    regime_window: int = 20
    """Rolling window (observations) for lightweight regime detection."""

    regime_override: Literal["calm", "turbulent"] | None = None
    """Pre-computed regime.  When set, skips the auto-detector."""


def detect_recent_regime(
    log_returns: Sequence[float],
    window_size: int,
) -> Literal["calm", "turbulent"]:
    """Compare recent sigma (last window_size obs) vs full-series sigma.

    Returns 'calm' if recent sigma < full sigma, else 'turbulent'.
    Falls back to 'turbulent' if insufficient data.
    """
    if len(log_returns) < window_size * 2:
        return "turbulent"
    recent = log_returns[-window_size:]
    recent_sse = sum(r * r for r in recent)
    recent_sigma = math.sqrt(recent_sse / len(recent))

    full_sse = sum(r * r for r in log_returns)
    full_sigma = math.sqrt(full_sse / len(log_returns))

    return "calm" if recent_sigma < full_sigma else "turbulent"


def compute_garch_scales(
    log_returns: Sequence[float],
    horizon_days: int,
    opts: GarchClampOptions | None = None,
) -> list[float]:
    """Return per-day GARCH volatility scalars for `horizon_days` steps.

    Returns an empty list when input is too short (< 5 obs) or has zero
    variance, so callers can fall back to constant-sigma behaviour.

    When `opts` is None (or GarchClampOptions()), preserves pre-R5
    byte-identical clamp [0.33, 3.0] with no horizon decay.
    """
    if horizon_days <= 0:
        return []
    if len(log_returns) < 5:
        return []

    sse = sum(r * r for r in log_returns)
    sample_var = sse / len(log_returns)
    if not (sample_var > 0) or not math.isfinite(sample_var):
        return []
    sample_sigma = math.sqrt(sample_var)

    try:
        params = fit_garch11(list(log_returns))
    except Exception:
        return []

    last = log_returns[-1]
    persistence = params.alpha + params.beta
    h1 = params.omega + params.alpha * last * last + params.beta * params.h0

    sigmas: list[float] = [math.sqrt(max(0.0, h1))]
    h = h1
    for _ in range(1, horizon_days):
        h = params.omega + persistence * h
        sigmas.append(math.sqrt(max(0.0, h)))

    if opts is None:
        opts = GarchClampOptions()

    cap = opts.horizon_cap
    ceiling_pair = opts.ceiling

    if opts.regime_override is not None:
        regime: Literal["calm", "turbulent"] = opts.regime_override
    elif ceiling_pair is not None:
        regime = detect_recent_regime(log_returns, opts.regime_window)
    else:
        regime = "turbulent"

    ceiling_high = ceiling_pair[0 if regime == "calm" else 1] if ceiling_pair else 3.0

    scales: list[float] = []
    for d_idx, s in enumerate(sigmas):
        k = s / sample_sigma
        if not math.isfinite(k) or k <= 0:
            scales.append(1.0)
            continue
        k = min(ceiling_high, max(0.33, k))

        if cap is not None and cap > 0 and (d_idx + 1) > cap:
            d = d_idx + 1
            raw_blend = 1.0 - (d - cap) / (2.0 * cap)
            blend = max(0.0, min(1.0, raw_blend))
            k = 1.0 + blend * (k - 1.0)

        scales.append(k)

    return scales
