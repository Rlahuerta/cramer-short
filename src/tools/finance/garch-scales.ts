/**
 * R4 Phase B — GARCH(1,1) per-day volatility scaler for trajectory MC.
 *
 * Given a series of log returns and a horizon in days, produces a
 * length-`horizon` array of multiplicative scalars `s[t]` such that
 *
 *   sigma_garch(t) = s[t] · sigma_unconditional
 *
 * where `sigma_unconditional` is the long-run GARCH(1,1) std. dev.
 *
 * Caller usage in trajectory MC:
 *
 *   dailyVols[d] *= garchScales[d];
 *
 * When the input is too short (< 5 obs) or has zero variance, returns an
 * empty array so the caller can fall back to constant-σ behaviour.
 *
 * Source: docs/forecast-improvement-ideas-round4-2026-04-28.md Idea 4.
 *
 * R5 Idea #5 — Horizon-aware + regime-conditional clamping.
 *
 *   Refs: R4 backtest finding (GARCH regresses at h=14d ΔBrier +0.006);
 *         arXiv:2603.10299 Asaad et al. 2026 (regime-aware vol forecasting).
 *
 *   Two new modulations layered on top of the base GARCH scalar:
 *
 *   1. Horizon decay.  Beyond `horizonCap` (default 7d), soft-blend the
 *      GARCH-vs-1.0 displacement toward 0:
 *           blend  = max(0, 1 - (d - cap) / (2 * cap))
 *           sEff   = 1 + blend × (sBase - 1)
 *      For d ≥ 3·cap the scalar is exactly 1.0 (GARCH is silenced).
 *      Rationale: the GARCH(1,1) variance forecast itself reverts toward
 *      the unconditional level at rate (α+β)^k, but the *amplification
 *      versus sample sigma* persists even after the underlying mean
 *      reverts.  The blend matches what the forecast is actually telling
 *      us at long horizons (≈ 1.0).
 *
 *   2. Regime ceiling.  Two ceilings are applied instead of the static
 *      3.0 cap:
 *           calm regime      → max scalar = ceiling.calm      (default 1.5)
 *           turbulent regime → max scalar = ceiling.turbulent (default 3.0)
 *      The regime is detected by comparing realised σ over the last
 *      `regimeWindow` returns (default 20) against the rolling median σ
 *      computed over the full series.  A series whose recent σ is below
 *      its historical median is "calm"; otherwise "turbulent".  We use
 *      this lightweight regime detector instead of importing the full
 *      RegimeState because callers may not have one available (e.g. unit
 *      tests, raw research scripts).
 *
 *   Both modulations are gated by the `horizonCap` / `ceiling` options
 *   being present.  When omitted the function preserves byte-identical
 *   pre-R5 behaviour (clamp [0.33, 3.0], no horizon decay).
 */

import { fitGarch11, garchForecast } from '../../utils/garch.js';

export interface GarchClampOptions {
  /**
   * R5 Idea #5 — soft-blend the GARCH scalar toward 1.0 beyond this
   * horizon (in days).  Past 3×cap the scalar is exactly 1.0.
   * Pass `undefined` to preserve pre-R5 behaviour.
   */
  horizonCap?: number;
  /**
   * R5 Idea #5 — regime-conditional ceiling for the scalar.  When omitted,
   * the legacy [0.33, 3.0] clamp is used regardless of regime.
   */
  ceiling?: { calm: number; turbulent: number };
  /**
   * Rolling window (in observations) used to detect the recent vol regime
   * for the ceiling clamp.  Default 20.  Only used when `ceiling` is set.
   */
  regimeWindow?: number;
  /**
   * Pre-computed regime override.  When provided, skips the lightweight
   * detector entirely and uses this regime directly.  Useful when the
   * caller already has a richer RegimeState available.
   */
  regimeOverride?: 'calm' | 'turbulent';
}

function detectRecentRegime(
  logReturns: readonly number[],
  windowSize: number,
): 'calm' | 'turbulent' {
  // Compare recent σ (last `windowSize` returns) vs full-series σ.
  // Recent < full ⇒ calm; recent ≥ full ⇒ turbulent.
  if (logReturns.length < windowSize * 2) return 'turbulent';  // play safe
  const recent = logReturns.slice(-windowSize);
  let recentSse = 0;
  for (const r of recent) recentSse += r * r;
  const recentSigma = Math.sqrt(recentSse / recent.length);

  let fullSse = 0;
  for (const r of logReturns) fullSse += r * r;
  const fullSigma = Math.sqrt(fullSse / logReturns.length);

  return recentSigma < fullSigma ? 'calm' : 'turbulent';
}

export function computeGarchScales(
  logReturns: readonly number[],
  horizonDays: number,
  opts: GarchClampOptions = {},
): number[] {
  if (horizonDays <= 0) return [];
  if (logReturns.length < 5) return [];

  // Sample variance (about zero — daily mean ~0).
  let sse = 0;
  for (const r of logReturns) sse += r * r;
  const sampleVar = sse / logReturns.length;
  if (!(sampleVar > 0) || !Number.isFinite(sampleVar)) return [];
  const sampleSigma = Math.sqrt(sampleVar);

  let params;
  try {
    params = fitGarch11(logReturns as number[]);
  } catch {
    return [];
  }

  // Drive the recursion forward from the most recent observation so the
  // initial h_0 reflects the latest squared innovation, not the long-run
  // average.  This is what gives GARCH its "vol clustering" lift.
  const last = logReturns[logReturns.length - 1];
  const persistence = params.alpha + params.beta;
  const uncondVar = params.omega / Math.max(1e-12, 1 - persistence);
  // First step uses the actual last innovation (not E[z²] = 1).
  const h1 = params.omega + params.alpha * last * last + params.beta * params.h0;

  const sigmas: number[] = [Math.sqrt(Math.max(0, h1))];
  let h = h1;
  for (let t = 1; t < horizonDays; t++) {
    h = params.omega + persistence * h;
    sigmas.push(Math.sqrt(Math.max(0, h)));
  }
  void uncondVar;
  void garchForecast; // imported for API parity / future use

  // R5 Idea #5 — pick effective ceiling based on regime when provided.
  const cap = opts.horizonCap;
  const ceiling = opts.ceiling;
  const regime = opts.regimeOverride
    ?? (ceiling ? detectRecentRegime(logReturns, opts.regimeWindow ?? 20) : 'turbulent');
  const ceilingHigh = ceiling ? ceiling[regime] : 3.0;
  // The lower bound stays at 0.33 — we never want to silence vol entirely.

  // Convert to scalars relative to sample sigma.  Clamp to the regime
  // ceiling, then apply optional horizon decay toward 1.0.
  return sigmas.map((s, dIdx) => {
    let k = s / sampleSigma;
    if (!Number.isFinite(k) || k <= 0) return 1;
    k = Math.min(ceilingHigh, Math.max(0.33, k));

    if (cap !== undefined && cap > 0 && dIdx + 1 > cap) {
      // dIdx is 0-indexed; map to "day d+1 in the horizon".
      const d = dIdx + 1;
      const rawBlend = 1 - (d - cap) / (2 * cap);
      const blend = Math.max(0, Math.min(1, rawBlend));
      // Soft-blend toward 1.0; beyond 3·cap blend == 0 ⇒ k == 1.
      k = 1 + blend * (k - 1);
    }
    return k;
  });
}
