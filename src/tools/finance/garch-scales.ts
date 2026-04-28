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
 */

import { fitGarch11, garchForecast } from '../../utils/garch.js';

export function computeGarchScales(logReturns: readonly number[], horizonDays: number): number[] {
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

  // Convert to scalars relative to sample sigma.  Clamp to a sane range to
  // protect against pathological short-history fits.
  return sigmas.map(s => {
    const k = s / sampleSigma;
    if (!Number.isFinite(k) || k <= 0) return 1;
    return Math.min(3, Math.max(0.33, k));
  });
}
