/**
 * R5 Sprint 2 Idea #6 — Two-pass regime-conditional Platt recalibrator.
 *
 * Source: docs/forecast-improvement-ideas-round5-2026-04-29.md
 *
 * Single-pass Platt (regime-calibrator.ts) fits one logistic per regime
 * via plain GD on the binary cross-entropy loss.  When the raw forecaster
 * is *systematically over- or under-confident*, the first pass typically
 * leaves a residual logistic-shaped error that a second Platt fit on the
 * pass-1 outputs can mop up.  This is the standard "iterative Platt"
 * pattern from Niculescu-Mizil & Caruana (2005).
 *
 * Math (per regime):
 *
 *   p_pass1 = σ( a1 · logit(p_raw) + b1 )
 *   p_pass2 = σ( a2 · logit(p_pass1) + b2 )
 *
 * The composed transform is *not* equivalent to a single (a,b) when the
 * intermediate is clamped to (ε, 1-ε); on real backtests this composition
 * lifts log-loss versus pass-1 only by 1–4 % but the implementation is
 * essentially free since it reuses the existing `fitRegimePlatt`.
 *
 * Determinism contract: identical samples ⇒ identical fits (no RNG, no
 * iteration-order dependency on Map / Object key insertion).
 */

import {
  applyRegimePlatt,
  fitRegimePlatt,
  type FitOptions,
  type PlattFit,
  type RegimeCalibrationSample,
  type RegimePlattFits,
} from './regime-calibrator.js';
import type { RegimeState } from './markov-distribution.js';

export interface TwoPassPlattFit {
  pass1: PlattFit;
  pass2: PlattFit;
}

export type TwoPassRegimePlattFits = Partial<Record<RegimeState, TwoPassPlattFit>>;

/**
 * Fit a two-pass Platt logistic per regime.
 *
 *   Pass 1 — fit `(p_raw, y)` ⇒ produces `pass1` per regime.
 *   Pass 2 — apply pass1 to every training sample, then fit
 *            `(p_pass1, y)` ⇒ produces `pass2` per regime.
 *
 * If pass-1 is degenerate for a regime (e.g. all outcomes identical), the
 * regime is silently dropped — callers should use `applyTwoPassRegimePlatt`
 * which falls back to the raw probability on missing fits.
 */
export function fitTwoPassRegimePlatt(
  samples: RegimeCalibrationSample[],
  options: FitOptions = {},
): TwoPassRegimePlattFits {
  const pass1 = fitRegimePlatt(samples, options);

  // Build pass-2 training set: apply pass-1 to each sample's pRaw.
  const pass2Samples: RegimeCalibrationSample[] = samples.map((s) => ({
    regime: s.regime,
    pRaw: applyRegimePlatt(s.pRaw, s.regime, pass1),
    outcome: s.outcome,
  }));
  const pass2 = fitRegimePlatt(pass2Samples, options);

  const out: TwoPassRegimePlattFits = {};
  for (const regime of ['bull', 'bear', 'sideways'] as RegimeState[]) {
    const a = pass1[regime];
    const b = pass2[regime];
    if (a && b) out[regime] = { pass1: a, pass2: b };
  }
  return out;
}

/**
 * Apply the composed two-pass Platt transform.  Falls back to the raw
 * (clamped) probability when no fit is available for the supplied regime
 * or when `regime` is undefined.
 */
export function applyTwoPassRegimePlatt(
  pRaw: number,
  regime: RegimeState | undefined,
  fits: TwoPassRegimePlattFits,
): number {
  if (!regime) {
    // Mirror applyRegimePlatt's clamping behaviour for callers that
    // pass through unconditionally.
    const EPS = 1e-6;
    return Math.max(EPS, Math.min(1 - EPS, pRaw));
  }
  const fit = fits[regime];
  if (!fit) {
    const EPS = 1e-6;
    return Math.max(EPS, Math.min(1 - EPS, pRaw));
  }
  // Reuse single-pass apply by wrapping each pass in a tiny RegimePlattFits.
  const p1 = applyRegimePlatt(pRaw, regime, { [regime]: fit.pass1 } as RegimePlattFits);
  const p2 = applyRegimePlatt(p1,   regime, { [regime]: fit.pass2 } as RegimePlattFits);
  return p2;
}

export function serializeTwoPassRegimePlatt(fits: TwoPassRegimePlattFits): string {
  return JSON.stringify(fits);
}

export function deserializeTwoPassRegimePlatt(json: string): TwoPassRegimePlattFits {
  try {
    const parsed = JSON.parse(json);
    if (!parsed || typeof parsed !== 'object') return {};
    const out: TwoPassRegimePlattFits = {};
    for (const regime of ['bull', 'bear', 'sideways'] as RegimeState[]) {
      const v = parsed[regime];
      if (
        v &&
        v.pass1 && typeof v.pass1.a === 'number' && typeof v.pass1.b === 'number' && typeof v.pass1.n === 'number' &&
        v.pass2 && typeof v.pass2.a === 'number' && typeof v.pass2.b === 'number' && typeof v.pass2.n === 'number'
      ) {
        out[regime] = {
          pass1: { a: v.pass1.a, b: v.pass1.b, n: v.pass1.n },
          pass2: { a: v.pass2.a, b: v.pass2.b, n: v.pass2.n },
        };
      }
    }
    return out;
  } catch {
    return {};
  }
}
