/**
 * Round-4 Idea 3 — Regime-conditional Platt recalibrator.
 *
 * Source: docs/forecast-improvement-ideas-round4-2026-04-28.md
 *
 * Fits one Platt-style 2-param logistic per regime (Bull / Bear / Sideways)
 * on (raw probability, binary outcome) training pairs. At inference time,
 * the fit for the current regime is applied as a post-processing step to
 * the raw `pUp`. Total parameters: 6 across all regimes — well below
 * over-fit risk on typical backtest histories (hundreds of samples).
 *
 * Math:
 *   p_calibrated = σ( a · logit(p_raw) + b )
 *
 * Fit method: gradient descent on the binary cross-entropy loss with
 *
 *   L = − Σ [ y·log(p_cal) + (1−y)·log(1 − p_cal) ]
 *
 * This is a tiny convex problem so plain GD with a small learning rate
 * converges in a few hundred iterations and stays deterministic.
 */

import type { RegimeState } from './markov-distribution.js';
import {
  createForecastLabAssetScopedRuntimeDefaults,
  type ForecastLabRuntimeAssetScope,
} from './forecast-lab-runtime-defaults.js';

const EPS = 1e-6;

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
const logit = (p: number): number => {
  const c = Math.max(EPS, Math.min(1 - EPS, p));
  return Math.log(c / (1 - c));
};

export interface RegimeCalibrationSample {
  regime: RegimeState;
  /** Raw probability from the un-calibrated forecaster, in (0, 1). */
  pRaw: number;
  /** Realised binary outcome, 0 or 1. */
  outcome: 0 | 1;
}

export interface PlattFit {
  /** Slope on logit(p_raw); a < 1 ⇒ shrinkage toward 0.5. */
  a: number;
  /** Bias term; b > 0 shifts predictions upward. */
  b: number;
  /** Number of training samples used. */
  n: number;
}

export type RegimePlattFits = Partial<Record<RegimeState, PlattFit>>;

export interface FitOptions {
  /** Below this many samples per regime, skip the fit (default 30). */
  minSamplesPerRegime?: number;
  /** Gradient-descent learning rate (default 0.05). */
  learningRate?: number;
  /** Max iterations (default 500). */
  maxIter?: number;
  /** Convergence tolerance on parameter delta (default 1e-6). */
  tol?: number;
}

export const FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS: Required<FitOptions> = {
  minSamplesPerRegime: 18,
  learningRate: 0.05,
  maxIter: 500,
  tol: 1e-6,
};

const forecastLabRegimeCalibratorRuntimeDefaults = createForecastLabAssetScopedRuntimeDefaults(
  FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS,
);

export function resolveForecastLabRegimeCalibratorDefaults(
  assetScope?: ForecastLabRuntimeAssetScope,
): Required<FitOptions> {
  return forecastLabRegimeCalibratorRuntimeDefaults.resolve(assetScope);
}

export function getForecastLabRegimeCalibratorRuntimeDefaults(
  assetScope: ForecastLabRuntimeAssetScope,
): Partial<Required<FitOptions>> | undefined {
  return forecastLabRegimeCalibratorRuntimeDefaults.get(assetScope);
}

export function setForecastLabRegimeCalibratorRuntimeDefaults(
  assetScope: ForecastLabRuntimeAssetScope,
  overrides?: Partial<Required<FitOptions>>,
): void {
  forecastLabRegimeCalibratorRuntimeDefaults.set(assetScope, overrides);
}

/**
 * Fit a Platt logistic on a single regime's samples.
 * Returns null if input is degenerate (all same outcome ⇒ unbounded fit).
 */
function fitOneRegime(
  pRaws: number[],
  outcomes: number[],
  opts: Required<FitOptions>,
): PlattFit | null {
  if (pRaws.length === 0) return null;
  // Degeneracy guard: if every outcome is identical, the optimal Platt
  // would saturate to ±∞. Skip and let raw passthrough handle it.
  const sumY = outcomes.reduce((s, y) => s + y, 0);
  if (sumY === 0 || sumY === outcomes.length) return null;
  const x = pRaws.map(logit);
  let a = 1;
  let b = 0;
  for (let it = 0; it < opts.maxIter; it++) {
    let gradA = 0;
    let gradB = 0;
    for (let i = 0; i < x.length; i++) {
      const z = a * x[i] + b;
      const p = sigmoid(z);
      const err = p - outcomes[i];
      gradA += err * x[i];
      gradB += err;
    }
    gradA /= x.length;
    gradB /= x.length;
    const aNew = a - opts.learningRate * gradA;
    const bNew = b - opts.learningRate * gradB;
    if (Math.abs(aNew - a) < opts.tol && Math.abs(bNew - b) < opts.tol) {
      a = aNew;
      b = bNew;
      break;
    }
    a = aNew;
    b = bNew;
  }
  return { a, b, n: pRaws.length };
}

/**
 * Fit one Platt logistic per regime present in `samples`. Regimes with
 * fewer than `minSamplesPerRegime` (default 30) samples are skipped.
 *
 * Deterministic: identical input ⇒ identical fits.
 */
export function fitRegimePlatt(
  samples: RegimeCalibrationSample[],
  options: FitOptions = {},
): RegimePlattFits {
  const opts: Required<FitOptions> = { ...resolveForecastLabRegimeCalibratorDefaults(), ...options };
  const buckets: Record<RegimeState, { p: number[]; y: number[] }> = {
    bull: { p: [], y: [] },
    bear: { p: [], y: [] },
    sideways: { p: [], y: [] },
  };
  for (const s of samples) {
    const b = buckets[s.regime];
    if (!b) continue;
    b.p.push(s.pRaw);
    b.y.push(s.outcome);
  }
  const out: RegimePlattFits = {};
  for (const regime of ['bull', 'bear', 'sideways'] as RegimeState[]) {
    const { p, y } = buckets[regime];
    if (p.length < opts.minSamplesPerRegime) continue;
    const fit = fitOneRegime(p, y, opts);
    if (fit) out[regime] = fit;
  }
  return out;
}

/**
 * Apply the regime-specific Platt fit to a raw probability. Falls back
 * to the raw value if no fit exists for the supplied regime, or if
 * `regime` is undefined.
 *
 * Inputs and outputs are clamped to (EPS, 1-EPS) for numerical safety.
 */
export function applyRegimePlatt(
  pRaw: number,
  regime: RegimeState | undefined,
  fits: RegimePlattFits,
): number {
  // Pre-clamp input so logit() never blows up.
  const pClamped = Math.max(EPS, Math.min(1 - EPS, pRaw));
  if (!regime) return pClamped;
  const fit = fits[regime];
  if (!fit) return pClamped;
  return sigmoid(fit.a * logit(pClamped) + fit.b);
}

/**
 * Serialise to JSON for persistence under data/calibration/.
 */
export function serializeRegimePlatt(fits: RegimePlattFits): string {
  return JSON.stringify(fits);
}

/**
 * Restore from JSON; safe to call on missing or malformed input.
 */
export function deserializeRegimePlatt(json: string): RegimePlattFits {
  try {
    const parsed = JSON.parse(json);
    if (!parsed || typeof parsed !== 'object') return {};
    const out: RegimePlattFits = {};
    for (const regime of ['bull', 'bear', 'sideways'] as RegimeState[]) {
      const v = parsed[regime];
      if (v && typeof v.a === 'number' && typeof v.b === 'number' && typeof v.n === 'number') {
        out[regime] = { a: v.a, b: v.b, n: v.n };
      }
    }
    return out;
  } catch {
    return {};
  }
}
