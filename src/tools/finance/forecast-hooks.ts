/**
 * Feature-flagged hooks that bring W3 Hawkes (jump-intensity self-excitation)
 * and ADWIN (adaptive history-window trimming) into the Markov-distribution
 * pipeline.
 *
 * Both hooks are pure helpers — the calling code in markov-distribution.ts
 * remains responsible for actually consuming the trimmed prices / amplified
 * jump events.  When the flags are off, neither helper is called and the
 * pipeline is byte-identical to the pre-W3 implementation.
 */

import { Adwin } from '../../utils/adwin.js';
import { detectKswinDrift, type KswinOptions } from '../../utils/kswin.js';
import { fitHawkesMLE, type HawkesFit } from './hawkes.js';
import type { JumpEventSpec } from './jump-diffusion.js';

// ---------------------------------------------------------------------------
// ADWIN: drift-aware history trimming
// ---------------------------------------------------------------------------

export interface AdwinTrimResult {
  /** Number of trailing prices kept after trimming. */
  keptPrices: number;
  /** Number of prices dropped from the front. */
  droppedPrices: number;
  /** Whether ADWIN actually shortened the window. */
  trimmed: boolean;
  /** Length of the input history. */
  totalPrices: number;
}

/**
 * Apply ADWIN drift detection to the log-return series of `historicalPrices`.
 * Returns the *suffix* of the price array that ADWIN considers stationary,
 * plus diagnostics.
 *
 * Invariants:
 *   - When the input has fewer than `minHistory` (default 60) prices, the
 *     input is returned unchanged.
 *   - The returned price array always contains ≥ `minKeep` (default 60)
 *     trailing prices, even if ADWIN suggests a shorter window.
 *   - When ADWIN detects no drift, the returned array equals the input.
 */
export function applyAdwinTrim(
  historicalPrices: number[],
  delta = 0.05,
  opts: { minHistory?: number; minKeep?: number } = {},
): { trimmedPrices: number[]; result: AdwinTrimResult } {
  const minHistory = opts.minHistory ?? 60;
  const minKeep = opts.minKeep ?? 60;
  const totalPrices = historicalPrices.length;

  if (totalPrices < minHistory) {
    return {
      trimmedPrices: historicalPrices,
      result: { keptPrices: totalPrices, droppedPrices: 0, trimmed: false, totalPrices },
    };
  }

  // Log returns; one fewer than prices.
  const logReturns: number[] = new Array(totalPrices - 1);
  for (let i = 1; i < totalPrices; i++) {
    logReturns[i - 1] = Math.log(historicalPrices[i] / historicalPrices[i - 1]);
  }

  const adwin = new Adwin({ delta });
  for (const r of logReturns) {
    adwin.add(r);
  }

  // ADWIN's `size()` reports how many of the most-recent samples remain in
  // the window after compression + drift-driven drops.
  const adwinKeep = adwin.size();
  const desiredKeepPrices = Math.min(totalPrices, Math.max(minKeep, adwinKeep + 1));

  if (desiredKeepPrices >= totalPrices) {
    return {
      trimmedPrices: historicalPrices,
      result: { keptPrices: totalPrices, droppedPrices: 0, trimmed: false, totalPrices },
    };
  }

  const trimmed = historicalPrices.slice(totalPrices - desiredKeepPrices);
  return {
    trimmedPrices: trimmed,
    result: {
      keptPrices: trimmed.length,
      droppedPrices: totalPrices - trimmed.length,
      trimmed: true,
      totalPrices,
    },
  };
}

// ---------------------------------------------------------------------------
// KSWIN: variance-aware drift trimming (R4 Idea 1)
// ---------------------------------------------------------------------------

export interface KswinTrimResult {
  keptPrices: number;
  droppedPrices: number;
  trimmed: boolean;
  totalPrices: number;
  /** Max KS-D observed across split points (diagnostic). */
  maxD: number;
  /** Critical D under the chosen alpha (diagnostic). */
  criticalD: number;
}

/**
 * Apply KSWIN drift detection to the |log-return| series of `historicalPrices`
 * and return the recent stationary suffix.  Operates on |returns| so the
 * detector is sensitive to *variance* shifts (where ADWIN, being mean-aware,
 * stays silent).
 *
 * Invariants mirror {@link applyAdwinTrim}: short histories pass through
 * untouched; the returned suffix always contains ≥ `minKeep` prices.
 */
export function applyKswinTrim(
  historicalPrices: number[],
  opts: KswinOptions & { minHistory?: number } = {},
): { trimmedPrices: number[]; result: KswinTrimResult } {
  const minHistory = opts.minHistory ?? 60;
  const minKeep = opts.minKeep ?? 60;
  const totalPrices = historicalPrices.length;

  if (totalPrices < minHistory) {
    return {
      trimmedPrices: historicalPrices,
      result: {
        keptPrices: totalPrices, droppedPrices: 0, trimmed: false, totalPrices,
        maxD: 0, criticalD: 0,
      },
    };
  }

  const absLogReturns: number[] = new Array(totalPrices - 1);
  for (let i = 1; i < totalPrices; i++) {
    absLogReturns[i - 1] = Math.abs(Math.log(historicalPrices[i] / historicalPrices[i - 1]));
  }

  const drift = detectKswinDrift(absLogReturns, { ...opts, minKeep });
  if (!drift.drift) {
    return {
      trimmedPrices: historicalPrices,
      result: {
        keptPrices: totalPrices, droppedPrices: 0, trimmed: false, totalPrices,
        maxD: drift.maxD, criticalD: drift.criticalD,
      },
    };
  }

  // drift.keepCount counts returns; map back to prices (returns + 1 boundary).
  const desiredKeepPrices = Math.min(totalPrices, Math.max(minKeep, drift.keepCount + 1));
  if (desiredKeepPrices >= totalPrices) {
    return {
      trimmedPrices: historicalPrices,
      result: {
        keptPrices: totalPrices, droppedPrices: 0, trimmed: false, totalPrices,
        maxD: drift.maxD, criticalD: drift.criticalD,
      },
    };
  }

  const trimmed = historicalPrices.slice(totalPrices - desiredKeepPrices);
  return {
    trimmedPrices: trimmed,
    result: {
      keptPrices: trimmed.length,
      droppedPrices: totalPrices - trimmed.length,
      trimmed: true,
      totalPrices,
      maxD: drift.maxD,
      criticalD: drift.criticalD,
    },
  };
}

// ---------------------------------------------------------------------------
// Hawkes: self-exciting jump-intensity amplification
// ---------------------------------------------------------------------------

export interface HawkesAmplificationResult {
  /** Indices in the historical-return series treated as jump events. */
  jumpIndices: number[];
  /** Fitted Hawkes parameters; null when the fit was not attempted (too few jumps). */
  fit: HawkesFit | null;
  /** Multiplier applied to each event's `dailyIntensity`.  1.0 ⇒ no change. */
  intensityMultiplier: number;
  /** Empirical mean of |log-return| on jump days, used for synthesizing endogenous jumps. */
  meanLogJump: number;
  /** Empirical std of log-return on jump days. */
  stdLogJump: number;
  /** Endogenous jump event synthesized from history (only when no caller-supplied
   *  events exist and clustering was detected). Null otherwise. */
  endogenousJump: JumpEventSpec | null;
}

/**
 * Detect historical jump days (|log return| > k·σ), fit a Hawkes(μ, α, β) to
 * those event times, and return both an intensity-amplification multiplier
 * (for caller-supplied `JumpEventSpec`s) and an optional synthesized
 * endogenous jump event.
 *
 * The multiplier captures self-excitation: a stationary Hawkes process has
 * long-run intensity μ/(1−α/β), so the amplification factor is
 * `1 / (1 − branchingRatio)`, clamped to [1, maxMultiplier].
 *
 * When fewer than `minJumpsForFit` (default 5) historical jumps are detected,
 * no fit is performed and the multiplier defaults to 1.0 with a null fit.
 */
export function applyHawkesAmplification(
  historicalPrices: number[],
  opts: {
    sigmaThreshold?: number;
    minJumpsForFit?: number;
    maxMultiplier?: number;
    minMultiplier?: number;
  } = {},
): HawkesAmplificationResult {
  const sigmaThreshold = opts.sigmaThreshold ?? 3;
  const minJumpsForFit = opts.minJumpsForFit ?? 5;
  const maxMultiplier = opts.maxMultiplier ?? 3;
  const minMultiplier = opts.minMultiplier ?? 1;

  if (historicalPrices.length < 30) {
    return {
      jumpIndices: [],
      fit: null,
      intensityMultiplier: 1,
      meanLogJump: 0,
      stdLogJump: 0,
      endogenousJump: null,
    };
  }

  const logReturns: number[] = new Array(historicalPrices.length - 1);
  for (let i = 1; i < historicalPrices.length; i++) {
    logReturns[i - 1] = Math.log(historicalPrices[i] / historicalPrices[i - 1]);
  }

  // Empirical std (mean is ~0 on log returns over short horizons).
  let mean = 0;
  for (const r of logReturns) mean += r;
  mean /= logReturns.length;
  let variance = 0;
  for (const r of logReturns) variance += (r - mean) ** 2;
  variance /= Math.max(1, logReturns.length - 1);
  const sigma = Math.sqrt(variance);

  if (!(sigma > 0)) {
    return {
      jumpIndices: [],
      fit: null,
      intensityMultiplier: 1,
      meanLogJump: 0,
      stdLogJump: 0,
      endogenousJump: null,
    };
  }

  const jumpIndices: number[] = [];
  const jumpReturns: number[] = [];
  for (let i = 0; i < logReturns.length; i++) {
    if (Math.abs(logReturns[i] - mean) > sigmaThreshold * sigma) {
      jumpIndices.push(i);
      jumpReturns.push(logReturns[i]);
    }
  }

  if (jumpIndices.length < minJumpsForFit) {
    return {
      jumpIndices,
      fit: null,
      intensityMultiplier: 1,
      meanLogJump: 0,
      stdLogJump: 0,
      endogenousJump: null,
    };
  }

  // Convert event indices to a "time" axis in days (one return = one day).
  const eventTimes = jumpIndices.map(i => i + 1);
  const horizonDays = logReturns.length;

  let fit: HawkesFit | null = null;
  try {
    fit = fitHawkesMLE(eventTimes, horizonDays);
  } catch {
    fit = null;
  }

  let multiplier = 1;
  if (fit && fit.isStable && fit.alpha > 0) {
    const branching = fit.alpha / fit.beta;
    if (branching > 0 && branching < 0.95) {
      const raw = 1 / (1 - branching);
      multiplier = Math.min(maxMultiplier, Math.max(minMultiplier, raw));
    }
  }

  // Empirical jump magnitude stats (used for synthesized endogenous jump).
  let jumpMean = 0;
  for (const r of jumpReturns) jumpMean += r;
  jumpMean /= jumpReturns.length;
  let jumpVar = 0;
  for (const r of jumpReturns) jumpVar += (r - jumpMean) ** 2;
  jumpVar /= Math.max(1, jumpReturns.length - 1);
  const jumpStd = Math.sqrt(jumpVar);

  let endogenousJump: JumpEventSpec | null = null;
  if (fit && fit.isStable && multiplier > 1.01) {
    // Long-run mean intensity per day from the fitted Hawkes process.
    const longRunIntensity = fit.mu / (1 - fit.alpha / fit.beta);
    const dailyIntensity = Math.min(0.95, Math.max(0, longRunIntensity));
    if (dailyIntensity > 1e-4) {
      endogenousJump = {
        id: 'hawkes-endogenous',
        dailyIntensity,
        meanLogJump: jumpMean,
        stdLogJump: Math.max(jumpStd, sigma * 0.5),
      };
    }
  }

  return {
    jumpIndices,
    fit,
    intensityMultiplier: multiplier,
    meanLogJump: jumpMean,
    stdLogJump: jumpStd,
    endogenousJump,
  };
}

/**
 * Apply the Hawkes intensity multiplier to a list of caller-supplied jump
 * events.  Each event's `dailyIntensity` is scaled and clamped to ≤ 0.95.
 */
export function amplifyJumpEvents(
  events: readonly JumpEventSpec[],
  multiplier: number,
): JumpEventSpec[] {
  if (multiplier <= 1) return events.map(e => ({ ...e }));
  return events.map(e => ({
    ...e,
    dailyIntensity: Math.min(0.95, e.dailyIntensity * multiplier),
  }));
}
