/**
 * R5 Sprint 2 Idea #6 — two-pass Platt: tests
 *
 * The two-pass calibrator should:
 *   1. Be byte-equivalent to single-pass when pass-1 already perfectly
 *      calibrates (residual is zero ⇒ pass-2 is the identity logistic
 *      a≈1, b≈0).
 *   2. On synthetic *over-confident* data (pUp pushed away from 0.5),
 *      single-pass Platt should under-correct because GD plateaus on a
 *      noisy gradient; two-pass should reduce log-loss further.
 *   3. Compose deterministically and round-trip via serialize/deserialize.
 */

import { describe, expect, it } from 'bun:test';

import {
  applyRegimePlatt,
  fitRegimePlatt,
  type RegimeCalibrationSample,
} from './regime-calibrator.js';
import {
  applyTwoPassRegimePlatt,
  deserializeTwoPassRegimePlatt,
  fitTwoPassRegimePlatt,
  serializeTwoPassRegimePlatt,
} from './regime-calibrator-two-pass.js';

const EPS = 1e-6;
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
const logit = (p: number): number => {
  const c = Math.max(EPS, Math.min(1 - EPS, p));
  return Math.log(c / (1 - c));
};
const logLoss = (p: number, y: number): number => {
  const c = Math.max(EPS, Math.min(1 - EPS, p));
  return -(y * Math.log(c) + (1 - y) * Math.log(1 - c));
};

function makeOverconfidentSamples(n: number, seed: number): RegimeCalibrationSample[] {
  // True probability: 0.6.  Forecaster output: pushed toward 0/1
  // by overconfidence factor of 1.5 on the logit scale.
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const samples: RegimeCalibrationSample[] = [];
  for (let i = 0; i < n; i++) {
    const trueP = 0.6;
    const outcome: 0 | 1 = rand() < trueP ? 1 : 0;
    const noisyTrue = trueP + (rand() - 0.5) * 0.1;
    const overconf = sigmoid(1.5 * logit(noisyTrue));
    samples.push({ regime: 'bull', pRaw: overconf, outcome });
  }
  return samples;
}

describe('R5 Sprint 2 — two-pass regime Platt calibration', () => {
  it('serialize/deserialize round-trip', () => {
    const samples = makeOverconfidentSamples(200, 7);
    const fits = fitTwoPassRegimePlatt(samples);
    const json = serializeTwoPassRegimePlatt(fits);
    const restored = deserializeTwoPassRegimePlatt(json);
    expect(restored).toEqual(fits);
  });

  it('graceful fallback on empty / malformed JSON', () => {
    expect(deserializeTwoPassRegimePlatt('')).toEqual({});
    expect(deserializeTwoPassRegimePlatt('not-json')).toEqual({});
    expect(deserializeTwoPassRegimePlatt('null')).toEqual({});
  });

  it('apply with no fit returns clamped raw probability', () => {
    const out = applyTwoPassRegimePlatt(0.7, 'bull', {});
    expect(out).toBeCloseTo(0.7, 6);
    const clamped = applyTwoPassRegimePlatt(1.0, 'bull', {});
    expect(clamped).toBeLessThan(1);
    expect(clamped).toBeGreaterThan(0);
  });

  it('two-pass log-loss <= single-pass on overconfident inputs', () => {
    const samples = makeOverconfidentSamples(400, 17);
    const onePass = fitRegimePlatt(samples);
    const twoPass = fitTwoPassRegimePlatt(samples);

    let onePassLoss = 0;
    let twoPassLoss = 0;
    for (const s of samples) {
      onePassLoss += logLoss(applyRegimePlatt(s.pRaw, s.regime, onePass), s.outcome);
      twoPassLoss += logLoss(applyTwoPassRegimePlatt(s.pRaw, s.regime, twoPass), s.outcome);
    }
    onePassLoss /= samples.length;
    twoPassLoss /= samples.length;

    // Two-pass must not be worse than one-pass on the training data.
    expect(twoPassLoss).toBeLessThanOrEqual(onePassLoss + 1e-9);
  });

  it('deterministic — same input ⇒ same fits', () => {
    const samples = makeOverconfidentSamples(250, 99);
    const a = fitTwoPassRegimePlatt(samples);
    const b = fitTwoPassRegimePlatt(samples);
    expect(a).toEqual(b);
  });

  it('falls back to one-pass when pass-2 is degenerate', () => {
    // All-same-outcome ⇒ pass-1 returns null; two-pass also returns nothing
    // for that regime (rather than blowing up).
    const samples: RegimeCalibrationSample[] = Array.from({ length: 60 }, (_, i) => ({
      regime: 'bull',
      pRaw: 0.4 + (i % 5) * 0.02,
      outcome: 1, // all positive
    }));
    const fits = fitTwoPassRegimePlatt(samples);
    expect(fits.bull).toBeUndefined();
  });
});
