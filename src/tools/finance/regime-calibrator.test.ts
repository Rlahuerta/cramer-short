/**
 * Round-4 Idea 3 — Regime-conditional Platt recalibrator.
 *
 * TDD spec for src/tools/finance/regime-calibrator.ts
 *
 * Platt scaling fits a 2-param logistic to (raw probability, outcome) pairs:
 *
 *   p_calibrated = sigmoid( a · logit(p_raw) + b )
 *
 * Per-regime fits let calibration adapt to regime-specific bias patterns
 * (e.g., over-confidence in bull markets, under-confidence in chop).
 */

import { describe, expect, test } from 'bun:test';
import {
  fitRegimePlatt,
  applyRegimePlatt,
  type RegimeCalibrationSample,
  type RegimePlattFits,
} from './regime-calibrator.js';

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
const logit = (p: number): number => {
  const c = Math.max(1e-6, Math.min(1 - 1e-6, p));
  return Math.log(c / (1 - c));
};

describe('regime-calibrator', () => {
  test('fitRegimePlatt produces one fit per regime present in samples', () => {
    const samples: RegimeCalibrationSample[] = [
      { regime: 'bull', pRaw: 0.6, outcome: 1 },
      { regime: 'bull', pRaw: 0.7, outcome: 1 },
      { regime: 'bull', pRaw: 0.4, outcome: 0 },
      { regime: 'bull', pRaw: 0.3, outcome: 0 },
      { regime: 'bear', pRaw: 0.55, outcome: 0 },
      { regime: 'bear', pRaw: 0.65, outcome: 0 },
      { regime: 'bear', pRaw: 0.45, outcome: 1 },
      { regime: 'bear', pRaw: 0.35, outcome: 1 },
    ];
    const fits = fitRegimePlatt(samples, { minSamplesPerRegime: 4 });
    expect(fits.bull).toBeDefined();
    expect(fits.bear).toBeDefined();
    expect(fits.sideways).toBeUndefined();
  });

  test('fits with too few samples per regime are skipped (min 10)', () => {
    const samples: RegimeCalibrationSample[] = [
      { regime: 'sideways', pRaw: 0.5, outcome: 1 },
      { regime: 'sideways', pRaw: 0.6, outcome: 0 },
    ];
    const fits = fitRegimePlatt(samples, { minSamplesPerRegime: 10 });
    expect(fits.sideways).toBeUndefined();
  });

  test('applyRegimePlatt returns raw probability when no fit exists for regime', () => {
    const fits: RegimePlattFits = {};
    const out = applyRegimePlatt(0.7, 'bull', fits);
    expect(out).toBeCloseTo(0.7, 6);
  });

  test('applyRegimePlatt with identity fit (a=1, b=0) returns raw', () => {
    const fits: RegimePlattFits = { bull: { a: 1, b: 0, n: 100 } };
    expect(applyRegimePlatt(0.7, 'bull', fits)).toBeCloseTo(0.7, 6);
    expect(applyRegimePlatt(0.3, 'bull', fits)).toBeCloseTo(0.3, 6);
  });

  test('applyRegimePlatt with shrinkage fit (a<1) compresses toward 0.5', () => {
    const fits: RegimePlattFits = { bull: { a: 0.5, b: 0, n: 100 } };
    const calibrated = applyRegimePlatt(0.9, 'bull', fits);
    // a=0.5 with b=0 ⇒ logit(p_cal) = 0.5·logit(0.9) ⇒ p_cal between 0.5 and 0.9.
    expect(calibrated).toBeGreaterThan(0.5);
    expect(calibrated).toBeLessThan(0.9);
    // Exact: sigmoid(0.5 · logit(0.9))
    expect(calibrated).toBeCloseTo(sigmoid(0.5 * logit(0.9)), 6);
  });

  test('fitted Platt improves Brier on synthetic miscalibrated data', () => {
    // Simulate a miscalibrated raw probability stream:
    //   true P(up) = 0.6, but raw model predicts 0.85 in bull regime.
    // Platt should pull predictions back toward truth ⇒ Brier drops.
    const truePUp = 0.6;
    const overconfidentRaw = 0.85;
    const samples: RegimeCalibrationSample[] = [];
    // Reproducible PRNG so test is deterministic.
    let seed = 42;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0xffffffff;
    };
    for (let i = 0; i < 200; i++) {
      samples.push({
        regime: 'bull',
        pRaw: overconfidentRaw,
        outcome: rng() < truePUp ? 1 : 0,
      });
    }
    const fits = fitRegimePlatt(samples);
    expect(fits.bull).toBeDefined();
    // Brier on raw (overconfident) vs calibrated.
    const brier = (preds: number[], outcomes: number[]): number => {
      let sse = 0;
      for (let i = 0; i < preds.length; i++) sse += (preds[i] - outcomes[i]) ** 2;
      return sse / preds.length;
    };
    const outcomes = samples.map(s => s.outcome);
    const rawPreds = samples.map(s => s.pRaw);
    const calPreds = samples.map(s => applyRegimePlatt(s.pRaw, s.regime, fits));
    const brierRaw = brier(rawPreds, outcomes);
    const brierCal = brier(calPreds, outcomes);
    // Calibrated should beat raw by a meaningful margin (>1pp) on this synthetic data.
    expect(brierCal).toBeLessThan(brierRaw - 0.01);
    // And the calibrated mean should be near the true rate.
    const calMean = calPreds.reduce((s, x) => s + x, 0) / calPreds.length;
    expect(Math.abs(calMean - truePUp)).toBeLessThan(0.05);
  });

  test('per-regime fits are independent — bear miscalibration does not leak to bull', () => {
    // Bull is well-calibrated (raw ≈ truth). Bear is over-confident.
    // After fit, bull probabilities should pass through nearly unchanged,
    // bear should be pulled toward truth.
    let seed = 7;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0xffffffff;
    };
    const samples: RegimeCalibrationSample[] = [];
    // 150 well-calibrated bull samples (raw=0.65 ≈ truth=0.65)
    for (let i = 0; i < 150; i++) {
      samples.push({ regime: 'bull', pRaw: 0.65, outcome: rng() < 0.65 ? 1 : 0 });
    }
    // 150 over-confident bear samples (raw=0.20 but truth=0.50 — wildly wrong)
    for (let i = 0; i < 150; i++) {
      samples.push({ regime: 'bear', pRaw: 0.20, outcome: rng() < 0.50 ? 1 : 0 });
    }
    const fits = fitRegimePlatt(samples);
    expect(fits.bull).toBeDefined();
    expect(fits.bear).toBeDefined();
    // Bull pass-through: calibrated ≈ raw within tolerance
    const bullCal = applyRegimePlatt(0.65, 'bull', fits);
    expect(Math.abs(bullCal - 0.65)).toBeLessThan(0.10);
    // Bear correction: calibrated should be much closer to 0.50 than raw 0.20
    const bearCal = applyRegimePlatt(0.20, 'bear', fits);
    expect(bearCal).toBeGreaterThan(0.30);
  });

  test('applyRegimePlatt clamps inputs and outputs to (0, 1)', () => {
    const fits: RegimePlattFits = { bull: { a: 1, b: 0, n: 100 } };
    expect(applyRegimePlatt(0, 'bull', fits)).toBeGreaterThan(0);
    expect(applyRegimePlatt(1, 'bull', fits)).toBeLessThan(1);
    expect(applyRegimePlatt(-0.5, 'bull', fits)).toBeGreaterThan(0);
    expect(applyRegimePlatt(2, 'bull', fits)).toBeLessThan(1);
  });

  test('fitRegimePlatt is deterministic for the same input', () => {
    const samples: RegimeCalibrationSample[] = [];
    for (let i = 0; i < 50; i++) {
      samples.push({ regime: 'bull', pRaw: 0.4 + (i % 5) * 0.05, outcome: i % 3 === 0 ? 1 : 0 });
    }
    const f1 = fitRegimePlatt(samples);
    const f2 = fitRegimePlatt(samples);
    expect(f1.bull?.a).toBeCloseTo(f2.bull?.a ?? 0, 9);
    expect(f1.bull?.b).toBeCloseTo(f2.bull?.b ?? 0, 9);
  });

  test('fitRegimePlatt returns empty object for empty samples', () => {
    expect(fitRegimePlatt([])).toEqual({});
  });

  test('serializeRegimePlatt + deserializeRegimePlatt round-trip', async () => {
    const { serializeRegimePlatt, deserializeRegimePlatt } = await import('./regime-calibrator.js');
    const fits: RegimePlattFits = {
      bull: { a: 0.8, b: 0.1, n: 120 },
      bear: { a: 0.6, b: -0.05, n: 90 },
    };
    const json = serializeRegimePlatt(fits);
    const restored = deserializeRegimePlatt(json);
    expect(restored.bull?.a).toBeCloseTo(0.8, 9);
    expect(restored.bear?.b).toBeCloseTo(-0.05, 9);
  });
});
