/**
 * Unit tests for backtest metrics.
 * Uses synthetic data with analytically known correct answers.
 */

import { describe, it, expect } from 'bun:test';
import {
  brierScore,
  reliabilityBins,
  maxReliabilityDeviation,
  ciCoverage,
  directionalAccuracy,
  selectiveDirectionalAccuracy,
  computeRCCurve,
  expectedReturnCorrelation,
  sharpness,
  gofPassRate,
  generateReport,
  optimizeThresholds,
  type BacktestStep,
} from './metrics.js';

// ---------------------------------------------------------------------------
// Helpers — create synthetic BacktestStep data
// ---------------------------------------------------------------------------

function makeStep(overrides: Partial<BacktestStep> = {}): BacktestStep {
  return {
    t: 0,
    predictedProb: 0.5,
    actualBinary: 1,
    predictedReturn: 0.02,
    actualReturn: 0.03,
    ciLower: 90,
    ciUpper: 110,
    realizedPrice: 100,
    recommendation: 'BUY',
    gofPasses: null,
    confidence: 0.5,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// brierScore
// ---------------------------------------------------------------------------

describe('brierScore', () => {
  it('returns 0 for perfect predictions', () => {
    const steps = [
      makeStep({ predictedProb: 1, actualBinary: 1 }),
      makeStep({ predictedProb: 0, actualBinary: 0 }),
    ];
    expect(brierScore(steps)).toBeCloseTo(0, 10);
  });

  it('returns 1 for worst possible predictions', () => {
    const steps = [
      makeStep({ predictedProb: 1, actualBinary: 0 }),
      makeStep({ predictedProb: 0, actualBinary: 1 }),
    ];
    expect(brierScore(steps)).toBeCloseTo(1, 10);
  });

  it('returns 0.25 for coin-flip predictions', () => {
    // P=0.5 always, mix of outcomes → (0.5-1)²+(0.5-0)² / 2 = (0.25+0.25)/2 = 0.25
    const steps = [
      makeStep({ predictedProb: 0.5, actualBinary: 1 }),
      makeStep({ predictedProb: 0.5, actualBinary: 0 }),
    ];
    expect(brierScore(steps)).toBeCloseTo(0.25, 10);
  });

  it('returns 1 for empty input', () => {
    expect(brierScore([])).toBe(1);
  });

  it('handles intermediate predictions correctly', () => {
    // P=0.7, actual=1 → (0.7-1)²=0.09
    // P=0.3, actual=0 → (0.3-0)²=0.09
    const steps = [
      makeStep({ predictedProb: 0.7, actualBinary: 1 }),
      makeStep({ predictedProb: 0.3, actualBinary: 0 }),
    ];
    expect(brierScore(steps)).toBeCloseTo(0.09, 6);
  });
});

// ---------------------------------------------------------------------------
// reliabilityBins
// ---------------------------------------------------------------------------

describe('reliabilityBins', () => {
  it('returns correct number of bins', () => {
    const bins = reliabilityBins([makeStep()], 10);
    expect(bins).toHaveLength(10);
  });

  it('places predictions in correct bins', () => {
    const steps = [
      makeStep({ predictedProb: 0.15, actualBinary: 1 }),
      makeStep({ predictedProb: 0.15, actualBinary: 0 }),
      makeStep({ predictedProb: 0.85, actualBinary: 1 }),
    ];
    const bins = reliabilityBins(steps, 10);
    // 0.15 goes in bin [0.1, 0.2)
    expect(bins[1].count).toBe(2);
    expect(bins[1].meanPredicted).toBeCloseTo(0.15);
    expect(bins[1].actualFrequency).toBeCloseTo(0.5);
    // 0.85 goes in bin [0.8, 0.9)
    expect(bins[8].count).toBe(1);
    expect(bins[8].actualFrequency).toBeCloseTo(1.0);
  });

  it('perfectly calibrated data has low maxReliabilityDeviation', () => {
    // Create steps where predicted ≈ actual frequency
    const steps: BacktestStep[] = [];
    for (let i = 0; i < 100; i++) {
      const p = (i + 0.5) / 100;
      const actual = Math.random() < p ? 1 : 0;
      steps.push(makeStep({ predictedProb: p, actualBinary: actual }));
    }
    // With 100 samples, deviation should be moderate (not zero due to randomness)
    const bins = reliabilityBins(steps);
    const dev = maxReliabilityDeviation(bins, 1);
    // Should be < 0.4 (generous for small sample)
    expect(dev).toBeLessThan(0.4);
  });
});

// ---------------------------------------------------------------------------
// ciCoverage
// ---------------------------------------------------------------------------

describe('ciCoverage', () => {
  it('returns 1.0 when all prices are within CI', () => {
    const steps = [
      makeStep({ ciLower: 90, ciUpper: 110, realizedPrice: 100 }),
      makeStep({ ciLower: 90, ciUpper: 110, realizedPrice: 95 }),
    ];
    expect(ciCoverage(steps)).toBeCloseTo(1.0);
  });

  it('returns 0.0 when no prices are within CI', () => {
    const steps = [
      makeStep({ ciLower: 90, ciUpper: 95, realizedPrice: 100 }),
      makeStep({ ciLower: 105, ciUpper: 110, realizedPrice: 100 }),
    ];
    expect(ciCoverage(steps)).toBeCloseTo(0.0);
  });

  it('returns correct fraction for mixed coverage', () => {
    const steps = [
      makeStep({ ciLower: 90, ciUpper: 110, realizedPrice: 100 }),  // covered
      makeStep({ ciLower: 90, ciUpper: 95,  realizedPrice: 100 }),  // not covered
      makeStep({ ciLower: 90, ciUpper: 110, realizedPrice: 90 }),   // covered (at boundary)
    ];
    expect(ciCoverage(steps)).toBeCloseTo(2 / 3);
  });

  it('returns 0 for empty input', () => {
    expect(ciCoverage([])).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// directionalAccuracy
// ---------------------------------------------------------------------------

describe('directionalAccuracy', () => {
  it('returns 1.0 for perfect directional calls', () => {
    const steps = [
      makeStep({ recommendation: 'BUY',  actualReturn:  0.05 }),
      makeStep({ recommendation: 'SELL', actualReturn: -0.04 }),
      makeStep({ recommendation: 'HOLD', actualReturn:  0.01 }),
    ];
    expect(directionalAccuracy(steps)).toBeCloseTo(1.0);
  });

  it('returns 0.0 for completely wrong calls', () => {
    const steps = [
      makeStep({ recommendation: 'BUY',  actualReturn: -0.05 }),
      makeStep({ recommendation: 'SELL', actualReturn:  0.05 }),
      makeStep({ recommendation: 'HOLD', actualReturn:  0.10 }),  // moved too much
    ];
    expect(directionalAccuracy(steps)).toBeCloseTo(0.0);
  });

  it('HOLD is correct when |return| < threshold', () => {
    const steps = [
      makeStep({ recommendation: 'HOLD', actualReturn:  0.02 }),
      makeStep({ recommendation: 'HOLD', actualReturn: -0.01 }),
    ];
    expect(directionalAccuracy(steps, 0.03)).toBeCloseTo(1.0);
  });
});

// ---------------------------------------------------------------------------
// expectedReturnCorrelation
// ---------------------------------------------------------------------------

describe('expectedReturnCorrelation', () => {
  it('returns 1.0 for perfectly correlated predictions', () => {
    const steps = [
      makeStep({ predictedReturn: 0.01, actualReturn: 0.02 }),
      makeStep({ predictedReturn: 0.03, actualReturn: 0.06 }),
      makeStep({ predictedReturn: 0.05, actualReturn: 0.10 }),
    ];
    expect(expectedReturnCorrelation(steps)).toBeCloseTo(1.0, 5);
  });

  it('returns -1.0 for perfectly anti-correlated', () => {
    const steps = [
      makeStep({ predictedReturn: 0.01, actualReturn: -0.01 }),
      makeStep({ predictedReturn: 0.03, actualReturn: -0.03 }),
      makeStep({ predictedReturn: 0.05, actualReturn: -0.05 }),
    ];
    expect(expectedReturnCorrelation(steps)).toBeCloseTo(-1.0, 5);
  });

  it('returns ~0 for uncorrelated data', () => {
    // Orthogonal signals
    const steps = [
      makeStep({ predictedReturn: 0.01, actualReturn:  0.01 }),
      makeStep({ predictedReturn: 0.01, actualReturn: -0.01 }),
      makeStep({ predictedReturn:-0.01, actualReturn:  0.01 }),
      makeStep({ predictedReturn:-0.01, actualReturn: -0.01 }),
    ];
    expect(expectedReturnCorrelation(steps)).toBeCloseTo(0, 5);
  });

  it('returns 0 for fewer than 3 steps', () => {
    expect(expectedReturnCorrelation([makeStep()])).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// sharpness
// ---------------------------------------------------------------------------

describe('sharpness', () => {
  it('narrower CIs produce lower sharpness', () => {
    const narrow = [makeStep({ ciLower: 98, ciUpper: 102, realizedPrice: 100 })];
    const wide   = [makeStep({ ciLower: 80, ciUpper: 120, realizedPrice: 100 })];
    expect(sharpness(narrow)).toBeLessThan(sharpness(wide));
  });

  it('computes correct relative width', () => {
    // (110-90)/100 = 0.2
    const steps = [makeStep({ ciLower: 90, ciUpper: 110, realizedPrice: 100 })];
    expect(sharpness(steps)).toBeCloseTo(0.2);
  });
});

// ---------------------------------------------------------------------------
// gofPassRate
// ---------------------------------------------------------------------------

describe('gofPassRate', () => {
  it('returns null when no steps have GOF results', () => {
    expect(gofPassRate([makeStep({ gofPasses: null })])).toBeNull();
  });

  it('returns correct fraction', () => {
    const steps = [
      makeStep({ gofPasses: true }),
      makeStep({ gofPasses: true }),
      makeStep({ gofPasses: false }),
      makeStep({ gofPasses: null }),  // ignored
    ];
    expect(gofPassRate(steps)).toBeCloseTo(2 / 3);
  });
});

// ---------------------------------------------------------------------------
// generateReport
// ---------------------------------------------------------------------------

describe('generateReport', () => {
  it('produces a complete report with all fields', () => {
    const steps = [
      makeStep({ predictedProb: 0.7, actualBinary: 1, predictedReturn: 0.02, actualReturn: 0.03 }),
      makeStep({ predictedProb: 0.3, actualBinary: 0, predictedReturn: -0.01, actualReturn: -0.02 }),
    ];
    const report = generateReport('SPY', 30, steps);

    expect(report.ticker).toBe('SPY');
    expect(report.horizon).toBe(30);
    expect(report.totalSteps).toBe(2);
    expect(typeof report.brierScore).toBe('number');
    expect(typeof report.ciCoverage).toBe('number');
    expect(typeof report.directionalAccuracy).toBe('number');
    expect(typeof report.expectedReturnCorrelation).toBe('number');
    expect(typeof report.sharpness).toBe('number');
    expect(report.reliabilityBins).toHaveLength(10);
  });
});

// ---------------------------------------------------------------------------
// optimizeThresholds
// ---------------------------------------------------------------------------

describe('optimizeThresholds', () => {
  it('finds optimal thresholds for strongly directional data', () => {
    // All steps have large positive predicted return and positive actual return
    // Best threshold should be low (to capture them as BUY)
    const steps = Array.from({ length: 20 }, () =>
      makeStep({ predictedReturn: 0.08, actualReturn: 0.05, recommendation: 'BUY' }),
    );
    const result = optimizeThresholds(steps);
    expect(result.bestAccuracy).toBeGreaterThan(0.9);
    expect(result.bestBuyThreshold).toBeLessThanOrEqual(0.07);
  });

  it('returns grid with all combinations', () => {
    const steps = [makeStep()];
    const result = optimizeThresholds(steps, [0.01, 0.02], [0.01, 0.02]);
    expect(result.grid).toHaveLength(4); // 2 × 2
  });

  it('handles empty steps array', () => {
    const result = optimizeThresholds([]);
    expect(result.bestAccuracy).toBe(0);
  });

  it('prefers HOLD when returns are near zero', () => {
    const steps = Array.from({ length: 20 }, () =>
      makeStep({ predictedReturn: 0.001, actualReturn: 0.005, recommendation: 'HOLD' }),
    );
    const result = optimizeThresholds(steps);
    // With near-zero predicted returns, best strategy is HOLD with tight buy threshold
    // actualReturn=0.005 < holdBand=0.03 so HOLD is correct
    expect(result.bestAccuracy).toBeGreaterThan(0.5);
  });
});

// ---------------------------------------------------------------------------
// selectiveDirectionalAccuracy (Idea M)
// ---------------------------------------------------------------------------

describe('selectiveDirectionalAccuracy', () => {
  it('returns full coverage at threshold 0', () => {
    const steps = [
      makeStep({ confidence: 0.8, recommendation: 'BUY', actualReturn: 0.05 }),
      makeStep({ confidence: 0.2, recommendation: 'BUY', actualReturn: -0.03 }),
    ];
    const result = selectiveDirectionalAccuracy(steps, 0);
    expect(result.coverage).toBe(1.0);
    expect(result.selected).toBe(2);
    expect(result.accuracy).toBe(0.5); // 1 correct out of 2
  });

  it('filters low-confidence predictions', () => {
    const steps = [
      makeStep({ confidence: 0.8, recommendation: 'BUY', actualReturn: 0.05 }),  // correct, included
      makeStep({ confidence: 0.2, recommendation: 'BUY', actualReturn: -0.03 }), // wrong, excluded
      makeStep({ confidence: 0.9, recommendation: 'SELL', actualReturn: -0.02 }), // correct, included
    ];
    const result = selectiveDirectionalAccuracy(steps, 0.5);
    expect(result.selected).toBe(2);
    expect(result.coverage).toBeCloseTo(2 / 3, 5);
    expect(result.accuracy).toBe(1.0); // 2/2 correct
  });

  it('returns 0 coverage when all below threshold', () => {
    const steps = [
      makeStep({ confidence: 0.1 }),
      makeStep({ confidence: 0.2 }),
    ];
    const result = selectiveDirectionalAccuracy(steps, 0.5);
    expect(result.selected).toBe(0);
    expect(result.coverage).toBe(0);
    expect(result.accuracy).toBe(0);
  });

  it('handles empty steps', () => {
    const result = selectiveDirectionalAccuracy([], 0.5);
    expect(result.total).toBe(0);
    expect(result.accuracy).toBe(0);
  });

  it('accuracy improves when low-confidence wrong predictions are filtered', () => {
    // Simulate a model that is accurate when confident, wrong when not
    const steps = [
      // High confidence, all correct
      makeStep({ confidence: 0.9, recommendation: 'BUY', actualReturn: 0.04 }),
      makeStep({ confidence: 0.8, recommendation: 'SELL', actualReturn: -0.03 }),
      makeStep({ confidence: 0.85, recommendation: 'BUY', actualReturn: 0.02 }),
      // Low confidence, all wrong
      makeStep({ confidence: 0.1, recommendation: 'BUY', actualReturn: -0.05 }),
      makeStep({ confidence: 0.15, recommendation: 'SELL', actualReturn: 0.03 }),
    ];
    const all = selectiveDirectionalAccuracy(steps, 0);
    const selective = selectiveDirectionalAccuracy(steps, 0.5);
    expect(all.accuracy).toBe(3 / 5); // 60%
    expect(selective.accuracy).toBe(1.0); // 100% — filtered out the wrong ones
    expect(selective.coverage).toBe(3 / 5); // 60% coverage
  });

  it('HOLD is correct when |actualReturn| < holdThreshold', () => {
    const steps = [
      makeStep({ confidence: 0.7, recommendation: 'HOLD', actualReturn: 0.01 }),
    ];
    const result = selectiveDirectionalAccuracy(steps, 0.5, 0.03);
    expect(result.accuracy).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// computeRCCurve
// ---------------------------------------------------------------------------

describe('computeRCCurve', () => {
  it('returns decreasing coverage with increasing threshold', () => {
    const steps = Array.from({ length: 10 }, (_, i) =>
      makeStep({ confidence: i / 10, recommendation: 'BUY', actualReturn: 0.05 }),
    );
    const curve = computeRCCurve(steps);
    // Coverage should be non-increasing
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i].coverage).toBeLessThanOrEqual(curve[i - 1].coverage);
    }
  });

  it('returns 100% coverage at threshold 0', () => {
    const steps = [makeStep({ confidence: 0.5 })];
    const curve = computeRCCurve(steps);
    expect(curve[0].threshold).toBe(0);
    expect(curve[0].coverage).toBe(1.0);
  });

  it('returns correct number of points', () => {
    const steps = [makeStep()];
    const curve = computeRCCurve(steps, [0, 0.25, 0.5, 0.75]);
    expect(curve).toHaveLength(4);
  });
});
