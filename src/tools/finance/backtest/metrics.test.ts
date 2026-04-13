/**
 * Unit tests for backtest metrics.
 * Uses synthetic data with analytically known correct answers.
 */

import { describe, it, expect } from 'bun:test';
import {
  balancedDirectionalAccuracy,
  brierScore,
  bucketByAnchorQuality,
  bucketByConfidence,
  bucketByDivergence,
  bucketByEnsembleConsensus,
  bucketByHmmConverged,
  bucketByMoveDirection,
  bucketByMoveMagnitude,
  bucketByPUpBand,
  bucketByRegime,
  bucketByRecommendation,
  bucketByStructuralBreak,
  bucketByTrendVsChop,
  bucketByValidationMetric,
  bucketByVolatility,
  calibratedPUpDirectionalAccuracy,
  ciCoverage,
  computeFailureDecomposition,
  directionalAccuracy,
  computeRCCurve,
  expectedReturnCorrelation,
  generateReport,
  gofPassRate,
  maxReliabilityDeviation,
  meanEdge,
  mulberry32,
  optimizeThresholds,
  bootstrapDirectionalCI,
  bootstrapMetricCI,
  bootstrapBrierCI,
  bootstrapCIcoverageCI,
  pUpDirectionalAccuracy,
  rawPUpDirectionalAccuracy,
  reliabilityBins,
  selectiveDirectionalAccuracy,
  selectivePUpAccuracy,
  selectiveRawPUpAccuracy,
  sharpness,
  type BacktestStep,
  type BootstrapCI,
  type DecisionSource,
  type ProbabilitySource,
  type ProvenanceSummary,
  DEFAULT_PUP_BANDS,
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
    probabilitySource: 'calibrated',
    decisionSource: 'default',
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
    const rand = mulberry32(42);
    const steps: BacktestStep[] = [];
    for (let i = 0; i < 100; i++) {
      const p = (i + 0.5) / 100;
      const actual = rand() < p ? 1 : 0;
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

describe('balancedDirectionalAccuracy', () => {
  it('averages per-class recall across supported classes', () => {
    const steps = [
      makeStep({ recommendation: 'BUY', actualReturn: 0.05 }),
      makeStep({ recommendation: 'SELL', actualReturn: 0.04 }),
      makeStep({ recommendation: 'HOLD', actualReturn: 0.01 }),
      makeStep({ recommendation: 'BUY', actualReturn: -0.05 }),
      makeStep({ recommendation: 'SELL', actualReturn: -0.06 }),
    ];

    expect(balancedDirectionalAccuracy(steps)).toBeCloseTo((0.5 + 1 + 0.5) / 3, 10);
  });

  it('ignores classes with zero support', () => {
    const steps = [
      makeStep({ recommendation: 'BUY', actualReturn: 0.05 }),
      makeStep({ recommendation: 'SELL', actualReturn: 0.04 }),
    ];

    expect(balancedDirectionalAccuracy(steps)).toBeCloseTo(0.5, 10);
  });

  it('returns 0 for empty input', () => {
    expect(balancedDirectionalAccuracy([])).toBe(0);
  });
});

describe('meanEdge', () => {
  it('computes signed realized edge in the chosen direction', () => {
    const steps = [
      makeStep({ recommendation: 'BUY', actualReturn: 0.05 }),
      makeStep({ recommendation: 'SELL', actualReturn: -0.02 }),
      makeStep({ recommendation: 'HOLD', actualReturn: 0.04 }),
    ];

    expect(meanEdge(steps)).toBeCloseTo((0.05 + 0.02 - 0.04) / 3, 10);
  });

  it('returns 0 for empty input', () => {
    expect(meanEdge([])).toBe(0);
  });
});

describe('bucket helpers', () => {
  it('buckets confidence into the fixed PR1 ranges', () => {
    const rows = bucketByConfidence([
      makeStep({ confidence: 0.05 }),
      makeStep({ confidence: 0.25 }),
      makeStep({ confidence: 0.45 }),
      makeStep({ confidence: 0.65 }),
      makeStep({ confidence: 1.0 }),
    ]);

    expect(rows.map(row => row.label)).toEqual([
      '[0.00, 0.20)',
      '[0.20, 0.40)',
      '[0.40, 0.60)',
      '[0.60, 0.80)',
      '[0.80, 1.00]',
    ]);
    expect(rows.map(row => row.count)).toEqual([1, 1, 1, 1, 1]);
    expect(rows.map(row => row.fraction)).toEqual([0.2, 0.2, 0.2, 0.2, 0.2]);
  });

  it('buckets volatility into low medium and high thirds', () => {
    const rows = bucketByVolatility([
      makeStep({ actualReturn: 0.01 }),
      makeStep({ actualReturn: -0.02 }),
      makeStep({ actualReturn: 0.03 }),
      makeStep({ actualReturn: -0.04 }),
      makeStep({ actualReturn: 0.05 }),
      makeStep({ actualReturn: -0.06 }),
    ]);

    expect(rows.map(row => row.label)).toEqual(['low', 'medium', 'high']);
    expect(rows.map(row => row.count)).toEqual([2, 2, 2]);
  });

  it('buckets anchor quality and preserves unknown rows', () => {
    const rows = bucketByAnchorQuality([
      makeStep({ anchorQuality: 'good' }),
      makeStep({ anchorQuality: 'sparse' }),
      makeStep({ anchorQuality: 'none' }),
      makeStep({ anchorQuality: undefined }),
    ]);

    expect(rows.map(row => row.label)).toEqual(['good', 'sparse', 'none', 'unknown']);
    expect(rows.map(row => row.count)).toEqual([1, 1, 1, 1]);
  });

  it('buckets regimes deterministically and keeps unknown last', () => {
    const rows = bucketByRegime([
      makeStep({ regime: 'bull' }),
      makeStep({ regime: 'bear' }),
      makeStep({ regime: 'bull' }),
      makeStep({ regime: undefined }),
    ]);

    expect(rows.map(row => row.label)).toEqual(['bear', 'bull', 'unknown']);
    expect(rows.map(row => row.count)).toEqual([1, 2, 1]);
  });

  it('buckets validation metrics and handles sparse metadata', () => {
    const rows = bucketByValidationMetric([
      makeStep({ validationMetric: 'daily_return' }),
      makeStep({ validationMetric: 'horizon_return' }),
      makeStep({ validationMetric: undefined }),
    ]);

    expect(rows.map(row => row.label)).toEqual(['daily_return', 'horizon_return', 'unknown']);
    expect(rows.map(row => row.count)).toEqual([1, 1, 1]);
  });
});

describe('computeFailureDecomposition', () => {
  it('returns the expected slice keys and stable counts', () => {
    const report = computeFailureDecomposition([
      makeStep({ confidence: 0.1, actualReturn: 0.01, recommendation: 'HOLD', regime: 'bull', anchorQuality: 'good', validationMetric: 'daily_return' }),
      makeStep({ confidence: 0.3, actualReturn: -0.02, recommendation: 'SELL', regime: 'bear', anchorQuality: 'sparse', validationMetric: 'horizon_return' }),
      makeStep({ confidence: 0.7, actualReturn: -0.05, recommendation: 'SELL', regime: 'bull', anchorQuality: 'none', validationMetric: 'daily_return' }),
      makeStep({ confidence: 0.9, actualReturn: 0.08, recommendation: 'BUY', regime: undefined, anchorQuality: undefined, validationMetric: undefined }),
    ]);

    expect(report.totalSteps).toBe(4);
    expect(report.slices.map(slice => slice.key)).toEqual([
      'regime',
      'volatility',
      'moveMagnitude',
      'moveDirection',
      'confidence',
      'anchorQuality',
      'recommendation',
      'trendVsChop',
      'validationMetric',
      'structuralBreak',
      'divergence',
      'hmmConverged',
      'ensembleConsensus',
      'pUpBand',
    ]);

    const regimeRows = report.slices[0].rows.map(row => ({ label: row.label, count: row.count, fraction: row.fraction }));
    expect(regimeRows).toEqual([
      { label: 'bear', count: 1, fraction: 0.25 },
      { label: 'bull', count: 2, fraction: 0.5 },
      { label: 'unknown', count: 1, fraction: 0.25 },
    ]);

    const validationRows = report.slices[8].rows.map(row => ({ label: row.label, count: row.count }));
    expect(validationRows).toEqual([
      { label: 'daily_return', count: 2 },
      { label: 'horizon_return', count: 1 },
      { label: 'unknown', count: 1 },
    ]);
  });

  it('handles empty input without throwing', () => {
    const report = computeFailureDecomposition([]);

    expect(report.totalSteps).toBe(0);
    expect(report.slices).toHaveLength(14);
    expect(report.slices[0].rows).toEqual([]);
    expect(report.slices[1].rows.map(row => row.count)).toEqual([0, 0, 0]);
    expect(report.slices[4].rows.map(row => row.count)).toEqual([0, 0, 0, 0, 0]);
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
    expect(typeof report.balancedDirectionalAccuracy).toBe('number');
    expect(typeof report.meanEdge).toBe('number');
    expect(report.failureDecomposition?.slices).toHaveLength(14);
  });

  it('aggregates provenanceSummary from steps', () => {
    const steps = [
      makeStep({ decisionSource: 'default', probabilitySource: 'calibrated' }),
      makeStep({ decisionSource: 'crypto-short-horizon-raw', probabilitySource: 'calibrated' }),
      makeStep({ decisionSource: 'default' }),
    ];
    const report = generateReport('BTC-USD', 7, steps);

    expect(report.provenanceSummary).toBeDefined();
    expect(report.provenanceSummary!.decisionSources.default).toBe(2);
    expect(report.provenanceSummary!.decisionSources['crypto-short-horizon-raw']).toBe(1);
    expect(report.provenanceSummary!.probabilitySources.calibrated).toBe(3);
  });

  it('defaults provenance to calibrated/default when absent', () => {
    const steps = [
      makeStep({ decisionSource: undefined, probabilitySource: undefined }),
    ];
    const report = generateReport('BTC-USD', 7, steps);

    expect(report.provenanceSummary!.decisionSources.default).toBe(1);
    expect(report.provenanceSummary!.probabilitySources.calibrated).toBe(1);
  });

  it('surfaces whether the trend-only break-confidence experiment was active', () => {
    const report = generateReport('SPY', 14, [
      makeStep({ trendPenaltyOnlyBreakConfidenceActive: false }),
      makeStep({ trendPenaltyOnlyBreakConfidenceActive: true }),
    ]);

    expect(report.trendPenaltyOnlyBreakConfidenceActive).toBe(true);
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

// ---------------------------------------------------------------------------
// bootstrapMetricCI
// ---------------------------------------------------------------------------

describe('bootstrapMetricCI', () => {
  it('returns CI near 100% for all-correct directional data', () => {
    const steps = Array.from({ length: 50 }, () =>
      makeStep({ recommendation: 'BUY', actualReturn: 0.05 }),
    );
    const ci = bootstrapDirectionalCI(steps);
    expect(ci.lower).toBeCloseTo(1.0, 2);
    expect(ci.median).toBeCloseTo(1.0, 2);
    expect(ci.upper).toBeCloseTo(1.0, 2);
    expect(ci.nResamples).toBe(1000);
  });

  it('returns CI around 50% for random-like data', () => {
    // Half correct, half wrong → directional accuracy ≈ 0.5
    const steps = Array.from({ length: 100 }, (_, i) =>
      makeStep({
        recommendation: 'BUY',
        actualReturn: i % 2 === 0 ? 0.05 : -0.05,
      }),
    );
    const ci = bootstrapDirectionalCI(steps);
    expect(ci.lower).toBeGreaterThan(0.3);
    expect(ci.upper).toBeLessThan(0.7);
    expect(ci.median).toBeCloseTo(0.5, 1);
  });

  it('is reproducible with the same seed', () => {
    const steps = Array.from({ length: 30 }, (_, i) =>
      makeStep({
        recommendation: 'BUY',
        actualReturn: i % 3 === 0 ? -0.02 : 0.04,
      }),
    );
    const a = bootstrapMetricCI(steps, directionalAccuracy, 500, 42);
    const b = bootstrapMetricCI(steps, directionalAccuracy, 500, 42);
    expect(a.lower).toBe(b.lower);
    expect(a.median).toBe(b.median);
    expect(a.upper).toBe(b.upper);
  });

  it('produces different results with different seeds', () => {
    // Use enough variation that bootstrap resamples are sensitive to seed
    const steps = Array.from({ length: 60 }, (_, i) =>
      makeStep({
        recommendation: 'BUY',
        actualReturn: i % 5 === 0 ? -0.08 : i % 3 === 0 ? -0.02 : 0.04,
      }),
    );
    const a = bootstrapMetricCI(steps, directionalAccuracy, 1000, 1);
    const b = bootstrapMetricCI(steps, directionalAccuracy, 1000, 99999);
    // With 1000 resamples and different seeds, percentiles should differ
    const allSame = a.lower === b.lower && a.median === b.median && a.upper === b.upper;
    expect(allSame).toBe(false);
  });

  it('returns zeros for empty input', () => {
    const ci = bootstrapMetricCI([], directionalAccuracy);
    expect(ci.lower).toBe(0);
    expect(ci.median).toBe(0);
    expect(ci.upper).toBe(0);
    expect(ci.nResamples).toBe(1000);
  });

  it('handles single element', () => {
    const steps = [makeStep({ recommendation: 'BUY', actualReturn: 0.05 })];
    const ci = bootstrapDirectionalCI(steps);
    // Single correct step: every resample is identical → CI = [1, 1]
    expect(ci.lower).toBe(1.0);
    expect(ci.median).toBe(1.0);
    expect(ci.upper).toBe(1.0);
  });

  it('lower ≤ median ≤ upper', () => {
    const steps = Array.from({ length: 40 }, (_, i) =>
      makeStep({
        recommendation: i % 2 === 0 ? 'BUY' : 'SELL',
        actualReturn: i % 3 === 0 ? 0.05 : -0.03,
      }),
    );
    const ci = bootstrapDirectionalCI(steps);
    expect(ci.lower).toBeLessThanOrEqual(ci.median);
    expect(ci.median).toBeLessThanOrEqual(ci.upper);
  });

  it('respects nResamples parameter', () => {
    const steps = Array.from({ length: 20 }, () =>
      makeStep({ recommendation: 'BUY', actualReturn: 0.05 }),
    );
    const ci = bootstrapMetricCI(steps, directionalAccuracy, 200);
    expect(ci.nResamples).toBe(200);
  });
});

// ---------------------------------------------------------------------------
// bootstrapBrierCI / bootstrapCIcoverageCI wrappers
// ---------------------------------------------------------------------------

describe('bootstrap convenience wrappers', () => {
  const steps = Array.from({ length: 40 }, (_, i) =>
    makeStep({
      predictedProb: 0.6,
      actualBinary: i % 2,
      ciLower: 90,
      ciUpper: 110,
      realizedPrice: 100,
    }),
  );

  it('bootstrapBrierCI returns valid CI', () => {
    const ci = bootstrapBrierCI(steps);
    expect(ci.lower).toBeGreaterThanOrEqual(0);
    expect(ci.upper).toBeLessThanOrEqual(1);
    expect(ci.lower).toBeLessThanOrEqual(ci.upper);
  });

  it('bootstrapCIcoverageCI returns valid CI', () => {
    const ci = bootstrapCIcoverageCI(steps);
    expect(ci.lower).toBeGreaterThanOrEqual(0);
    expect(ci.upper).toBeLessThanOrEqual(1);
    expect(ci.lower).toBeLessThanOrEqual(ci.upper);
  });
});

// ---------------------------------------------------------------------------
// PR3A: calibrated vs raw P(up) directional accuracy
// ---------------------------------------------------------------------------

describe('calibratedPUpDirectionalAccuracy', () => {
  it('returns 1.0 when calibrated prob correctly predicts direction', () => {
    const steps = [
      makeStep({ predictedProb: 0.7, actualBinary: 1 }),
      makeStep({ predictedProb: 0.3, actualBinary: 0 }),
    ];
    expect(calibratedPUpDirectionalAccuracy(steps)).toBeCloseTo(1.0);
  });

  it('returns 0.0 when calibrated prob is always wrong', () => {
    const steps = [
      makeStep({ predictedProb: 0.7, actualBinary: 0 }),
      makeStep({ predictedProb: 0.3, actualBinary: 1 }),
    ];
    expect(calibratedPUpDirectionalAccuracy(steps)).toBeCloseTo(0.0);
  });

  it('counts tie (0.5) as correct for actualBinary=1', () => {
    const steps = [
      makeStep({ predictedProb: 0.5, actualBinary: 1 }),
    ];
    expect(calibratedPUpDirectionalAccuracy(steps)).toBeCloseTo(1.0);
  });

  it('pUpDirectionalAccuracy is an alias for calibratedPUpDirectionalAccuracy', () => {
    const steps = [
      makeStep({ predictedProb: 0.7, actualBinary: 1 }),
      makeStep({ predictedProb: 0.3, actualBinary: 0 }),
    ];
    expect(pUpDirectionalAccuracy(steps)).toBe(calibratedPUpDirectionalAccuracy(steps));
  });
});

describe('rawPUpDirectionalAccuracy', () => {
  it('returns 1.0 when raw prob correctly predicts direction', () => {
    const steps = [
      makeStep({ rawPredictedProb: 0.8, actualBinary: 1 }),
      makeStep({ rawPredictedProb: 0.2, actualBinary: 0 }),
    ];
    expect(rawPUpDirectionalAccuracy(steps)).toBeCloseTo(1.0);
  });

  it('returns 0.0 when raw prob is always wrong', () => {
    const steps = [
      makeStep({ rawPredictedProb: 0.8, actualBinary: 0 }),
      makeStep({ rawPredictedProb: 0.2, actualBinary: 1 }),
    ];
    expect(rawPUpDirectionalAccuracy(steps)).toBeCloseTo(0.0);
  });

  it('falls back to predictedProb when rawPredictedProb is absent', () => {
    const steps = [
      makeStep({ predictedProb: 0.7, actualBinary: 1 }),
      makeStep({ predictedProb: 0.3, actualBinary: 0 }),
    ];
    expect(rawPUpDirectionalAccuracy(steps)).toBeCloseTo(1.0);
  });

  it('raw and calibrated can differ when raw is more discriminative', () => {
    const steps = [
      makeStep({ predictedProb: 0.52, actualBinary: 1, rawPredictedProb: 0.85 }),
      makeStep({ predictedProb: 0.51, actualBinary: 1, rawPredictedProb: 0.20 }),
    ];
    // calibrated: both prob > 0.5 → both predicted UP → both correct → 100%
    expect(calibratedPUpDirectionalAccuracy(steps)).toBeCloseTo(1.0);
    // raw: step1 UP correct, step2 DOWN correct (0.20 < 0.5, actual=1=UP, wrong) → 50%
    expect(rawPUpDirectionalAccuracy(steps)).toBeCloseTo(0.5);
  });

  it('returns 0 for empty input', () => {
    expect(rawPUpDirectionalAccuracy([])).toBe(0);
  });
});

describe('selectiveRawPUpAccuracy', () => {
  it('filters by confidence on raw probability', () => {
    const steps = [
      makeStep({ confidence: 0.9, rawPredictedProb: 0.85, actualBinary: 1 }),
      makeStep({ confidence: 0.1, rawPredictedProb: 0.20, actualBinary: 0 }),
    ];
    const result = selectiveRawPUpAccuracy(steps, 0.5);
    expect(result.selected).toBe(1);
    expect(result.accuracy).toBe(1.0);
  });

  it('returns zeros for empty input', () => {
    const result = selectiveRawPUpAccuracy([], 0.5);
    expect(result.accuracy).toBe(0);
    expect(result.coverage).toBe(0);
    expect(result.selected).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// PR3A: P(up)-band buckets
// ---------------------------------------------------------------------------

describe('bucketByPUpBand', () => {
  it('assigns steps to correct bands', () => {
    const steps = [
      makeStep({ rawPredictedProb: 0.30 }),
      makeStep({ rawPredictedProb: 0.47 }),
      makeStep({ rawPredictedProb: 0.52 }),
      makeStep({ rawPredictedProb: 0.70 }),
    ];
    const rows = bucketByPUpBand(steps, 0.03, DEFAULT_PUP_BANDS, 'raw');
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel['<0.45']).toBe(1);
    expect(byLabel['0.45–0.50']).toBe(1);
    expect(byLabel['0.50–0.55']).toBe(1);
    expect(byLabel['>0.55']).toBe(1);
  });

  it('uses predictedProb as fallback when rawPredictedProb is absent', () => {
    const steps = [
      makeStep({ predictedProb: 0.30 }),
      makeStep({ predictedProb: 0.47 }),
      makeStep({ predictedProb: 0.52 }),
      makeStep({ predictedProb: 0.70 }),
    ];
    const rows = bucketByPUpBand(steps, 0.03, DEFAULT_PUP_BANDS, 'raw');
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel['<0.45']).toBe(1);
    expect(byLabel['0.45–0.50']).toBe(1);
    expect(byLabel['0.50–0.55']).toBe(1);
    expect(byLabel['>0.55']).toBe(1);
  });

  it('uses DEFAULT_PUP_BANDS labels', () => {
    const rows = bucketByPUpBand([]);
    expect(rows.map(r => r.label)).toEqual(DEFAULT_PUP_BANDS.map(b => b.label));
  });

  it('uses calibrated predictedProb by default', () => {
    const steps = [
      makeStep({ predictedProb: 0.48, rawPredictedProb: 0.70 }),
      makeStep({ predictedProb: 0.53, rawPredictedProb: 0.20 }),
    ];
    const rows = bucketByPUpBand(steps);
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel['0.45–0.50']).toBe(1);
    expect(byLabel['0.50–0.55']).toBe(1);
  });

  it('computes directional accuracy per band', () => {
    const steps = [
      makeStep({ rawPredictedProb: 0.30, actualBinary: 0 }), // <0.45, DOWN correct
      makeStep({ rawPredictedProb: 0.80, actualBinary: 1 }), // >0.55, UP correct
    ];
    const rows = bucketByPUpBand(steps, 0.03, DEFAULT_PUP_BANDS, 'raw');
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.directionalAccuracy]));
    expect(byLabel['<0.45']).toBe(1.0);
    expect(byLabel['>0.55']).toBe(1.0);
  });
});

// ---------------------------------------------------------------------------
// PR3A: structural break / HMM / ensemble bucket helpers
// ---------------------------------------------------------------------------

describe('bucketByStructuralBreak', () => {
  it('assigns true/false/unknown buckets', () => {
    const steps = [
      makeStep({ structuralBreakDetected: true }),
      makeStep({ structuralBreakDetected: false }),
      makeStep({ structuralBreakDetected: undefined }),
    ];
    const rows = bucketByStructuralBreak(steps);
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel['true']).toBe(1);
    expect(byLabel['false']).toBe(1);
    expect(byLabel['unknown']).toBe(1);
  });

  it('handles empty input', () => {
    const rows = bucketByStructuralBreak([]);
    expect(rows).toHaveLength(3);
  });
});

describe('bucketByDivergence', () => {
  it('buckets original or current structural-break divergence', () => {
    const steps = [
      makeStep({ structuralBreakDivergence: 0.04 }),
      makeStep({ structuralBreakDivergence: 0.08 }),
      makeStep({ originalStructuralBreakDivergence: 0.12 }),
      makeStep({ structuralBreakDivergence: 0.25 }),
      makeStep({ structuralBreakDivergence: null }),
    ];
    const rows = bucketByDivergence(steps);
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel['<0.05']).toBe(1);
    expect(byLabel['0.05–0.10']).toBe(1);
    expect(byLabel['0.10–0.20']).toBe(1);
    expect(byLabel['≥0.20']).toBe(1);
    expect(byLabel['unknown']).toBe(1);
  });
});

describe('bucketByMoveMagnitude', () => {
  it('buckets realized-move magnitudes into fixed ranges', () => {
    const steps = [
      makeStep({ actualReturn: 0.01 }),
      makeStep({ actualReturn: -0.03 }),
      makeStep({ actualReturn: 0.07 }),
      makeStep({ actualReturn: -0.15 }),
    ];
    const rows = bucketByMoveMagnitude(steps);
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel['<2%']).toBe(1);
    expect(byLabel['2–5%']).toBe(1);
    expect(byLabel['5–10%']).toBe(1);
    expect(byLabel['≥10%']).toBe(1);
  });
});

describe('bucketByMoveDirection', () => {
  it('classifies realized moves as up/down/flat', () => {
    const steps = [
      makeStep({ actualReturn: 0.05 }),
      makeStep({ actualReturn: -0.05 }),
      makeStep({ actualReturn: 0.01 }),
    ];
    const rows = bucketByMoveDirection(steps);
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel.up).toBe(1);
    expect(byLabel.down).toBe(1);
    expect(byLabel.flat).toBe(1);
  });
});

describe('bucketByRecommendation', () => {
  it('groups by BUY/HOLD/SELL recommendation', () => {
    const steps = [
      makeStep({ recommendation: 'BUY' }),
      makeStep({ recommendation: 'HOLD' }),
      makeStep({ recommendation: 'SELL' }),
    ];
    const rows = bucketByRecommendation(steps);
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel.BUY).toBe(1);
    expect(byLabel.HOLD).toBe(1);
    expect(byLabel.SELL).toBe(1);
  });
});

describe('bucketByTrendVsChop', () => {
  it('groups bull/bear as trending and sideways as chop', () => {
    const steps = [
      makeStep({ regime: 'bull' }),
      makeStep({ regime: 'bear' }),
      makeStep({ regime: 'sideways' }),
      makeStep({ regime: undefined }),
    ];
    const rows = bucketByTrendVsChop(steps);
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel.trending).toBe(2);
    expect(byLabel.chop).toBe(1);
    expect(byLabel.unknown).toBe(1);
  });
});

describe('bucketByHmmConverged', () => {
  it('assigns true/false/unknown buckets', () => {
    const steps = [
      makeStep({ hmmConverged: true }),
      makeStep({ hmmConverged: false }),
      makeStep({ hmmConverged: null }),
    ];
    const rows = bucketByHmmConverged(steps);
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel['true']).toBe(1);
    expect(byLabel['false']).toBe(1);
    expect(byLabel['unknown']).toBe(1);
  });
});

describe('bucketByEnsembleConsensus', () => {
  it('assigns low/medium/high/unknown buckets', () => {
    const steps = [
      makeStep({ ensembleConsensus: 0.2 }),
      makeStep({ ensembleConsensus: 0.5 }),
      makeStep({ ensembleConsensus: 0.8 }),
      makeStep({ ensembleConsensus: null }),
    ];
    const rows = bucketByEnsembleConsensus(steps);
    const byLabel = Object.fromEntries(rows.map(r => [r.label, r.count]));
    expect(byLabel['low']).toBe(1);
    expect(byLabel['medium']).toBe(1);
    expect(byLabel['high']).toBe(1);
    expect(byLabel['unknown']).toBe(1);
  });
});

describe('computeFailureDecomposition PR3A slices', () => {
  it('includes structuralBreak, divergence, new move/trend slices, hmmConverged, ensembleConsensus, pUpBand slices', () => {
    const report = computeFailureDecomposition([
      makeStep({
        structuralBreakDetected: false,
        structuralBreakDivergence: 0.11,
        hmmConverged: true,
        ensembleConsensus: 0.6,
        rawPredictedProb: 0.55,
        regime: 'bull',
      }),
    ]);
    const keys = report.slices.map(s => s.key);
    expect(keys).toContain('structuralBreak');
    expect(keys).toContain('divergence');
    expect(keys).toContain('moveMagnitude');
    expect(keys).toContain('moveDirection');
    expect(keys).toContain('recommendation');
    expect(keys).toContain('trendVsChop');
    expect(keys).toContain('hmmConverged');
    expect(keys).toContain('ensembleConsensus');
    expect(keys).toContain('pUpBand');
  });

  it('has 14 slices total after Phase 2A additions', () => {
    const report = computeFailureDecomposition([]);
    expect(report.slices).toHaveLength(14);
  });
});

describe('generateReport provenance summary', () => {
  it('counts mixed replay/internal decision sources explicitly', () => {
    const steps = [
      makeStep({ decisionSource: 'default', probabilitySource: 'calibrated' }),
      makeStep({ decisionSource: 'replay-anchor', probabilitySource: 'calibrated' }),
      makeStep({ decisionSource: 'crypto-short-horizon-recency+replay-anchor', probabilitySource: 'calibrated' }),
      makeStep({ decisionSource: 'crypto-short-horizon-disagreement-blend+replay-anchor', probabilitySource: 'calibrated' }),
    ];

    const report = generateReport('BTC-USD', 14, steps);

    expect(report.provenanceSummary?.decisionSources.default).toBe(1);
    expect(report.provenanceSummary?.decisionSources['replay-anchor']).toBe(1);
    expect(report.provenanceSummary?.decisionSources['crypto-short-horizon-recency+replay-anchor']).toBe(1);
    expect(report.provenanceSummary?.decisionSources['crypto-short-horizon-disagreement-blend+replay-anchor']).toBe(1);
    expect(report.provenanceSummary?.probabilitySources.calibrated).toBe(4);
  });
});
