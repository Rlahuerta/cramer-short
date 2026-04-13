/**
 * Evaluation harness for Phase 5 hybrid structural-break fallback candidates.
 *
 * Tests the fallback plumbing in markov-distribution.ts (Work Package A)
 * and validates candidates against Phase 4 baseline (Work Package B/C).
 *
 * Run: bun test src/tools/finance/backtest/hybrid-break-fallback.test.ts
 * Run (grep): bun test src/tools/finance/backtest/hybrid-break-fallback.test.ts --grep "hybrid break fallback"
 */
import { readFileSync } from 'fs';
import { join } from 'path';
import { describe, expect, test } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import {
  buildConservativeFallbackMatrix,
  buildProfileFallbackMatrix,
  blendMatrices,
  computeBlendWeight,
  applyBreakFallbackCandidate,
  type BreakFallbackCandidate,
  type TransitionMatrix,
  buildDefaultMatrix,
  NUM_STATES,
} from '../markov-distribution.js';
import { walkForward } from './walk-forward.js';
import { evaluatePhase5Thresholds, runComparison } from './phase5-hybrid-break-fallback.js';

interface FixtureTickerData {
  closes: number[];
  dates: string[];
}

interface FixtureData {
  tickers: Record<string, FixtureTickerData>;
}

// ---------------------------------------------------------------------------
// Work Package A: Fallback plumbing unit tests
// ---------------------------------------------------------------------------

describe('hybrid break fallback', () => {
  test('buildConservativeFallbackMatrix produces valid stochastic matrix', () => {
    for (const diagonal of [0.55, 0.60, 0.65]) {
      const matrix = buildConservativeFallbackMatrix(diagonal);
      expect(matrix.length).toBe(NUM_STATES);
      for (let i = 0; i < NUM_STATES; i++) {
        const row = matrix[i];
        const rowSum = row.reduce((a, b) => a + b, 0);
        expect(Math.abs(rowSum - 1.0)).toBeLessThan(1e-10);
        expect(row[i]).toBeCloseTo(diagonal, 10);
        const offDiag = (1 - diagonal) / (NUM_STATES - 1);
        for (let j = 0; j < NUM_STATES; j++) {
          if (j !== i) {
            expect(row[j]).toBeCloseTo(offDiag, 10);
          }
        }
      }
    }
  });

  test('buildProfileFallbackMatrix produces per-asset-type diagonals', () => {
    const profileDiagonals = { equity: 0.60, etf: 0.55, commodity: 0.65, crypto: 0.70 };
    for (const assetType of ['equity', 'etf', 'commodity', 'crypto'] as const) {
      const matrix = buildProfileFallbackMatrix(assetType, profileDiagonals);
      const expectedDiag = profileDiagonals[assetType];
      for (let i = 0; i < NUM_STATES; i++) {
        const row = matrix[i];
        const rowSum = row.reduce((a, b) => a + b, 0);
        expect(Math.abs(rowSum - 1.0)).toBeLessThan(1e-10);
        expect(row[i]).toBeCloseTo(expectedDiag, 10);
        const offDiag = (1 - expectedDiag) / (NUM_STATES - 1);
        for (let j = 0; j < NUM_STATES; j++) {
          if (j !== i) {
            expect(row[j]).toBeCloseTo(offDiag, 10);
          }
        }
      }
    }
  });

  test('blendMatrices produces convex combination', () => {
    const A = buildConservativeFallbackMatrix(0.55);
    const B = buildConservativeFallbackMatrix(0.70);
    const blended = blendMatrices(0.5, A, B);
    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(blended[i][j]).toBeCloseTo(0.5 * A[i][j] + 0.5 * B[i][j], 10);
      }
      const rowSum = blended[i].reduce((a, b) => a + b, 0);
      expect(Math.abs(rowSum - 1.0)).toBeLessThan(1e-10);
    }
  });

  test('blendMatrices at λ=0 returns B, λ=1 returns A', () => {
    const A = buildConservativeFallbackMatrix(0.55);
    const B = buildConservativeFallbackMatrix(0.70);
    const zeroLambda = blendMatrices(0, A, B);
    const oneLambda = blendMatrices(1, A, B);
    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(zeroLambda[i][j]).toBeCloseTo(B[i][j], 10);
        expect(oneLambda[i][j]).toBeCloseTo(A[i][j], 10);
      }
    }
  });

  test('computeBlendWeight returns 0 for divergence below 0.05', () => {
    const weights = { mild: 0.25, medium: 0.50, high: 0.75 };
    expect(computeBlendWeight(0.03, weights)).toBe(0);
    expect(computeBlendWeight(0.049, weights)).toBe(0);
  });

  test('computeBlendWeight returns appropriate bucket values', () => {
    const weights = { mild: 0.25, medium: 0.50, high: 0.75 };
    expect(computeBlendWeight(0.07, weights)).toBe(0.25);
    expect(computeBlendWeight(0.15, weights)).toBe(0.50);
    expect(computeBlendWeight(0.25, weights)).toBe(0.75);
  });

  test('default behavior unchanged when no candidate supplied (hard mode, divergence above threshold)', () => {
    // When the default matrix is used (buildDefaultMatrix), it should be identical
    // to what happens when no candidate is supplied and a break is detected.
    // The default matrix has diagonal=0.6
    const defaultMatrix = buildDefaultMatrix();
    const estimatedMatrix: TransitionMatrix = [
      [0.7, 0.2, 0.1],
      [0.15, 0.6, 0.25],
      [0.1, 0.3, 0.6],
    ];
    // No candidate → break detected → buildDefaultMatrix()
    // With candidate mode='hard', conservativeDiagonal=0.60, conservativeWeight=1.0 → same result
    const hardCandidate: BreakFallbackCandidate = {
      id: 'test-hard-060',
      mode: 'hard',
      conservativeDiagonal: 0.60,
      profileDiagonals: { equity: 0.60, etf: 0.60, commodity: 0.60, crypto: 0.60 },
      conservativeWeight: 1.0,
      severityWeights: { mild: 1.0, medium: 1.0, high: 1.0 },
    };

    const result = applyBreakFallbackCandidate(estimatedMatrix, 0.15, hardCandidate, 'equity');
    // In hard mode, the result should be the hybrid fallback = conservative*1.0 + profile*0.0 = conservative
    // With conservativeWeight=1.0, hybridFallback = 1.0*conservative + 0.0*profile = conservative
    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(result[i][j]).toBeCloseTo(defaultMatrix[i][j], 10);
      }
    }
  });

  test('blended mode produces convex combination of estimated and fallback', () => {
    const estimatedMatrix: TransitionMatrix = [
      [0.7, 0.2, 0.1],
      [0.15, 0.6, 0.25],
      [0.1, 0.3, 0.6],
    ];
    const blendedCandidate: BreakFallbackCandidate = {
      id: 'test-blended',
      mode: 'blended',
      conservativeDiagonal: 0.60,
      profileDiagonals: { equity: 0.65, etf: 0.55, commodity: 0.60, crypto: 0.70 },
      conservativeWeight: 0.50,
      severityWeights: { mild: 0.25, medium: 0.50, high: 0.75 },
    };

    // With divergence=0.15 (medium bucket), blend weight = 0.50
    const result = applyBreakFallbackCandidate(estimatedMatrix, 0.15, blendedCandidate, 'equity');

    // Expected: (1-0.50)*estimated + 0.50*hybridFallback
    // hybridFallback for equity = 0.50*conservative(diag=0.60) + 0.50*profile(diag=0.65)
    const conservative = buildConservativeFallbackMatrix(0.60);
    const profile = buildProfileFallbackMatrix('equity', blendedCandidate.profileDiagonals);
    const hybridFallback = blendMatrices(0.50, conservative, profile);
    const expected = blendMatrices(1 - 0.50, estimatedMatrix, hybridFallback);

    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(result[i][j]).toBeCloseTo(expected[i][j], 10);
      }
    }

    // Result should be a valid stochastic matrix
    for (let i = 0; i < NUM_STATES; i++) {
      const rowSum = result[i].reduce((a, b) => a + b, 0);
      expect(Math.abs(rowSum - 1.0)).toBeLessThan(1e-10);
    }
  });

  test('blended_capped mode caps blend weight', () => {
    const estimatedMatrix: TransitionMatrix = [
      [0.7, 0.2, 0.1],
      [0.15, 0.6, 0.25],
      [0.1, 0.3, 0.6],
    ];
    const cappedCandidate: BreakFallbackCandidate = {
      id: 'test-capped',
      mode: 'blended_capped',
      conservativeDiagonal: 0.60,
      profileDiagonals: { equity: 0.65, etf: 0.55, commodity: 0.60, crypto: 0.70 },
      conservativeWeight: 0.50,
      severityWeights: { mild: 0.25, medium: 0.50, high: 0.75 },
      maxBlendWeight: 0.60,
    };

    // With divergence=0.25 (high bucket), raw blend weight = 0.75, but capped at 0.60
    const result = applyBreakFallbackCandidate(estimatedMatrix, 0.25, cappedCandidate, 'equity');

    const conservative = buildConservativeFallbackMatrix(0.60);
    const profile = buildProfileFallbackMatrix('equity', cappedCandidate.profileDiagonals);
    const hybridFallback = blendMatrices(0.50, conservative, profile);
    const expected = blendMatrices(1 - 0.60, estimatedMatrix, hybridFallback);

    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(result[i][j]).toBeCloseTo(expected[i][j], 10);
      }
    }
  });

  test('blended mode with divergence below 0.05 returns estimated matrix unchanged', () => {
    const estimatedMatrix: TransitionMatrix = [
      [0.7, 0.2, 0.1],
      [0.15, 0.6, 0.25],
      [0.1, 0.3, 0.6],
    ];
    const candidate: BreakFallbackCandidate = {
      id: 'test-low-div',
      mode: 'blended',
      conservativeDiagonal: 0.60,
      profileDiagonals: { equity: 0.65, etf: 0.55, commodity: 0.60, crypto: 0.70 },
      conservativeWeight: 0.50,
      severityWeights: { mild: 0.25, medium: 0.50, high: 0.75 },
    };

    const result = applyBreakFallbackCandidate(estimatedMatrix, 0.03, candidate, 'equity');
    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(result[i][j]).toBeCloseTo(estimatedMatrix[i][j], 10);
      }
    }
  });

  test('walkForward surfaces fallback candidate provenance on experiment runs', async () => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;
    const prices = fixture.tickers.SPY.closes;

    const result = await walkForward({
      ticker: 'SPY',
      prices,
      horizon: 14,
      warmup: 120,
      stride: 5,
      trendPenaltyOnlyBreakConfidence: true,
      breakFallbackCandidate: {
        id: 'C60',
        mode: 'hard',
        conservativeDiagonal: 0.60,
        profileDiagonals: { equity: 0.60, etf: 0.60, commodity: 0.60, crypto: 0.60 },
        conservativeWeight: 1.0,
        severityWeights: { mild: 1.0, medium: 1.0, high: 1.0 },
      },
    });

    expect(result.errors).toHaveLength(0);
    expect(result.steps.length).toBeGreaterThan(0);
    expect(result.steps.every(step => step.breakFallbackCandidateId === 'C60')).toBe(true);
    expect(result.steps.some(step => step.breakFallbackMode === 'hard')).toBe(true);
  });

  test('Phase 5 threshold evaluation enforces the approved sideways and horizon guardrails', () => {
    const baselinePerHorizon = [
      {
        horizon: 7,
        steps: 10,
        breakSteps: 8,
        overall: { directionalAccuracy: 0.60, brierScore: 0.24, ciCoverage: 0.94, avgConfidence: 0.25, directionalCi: { lower: 0.5, median: 0.6, upper: 0.7, nResamples: 10 } },
        breakContext: { directionalAccuracy: 0.62, brierScore: 0.23, ciCoverage: 0.94, avgConfidence: 0.24, directionalCi: { lower: 0.52, median: 0.62, upper: 0.72, nResamples: 10 } },
        nonBreak: { directionalAccuracy: 0.55, brierScore: 0.25, ciCoverage: 0.92, avgConfidence: 0.35, directionalCi: { lower: 0.45, median: 0.55, upper: 0.65, nResamples: 10 } },
        breakTrending: { directionalAccuracy: 0.58, brierScore: 0.24, ciCoverage: 0.92, avgConfidence: 0.21, directionalCi: { lower: 0.48, median: 0.58, upper: 0.68, nResamples: 10 } },
        breakChop: { directionalAccuracy: 0.70, brierScore: 0.20, ciCoverage: 0.99, avgConfidence: 0.33, directionalCi: { lower: 0.60, median: 0.70, upper: 0.80, nResamples: 10 } },
        rc020: { threshold: 0.2, accuracy: 0.62, coverage: 0.60, n: 5 },
        rc030: { threshold: 0.3, accuracy: 0.63, coverage: 0.25, n: 2 },
      },
      {
        horizon: 14,
        steps: 10,
        breakSteps: 8,
        overall: { directionalAccuracy: 0.65, brierScore: 0.23, ciCoverage: 0.94, avgConfidence: 0.25, directionalCi: { lower: 0.55, median: 0.65, upper: 0.75, nResamples: 10 } },
        breakContext: { directionalAccuracy: 0.66, brierScore: 0.22, ciCoverage: 0.94, avgConfidence: 0.24, directionalCi: { lower: 0.56, median: 0.66, upper: 0.76, nResamples: 10 } },
        nonBreak: { directionalAccuracy: 0.60, brierScore: 0.24, ciCoverage: 0.92, avgConfidence: 0.35, directionalCi: { lower: 0.5, median: 0.6, upper: 0.7, nResamples: 10 } },
        breakTrending: { directionalAccuracy: 0.61, brierScore: 0.23, ciCoverage: 0.92, avgConfidence: 0.21, directionalCi: { lower: 0.51, median: 0.61, upper: 0.71, nResamples: 10 } },
        breakChop: { directionalAccuracy: 0.74, brierScore: 0.19, ciCoverage: 1.0, avgConfidence: 0.33, directionalCi: { lower: 0.64, median: 0.74, upper: 0.84, nResamples: 10 } },
        rc020: { threshold: 0.2, accuracy: 0.66, coverage: 0.62, n: 5 },
        rc030: { threshold: 0.3, accuracy: 0.67, coverage: 0.24, n: 2 },
      },
      {
        horizon: 30,
        steps: 10,
        breakSteps: 8,
        overall: { directionalAccuracy: 0.67, brierScore: 0.22, ciCoverage: 0.94, avgConfidence: 0.28, directionalCi: { lower: 0.57, median: 0.67, upper: 0.77, nResamples: 10 } },
        breakContext: { directionalAccuracy: 0.68, brierScore: 0.22, ciCoverage: 0.94, avgConfidence: 0.25, directionalCi: { lower: 0.58, median: 0.68, upper: 0.78, nResamples: 10 } },
        nonBreak: { directionalAccuracy: 0.61, brierScore: 0.24, ciCoverage: 0.92, avgConfidence: 0.35, directionalCi: { lower: 0.51, median: 0.61, upper: 0.71, nResamples: 10 } },
        breakTrending: { directionalAccuracy: 0.63, brierScore: 0.23, ciCoverage: 0.92, avgConfidence: 0.21, directionalCi: { lower: 0.53, median: 0.63, upper: 0.73, nResamples: 10 } },
        breakChop: { directionalAccuracy: 0.75, brierScore: 0.19, ciCoverage: 0.99, avgConfidence: 0.33, directionalCi: { lower: 0.65, median: 0.75, upper: 0.85, nResamples: 10 } },
        rc020: { threshold: 0.2, accuracy: 0.68, coverage: 0.64, n: 5 },
        rc030: { threshold: 0.3, accuracy: 0.69, coverage: 0.26, n: 2 },
      },
    ];

    const guardrailFailure = evaluatePhase5Thresholds({
      candidate: {
        overall: { directionalAccuracy: 0.66, brierScore: 0.225, ciCoverage: 0.94, avgConfidence: 0.28, directionalCi: { lower: 0.56, median: 0.66, upper: 0.76, nResamples: 10 } },
        deltaVsBaseline: {
          breakTrendingDirectionalAccuracy: 0.021,
          nonBreakDirectionalAccuracy: 0,
          overallBrier: 0.001,
          overallCiCoverage: 0,
          breakChopRC020Accuracy: -0.015,
          breakChopRC020Coverage: -0.01,
          breakContextRC020Accuracy: 0.01,
          breakContextRC020Coverage: 0.02,
        },
        perHorizon: [
          { ...baselinePerHorizon[0], breakContext: { ...baselinePerHorizon[0].breakContext, directionalAccuracy: 0.60 } },
          { ...baselinePerHorizon[1], breakContext: { ...baselinePerHorizon[1].breakContext, directionalAccuracy: 0.67 } },
          { ...baselinePerHorizon[2], breakContext: { ...baselinePerHorizon[2].breakContext, directionalAccuracy: 0.69 } },
        ],
      },
      baselinePerHorizon,
    });

    expect(guardrailFailure.passes).toBe(false);
    expect(guardrailFailure.failureReasons.some(reason => reason.includes('break+sideways RC@0.2 accuracy'))).toBe(true);
    expect(guardrailFailure.failureReasons.some(reason => reason.includes('horizon guardrail failed'))).toBe(true);

    const passing = evaluatePhase5Thresholds({
      candidate: {
        overall: { directionalAccuracy: 0.68, brierScore: 0.223, ciCoverage: 0.94, avgConfidence: 0.29, directionalCi: { lower: 0.58, median: 0.68, upper: 0.78, nResamples: 10 } },
        deltaVsBaseline: {
          breakTrendingDirectionalAccuracy: 0.022,
          nonBreakDirectionalAccuracy: 0,
          overallBrier: 0.001,
          overallCiCoverage: 0,
          breakChopRC020Accuracy: -0.005,
          breakChopRC020Coverage: -0.01,
          breakContextRC020Accuracy: 0.01,
          breakContextRC020Coverage: 0.02,
        },
        perHorizon: [
          { ...baselinePerHorizon[0], breakContext: { ...baselinePerHorizon[0].breakContext, directionalAccuracy: 0.65 } },
          { ...baselinePerHorizon[1], breakContext: { ...baselinePerHorizon[1].breakContext, directionalAccuracy: 0.68 } },
          { ...baselinePerHorizon[2], breakContext: { ...baselinePerHorizon[2].breakContext, directionalAccuracy: 0.70 } },
        ],
      },
      baselinePerHorizon,
    });

    expect(passing.passes).toBe(true);
    expect(passing.failureReasons).toHaveLength(0);
  });

  integrationIt('evaluates the full Phase 5 candidate family against the Phase 4 baseline', async () => {
    const artifact = await runComparison();

    expect(artifact.baseline.candidateId).toBe('phase4-control');
    expect(artifact.candidates).toHaveLength(9);
    expect(artifact.winner.candidateId).toBeNull();
    expect(artifact.winner.reason).toContain('No candidate passed');

    const bestHybrid = artifact.candidates.find(candidate => candidate.candidateId === 'HYB_L025_M050_H075_lambda025');
    expect(bestHybrid).toBeDefined();
    expect(bestHybrid!.deltaVsBaseline.breakTrendingDirectionalAccuracy).toBeGreaterThan(0.004);
    expect(bestHybrid!.deltaVsBaseline.breakTrendingDirectionalAccuracy).toBeLessThan(0.006);
    expect(bestHybrid!.deltaVsBaseline.overallBrier).toBeGreaterThan(0.006);
    expect(bestHybrid!.deltaVsBaseline.overallBrier).toBeLessThan(0.0062);
    expect(bestHybrid!.passesThresholds).toBe(false);
    expect(bestHybrid!.failureReasons.some(reason => reason.includes('break+trending gain'))).toBe(true);
    expect(bestHybrid!.failureReasons.some(reason => reason.includes('overall Brier'))).toBe(true);

    const control = artifact.candidates.find(candidate => candidate.candidateId === 'C60');
    expect(control).toBeDefined();
    expect(control!.deltaVsBaseline.breakTrendingDirectionalAccuracy).toBeCloseTo(0, 10);
    expect(control!.passesThresholds).toBe(false);
  }, 900_000);
});
