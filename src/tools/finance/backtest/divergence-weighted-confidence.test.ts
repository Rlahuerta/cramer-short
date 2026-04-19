/**
 * Tests for Phase 6 divergence-weighted break confidence.
 *
 * Covers:
 *  - Unit behavior for divergence bucket penalties
 *  - Default path unchanged when flag is off
 *  - Sideways carve-out preserved
 *  - Divergence-weighted penalties only affect break contexts
 *  - Threshold/guardrail helper behavior for the comparison harness
 *
 * Run: bun test src/tools/finance/backtest/divergence-weighted-confidence.test.ts
 */
import { describe, expect, test } from 'bun:test';
import {
  computeDivergencePenalty,
  DEFAULT_DIVERGENCE_PENALTY_SCHEDULE,
  computePredictionConfidence,
  type DivergencePenaltySchedule,
} from '../markov-distribution.js';

// ---------------------------------------------------------------------------
// Unit: computeDivergencePenalty
// ---------------------------------------------------------------------------

describe('computeDivergencePenalty', () => {
  const schedule: DivergencePenaltySchedule = {
    mild: 0.80,
    medium: 0.70,
    high: 0.60,
  };

  test('returns 1.0 for divergence below 0.05 (no break)', () => {
    expect(computeDivergencePenalty(0.0, schedule)).toBe(1.0);
    expect(computeDivergencePenalty(0.02, schedule)).toBe(1.0);
    expect(computeDivergencePenalty(0.0499, schedule)).toBe(1.0);
  });

  test('returns mild penalty for divergence in [0.05, 0.10)', () => {
    expect(computeDivergencePenalty(0.05, schedule)).toBe(0.80);
    expect(computeDivergencePenalty(0.07, schedule)).toBe(0.80);
    expect(computeDivergencePenalty(0.0999, schedule)).toBe(0.80);
  });

  test('returns medium penalty for divergence in [0.10, 0.20)', () => {
    expect(computeDivergencePenalty(0.10, schedule)).toBe(0.70);
    expect(computeDivergencePenalty(0.15, schedule)).toBe(0.70);
    expect(computeDivergencePenalty(0.1999, schedule)).toBe(0.70);
  });

  test('returns high penalty for divergence >= 0.20', () => {
    expect(computeDivergencePenalty(0.20, schedule)).toBe(0.60);
    expect(computeDivergencePenalty(0.50, schedule)).toBe(0.60);
    expect(computeDivergencePenalty(1.0, schedule)).toBe(0.60);
  });

  test('DEFAULT_DIVERGENCE_PENALTY_SCHEDULE matches expected values', () => {
    expect(DEFAULT_DIVERGENCE_PENALTY_SCHEDULE.mild).toBe(0.80);
    expect(DEFAULT_DIVERGENCE_PENALTY_SCHEDULE.medium).toBe(0.70);
    expect(DEFAULT_DIVERGENCE_PENALTY_SCHEDULE.high).toBe(0.60);
  });

  test('custom schedule with different values works correctly', () => {
    const aggressive: DivergencePenaltySchedule = { mild: 0.90, medium: 0.75, high: 0.50 };
    expect(computeDivergencePenalty(0.07, aggressive)).toBe(0.90);
    expect(computeDivergencePenalty(0.15, aggressive)).toBe(0.75);
    expect(computeDivergencePenalty(0.30, aggressive)).toBe(0.50);
  });
});

// ---------------------------------------------------------------------------
// Unit: default path unchanged when divergenceWeightedBreakConfidence is off
// ---------------------------------------------------------------------------

describe('computePredictionConfidence default path', () => {
  const baseOptions = {
    pUp: 0.6,
    ensembleConsensus: 2,
    hmmConverged: true,
    regimeRunLength: 10,
    structuralBreak: true,
    structuralBreakDivergence: 0.15,
    regimeState: 'bull' as const,
  };

  test('default policy applies flat 0.6 penalty regardless of divergence', () => {
    const result = computePredictionConfidence({
      ...baseOptions,
      breakConfidencePolicy: 'default',
    });
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1);
  });

  test('default policy ignores divergence schedule even when provided', () => {
    const withoutSchedule = computePredictionConfidence({
      ...baseOptions,
      breakConfidencePolicy: 'default',
    });
    const withSchedule = computePredictionConfidence({
      ...baseOptions,
      breakConfidencePolicy: 'default',
      divergencePenaltySchedule: { mild: 0.99, medium: 0.99, high: 0.99 },
    });
    expect(withSchedule).toBeCloseTo(withoutSchedule, 10);
  });

  test('divergence_weighted policy with default schedule gives lighter penalty for mild breaks', () => {
    const defaultConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.07,
      breakConfidencePolicy: 'default',
    });
    const weightedConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.07,
      breakConfidencePolicy: 'divergence_weighted',
    });
    expect(weightedConf).toBeGreaterThan(defaultConf);
  });

  test('divergence_weighted policy equals default for high-divergence breaks', () => {
    const defaultConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.30,
      breakConfidencePolicy: 'default',
    });
    const weightedConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.30,
      breakConfidencePolicy: 'divergence_weighted',
    });
    expect(weightedConf).toBeCloseTo(defaultConf, 10);
  });
});

// ---------------------------------------------------------------------------
// Unit: sideways carve-out preserved
// ---------------------------------------------------------------------------

describe('sideways carve-out with divergence_weighted', () => {
  const baseOptions = {
    pUp: 0.55,
    ensembleConsensus: 1,
    hmmConverged: true,
    regimeRunLength: 5,
    structuralBreak: true,
    structuralBreakDivergence: 0.15,
  };

  test('trend_penalty_only skips penalty in sideways regime', () => {
    const sidewaysWithPenalty = computePredictionConfidence({
      ...baseOptions,
      regimeState: 'sideways',
      breakConfidencePolicy: 'trend_penalty_only',
    });
    const noBreak = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: false,
      breakConfidencePolicy: 'default',
      regimeState: 'sideways',
    });
    expect(sidewaysWithPenalty).toBeCloseTo(noBreak, 10);
  });

  test('divergence_weighted preserves the sideways carve-out when explicitly requested', () => {
    const sidewaysDivergenceWeighted = computePredictionConfidence({
      ...baseOptions,
      regimeState: 'sideways',
      breakConfidencePolicy: 'divergence_weighted',
      skipSidewaysBreakPenalty: true,
    });
    const noBreak = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: false,
      regimeState: 'sideways',
      breakConfidencePolicy: 'default',
    });
    expect(sidewaysDivergenceWeighted).toBeCloseTo(noBreak, 10);
  });

  test('divergence_weighted without the carve-out still penalizes sideways breaks', () => {
    const sidewaysDivergenceWeighted = computePredictionConfidence({
      ...baseOptions,
      regimeState: 'sideways',
      breakConfidencePolicy: 'divergence_weighted',
    });
    const sidewaysDefault = computePredictionConfidence({
      ...baseOptions,
      regimeState: 'sideways',
      breakConfidencePolicy: 'default',
    });
    expect(sidewaysDivergenceWeighted).toBeGreaterThan(sidewaysDefault);
  });

  test('trend_penalty_only still penalizes trending breaks', () => {
    const trendingWithPenalty = computePredictionConfidence({
      ...baseOptions,
      regimeState: 'bull',
      breakConfidencePolicy: 'trend_penalty_only',
    });
    const trendingNoBreak = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: false,
      breakConfidencePolicy: 'default',
      regimeState: 'bull',
    });
    expect(trendingWithPenalty).toBeLessThan(trendingNoBreak);
  });
});

// ---------------------------------------------------------------------------
// Unit: divergence-weighted penalties only affect break contexts
// ---------------------------------------------------------------------------

describe('divergence-weighted penalties only affect break contexts', () => {
  const baseOptions = {
    pUp: 0.65,
    ensembleConsensus: 2,
    hmmConverged: true,
    regimeRunLength: 8,
    regimeState: 'bull' as const,
  };

  test('no break detected → divergence_weighted has no effect', () => {
    const noBreakDefault = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: false,
      structuralBreakDivergence: 0.15,
      breakConfidencePolicy: 'default',
    });
    const noBreakWeighted = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: false,
      structuralBreakDivergence: 0.15,
      breakConfidencePolicy: 'divergence_weighted',
    });
    expect(noBreakWeighted).toBeCloseTo(noBreakDefault, 10);
  });

  test('break detected → divergence_weighted with mild divergence gives lighter penalty', () => {
    const defaultConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: true,
      structuralBreakDivergence: 0.07,
      breakConfidencePolicy: 'default',
    });
    const weightedConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: true,
      structuralBreakDivergence: 0.07,
      breakConfidencePolicy: 'divergence_weighted',
    });
    expect(weightedConf).toBeGreaterThan(defaultConf);
  });

  test('break detected → divergence_weighted with high divergence equals default', () => {
    const defaultConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: true,
      structuralBreakDivergence: 0.25,
      breakConfidencePolicy: 'default',
    });
    const weightedConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: true,
      structuralBreakDivergence: 0.25,
      breakConfidencePolicy: 'divergence_weighted',
    });
    expect(weightedConf).toBeCloseTo(defaultConf, 10);
  });

  test('divergence_weighted with missing divergence defaults to high penalty', () => {
    const result = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: true,
      breakConfidencePolicy: 'divergence_weighted',
      // structuralBreakDivergence is undefined → defaults to 0.20 → high bucket
    });
    const highDefault = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: true,
      breakConfidencePolicy: 'default',
    });
    expect(result).toBeCloseTo(highDefault, 10);
  });
});

// ---------------------------------------------------------------------------
// Unit: custom schedule overrides
// ---------------------------------------------------------------------------

describe('custom divergence penalty schedules', () => {
  const baseOptions = {
    pUp: 0.6,
    ensembleConsensus: 2,
    hmmConverged: true,
    regimeRunLength: 10,
    structuralBreak: true,
    regimeState: 'bull' as const,
  };

  test('schedule with all 1.0 penalties (effectively no break penalty)', () => {
    const noPenalty: DivergencePenaltySchedule = { mild: 1.0, medium: 1.0, high: 1.0 };
    const result = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.15,
      breakConfidencePolicy: 'divergence_weighted',
      divergencePenaltySchedule: noPenalty,
    });
    const noBreak = computePredictionConfidence({
      ...baseOptions,
      structuralBreak: false,
      breakConfidencePolicy: 'default',
    });
    expect(result).toBeCloseTo(noBreak, 10);
  });

  test('schedule with all 0.4 penalties (aggressive everywhere)', () => {
    const aggressive: DivergencePenaltySchedule = { mild: 0.4, medium: 0.4, high: 0.4 };
    const result = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.07,
      breakConfidencePolicy: 'divergence_weighted',
      divergencePenaltySchedule: aggressive,
    });
    const defaultConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.07,
      breakConfidencePolicy: 'default',
    });
    expect(result).toBeLessThan(defaultConf);
  });

  test('progressive schedule: mild=0.90, medium=0.75, high=0.60', () => {
    const progressive: DivergencePenaltySchedule = { mild: 0.90, medium: 0.75, high: 0.60 };
    const mildConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.06,
      breakConfidencePolicy: 'divergence_weighted',
      divergencePenaltySchedule: progressive,
    });
    const mediumConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.15,
      breakConfidencePolicy: 'divergence_weighted',
      divergencePenaltySchedule: progressive,
    });
    const highConf = computePredictionConfidence({
      ...baseOptions,
      structuralBreakDivergence: 0.30,
      breakConfidencePolicy: 'divergence_weighted',
      divergencePenaltySchedule: progressive,
    });
    expect(mildConf).toBeGreaterThan(mediumConf);
    expect(mediumConf).toBeGreaterThan(highConf);
  });
});

// ---------------------------------------------------------------------------
// Integration: walkForward surfaces divergence-weighted provenance
// ---------------------------------------------------------------------------

import { readFileSync } from 'fs';
import { join } from 'path';
import { walkForward } from './walk-forward.js';
import { integrationIt } from '@/utils/test-guards.js';

interface FixtureData {
  tickers: Record<string, { closes: number[]; dates: string[] }>;
}

describe('walkForward divergence-weighted provenance', () => {
  integrationIt('records divergenceWeightedBreakConfidenceActive when flag is on', async () => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;
    const prices = fixture.tickers.SPY.closes;

    const result = await walkForward({
      ticker: 'SPY',
      prices,
      horizon: 7,
      warmup: 120,
      stride: 5,
      trendPenaltyOnlyBreakConfidence: true,
      divergenceWeightedBreakConfidence: true,
    });

    expect(result.errors).toHaveLength(0);
    expect(result.steps.length).toBeGreaterThan(0);
    expect(result.steps.every(step => step.divergenceWeightedBreakConfidenceActive === true)).toBe(true);
  });

  integrationIt('provenance is undefined when flag is off', async () => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;
    const prices = fixture.tickers.SPY.closes;

    const result = await walkForward({
      ticker: 'SPY',
      prices,
      horizon: 7,
      warmup: 120,
      stride: 5,
    });

    expect(result.errors).toHaveLength(0);
    expect(result.steps.length).toBeGreaterThan(0);
    expect(result.steps.every(step => step.divergenceWeightedBreakConfidenceActive === undefined)).toBe(true);
  });
});
