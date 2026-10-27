import { describe, expect, it } from 'bun:test';
import type { BacktestStep } from './metrics.js';
import {
  computeMetricDelta,
  renderArmMetricTable,
  renderDeltaMetricTable,
  summarizeBacktestSteps,
  summarizeCalibrationSteps,
} from './comparison-harness.js';

function makeStep(overrides: Partial<BacktestStep> = {}): BacktestStep {
  return {
    t: 0,
    predictedProb: 0.55,
    actualBinary: 1,
    predictedReturn: 0.02,
    actualReturn: 0.03,
    ciLower: 80,
    ciUpper: 120,
    central90CiLower: 98,
    central90CiUpper: 102,
    realizedPrice: 100,
    recommendation: 'BUY',
    gofPasses: null,
    confidence: 0.30,
    probabilitySource: 'calibrated',
    decisionSource: 'default',
    ...overrides,
  };
}

describe('comparison-harness central-90 coverage', () => {
  // The wide helper interval covers all four steps; the literal central-90 covers two.
  const steps = [
    makeStep({ realizedPrice: 100 }),
    makeStep({ realizedPrice: 101 }),
    makeStep({ realizedPrice: 105 }),
    makeStep({ realizedPrice: 95 }),
  ];

  it('scores the literal central-90 interval separately from the helper interval', () => {
    const m = summarizeBacktestSteps(steps);
    expect(m.ciCoverage).toBe(1);
    expect(m.central90Coverage).toBe(0.5);
  });

  it('shows central-90 coverage in the arm and delta tables', () => {
    const base = { label: 'base', perHorizon: [{ horizon: 7, metrics: summarizeBacktestSteps(steps) }] };
    const ext = { label: 'ext', perHorizon: [{ horizon: 7, metrics: summarizeBacktestSteps(steps.slice(0, 2)) }] };

    const arm = renderArmMetricTable(base);
    expect(arm[0]).toContain('| Central-90 |');
    expect(arm[2]).toContain('| 0.500 |');

    const delta = computeMetricDelta(base, ext);
    expect(delta[0].central90CoverageDelta).toBe(0.5);
    const deltaTable = renderDeltaMetricTable(delta);
    expect(deltaTable[0]).toContain('| ΔCentral-90 |');
    expect(deltaTable[2]).toContain('| +0.500 |');
  });
});

describe('summarizeCalibrationSteps', () => {
  // Realized 108 is inside the wide helper interval (80..120) but outside central-90 (98..102).
  const steps = [
    makeStep({ realizedPrice: 100, structuralBreakDetected: true }),
    makeStep({ realizedPrice: 108, structuralBreakDetected: true }),
    makeStep({ realizedPrice: 101 }),
    makeStep({ realizedPrice: 108 }),
  ];

  it('keeps the helper-interval coverage numbers and adds central-90 numbers beside them (PI-16)', () => {
    const summary = summarizeCalibrationSteps(steps, {
      targetCiCoverage: 0.9,
      isBreakStep: step => step.structuralBreakDetected === true,
    });

    // Existing helper-interval numbers are unchanged: it covers every step.
    expect(summary.ciCoverage).toBe(1);
    expect(summary.coverageError).toBeCloseTo(0.1, 10);
    expect(summary.breakCiCoverage).toBe(1);
    expect(summary.breakCoverageError).toBeCloseTo(0.1, 10);

    // Central-90 covers 100 and 101 only; one of the two break steps.
    expect(summary.central90Coverage).toBe(0.5);
    expect(summary.central90CoverageError).toBeCloseTo(0.4, 10);
    expect(summary.breakCentral90Coverage).toBe(0.5);
    expect(summary.breakCentral90CoverageError).toBeCloseTo(0.4, 10);
  });

  it('reports no break central-90 numbers when no break step is selected', () => {
    const summary = summarizeCalibrationSteps(steps, { targetCiCoverage: 0.9 });

    expect(summary.breakCount).toBe(0);
    expect(summary.breakCentral90Coverage).toBeNull();
    expect(summary.breakCentral90CoverageError).toBeNull();
  });
});
