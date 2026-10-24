import { describe, expect, it } from 'bun:test';
import type { BacktestStep } from './metrics.js';
import {
  computeMetricDelta,
  renderArmMetricTable,
  renderDeltaMetricTable,
  summarizeBacktestSteps,
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
