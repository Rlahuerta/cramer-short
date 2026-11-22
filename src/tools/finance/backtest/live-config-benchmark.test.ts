import { describe, expect, it } from 'bun:test';
import {
  compareBenchmarkRuns,
  type BenchmarkCell,
  type BenchmarkRun,
  type BenchmarkStep,
} from './live-config-benchmark.js';

function makeCell(ticker: string, horizon: number, steps: BenchmarkStep[]): BenchmarkCell {
  return { ticker, horizon, stride: 5, steps };
}

function makeRun(label: string, cells: BenchmarkCell[]): BenchmarkRun {
  return { label, generatedAt: '2026-11-22T00:00:00.000Z', cells };
}

// 40 steps per cell, outcomes alternating, a forecast slightly on the right side of 0.5.
function baselineSteps(n = 40, offset = 0): BenchmarkStep[] {
  return Array.from({ length: n }, (_, i) => {
    const actualBinary = (i + offset) % 2 === 0 ? 1 : 0;
    return { t: 100 + 5 * i, predictedProb: actualBinary === 1 ? 0.55 : 0.45, actualBinary, recommendation: 'BUY' };
  });
}

function shiftTowardOutcome(steps: BenchmarkStep[], by: number): BenchmarkStep[] {
  return steps.map((s) => ({
    ...s,
    predictedProb: s.actualBinary === 1 ? s.predictedProb + by : s.predictedProb - by,
  }));
}

const base = makeRun('base', [makeCell('SPY', 7, baselineSteps()), makeCell('BTC-USD', 14, baselineSteps(40, 1))]);

describe('compareBenchmarkRuns', () => {
  it('reports no change and no harm for identical runs', () => {
    const result = compareBenchmarkRuns(base, base);
    expect(result.pairedSteps).toBe(80);
    expect(result.meanBrierDelta).toBe(0);
    expect(result.ci95).toEqual([0, 0]);
    expect(result.abstainRateDelta).toBe(0);
    expect(result.verdict).toBe('no-harm');
  });

  it('calls a candidate that is closer to every outcome an improvement', () => {
    const better = makeRun('better', base.cells.map((c) => ({ ...c, steps: shiftTowardOutcome(c.steps, 0.1) })));
    const result = compareBenchmarkRuns(base, better);
    expect(result.meanBrierDelta).toBeLessThan(0);
    expect(result.ci95[1]).toBeLessThan(0);
    expect(result.verdict).toBe('improvement');
  });

  it('calls a candidate that is further from every outcome harm', () => {
    const worse = makeRun('worse', base.cells.map((c) => ({ ...c, steps: shiftTowardOutcome(c.steps, -0.1) })));
    const result = compareBenchmarkRuns(base, worse);
    expect(result.ci95[0]).toBeGreaterThan(0);
    expect(result.verdict).toBe('harm');
  });

  it('rejects buying accuracy by abstaining more than 2pp', () => {
    const abstaining = makeRun('abstaining', base.cells.map((c) => ({
      ...c,
      steps: shiftTowardOutcome(c.steps, 0.1).map((s, i) => (i % 10 === 0 ? { ...s, recommendation: 'HOLD' as const } : s)),
    })));
    const result = compareBenchmarkRuns(base, abstaining);
    expect(result.abstainRateDelta).toBeCloseTo(0.1, 10);
    expect(result.verdict).toBe('harm');
  });

  it('refuses to compare runs whose cells or steps do not line up', () => {
    const missingCell = makeRun('missing', [base.cells[0]]);
    expect(() => compareBenchmarkRuns(base, missingCell)).toThrow(/BTC-USD\|14/);
    const shifted = makeRun('shifted', base.cells.map((c) => ({ ...c, steps: c.steps.map((s) => ({ ...s, t: s.t + 1 })) })));
    expect(() => compareBenchmarkRuns(base, shifted)).toThrow(/t=/);
  });

  it('is deterministic for a seed', () => {
    const noisy = makeRun('noisy', base.cells.map((c) => ({
      ...c,
      steps: c.steps.map((s, i) => ({ ...s, predictedProb: s.predictedProb + (i % 3 === 0 ? 0.04 : -0.03) })),
    })));
    expect(compareBenchmarkRuns(base, noisy, { seed: 7 })).toEqual(compareBenchmarkRuns(base, noisy, { seed: 7 }));
  });
});
