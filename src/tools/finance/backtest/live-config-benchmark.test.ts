import { describe, expect, it } from 'bun:test';
import { getBtcShortHorizonLivePolicy } from '../markov-distribution/live-policies.js';
import {
  compareBenchmarkRuns,
  liveBenchmarkCells,
  runLiveConfigBenchmark,
  type BenchmarkCell,
  type BenchmarkRun,
  type BenchmarkStep,
} from './live-config-benchmark.js';
import { xiphosBtcWalkForwardConfig } from './xiphos-btc-preset.js';

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

describe('liveBenchmarkCells', () => {
  it('covers six tickers at 1, 7 and 14 days with the history the live tool fetches', () => {
    const cells = liveBenchmarkCells();
    expect(cells).toHaveLength(18);
    const historyOf = (ticker: string) => [...new Set(cells.filter((c) => c.ticker === ticker).map((c) => c.maxHistoryPrices))];
    expect(historyOf('SPY')).toEqual([83]);
    expect(historyOf('GLD')).toEqual([174]);
    expect(historyOf('BTC-USD')).toEqual([252]);
  });

  it("configures BTC like the live tool: the xiphos profile, then each horizon's live break policy", () => {
    for (const cell of liveBenchmarkCells().filter((c) => c.ticker === 'BTC-USD')) {
      const policy = getBtcShortHorizonLivePolicy('BTC-USD', cell.horizon)!;
      expect(cell.config.btcBreakDivergenceThreshold).toBe(policy.breakDivergenceThreshold);
      expect(cell.config.postBreakShortWindow).toBe(policy.rerunOnBreak);
      expect(cell.config.postBreakWindowSize).toBe(policy.rerunWindowDays);
      expect(cell.config.btcReturnThresholdMultiplier).toBe(xiphosBtcWalkForwardConfig().btcReturnThresholdMultiplier);
      expect(cell.config.enableGarchVol).toBe(false);
    }
  });
});

describe('runLiveConfigBenchmark', () => {
  it('turns a walk-forward run into cells the decision rule can pair', async () => {
    const prices = Array.from({ length: 140 }, (_, i) => 100 * Math.exp(0.001 * i + 0.02 * Math.sin(i / 3)));
    const run = await runLiveConfigBenchmark('smoke', { SPY: prices }, [
      { ticker: 'SPY', horizon: 1, maxHistoryPrices: 60, config: {} },
    ]);
    expect(run.cells).toHaveLength(1);
    expect(run.cells[0].steps.length).toBeGreaterThan(0);
    expect(run.cells[0].steps[0].t).toBe(60);
    expect(compareBenchmarkRuns(run, run).verdict).toBe('no-harm');
  });
});
