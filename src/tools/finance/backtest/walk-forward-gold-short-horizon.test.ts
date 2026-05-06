import { describe, beforeAll, expect } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { integrationIt } from '@/utils/test-guards.js';
import { walkForward, type WalkForwardResult } from './walk-forward.js';
import { brierScore, ciCoverage, directionalAccuracy, type BacktestStep } from './metrics.js';

const TICKER = 'GLD';
const HORIZONS = [1, 2, 3, 7, 14] as const;
const STRIDE = 5;
const WARMUP = 120;
const TIMEOUT = 480_000;

interface FixtureData {
  tickers: Record<string, {
    type: string;
    closes: number[];
    dates: string[];
    count: number;
    synthetic?: boolean;
  }>;
}

interface HorizonMetrics {
  steps: BacktestStep[];
  errors: number;
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  structuralBreakCount: number;
  abstainCount: number;
}

let fixture: FixtureData;

function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function countStructuralBreaks(steps: readonly BacktestStep[]): number {
  return steps.filter((step) => (step.originalStructuralBreakDetected ?? step.structuralBreakDetected) === true).length;
}

function countAbstains(steps: readonly BacktestStep[]): number {
  return steps.filter((step) => step.recommendation === 'HOLD').length;
}

async function runCurrentLaneForHorizon(horizon: number): Promise<HorizonMetrics> {
  const tickerData = fixture.tickers[TICKER];
  const result: WalkForwardResult = await walkForward({
    ticker: TICKER,
    prices: tickerData.closes,
    horizon,
    warmup: WARMUP,
    stride: STRIDE,
  });

  return {
    steps: result.steps,
    errors: result.errors.length,
    directionalAccuracy: result.steps.length > 0 ? directionalAccuracy(result.steps) : 0,
    brierScore: result.steps.length > 0 ? brierScore(result.steps) : 0,
    ciCoverage: result.steps.length > 0 ? ciCoverage(result.steps) : 0,
    structuralBreakCount: countStructuralBreaks(result.steps),
    abstainCount: countAbstains(result.steps),
  };
}

describe('Walk-forward GOLD short-horizon benchmark', () => {
  beforeAll(() => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));
  });

  integrationIt('emits machine-readable GLD short-horizon metrics across 1d/2d/3d/7d/14d', async () => {
    const metricsByHorizon: Record<string, Omit<HorizonMetrics, 'steps' | 'errors'>> = {};
    const lines: string[] = ['', '═══ GOLD SHORT-HORIZON BENCHMARK ═══'];

    for (const horizon of HORIZONS) {
      const metrics = await runCurrentLaneForHorizon(horizon);
      metricsByHorizon[`h${horizon}`] = {
        directionalAccuracy: metrics.directionalAccuracy,
        brierScore: metrics.brierScore,
        ciCoverage: metrics.ciCoverage,
        structuralBreakCount: metrics.structuralBreakCount,
        abstainCount: metrics.abstainCount,
      };

      lines.push(
        `${horizon}d │ steps=${String(metrics.steps.length).padStart(3)} │ errors=${String(metrics.errors).padStart(2)} │ dir=${formatPct(metrics.directionalAccuracy).padStart(6)} │ brier=${metrics.brierScore.toFixed(4)} │ ci=${formatPct(metrics.ciCoverage).padStart(6)} │ breaks=${String(metrics.structuralBreakCount).padStart(3)} │ abstain=${String(metrics.abstainCount).padStart(3)}`,
      );

      expect(metrics.errors).toBe(0);
      expect(metrics.steps.length).toBeGreaterThan(0);
    }

    lines.push('═══════════════════════════════════════');
    console.log(lines.join('\n'));
    if (process.env.FORECAST_LAB_OUTPUT_METRICS === '1') {
      console.log(`FORECAST_LAB_METRICS ${JSON.stringify(metricsByHorizon)}`);
    }
  }, TIMEOUT);
});
