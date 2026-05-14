import { describe, expect, it, beforeAll } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { integrationIt } from '@/utils/test-guards.js';
import { walkForward, type WalkForwardResult } from './walk-forward.js';
import { brierScore, ciCoverage, directionalAccuracy, type BacktestStep } from './metrics.js';

const TICKER = 'SOLUSD';
const HORIZONS = [1, 2, 3, 7, 14] as const;
const PRIMARY_HORIZONS = [1, 2, 3] as const;
const GUARDRAIL_HORIZONS = [7, 14] as const;
const PRIMARY_DIRECTIONAL_ACCURACY_WEIGHTS = {
  h1: 0.5,
  h2: 0.3,
  h3: 0.2,
} as const;
const STRIDE = 5;
const WARMUP = 120;
const HISTORY_DAYS = 365;
const TIMEOUT = 480_000;

interface FixtureData {
  tickers: Record<string, {
    closes: number[];
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

function scorePrimaryShortHorizons(metrics: Record<string, { directionalAccuracy: number }>): number {
  return (
    metrics.h1.directionalAccuracy * PRIMARY_DIRECTIONAL_ACCURACY_WEIGHTS.h1
    + metrics.h2.directionalAccuracy * PRIMARY_DIRECTIONAL_ACCURACY_WEIGHTS.h2
    + metrics.h3.directionalAccuracy * PRIMARY_DIRECTIONAL_ACCURACY_WEIGHTS.h3
  );
}

function loadPrices(): number[] {
  return fixture.tickers['SOL-USD']?.closes.slice(-HISTORY_DAYS) ?? [];
}

async function runCurrentLaneForHorizon(horizon: number): Promise<HorizonMetrics> {
  const prices = loadPrices();
  const result: WalkForwardResult = await walkForward({
    ticker: TICKER,
    prices,
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

describe('Walk-forward SOL short-horizon benchmark', () => {
  beforeAll(() => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'crypto-peer-prices.json');
    fixture = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;
  });

  it('keeps the SOL mutator contract centered on 1d/2d/3d while treating 7d/14d as guardrails', () => {
    const baseline = {
      h1: { directionalAccuracy: 0.51 },
      h2: { directionalAccuracy: 0.37 },
      h3: { directionalAccuracy: 0.35 },
      h7: { directionalAccuracy: 0.35 },
      h14: { directionalAccuracy: 0.28 },
    };
    const improvedGuardrailsOnly = {
      ...baseline,
      h7: { directionalAccuracy: 0.99 },
      h14: { directionalAccuracy: 0.99 },
    };
    const regressedPrimaryHorizons = {
      ...improvedGuardrailsOnly,
      h1: { directionalAccuracy: 0.48 },
      h2: { directionalAccuracy: 0.34 },
      h3: { directionalAccuracy: 0.31 },
    };

    expect(PRIMARY_HORIZONS).toEqual([1, 2, 3]);
    expect(GUARDRAIL_HORIZONS).toEqual([7, 14]);
    expect(Object.values(PRIMARY_DIRECTIONAL_ACCURACY_WEIGHTS).reduce((total, weight) => total + weight, 0)).toBeCloseTo(1);
    expect(scorePrimaryShortHorizons(improvedGuardrailsOnly)).toBeCloseTo(scorePrimaryShortHorizons(baseline));
    expect(scorePrimaryShortHorizons(regressedPrimaryHorizons)).toBeLessThan(scorePrimaryShortHorizons(baseline));
  });

  integrationIt('emits machine-readable SOL short-horizon metrics across 1d/2d/3d/7d/14d', async () => {
    const metricsByHorizon: Record<string, Omit<HorizonMetrics, 'steps' | 'errors'>> = {};
    const lines: string[] = ['', '═══ SOL SHORT-HORIZON BENCHMARK ═══'];

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

    lines.push('══════════════════════════════════════');
    console.log(lines.join('\n'));
    if (process.env.FORECAST_LAB_OUTPUT_METRICS === '1') {
      console.log(`FORECAST_LAB_METRICS ${JSON.stringify(metricsByHorizon)}`);
    }
  }, TIMEOUT);
});
