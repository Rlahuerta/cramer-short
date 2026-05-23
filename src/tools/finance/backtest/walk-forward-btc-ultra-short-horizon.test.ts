import { describe, beforeAll, expect } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { integrationIt } from '@/utils/test-guards.js';
import { MARKOV_PHASE0_BASELINES } from './markov-phase-baselines.js';
import { getBtcShortHorizonLivePolicy } from '../markov-distribution.js';
import { walkForward, type WalkForwardResult } from './walk-forward.js';
import { brierScore, ciCoverage, directionalAccuracy, type BacktestStep } from './metrics.js';

const TICKER = 'BTC-USD';
const HORIZONS = [1, 2, 3, 14] as const;
const STRIDE = 5;
const TIMEOUT = 360_000;

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
  abstainCount: number;
  rerunRate: number;
  breakDivergenceThreshold: number;
  historyDays: number;
  rerunOnBreak: boolean;
  rerunWindowDays: number | null;
}

let fixture: FixtureData;

function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function computeRerunRate(steps: readonly BacktestStep[]): number {
  if (steps.length === 0) return 0;
  return steps.filter((step) => step.structuralBreakRerunTriggered === true).length / steps.length;
}

function computeAbstainCount(steps: readonly BacktestStep[]): number {
  return steps.filter((step) => step.recommendation === 'HOLD').length;
}

async function runCurrentPolicyForHorizon(horizon: number): Promise<HorizonMetrics> {
  const tickerData = fixture.tickers[TICKER];
  const policy = getBtcShortHorizonLivePolicy(TICKER, horizon);

  expect(policy).toBeDefined();

  const result: WalkForwardResult = await walkForward({
    ticker: TICKER,
    prices: tickerData.closes,
    horizon,
    warmup: policy!.historyDays,
    stride: STRIDE,
    btcBreakDivergenceThreshold: policy!.breakDivergenceThreshold,
    postBreakShortWindow: policy!.rerunOnBreak,
    postBreakWindowSize: policy!.rerunWindowDays,
  });

  return {
    steps: result.steps,
    errors: result.errors.length,
    directionalAccuracy: result.steps.length > 0 ? directionalAccuracy(result.steps) : 0,
    brierScore: result.steps.length > 0 ? brierScore(result.steps) : 0,
    ciCoverage: result.steps.length > 0 ? ciCoverage(result.steps) : 0,
    abstainCount: computeAbstainCount(result.steps),
    rerunRate: computeRerunRate(result.steps),
    breakDivergenceThreshold: policy!.breakDivergenceThreshold,
    historyDays: policy!.historyDays,
    rerunOnBreak: policy!.rerunOnBreak,
    rerunWindowDays: policy!.rerunWindowDays ?? null,
  };
}

describe('Walk-forward BTC ultra-short-horizon live policy', () => {
  beforeAll(() => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));
  });

  integrationIt('emits machine-readable current BTC live-policy metrics across 1d/2d/3d/14d horizons', async () => {
    const metricsByHorizon: Record<string, Omit<HorizonMetrics, 'steps' | 'errors'>> = {};
    const lines: string[] = ['', '═══ BTC ULTRA-SHORT-HORIZON LIVE POLICY ═══'];

    for (const horizon of HORIZONS) {
      const metrics = await runCurrentPolicyForHorizon(horizon);
      metricsByHorizon[`h${horizon}`] = {
        directionalAccuracy: metrics.directionalAccuracy,
        brierScore: metrics.brierScore,
        ciCoverage: metrics.ciCoverage,
        abstainCount: metrics.abstainCount,
        rerunRate: metrics.rerunRate,
        breakDivergenceThreshold: metrics.breakDivergenceThreshold,
        historyDays: metrics.historyDays,
        rerunOnBreak: metrics.rerunOnBreak,
        rerunWindowDays: metrics.rerunWindowDays,
      };

      lines.push(
        `${horizon}d │ steps=${String(metrics.steps.length).padStart(3)} │ errors=${String(metrics.errors).padStart(2)} │ dir=${formatPct(metrics.directionalAccuracy).padStart(6)} │ brier=${metrics.brierScore.toFixed(4)} │ ci=${formatPct(metrics.ciCoverage).padStart(6)} │ abstain=${String(metrics.abstainCount).padStart(3)} │ rerun=${formatPct(metrics.rerunRate).padStart(6)} │ threshold=${metrics.breakDivergenceThreshold.toFixed(2)} │ history=${metrics.historyDays}d`,
      );

      expect(metrics.errors).toBe(0);
      expect(metrics.steps.length).toBeGreaterThan(0);
      if (horizon === 2) {
        expect(metrics.rerunOnBreak).toBe(true);
        expect(metrics.rerunWindowDays).toBe(120);
        expect(metrics.abstainCount).toBeLessThan(MARKOV_PHASE0_BASELINES.btc.h2.abstainCount);
        expect(metrics.directionalAccuracy).toBeGreaterThan(MARKOV_PHASE0_BASELINES.btc.h2.directionalAccuracy);
      }

      if (horizon === 3) {
        expect(metrics.rerunOnBreak).toBe(true);
        expect(metrics.rerunWindowDays).toBe(45);
        expect(metrics.abstainCount).toBeLessThan(MARKOV_PHASE0_BASELINES.btc.h3.abstainCount);
        expect(metrics.directionalAccuracy).toBeGreaterThanOrEqual(MARKOV_PHASE0_BASELINES.btc.h3.directionalAccuracy);
      }

      if (horizon === 14) {
        expect(metrics.rerunOnBreak).toBe(false);
        expect(metrics.rerunWindowDays).toBeNull();
        expect(metrics.breakDivergenceThreshold).toBeCloseTo(0.08);
        expect(metrics.abstainCount).toBeLessThan(MARKOV_PHASE0_BASELINES.btc.h14.abstainCount);
        expect(metrics.directionalAccuracy).toBeGreaterThanOrEqual(MARKOV_PHASE0_BASELINES.btc.h14.directionalAccuracy);
      }

    }

    lines.push('══════════════════════════════════════════');
    console.log(lines.join('\n'));
    if (process.env.FORECAST_LAB_OUTPUT_METRICS === '1') {
      console.log(`FORECAST_LAB_METRICS ${JSON.stringify(metricsByHorizon)}`);
    }
  }, TIMEOUT);
});
