import { describe, expect } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { integrationIt } from '@/utils/test-guards.js';
import { walkForward } from './walk-forward.js';
import { brierScore, ciCoverage, directionalAccuracy } from './metrics.js';

const TIMEOUT = 240_000;
const STRIDE = 5;

interface FixtureData {
  tickers: Record<string, {
    closes: number[];
  }>;
}

interface VariantConfig {
  label: string;
  warmup: number;
  btcBreakDivergenceThreshold?: number;
  postBreakShortWindow?: boolean;
  postBreakWindowSize?: number;
}

interface VariantMetrics {
  label: string;
  horizon: number;
  steps: number;
  errors: number;
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  rerunRate: number;
}

async function runVariant(
  prices: number[],
  horizon: number,
  config: VariantConfig,
): Promise<VariantMetrics> {
  const result = await walkForward({
    ticker: 'BTC-USD',
    prices,
    horizon,
    warmup: config.warmup,
    stride: STRIDE,
    btcBreakDivergenceThreshold: config.btcBreakDivergenceThreshold,
    postBreakShortWindow: config.postBreakShortWindow,
    postBreakWindowSize: config.postBreakWindowSize,
  });

  const rerunCount = result.steps.filter((step) => step.structuralBreakRerunTriggered).length;
  return {
    label: config.label,
    horizon,
    steps: result.steps.length,
    errors: result.errors.length,
    directionalAccuracy: directionalAccuracy(result.steps),
    brierScore: brierScore(result.steps),
    ciCoverage: ciCoverage(result.steps),
    rerunRate: result.steps.length > 0 ? rerunCount / result.steps.length : 0,
  };
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

describe('BTC live short-horizon policy walk-forward', () => {
  integrationIt('improves directional accuracy across 1d/2d/3d/14d on the BTC fixture', async () => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;
    const prices = fixture.tickers['BTC-USD']?.closes;

    expect(prices).toBeDefined();
    expect(prices.length).toBeGreaterThan(300);

    const baseline: VariantConfig = {
      label: 'baseline-120',
      warmup: 120,
    };

    const tunedByHorizon = new Map<number, VariantConfig>([
      [1, { label: 'btc-live-1d', warmup: 252, btcBreakDivergenceThreshold: 0.10, postBreakShortWindow: true, postBreakWindowSize: 60 }],
      [2, { label: 'btc-live-2d', warmup: 252, btcBreakDivergenceThreshold: 0.15 }],
      [3, { label: 'btc-live-3d', warmup: 252, btcBreakDivergenceThreshold: 0.20, postBreakShortWindow: true, postBreakWindowSize: 60 }],
      [14, { label: 'btc-live-14d', warmup: 252, btcBreakDivergenceThreshold: 0.15 }],
    ]);

    const horizons = [1, 2, 3, 14] as const;
    const lines = ['', '═══ BTC LIVE SHORT-HORIZON POLICY ═══'];

    for (const horizon of horizons) {
      const baselineMetrics = await runVariant(prices, horizon, baseline);
      const tunedMetrics = await runVariant(prices, horizon, tunedByHorizon.get(horizon)!);

      lines.push(
        `${horizon}d | ${baselineMetrics.label}: dir=${formatPct(baselineMetrics.directionalAccuracy)} brier=${baselineMetrics.brierScore.toFixed(4)} ci=${formatPct(baselineMetrics.ciCoverage)} rerun=${formatPct(baselineMetrics.rerunRate)}`,
      );
      lines.push(
        `${horizon}d | ${tunedMetrics.label}: dir=${formatPct(tunedMetrics.directionalAccuracy)} brier=${tunedMetrics.brierScore.toFixed(4)} ci=${formatPct(tunedMetrics.ciCoverage)} rerun=${formatPct(tunedMetrics.rerunRate)}`,
      );

      expect(baselineMetrics.errors).toBe(0);
      expect(tunedMetrics.errors).toBe(0);
      expect(tunedMetrics.directionalAccuracy).toBeGreaterThan(baselineMetrics.directionalAccuracy);

      if (horizon === 1) {
        expect(tunedMetrics.directionalAccuracy).toBeGreaterThanOrEqual(0.60);
        expect(tunedMetrics.rerunRate).toBeGreaterThan(0.50);
      }

      if (horizon === 2) {
        expect(tunedMetrics.directionalAccuracy).toBeGreaterThanOrEqual(0.50);
        expect(tunedMetrics.rerunRate).toBe(0);
      }

      if (horizon === 3) {
        expect(tunedMetrics.directionalAccuracy).toBeGreaterThanOrEqual(0.55);
        expect(tunedMetrics.rerunRate).toBeGreaterThan(0.20);
        expect(tunedMetrics.rerunRate).toBeLessThan(0.50);
      }

      if (horizon === 14) {
        expect(tunedMetrics.directionalAccuracy).toBeGreaterThanOrEqual(0.50);
        expect(tunedMetrics.rerunRate).toBe(0);
      }
    }

    lines.push('══════════════════════════════════════');
    console.log(lines.join('\n'));
  }, TIMEOUT);
});
