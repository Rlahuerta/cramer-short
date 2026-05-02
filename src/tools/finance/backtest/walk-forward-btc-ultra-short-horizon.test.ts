import { describe, beforeAll, expect } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { integrationIt } from '@/utils/test-guards.js';
import { walkForward, type WalkForwardResult } from './walk-forward.js';
import {
  brierScore,
  ciCoverage,
  directionalAccuracy,
  type BacktestStep,
} from './metrics.js';

const TICKER = 'BTC-USD';
const HORIZONS = [1, 2, 3] as const;
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

interface VariantDefinition {
  key: string;
  label: string;
  warmup: number;
  stride: number;
  transitionDecayOverride?: number;
  trendPenaltyOnlyBreakConfidence?: boolean;
}

interface VariantMetrics {
  steps: BacktestStep[];
  errors: number;
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
}

let fixture: FixtureData;

function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatDelta(value: number, digits = 1): string {
  const pct = value * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(digits)}pp`;
}

async function runVariantForHorizon(
  horizon: number,
  variant: VariantDefinition,
): Promise<VariantMetrics> {
  const tickerData = fixture.tickers[TICKER];
  const result: WalkForwardResult = await walkForward({
    ticker: TICKER,
    prices: tickerData.closes,
    horizon,
    warmup: variant.warmup,
    stride: variant.stride,
    transitionDecayOverride: variant.transitionDecayOverride,
    trendPenaltyOnlyBreakConfidence: variant.trendPenaltyOnlyBreakConfidence,
  });

  return {
    steps: result.steps,
    errors: result.errors.length,
    directionalAccuracy: result.steps.length > 0 ? directionalAccuracy(result.steps) : 0,
    brierScore: result.steps.length > 0 ? brierScore(result.steps) : 0,
    ciCoverage: result.steps.length > 0 ? ciCoverage(result.steps) : 0,
  };
}

describe('Walk-forward BTC ultra-short-horizon ablation', () => {
  beforeAll(() => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));
  });

  integrationIt('compares BTC-only variants across 1d/2d/3d horizons', async () => {
    const variants: VariantDefinition[] = [
      { key: 'baseline-120', label: 'baseline warmup=120 stride=3', warmup: 120, stride: 3 },
      { key: 'warmup-90', label: 'warmup=90 stride=3', warmup: 90, stride: 3 },
      { key: 'warmup-60', label: 'warmup=60 stride=3', warmup: 60, stride: 3 },
      { key: 'trend-penalty-only', label: 'warmup=120 trend-penalty-only', warmup: 120, stride: 3, trendPenaltyOnlyBreakConfidence: true },
      { key: 'decay-095', label: 'warmup=120 decay=0.95', warmup: 120, stride: 3, transitionDecayOverride: 0.95 },
    ];

    const lines: string[] = ['', '═══ BTC ULTRA-SHORT-HORIZON ABLATION ═══'];
    let totalRuns = 0;
    let totalErrors = 0;

    for (const horizon of HORIZONS) {
      lines.push('');
      lines.push(`BTC-USD horizon ${horizon}d`);
      lines.push('Variant                         │ Steps │ Errors │ Dir Acc │ Brier │ CI Cov │ ΔDir vs Base');
      lines.push('───────────────────────────────┼───────┼────────┼─────────┼───────┼────────┼─────────────');

      const metricsByVariant = new Map<string, VariantMetrics>();

      for (const variant of variants) {
        const metrics = await runVariantForHorizon(horizon, variant);
        metricsByVariant.set(variant.key, metrics);
        totalRuns += 1;
        totalErrors += metrics.errors;
      }

      const baseline = metricsByVariant.get('baseline-120');
      expect(baseline).toBeDefined();
      expect(baseline!.steps.length).toBeGreaterThan(0);

      for (const variant of variants) {
        const metrics = metricsByVariant.get(variant.key)!;
        const delta = metrics.directionalAccuracy - baseline!.directionalAccuracy;

        lines.push(
          `${variant.label.padEnd(31)} │ ${String(metrics.steps.length).padStart(5)} │ ${String(metrics.errors).padStart(6)} │ ${formatPct(metrics.directionalAccuracy).padStart(7)} │ ${metrics.brierScore.toFixed(3).padStart(5)} │ ${formatPct(metrics.ciCoverage).padStart(6)} │ ${formatDelta(delta).padStart(11)}`,
        );
      }
    }

    lines.push('');
    lines.push(`Completed ${totalRuns} BTC ultra-short-horizon variant runs.`);
    lines.push(`Total errors across runs: ${totalErrors}`);
    lines.push('══════════════════════════════════════════');
    console.log(lines.join('\n'));

    expect(totalRuns).toBe(HORIZONS.length * variants.length);
    expect(totalErrors).toBe(0);
  }, TIMEOUT);
});
