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

const TICKERS = ['SPY', 'QQQ', 'GLD', 'VOO', 'NVDA', 'MSFT'] as const;
const HORIZONS = [7, 14, 30] as const;
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

interface VariantDefinition {
  key: string;
  label: string;
  warmup: number;
  transitionDecayOverride?: number;
  postBreakShortWindow?: boolean;
  postBreakWindowSize?: number;
  trendPenaltyOnlyBreakConfidence?: boolean;
  regimeSpecificSigma?: boolean;
  regimeSpecificSigmaThreshold?: number;
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

async function runVariantForTickerHorizon(
  ticker: string,
  horizon: number,
  variant: VariantDefinition,
): Promise<VariantMetrics> {
  const tickerData = fixture.tickers[ticker];
  const result: WalkForwardResult = await walkForward({
    ticker,
    prices: tickerData.closes,
    horizon,
    warmup: variant.warmup,
    stride: 5,
    transitionDecayOverride: variant.transitionDecayOverride,
    postBreakShortWindow: variant.postBreakShortWindow,
    postBreakWindowSize: variant.postBreakWindowSize,
    trendPenaltyOnlyBreakConfidence: variant.trendPenaltyOnlyBreakConfidence,
    regimeSpecificSigma: variant.regimeSpecificSigma,
    regimeSpecificSigmaThreshold: variant.regimeSpecificSigmaThreshold,
  });

  return {
    steps: result.steps,
    errors: result.errors.length,
    directionalAccuracy: result.steps.length > 0 ? directionalAccuracy(result.steps) : 0,
    brierScore: result.steps.length > 0 ? brierScore(result.steps) : 0,
    ciCoverage: result.steps.length > 0 ? ciCoverage(result.steps) : 0,
  };
}

describe('Walk-forward short-horizon ablation (Phase 1 Discovery B1)', () => {
  beforeAll(() => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));
  });

  integrationIt('compares warmup, decay, and post-break variants across 7d/14d/30d', async () => {
    const variants: VariantDefinition[] = [
      { key: 'baseline-120', label: 'baseline warmup=120', warmup: 120 },
      { key: 'trend-penalty-only', label: 'warmup=120 trend-penalty-only', warmup: 120, trendPenaltyOnlyBreakConfidence: true },
      { key: 'warmup-90', label: 'warmup=90', warmup: 90 },
      { key: 'warmup-60', label: 'warmup=60', warmup: 60 },
      { key: 'decay-095', label: 'warmup=120 decay=0.95', warmup: 120, transitionDecayOverride: 0.95 },
      { key: 'post-break-60', label: 'warmup=120 postBreakShortWindow=60', warmup: 120, postBreakShortWindow: true, postBreakWindowSize: 60 },
      { key: 'ph7-sigma-t55', label: 'warmup=120 trend+sigma0.55', warmup: 120, trendPenaltyOnlyBreakConfidence: true, regimeSpecificSigma: true, regimeSpecificSigmaThreshold: 0.55 },
      { key: 'ph7-sigma-t60', label: 'warmup=120 trend+sigma0.60', warmup: 120, trendPenaltyOnlyBreakConfidence: true, regimeSpecificSigma: true, regimeSpecificSigmaThreshold: 0.60 },
      { key: 'ph7-sigma-t70', label: 'warmup=120 trend+sigma0.70', warmup: 120, trendPenaltyOnlyBreakConfidence: true, regimeSpecificSigma: true, regimeSpecificSigmaThreshold: 0.70 },
    ];

    const lines: string[] = ['', '═══ WALK-FORWARD SHORT-HORIZON ABLATION ═══'];
    let totalRuns = 0;
    let totalErrors = 0;
    let breakSidewaysCount = 0;
    let breakSidewaysChangedCount = 0;
    let breakTrendingCount = 0;
    let regimeSpecificSigmaProvenanceCount = 0;
    let regimeSpecificSigmaActiveCount = 0;
    const metricsCache = new Map<string, VariantMetrics>();

    const cacheKey = (ticker: string, horizon: number, variant: VariantDefinition) =>
      `${ticker}:${horizon}:${variant.key}`;

    for (const horizon of HORIZONS) {
      lines.push('');
      lines.push(`Horizon ${horizon}d`);
      lines.push('Ticker  │ Variant                         │ Steps │ Errors │ Dir Acc │ Brier │ CI Cov │ ΔDir vs Base');
      lines.push('───────┼─────────────────────────────────┼───────┼────────┼─────────┼───────┼────────┼─────────────');

      for (const ticker of TICKERS) {
        const metricsByVariant = new Map<string, VariantMetrics>();
        for (const variant of variants) {
          const key = cacheKey(ticker, horizon, variant);
          const cached = metricsCache.get(key);
          const metrics = cached ?? await runVariantForTickerHorizon(ticker, horizon, variant);
          metricsCache.set(key, metrics);
          metricsByVariant.set(variant.key, metrics);
          totalRuns++;
          totalErrors += metrics.errors;
        }

        const baseline = metricsByVariant.get('baseline-120');
        const trendPenaltyOnly = metricsByVariant.get('trend-penalty-only');
        expect(baseline).toBeDefined();
        expect(baseline!.steps.length).toBeGreaterThan(0);
        expect(trendPenaltyOnly).toBeDefined();

        const baselineSteps = baseline!.steps;
        const trendPenaltyOnlySteps = trendPenaltyOnly!.steps;
        expect(trendPenaltyOnlySteps).toHaveLength(baselineSteps.length);

        for (let i = 0; i < baselineSteps.length; i++) {
          const baseStep = baselineSteps[i];
          const experimentStep = trendPenaltyOnlySteps[i];
          const effectiveBreak = baseStep.originalStructuralBreakDetected ?? baseStep.structuralBreakDetected;

          expect(experimentStep.t).toBe(baseStep.t);
          expect(experimentStep.trendPenaltyOnlyBreakConfidenceActive).toBe(true);

          if (effectiveBreak && baseStep.regime === 'sideways') {
            breakSidewaysCount++;
            if (Math.abs(experimentStep.confidence - baseStep.confidence) > 1e-9) {
              breakSidewaysChangedCount++;
            }
            expect(experimentStep.confidence).toBeGreaterThanOrEqual(baseStep.confidence);
          } else {
            expect(experimentStep.confidence).toBeCloseTo(baseStep.confidence, 10);
            if (effectiveBreak && (baseStep.regime === 'bull' || baseStep.regime === 'bear')) {
              breakTrendingCount++;
            }
          }
        }

        // Phase 7: verify regime-specific sigma variants have expected step counts and provenance.
        // Activation itself is threshold- and universe-dependent; the unit tests cover
        // that activation can occur when the threshold is low enough.
        const ph7Variant = metricsByVariant.get('ph7-sigma-t55');
        if (ph7Variant) {
          expect(ph7Variant.steps).toHaveLength(baselineSteps.length);
          for (const step of ph7Variant.steps) {
            expect(step.trendPenaltyOnlyBreakConfidenceActive).toBe(true);
            expect(typeof step.regimeSpecificSigmaActive).toBe('boolean');
            regimeSpecificSigmaProvenanceCount++;
            if (step.regimeSpecificSigmaActive) {
              regimeSpecificSigmaActiveCount++;
            }
          }
        }

        for (const variant of variants) {
          const metrics = metricsByVariant.get(variant.key)!;
          const delta = baseline ? metrics.directionalAccuracy - baseline.directionalAccuracy : 0;
          lines.push(
            `${ticker.padEnd(6)} │ ${variant.label.padEnd(31)} │ ${String(metrics.steps.length).padStart(5)} │ ${String(metrics.errors).padStart(6)} │ ${formatPct(metrics.directionalAccuracy).padStart(7)} │ ${metrics.brierScore.toFixed(3).padStart(5)} │ ${formatPct(metrics.ciCoverage).padStart(6)} │ ${formatDelta(delta).padStart(11)}`,
          );
        }
      }

      lines.push('');
      lines.push(`Aggregate summary (${horizon}d):`);
      lines.push('Variant                         │ Dir Acc │ Brier │ CI Cov │ Promising');
      lines.push('───────────────────────────────┼─────────┼───────┼────────┼──────────');

      const baselineSteps = TICKERS.flatMap(ticker => metricsCache.get(cacheKey(ticker, horizon, variants[0]))?.steps ?? []);
      const baselineDir = directionalAccuracy(baselineSteps);
      const baselineBrier = brierScore(baselineSteps);

      for (const variant of variants) {
        const combinedSteps: BacktestStep[] = [];
        let combinedErrors = 0;

        for (const ticker of TICKERS) {
          const metrics = metricsCache.get(cacheKey(ticker, horizon, variant));
          if (!metrics) {
            throw new Error(`Missing cached metrics for ${ticker} ${horizon} ${variant.key}`);
          }
          combinedSteps.push(...metrics.steps);
          combinedErrors += metrics.errors;
        }

        const dir = combinedSteps.length > 0 ? directionalAccuracy(combinedSteps) : 0;
        const brier = combinedSteps.length > 0 ? brierScore(combinedSteps) : 0;
        const coverage = combinedSteps.length > 0 ? ciCoverage(combinedSteps) : 0;
        const promising =
          dir - baselineDir >= 0.02 &&
          brier - baselineBrier <= 0.01 &&
          coverage >= 0.85;

        lines.push(
          `${variant.label.padEnd(31)} │ ${formatPct(dir).padStart(7)} │ ${brier.toFixed(3).padStart(5)} │ ${formatPct(coverage).padStart(6)} │ ${promising ? 'YES' : 'no '} (${combinedErrors} err)`,
        );
      }
    }

    lines.push('');
    lines.push(`Completed ${totalRuns} ticker-horizon-variant runs.`);
    lines.push(`Total errors across runs: ${totalErrors}`);
    lines.push('═══════════════════════════════════════════════════');
    console.log(lines.join('\n'));

    expect(totalRuns).toBe(TICKERS.length * HORIZONS.length * variants.length);
    expect(totalErrors).toBe(0);
    expect(breakSidewaysCount).toBeGreaterThan(0);
    expect(breakSidewaysChangedCount).toBeGreaterThan(0);
    expect(breakTrendingCount).toBeGreaterThan(0);
    expect(regimeSpecificSigmaProvenanceCount).toBeGreaterThan(0);
  }, TIMEOUT);
});
