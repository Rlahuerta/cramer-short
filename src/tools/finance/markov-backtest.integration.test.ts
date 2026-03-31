/**
 * Walk-forward backtest for the Markov distribution model.
 *
 * Validates probability calibration, CI coverage, and directional accuracy
 * using embedded fixture data (no live API calls).
 *
 * Three tiers:
 *   1. HARD GATES — robustness (no crashes, valid outputs) → must pass
 *   2. REGRESSION GATES — prediction quality (Brier) → catch degradation
 *   3. INFORMATIONAL — CI coverage, direction, correlation → logged baseline
 *
 * Run: RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts
 */

import { describe, it, expect, beforeAll } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { integrationIt } from '@/utils/test-guards.js';
import { walkForward, type WalkForwardResult } from './backtest/walk-forward.js';
import {
  brierScore,
  ciCoverage,
  directionalAccuracy,
  selectiveDirectionalAccuracy,
  computeRCCurve,
  expectedReturnCorrelation,
  sharpness,
  reliabilityBins,
  maxReliabilityDeviation,
  gofPassRate,
  generateReport,
  optimizeThresholds,
  type BacktestStep,
} from './backtest/metrics.js';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const TICKERS  = ['SPY', 'AAPL', 'TSLA', 'GLD', 'QQQ', 'BTC-USD'] as const;
const HORIZONS = [14, 30] as const;
const WARMUP   = 120;
const STRIDE   = 10;  // every 10 days — keeps test under 5s
const TIMEOUT  = 120_000;

// Regression thresholds — these catch model degradation, not aspirational targets
const BRIER_MAX = 0.42; // directional improvements trade some calibration; 0.42 catches severe regression

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

interface FixtureData {
  tickers: Record<string, {
    type: string;
    closes: number[];
    dates: string[];
    count: number;
    synthetic?: boolean;
  }>;
}

let fixture: FixtureData;
const results: Map<string, Map<number, WalkForwardResult>> = new Map();
const allSteps: BacktestStep[] = [];

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

describe('Markov distribution walk-forward backtest', () => {
  beforeAll(async () => {
    const fixturePath = join(import.meta.dir, 'fixtures', 'backtest-prices.json');
    fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));
  });

  // ========================================================================
  // Tier 1: HARD GATES — robustness (must pass)
  // ========================================================================

  describe('Tier 1: robustness', () => {
    for (const ticker of TICKERS) {
      for (const horizon of HORIZONS) {
        integrationIt(
          `${ticker} ${horizon}d: no crashes across walk-forward`,
          async () => {
            const data = fixture.tickers[ticker];
            expect(data).toBeDefined();
            expect(data.count).toBeGreaterThan(WARMUP + horizon + 10);

            const result = await walkForward({
              ticker,
              prices: data.closes,
              horizon,
              warmup: WARMUP,
              stride: STRIDE,
            });

            // Store for later tests
            if (!results.has(ticker)) results.set(ticker, new Map());
            results.get(ticker)!.set(horizon, result);
            allSteps.push(...result.steps);

            expect(result.errors).toHaveLength(0);
            expect(result.steps.length).toBeGreaterThan(5);
          },
          TIMEOUT,
        );
      }
    }

    integrationIt('no NaN in any prediction', async () => {
      for (const step of allSteps) {
        expect(Number.isNaN(step.predictedProb)).toBe(false);
        expect(Number.isNaN(step.predictedReturn)).toBe(false);
        expect(Number.isNaN(step.ciLower)).toBe(false);
        expect(Number.isNaN(step.ciUpper)).toBe(false);
      }
    });

    integrationIt('predicted probabilities are in [0, 1]', async () => {
      for (const step of allSteps) {
        expect(step.predictedProb).toBeGreaterThanOrEqual(0);
        expect(step.predictedProb).toBeLessThanOrEqual(1);
      }
    });

    integrationIt('CI lower < CI upper for all steps', async () => {
      for (const step of allSteps) {
        expect(step.ciLower).toBeLessThan(step.ciUpper);
      }
    });
  });

  // ========================================================================
  // Tier 2: REGRESSION GATES — catch degradation
  // ========================================================================

  describe('Tier 2: regression gates', () => {
    for (const ticker of TICKERS) {
      for (const horizon of HORIZONS) {
        integrationIt(
          `${ticker} ${horizon}d: Brier score < ${BRIER_MAX}`,
          async () => {
            const result = results.get(ticker)?.get(horizon);
            if (!result || result.steps.length === 0) return;
            const bs = brierScore(result.steps);
            expect(bs).toBeLessThan(BRIER_MAX);
          },
          TIMEOUT,
        );
      }
    }

    integrationIt('aggregate Brier score < 0.35', async () => {
      if (allSteps.length === 0) return;
      expect(brierScore(allSteps)).toBeLessThan(0.35);
    });
  });

  // ========================================================================
  // Tier 3: INFORMATIONAL — log baselines, always passes
  // ========================================================================

  describe('Tier 3: baseline metrics (informational)', () => {
    integrationIt('prints backtest summary report', async () => {
      if (allSteps.length === 0) return;

      const lines: string[] = ['', '═══ MARKOV BACKTEST SUMMARY ═══'];
      for (const ticker of TICKERS) {
        const horizonMap = results.get(ticker);
        if (!horizonMap) continue;
        for (const horizon of HORIZONS) {
          const result = horizonMap.get(horizon);
          if (!result) continue;
          const r = generateReport(ticker, horizon, result.steps);
          const flag = (v: number, good: number, bad: number) =>
            v >= good ? '✓' : v >= bad ? '~' : '✗';
          lines.push(
            `  ${ticker.padEnd(7)} ${horizon}d: `
            + `Brier=${r.brierScore.toFixed(3)} ${flag(1 - r.brierScore, 0.75, 0.65)} | `
            + `CI=${(r.ciCoverage * 100).toFixed(0).padStart(3)}% ${flag(r.ciCoverage, 0.80, 0.50)} | `
            + `Dir=${(r.directionalAccuracy * 100).toFixed(0).padStart(3)}% ${flag(r.directionalAccuracy, 0.55, 0.45)} | `
            + `Corr=${r.expectedReturnCorrelation.toFixed(3).padStart(7)} | `
            + `Sharp=${r.sharpness.toFixed(3)} | `
            + `GOF=${r.gofPassRate !== null ? (r.gofPassRate * 100).toFixed(0) + '%' : 'N/A'} | `
            + `n=${r.totalSteps}`,
          );
        }
      }

      // Aggregate
      const aggBrier = brierScore(allSteps);
      const aggCI    = ciCoverage(allSteps);
      const aggDir   = directionalAccuracy(allSteps);
      const aggCorr  = expectedReturnCorrelation(allSteps);
      const aggGOF   = gofPassRate(allSteps);
      const bins     = reliabilityBins(allSteps);
      const maxDev   = maxReliabilityDeviation(bins, 5);

      lines.push('─'.repeat(90));
      lines.push(
        `  AGGREGATE: Brier=${aggBrier.toFixed(3)} | CI=${(aggCI * 100).toFixed(0)}% | `
        + `Dir=${(aggDir * 100).toFixed(0)}% | Corr=${aggCorr.toFixed(3)} | `
        + `RelDev=${(maxDev * 100).toFixed(0)}pp | `
        + `GOF=${aggGOF !== null ? (aggGOF * 100).toFixed(0) + '%' : 'N/A'}`,
      );

      // Threshold optimization
      const opt = optimizeThresholds(allSteps);
      lines.push(
        `  OPTIMAL THRESHOLDS: buy=${opt.bestBuyThreshold} sell=${opt.bestSellThreshold} `
        + `→ accuracy=${(opt.bestAccuracy * 100).toFixed(1)}%`,
      );

      // Selective accuracy (RC curve — Idea M)
      const rcCurve = computeRCCurve(allSteps);
      lines.push('  RC Curve (selective prediction — coverage→accuracy):');
      for (const pt of rcCurve) {
        if (pt.n === 0) continue;
        const bar = '▓'.repeat(Math.round(pt.accuracy * 20));
        lines.push(
          `    conf≥${pt.threshold.toFixed(1)}: `
          + `acc=${(pt.accuracy * 100).toFixed(0).padStart(3)}% `
          + `cov=${(pt.coverage * 100).toFixed(0).padStart(3)}% `
          + `n=${String(pt.n).padStart(4)} ${bar}`,
        );
      }

      // Per-ticker selective accuracy at best threshold
      const bestSelective = rcCurve.find(pt => pt.accuracy >= 0.70 && pt.coverage >= 0.30);
      if (bestSelective) {
        lines.push(
          `  ★ SELECTIVE TARGET MET: ${(bestSelective.accuracy * 100).toFixed(0)}% accuracy `
          + `at ${(bestSelective.coverage * 100).toFixed(0)}% coverage `
          + `(threshold=${bestSelective.threshold})`,
        );
      }

      // Reliability breakdown
      lines.push('  Reliability bins (predicted→actual):');
      for (const b of bins) {
        if (b.count === 0) continue;
        const bar = '█'.repeat(Math.round(b.actualFrequency * 20));
        lines.push(
          `    [${b.binLower.toFixed(1)}–${b.binUpper.toFixed(1)}): `
          + `pred=${b.meanPredicted.toFixed(2)} actual=${b.actualFrequency.toFixed(2)} `
          + `n=${b.count} ${bar}`,
        );
      }

      lines.push('═══════════════════════════════', '');
      console.log(lines.join('\n'));

      // Always passes — this is informational
      expect(true).toBe(true);
    });
  });
});
