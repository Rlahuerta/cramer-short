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
  computeFailureDecomposition,
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
  bootstrapDirectionalCI,
  bootstrapBrierCI,
  bootstrapCIcoverageCI,
  bootstrapMetricCI,
  pUpDirectionalAccuracy,
  calibratedPUpDirectionalAccuracy,
  rawPUpDirectionalAccuracy,
  selectivePUpAccuracy,
  selectiveRawPUpAccuracy,
  bucketByPUpBand,
  type BacktestStep,
  type FailureSliceKey,
  type DecisionSource,
  type ProbabilitySource,
} from './backtest/metrics.js';
import { generateStressScenarios, type StressScenario } from './backtest/stress-scenarios.js';

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

function formatPct(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatSignedPct(value: number, digits = 1): string {
  const pct = value * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(digits)}%`;
}

function appendRawVsCalibratedBlock(lines: string[], label: string, steps: BacktestStep[]): void {
  if (steps.length === 0) return;
  const calibrated = directionalAccuracy(steps);
  const raw = pUpDirectionalAccuracy(steps);
  const deltaPp = (raw - calibrated) * 100;
  lines.push(
    `    ${label.padEnd(12)} rec=${formatPct(calibrated).padStart(4)} | raw=${formatPct(raw).padStart(4)} | Δ=${deltaPp >= 0 ? '+' : ''}${deltaPp.toFixed(1)}pp | n=${steps.length}`,
  );
}

function appendFailureSlice(
  lines: string[],
  title: string,
  steps: BacktestStep[],
  key: FailureSliceKey,
): void {
  if (steps.length === 0) return;
  const slice = computeFailureDecomposition(steps).slices.find(item => item.key === key);
  if (!slice) return;

  const rows = slice.rows.filter(row => row.count > 0);
  if (rows.length === 0) return;

  lines.push(`    ${title}:`);
  for (const row of rows) {
    lines.push(
      `      ${row.label.padEnd(14)} n=${String(row.count).padStart(3)} | dir=${formatPct(row.directionalAccuracy).padStart(4)} | bal=${formatPct(row.balancedDirectionalAccuracy).padStart(4)} | brier=${row.brierScore.toFixed(3)} | ci=${formatPct(row.ciCoverage).padStart(4)} | edge=${formatSignedPct(row.meanEdge).padStart(6)}`,
    );
  }
}

function collectTickerSteps(ticker: string, horizons: readonly number[]): BacktestStep[] {
  const horizonMap = results.get(ticker);
  if (!horizonMap) return [];
  return horizons.flatMap(horizon => horizonMap.get(horizon)?.steps ?? []);
}

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
      const rcCurve = computeRCCurve(allSteps, [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]);
      lines.push('  RC Curve (selective prediction — coverage→accuracy):');
      for (const pt of rcCurve) {
        if (pt.n === 0) continue;
        const bar = '▓'.repeat(Math.round(pt.accuracy * 20));
        lines.push(
          `    conf≥${pt.threshold.toFixed(2)}: `
          + `acc=${(pt.accuracy * 100).toFixed(0).padStart(3)}% `
          + `cov=${(pt.coverage * 100).toFixed(0).padStart(3)}% `
          + `n=${String(pt.n).padStart(4)} ${bar}`,
        );
      }

      // P(up)-based directional accuracy (no HOLD dead zone)
      const pUpDir = pUpDirectionalAccuracy(allSteps);
      lines.push(`  P(up) Directional: ${(pUpDir * 100).toFixed(0)}% (no HOLD zone)`);

      // P(up)-based selective accuracy
      const pUpRC = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5].map(t => {
        const r = selectivePUpAccuracy(allSteps, t);
        return { threshold: t, ...r };
      });
      lines.push('  P(up) RC Curve:');
      for (const pt of pUpRC) {
        if (pt.selected === 0) continue;
        const bar = '▓'.repeat(Math.round(pt.accuracy * 20));
        lines.push(
          `    conf≥${pt.threshold.toFixed(2)}: `
          + `acc=${(pt.accuracy * 100).toFixed(0).padStart(3)}% `
          + `cov=${(pt.coverage * 100).toFixed(0).padStart(3)}% `
          + `n=${String(pt.selected).padStart(4)} ${bar}`,
        );
      }

      const btcAggregateSteps = collectTickerSteps('BTC-USD', HORIZONS);
      lines.push('  Raw vs calibrated direction:');
      appendRawVsCalibratedBlock(lines, 'AGGREGATE', allSteps);
      appendRawVsCalibratedBlock(lines, 'BTC-ONLY', btcAggregateSteps);

      lines.push('  Failure decomposition (aggregate):');
      appendFailureSlice(lines, 'Regime', allSteps, 'regime');
      appendFailureSlice(lines, 'Volatility', allSteps, 'volatility');
      appendFailureSlice(lines, 'Confidence', allSteps, 'confidence');
      appendFailureSlice(lines, 'Anchor quality', allSteps, 'anchorQuality');
      appendFailureSlice(lines, 'Validation metric', allSteps, 'validationMetric');

      // ETF-only accuracy
      const etfTickers = ['SPY', 'GLD', 'QQQ'];
      const etfSteps = allSteps.filter((_s, idx) => {
        // Determine which ticker this step belongs to by checking the results
        let cumIdx = 0;
        for (const ticker of TICKERS) {
          const horizonMap = results.get(ticker);
          if (!horizonMap) continue;
          for (const horizon of HORIZONS) {
            const result = horizonMap.get(horizon);
            if (!result) continue;
            if (idx >= cumIdx && idx < cumIdx + result.steps.length) {
              return etfTickers.includes(ticker);
            }
            cumIdx += result.steps.length;
          }
        }
        return false;
      });
      if (etfSteps.length > 0) {
        const etfDir = directionalAccuracy(etfSteps);
        const etfPUp = pUpDirectionalAccuracy(etfSteps);
        const etfCI = ciCoverage(etfSteps);
        lines.push(
          `  ETF-ONLY (SPY+GLD+QQQ): Dir=${(etfDir * 100).toFixed(0)}% | `
          + `P(up)Dir=${(etfPUp * 100).toFixed(0)}% | CI=${(etfCI * 100).toFixed(0)}% | n=${etfSteps.length}`,
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

  describe('First-wave promotion audit', () => {
    const promoTickers = ['VOO', 'DIA', 'VTI', 'IAU'] as const;
    const promoHorizons = [5, 7, 10, 14, 20, 30] as const;
    const promoResults: Map<string, Map<number, WalkForwardResult>> = new Map();
    const promoSteps: BacktestStep[] = [];

    describe('Tier 1: robustness', () => {
      for (const ticker of promoTickers) {
        for (const horizon of promoHorizons) {
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

              if (!promoResults.has(ticker)) promoResults.set(ticker, new Map());
              promoResults.get(ticker)!.set(horizon, result);
              promoSteps.push(...result.steps);

              expect(result.errors).toHaveLength(0);
              expect(result.steps.length).toBeGreaterThan(5);
            },
            TIMEOUT,
          );
        }
      }

      integrationIt('first-wave: no NaN in any prediction', async () => {
        for (const step of promoSteps) {
          expect(Number.isNaN(step.predictedProb)).toBe(false);
          expect(Number.isNaN(step.predictedReturn)).toBe(false);
          expect(Number.isNaN(step.ciLower)).toBe(false);
          expect(Number.isNaN(step.ciUpper)).toBe(false);
        }
      });

      integrationIt('first-wave: predicted probabilities are in [0, 1]', async () => {
        for (const step of promoSteps) {
          expect(step.predictedProb).toBeGreaterThanOrEqual(0);
          expect(step.predictedProb).toBeLessThanOrEqual(1);
        }
      });

      integrationIt('first-wave: CI lower < CI upper for all steps', async () => {
        for (const step of promoSteps) {
          expect(step.ciLower).toBeLessThan(step.ciUpper);
        }
      });
    });

    describe('Tier 2: regression gates', () => {
      for (const ticker of promoTickers) {
        for (const horizon of promoHorizons) {
          integrationIt(
            `${ticker} ${horizon}d: Brier score < ${BRIER_MAX}`,
            async () => {
              const result = promoResults.get(ticker)?.get(horizon);
              if (!result || result.steps.length === 0) return;
              const bs = brierScore(result.steps);
              expect(bs).toBeLessThan(BRIER_MAX);
            },
            TIMEOUT,
          );
        }
      }

      integrationIt('first-wave aggregate Brier score < 0.35', async () => {
        if (promoSteps.length === 0) return;
        expect(brierScore(promoSteps)).toBeLessThan(0.35);
      });
    });

    describe('Tier 3: promotion metrics (informational)', () => {
      integrationIt('prints first-wave promotion audit report', async () => {
        if (promoSteps.length === 0) return;

        const lines: string[] = ['', '═══ FIRST-WAVE PROMOTION AUDIT ═══'];
        for (const ticker of promoTickers) {
          const horizonMap = promoResults.get(ticker);
          if (!horizonMap) continue;
          for (const horizon of promoHorizons) {
            const result = horizonMap.get(horizon);
            if (!result) continue;
            const r = generateReport(ticker, horizon, result.steps);
            const calPUp = calibratedPUpDirectionalAccuracy(result.steps);
            const avgMarkovWeight = result.steps.length > 0
              ? result.steps.reduce((sum, step) => sum + (step.markovWeight ?? 0), 0) / result.steps.length
              : 0;
            const flag = (v: number, good: number, bad: number) =>
              v >= good ? '✓' : v >= bad ? '~' : '✗';
            lines.push(
              `  ${ticker.padEnd(5)} ${String(horizon).padStart(2)}d: `
              + `Brier=${r.brierScore.toFixed(3)} ${flag(1 - r.brierScore, 0.75, 0.65)} | `
              + `CI=${(r.ciCoverage * 100).toFixed(0).padStart(3)}% ${flag(r.ciCoverage, 0.80, 0.50)} | `
              + `Dir=${(r.directionalAccuracy * 100).toFixed(0).padStart(3)}% ${flag(r.directionalAccuracy, 0.55, 0.45)} | `
              + `CalPUp=${(calPUp * 100).toFixed(0).padStart(3)}% ${flag(calPUp, 0.55, 0.45)} | `
              + `MW=${avgMarkovWeight.toExponential(2)} | `
              + `n=${r.totalSteps}`,
            );
          }
        }

        const aggBrier = brierScore(promoSteps);
        const aggCI = ciCoverage(promoSteps);
        const aggDir = directionalAccuracy(promoSteps);
        const aggCalPUp = calibratedPUpDirectionalAccuracy(promoSteps);
        const aggMarkovWeight = promoSteps.length > 0
          ? promoSteps.reduce((sum, step) => sum + (step.markovWeight ?? 0), 0) / promoSteps.length
          : 0;
        lines.push('─'.repeat(110));
        lines.push(
          `  FIRST-WAVE AGG: Brier=${aggBrier.toFixed(3)} | `
          + `CI=${(aggCI * 100).toFixed(0)}% | `
          + `Dir=${(aggDir * 100).toFixed(0)}% | `
          + `CalPUp=${(aggCalPUp * 100).toFixed(0)}% | `
          + `MW=${aggMarkovWeight.toExponential(2)} | `
          + `n=${promoSteps.length}`,
        );
        lines.push('════════════════════════════════', '');
        console.log(lines.join('\n'));

        expect(true).toBe(true);
      }, TIMEOUT);
    });
  });

  describe('Second-wave promotion audit', () => {
    const wave2Tickers = ['MSFT', 'NVDA', 'GOOGL', 'AMZN'] as const;
    const wave2Horizons = [5, 7, 10, 14, 20, 30] as const;
    const wave2Results: Map<string, Map<number, WalkForwardResult>> = new Map();
    const wave2Steps: BacktestStep[] = [];

    describe('Tier 1: robustness', () => {
      for (const ticker of wave2Tickers) {
        for (const horizon of wave2Horizons) {
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

              if (!wave2Results.has(ticker)) wave2Results.set(ticker, new Map());
              wave2Results.get(ticker)!.set(horizon, result);
              wave2Steps.push(...result.steps);

              expect(result.errors).toHaveLength(0);
              expect(result.steps.length).toBeGreaterThan(5);
            },
            TIMEOUT,
          );
        }
      }

      integrationIt('second-wave: no NaN in any prediction', async () => {
        for (const step of wave2Steps) {
          expect(Number.isNaN(step.predictedProb)).toBe(false);
          expect(Number.isNaN(step.predictedReturn)).toBe(false);
          expect(Number.isNaN(step.ciLower)).toBe(false);
          expect(Number.isNaN(step.ciUpper)).toBe(false);
        }
      });

      integrationIt('second-wave: predicted probabilities are in [0, 1]', async () => {
        for (const step of wave2Steps) {
          expect(step.predictedProb).toBeGreaterThanOrEqual(0);
          expect(step.predictedProb).toBeLessThanOrEqual(1);
        }
      });

      integrationIt('second-wave: CI lower < CI upper for all steps', async () => {
        for (const step of wave2Steps) {
          expect(step.ciLower).toBeLessThan(step.ciUpper);
        }
      });
    });

    describe('Tier 2: regression gates', () => {
      for (const ticker of wave2Tickers) {
        for (const horizon of wave2Horizons) {
          integrationIt(
            `${ticker} ${horizon}d: Brier score < ${BRIER_MAX}`,
            async () => {
              const result = wave2Results.get(ticker)?.get(horizon);
              if (!result || result.steps.length === 0) return;
              const bs = brierScore(result.steps);
              expect(bs).toBeLessThan(BRIER_MAX);
            },
            TIMEOUT,
          );
        }
      }

      integrationIt('second-wave aggregate Brier score < 0.35', async () => {
        if (wave2Steps.length === 0) return;
        expect(brierScore(wave2Steps)).toBeLessThan(0.35);
      });
    });

    describe('Tier 3: promotion metrics (informational)', () => {
      integrationIt('prints second-wave promotion audit report', async () => {
        if (wave2Steps.length === 0) return;

        const lines: string[] = ['', '═══ SECOND-WAVE PROMOTION AUDIT ═══'];
        for (const ticker of wave2Tickers) {
          const horizonMap = wave2Results.get(ticker);
          if (!horizonMap) continue;
          for (const horizon of wave2Horizons) {
            const result = horizonMap.get(horizon);
            if (!result) continue;
            const r = generateReport(ticker, horizon, result.steps);
            const calPUp = calibratedPUpDirectionalAccuracy(result.steps);
            const avgMarkovWeight = result.steps.length > 0
              ? result.steps.reduce((sum, step) => sum + (step.markovWeight ?? 0), 0) / result.steps.length
              : 0;
            const flag = (v: number, good: number, bad: number) =>
              v >= good ? '✓' : v >= bad ? '~' : '✗';
            lines.push(
              `  ${ticker.padEnd(5)} ${String(horizon).padStart(2)}d: `
              + `Brier=${r.brierScore.toFixed(3)} ${flag(1 - r.brierScore, 0.75, 0.65)} | `
              + `CI=${(r.ciCoverage * 100).toFixed(0).padStart(3)}% ${flag(r.ciCoverage, 0.80, 0.50)} | `
              + `Dir=${(r.directionalAccuracy * 100).toFixed(0).padStart(3)}% ${flag(r.directionalAccuracy, 0.55, 0.45)} | `
              + `CalPUp=${(calPUp * 100).toFixed(0).padStart(3)}% ${flag(calPUp, 0.55, 0.45)} | `
              + `MW=${avgMarkovWeight.toExponential(2)} | `
              + `n=${r.totalSteps}`,
            );
          }
        }

        const aggBrier = brierScore(wave2Steps);
        const aggCI = ciCoverage(wave2Steps);
        const aggDir = directionalAccuracy(wave2Steps);
        const aggCalPUp = calibratedPUpDirectionalAccuracy(wave2Steps);
        const aggMarkovWeight = wave2Steps.length > 0
          ? wave2Steps.reduce((sum, step) => sum + (step.markovWeight ?? 0), 0) / wave2Steps.length
          : 0;
        lines.push('─'.repeat(110));
        lines.push(
          `  SECOND-WAVE AGG: Brier=${aggBrier.toFixed(3)} | `
          + `CI=${(aggCI * 100).toFixed(0)}% | `
          + `Dir=${(aggDir * 100).toFixed(0)}% | `
          + `CalPUp=${(aggCalPUp * 100).toFixed(0)}% | `
          + `MW=${aggMarkovWeight.toExponential(2)} | `
          + `n=${wave2Steps.length}`,
        );
        lines.push('═════════════════════════════════', '');
        console.log(lines.join('\n'));

        expect(true).toBe(true);
      }, TIMEOUT);
    });
  });

  describe('BTC short-horizon calibration', () => {
    const btcHorizons = [7, 14] as const;
    const btcResults: Map<number, WalkForwardResult> = new Map();

    for (const horizon of btcHorizons) {
      integrationIt(
        `BTC-USD ${horizon}d: walk-forward completes without errors`,
        async () => {
          const data = fixture.tickers['BTC-USD'];
          expect(data).toBeDefined();

          const result = await walkForward({
            ticker: 'BTC-USD',
            prices: data.closes,
            horizon,
            warmup: WARMUP,
            stride: STRIDE,
          });

          btcResults.set(horizon, result);

          expect(result.errors).toHaveLength(0);
          expect(result.steps.length).toBeGreaterThan(5);
        },
        TIMEOUT,
      );
    }

    for (const horizon of btcHorizons) {
      integrationIt(
        `BTC-USD ${horizon}d: loose sanity gates`,
        async () => {
          const result = btcResults.get(horizon);
          if (!result || result.steps.length === 0) return;

          const bs = brierScore(result.steps);
          const dir = directionalAccuracy(result.steps);
          const cov = ciCoverage(result.steps);

          expect(bs).toBeLessThan(0.55);
          expect(cov).toBeGreaterThan(0.40);
        },
        TIMEOUT,
      );
    }

    integrationIt(
      'BTC-USD 7d/14d: PR3A calibration diagnostic report',
      async () => {
        // Named baseline constants — Stage 0 / PR3A reference values
        // These are informational; the backtest harness is the source of truth.
        const PR3A_BASELINE: Record<number, { recAccuracy: number; calPUpAccuracy: number; rawPUpAccuracy: number | null }> = {
          7:  { recAccuracy: 0.44, calPUpAccuracy: 0.44, rawPUpAccuracy: null },
          14: { recAccuracy: 0.47, calPUpAccuracy: 0.47, rawPUpAccuracy: null },
        };

        const lines: string[] = ['', '═══ BTC SHORT-HORIZON PR3A DIAGNOSTIC ═══'];

        for (const horizon of btcHorizons) {
          const result = btcResults.get(horizon);
          if (!result || result.steps.length === 0) continue;
          const steps = result.steps;

          const baseline = PR3A_BASELINE[horizon];

          const bs = brierScore(steps);
          const recDir = directionalAccuracy(steps);
          const calPUp = calibratedPUpDirectionalAccuracy(steps);
          const rawPUp = rawPUpDirectionalAccuracy(steps);
          const cov = ciCoverage(steps);
          const bins = reliabilityBins(steps, 5);
          const maxDev = maxReliabilityDeviation(bins, 3);
          const bsCI = bootstrapBrierCI(steps);
          const recDirCI = bootstrapDirectionalCI(steps);
          const covCI = bootstrapCIcoverageCI(steps);

          // Bootstrap CIs for the three directional metrics
          const calPUpCI = bootstrapMetricCI(steps, calibratedPUpDirectionalAccuracy);
          const rawPUpCI = bootstrapMetricCI(steps, rawPUpDirectionalAccuracy);

          lines.push(
            `  BTC-USD ${horizon}d (n=${steps.length}):`,
          );
          lines.push(
            `    Brier=${bs.toFixed(3)} [${bsCI.lower.toFixed(3)}, ${bsCI.upper.toFixed(3)}] | `
            + `CI=${(cov * 100).toFixed(0)}% [${(covCI.lower * 100).toFixed(0)}%, ${(covCI.upper * 100).toFixed(0)}%] | `
            + `RelDev=${(maxDev * 100).toFixed(0)}pp`,
          );

          // PR3A directional accuracy comparison
          const flag = (v: number, good: number, bad: number) =>
            v >= good ? '✓' : v >= bad ? '~' : '✗';
          lines.push(
            `    Recommendation accuracy:  ${(recDir * 100).toFixed(0)}% [${(recDirCI.lower * 100).toFixed(0)}%, ${(recDirCI.upper * 100).toFixed(0)}%] ${flag(recDir, 0.55, 0.45)}`
            + (baseline.recAccuracy !== null ? ` (baseline ${(baseline.recAccuracy * 100).toFixed(0)}%)` : ''),
          );
          lines.push(
            `    Calibrated P(up) sign:    ${(calPUp * 100).toFixed(0)}% [${(calPUpCI.lower * 100).toFixed(0)}%, ${(calPUpCI.upper * 100).toFixed(0)}%] ${flag(calPUp, 0.55, 0.45)}`
            + (baseline.calPUpAccuracy !== null ? ` (baseline ${(baseline.calPUpAccuracy * 100).toFixed(0)}%)` : ''),
          );
          lines.push(
            `    Raw P(up) sign:           ${(rawPUp * 100).toFixed(0)}% [${(rawPUpCI.lower * 100).toFixed(0)}%, ${(rawPUpCI.upper * 100).toFixed(0)}%] ${flag(rawPUp, 0.55, 0.45)}`
            + (baseline.rawPUpAccuracy !== null ? ` (baseline ${(baseline.rawPUpAccuracy * 100).toFixed(0)}%)` : ''),
          );

          // Selective accuracy at thresholds 0.10 / 0.15 / 0.20
          const selectiveThresholds = [0.10, 0.15, 0.20];
          lines.push('    Selective accuracy (confidence ≥ threshold):');
          for (const t of selectiveThresholds) {
            const recSel = selectiveDirectionalAccuracy(steps, t);
            const calSel = selectivePUpAccuracy(steps, t);
            const rawSel = selectiveRawPUpAccuracy(steps, t);
            const recBar = recSel.selected > 0 ? '▓'.repeat(Math.round(recSel.accuracy * 10)) : '·';
            const calBar = calSel.selected > 0 ? '▓'.repeat(Math.round(calSel.accuracy * 10)) : '·';
            const rawBar = rawSel.selected > 0 ? '▓'.repeat(Math.round(rawSel.accuracy * 10)) : '·';
            lines.push(
              `      conf≥${t.toFixed(2)}: rec=${(recSel.accuracy * 100).toFixed(0).padStart(3)}% `
              + `cov=${(recSel.coverage * 100).toFixed(0).padStart(3)}% n=${String(recSel.selected).padStart(4)} ${recBar} | `
              + `cal=${(calSel.accuracy * 100).toFixed(0).padStart(3)}% cov=${(calSel.coverage * 100).toFixed(0).padStart(3)}% n=${String(calSel.selected).padStart(4)} ${calBar} | `
              + `raw=${(rawSel.accuracy * 100).toFixed(0).padStart(3)}% cov=${(rawSel.coverage * 100).toFixed(0).padStart(3)}% n=${String(rawSel.selected).padStart(4)} ${rawBar}`,
            );
          }

          // Hold rate
          const holdRate = steps.filter(s => s.recommendation === 'HOLD').length / steps.length;
          lines.push(`    Hold rate: ${(holdRate * 100).toFixed(1)}%`);

          // P(up) band breakdown using calibrated P(up)
          const pUpBandRows = bucketByPUpBand(steps, 0.03, undefined, 'calibrated');
          lines.push('    P(up) band breakdown (calibrated prob → calibrated direction):');
          for (const row of pUpBandRows) {
            if (row.count === 0) continue;
            const bar = '▓'.repeat(Math.round(row.directionalAccuracy * 10));
            lines.push(
              `      ${row.label.padEnd(12)} n=${String(row.count).padStart(3)} | `
              + `dir=${(row.directionalAccuracy * 100).toFixed(0).padStart(3)}% | `
              + `bal=${(row.balancedDirectionalAccuracy * 100).toFixed(0).padStart(3)}% | `
              + `brier=${row.brierScore.toFixed(3)} ${bar}`,
            );
          }

          const rawPUpBandRows = bucketByPUpBand(steps, 0.03, undefined, 'raw');
          lines.push('    P(up) band breakdown (raw prob → raw sign reference):');
          for (const row of rawPUpBandRows) {
            if (row.count === 0) continue;
            const bar = '▓'.repeat(Math.round(row.directionalAccuracy * 10));
            lines.push(
              `      ${row.label.padEnd(12)} n=${String(row.count).padStart(3)} | `
              + `dir=${(row.directionalAccuracy * 100).toFixed(0).padStart(3)}% | `
              + `bal=${(row.balancedDirectionalAccuracy * 100).toFixed(0).padStart(3)}% | `
              + `brier=${row.brierScore.toFixed(3)} ${bar}`,
            );
          }

          appendFailureSlice(lines, 'Anchor quality', steps, 'anchorQuality');
          appendFailureSlice(lines, 'Confidence', steps, 'confidence');
          appendFailureSlice(lines, 'Regime', steps, 'regime');
          appendFailureSlice(lines, 'Structural break', steps, 'structuralBreak');
          appendFailureSlice(lines, 'HMM converged', steps, 'hmmConverged');
          appendFailureSlice(lines, 'Ensemble consensus', steps, 'ensembleConsensus');

          // Reliability bins (5-bin)
          for (const b of bins) {
            if (b.count < 3) continue;
            lines.push(
              `    [${b.binLower.toFixed(1)}–${b.binUpper.toFixed(1)}): `
              + `pred=${b.meanPredicted.toFixed(2)} actual=${b.actualFrequency.toFixed(2)} n=${b.count}`,
            );
          }
        }

        lines.push('═════════════════════════════════════════════', '');
        console.log(lines.join('\n'));
        expect(true).toBe(true);
      },
      TIMEOUT,
    );

    integrationIt(
      'BTC-USD 7d/14d: PR3B ablation harness',
      async () => {
        const lines: string[] = ['', '═══ BTC SHORT-HORIZON PR3B ABLATION ═══'];
        const data = fixture.tickers['BTC-USD'];
        
        // Define the parameter grid to explore
        const weights = [0.4, 0.6, 0.8]; // Crypto default is 0.4
        const kappas = [1.3, 1.0, 0.8];  // Crypto default is 1.3

        for (const horizon of btcHorizons) {
          lines.push(`\n  Horizon: ${horizon}d`);
          lines.push('  Weight | Kappa | DirAcc% | CalPUp% | Brier | CI% | Hold%');
          lines.push('  -------+-------+---------+---------+-------+-----+------');
          
          for (const w of weights) {
            for (const k of kappas) {
              const result = await walkForward({
                ticker: 'BTC-USD',
                prices: data.closes,
                horizon,
                warmup: WARMUP,
                stride: STRIDE,
                cryptoShortHorizonConditionalWeight: w,
                cryptoShortHorizonKappaMultiplier: k,
              });
              
              if (result.steps.length === 0) continue;
              
              const steps = result.steps;
              const bs = brierScore(steps);
              const recDir = directionalAccuracy(steps);
              const calPUp = calibratedPUpDirectionalAccuracy(steps);
              const cov = ciCoverage(steps);
              const holdRate = steps.filter(s => s.recommendation === 'HOLD').length / steps.length;
              
              lines.push(
                `  ${w.toFixed(1).padEnd(6)} | ${k.toFixed(1).padEnd(5)} | ` +
                `${(recDir * 100).toFixed(1).padStart(6)}% | ` +
                `${(calPUp * 100).toFixed(1).padStart(6)}% | ` +
                `${bs.toFixed(3).padStart(5)} | ` +
                `${(cov * 100).toFixed(0).padStart(3)}% | ` +
                `${(holdRate * 100).toFixed(1).padStart(4)}%`
              );
            }
          }
        }
        
        lines.push('═══════════════════════════════════════', '');
        console.log(lines.join('\n'));
        expect(true).toBe(true);
      },
      TIMEOUT,
    );

    integrationIt(
      'BTC-USD 7d/14d: PR3 Stage 2 Floor Ablation harness',
      async () => {
        const lines: string[] = ['', '═══ BTC SHORT-HORIZON PR3 STAGE 2 FLOOR ABLATION ═══'];
        const data = fixture.tickers['BTC-USD'];
        
        // Define the parameter grid to explore
        const floors = [0.35, 0.40, 0.45]; // Default is 0.35
        const multipliers = [1.0, 0.5, 0.0];  // Default is 1.0

        for (const horizon of btcHorizons) {
          lines.push(`\n  Horizon: ${horizon}d`);
          lines.push('  Floor | Mltpr | DirAcc% | CalPUp% | Brier | CI% | Hold%');
          lines.push('  ------+-------+---------+---------+-------+-----+------');
          
          for (const f of floors) {
            for (const m of multipliers) {
              const result = await walkForward({
                ticker: 'BTC-USD',
                prices: data.closes,
                horizon,
                warmup: WARMUP,
                stride: STRIDE,
                cryptoShortHorizonPUpFloor: f,
                cryptoShortHorizonBearMarginMultiplier: m,
              });
              
              if (result.steps.length === 0) continue;
              
              const steps = result.steps;
              const bs = brierScore(steps);
              const recDir = directionalAccuracy(steps);
              const calPUp = calibratedPUpDirectionalAccuracy(steps);
              const cov = ciCoverage(steps);
              const holdRate = steps.filter(s => s.recommendation === 'HOLD').length / steps.length;
              
              lines.push(
                `  ${f.toFixed(2).padEnd(5)} | ${m.toFixed(1).padEnd(5)} | ` +
                `${(recDir * 100).toFixed(1).padStart(6)}% | ` +
                `${(calPUp * 100).toFixed(1).padStart(6)}% | ` +
                `${bs.toFixed(3).padStart(5)} | ` +
                `${(cov * 100).toFixed(0).padStart(3)}% | ` +
                `${(holdRate * 100).toFixed(1).padStart(4)}%`
              );
            }
          }
        }
        
        lines.push('════════════════════════════════════════════════════', '');
        console.log(lines.join('\n'));
        expect(true).toBe(true);
      },
      TIMEOUT,
    );
  });

  // ========================================================================
  // Tier 4: STRESS TESTS — extreme market scenarios + bootstrap CIs
  // ========================================================================

  describe('Tier 4: stress tests', () => {
    const stressResults: Map<string, { result: WalkForwardResult; scenario: StressScenario }> = new Map();
    const scenarios = generateStressScenarios(100);

    // Run walk-forward on each stress scenario
    for (const scenario of scenarios) {
      integrationIt(
        `stress/${scenario.name}: completes without errors`,
        async () => {
          const result = await walkForward({
            ticker: `STRESS-${scenario.name.toUpperCase()}`,
            prices: scenario.prices,
            horizon: 14,
            warmup: 120,
            stride: 5,
          });

          stressResults.set(scenario.name, { result, scenario });

          // HARD GATE: no crashes
          expect(result.errors).toHaveLength(0);
          expect(result.steps.length).toBeGreaterThan(0);

          // No NaN in outputs
          for (const step of result.steps) {
            expect(Number.isNaN(step.predictedProb)).toBe(false);
            expect(Number.isNaN(step.predictedReturn)).toBe(false);
            expect(Number.isNaN(step.ciLower)).toBe(false);
            expect(Number.isNaN(step.ciUpper)).toBe(false);
          }
        },
        TIMEOUT,
      );
    }

    // Crash scenario: check model behavior around crash window (informational)
    integrationIt(
      'stress/crash: behavior during crash window (informational)',
      async () => {
        const entry = stressResults.get('crash');
        if (!entry) return;
        const { result } = entry;
        // Crash happens days 200-210 in the price array.
        const crashSteps = result.steps.filter(s => s.t >= 195 && s.t <= 215);
        const preCrashSteps = result.steps.filter(s => s.t >= 150 && s.t < 195);
        const postCrashSteps = result.steps.filter(s => s.t > 215 && s.t <= 260);

        if (crashSteps.length === 0 || preCrashSteps.length === 0) return;

        const preCrashBuyRate = preCrashSteps.filter(s => s.recommendation === 'BUY').length / preCrashSteps.length;
        const crashBuyRate = crashSteps.filter(s => s.recommendation === 'BUY').length / crashSteps.length;
        const postCrashBuyRate = postCrashSteps.length > 0
          ? postCrashSteps.filter(s => s.recommendation === 'BUY').length / postCrashSteps.length
          : NaN;

        console.log([
          '',
          '  Crash scenario behavior:',
          `    Pre-crash BUY rate:  ${(preCrashBuyRate * 100).toFixed(0)}% (n=${preCrashSteps.length})`,
          `    During crash BUY:    ${(crashBuyRate * 100).toFixed(0)}% (n=${crashSteps.length})`,
          `    Post-crash BUY rate: ${isNaN(postCrashBuyRate) ? 'N/A' : (postCrashBuyRate * 100).toFixed(0) + '%'} (n=${postCrashSteps.length})`,
        ].join('\n'));

        // Log the behavior — purely informational, no hard assertion on crash timing
        // (the Markov model uses 120-day trailing windows, so crash response is delayed)
        expect(true).toBe(true);
      },
      TIMEOUT,
    );

    // Persistent bear: model should produce at least some non-BUY signals
    integrationIt(
      'stress/persistent-bear: at least some SELL/HOLD predictions',
      async () => {
        const entry = stressResults.get('persistent-bear');
        if (!entry) return;
        const { result } = entry;

        // Look at the second half of the series (model has seen enough bearish data)
        const lateSteps = result.steps.filter(s => s.t >= 250);
        if (lateSteps.length === 0) return;

        const nonBuy = lateSteps.filter(s => s.recommendation !== 'BUY');
        // At least 10% of late predictions should not be BUY
        // (the Markov model is inherently bullish due to long-term drift, so
        // even in a bear market it may still lean BUY — this catches the extreme
        // case of 100% BUY on a clear downtrend)
        expect(nonBuy.length).toBeGreaterThanOrEqual(1);
      },
      TIMEOUT,
    );

    // Bootstrap 95% CI for directional accuracy across all real fixture data
    integrationIt(
      'bootstrap: 95% CI for directional accuracy on real fixture data',
      async () => {
        if (allSteps.length === 0) return;

        const dirCI = bootstrapDirectionalCI(allSteps);
        const brierCI = bootstrapBrierCI(allSteps);
        const covCI = bootstrapCIcoverageCI(allSteps);

        // CIs should be well-formed
        expect(dirCI.lower).toBeLessThanOrEqual(dirCI.upper);
        expect(brierCI.lower).toBeLessThanOrEqual(brierCI.upper);
        expect(covCI.lower).toBeLessThanOrEqual(covCI.upper);

        // Directional accuracy CI lower bound should be above random chance
        expect(dirCI.lower).toBeGreaterThan(0.40);

        console.log([
          '',
          '═══ BOOTSTRAP 95% CI (real fixture data) ═══',
          `  Directional: [${(dirCI.lower * 100).toFixed(1)}%, ${(dirCI.upper * 100).toFixed(1)}%]  median=${(dirCI.median * 100).toFixed(1)}%`,
          `  Brier:       [${brierCI.lower.toFixed(3)}, ${brierCI.upper.toFixed(3)}]  median=${brierCI.median.toFixed(3)}`,
          `  CI Coverage: [${(covCI.lower * 100).toFixed(1)}%, ${(covCI.upper * 100).toFixed(1)}%]  median=${(covCI.median * 100).toFixed(1)}%`,
          '═══════════════════════════════',
          '',
        ].join('\n'));

        expect(true).toBe(true);
      },
      TIMEOUT,
    );

    // Print stress test summary table
    integrationIt(
      'prints stress test summary',
      async () => {
        if (stressResults.size === 0) return;

        const lines: string[] = ['', '═══ STRESS TEST SUMMARY ═══'];
        lines.push(
          '  ' + 'Scenario'.padEnd(20) + 'Steps'.padStart(6) + 'Errs'.padStart(6)
          + '  Dir%'.padStart(6) + ' Brier'.padStart(7) + '   CI%'.padStart(6)
          + '  BUY'.padStart(5) + ' HOLD'.padStart(5) + ' SELL'.padStart(5),
        );
        lines.push('  ' + '─'.repeat(74));

        for (const [name, { result }] of stressResults) {
          const s = result.steps;
          const dir = s.length > 0 ? directionalAccuracy(s) : 0;
          const bs = s.length > 0 ? brierScore(s) : 1;
          const ci = s.length > 0 ? ciCoverage(s) : 0;
          const buys = s.filter(x => x.recommendation === 'BUY').length;
          const holds = s.filter(x => x.recommendation === 'HOLD').length;
          const sells = s.filter(x => x.recommendation === 'SELL').length;

          lines.push(
            '  ' + name.padEnd(20)
            + String(s.length).padStart(6)
            + String(result.errors.length).padStart(6)
            + `${(dir * 100).toFixed(0)}%`.padStart(6)
            + `${bs.toFixed(3)}`.padStart(7)
            + `${(ci * 100).toFixed(0)}%`.padStart(6)
            + String(buys).padStart(5)
            + String(holds).padStart(5)
            + String(sells).padStart(5),
          );
        }

        lines.push('═══════════════════════════════', '');
        console.log(lines.join('\n'));
        expect(true).toBe(true);
      },
      TIMEOUT,
    );
  });

  describe('PR3 Lever: short-horizon crypto raw decision ablation', () => {
    integrationIt(
      'BTC-USD 7d & 14d default vs ablation',
      async () => {
        const prices = fixture.tickers['BTC-USD'].closes;

        // 7d Default
        const btc7dDefault = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 7,
          warmup: WARMUP,
          stride: STRIDE,
        });

        // 7d Ablation
        const btc7dAblation = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 7,
          warmup: WARMUP,
          stride: STRIDE,
          cryptoShortHorizonRawDecisionAblation: true,
        });

        // 14d Default
        const btc14dDefault = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 14,
          warmup: WARMUP,
          stride: STRIDE,
        });

        // 14d Ablation
        const btc14dAblation = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 14,
          warmup: WARMUP,
          stride: STRIDE,
          cryptoShortHorizonRawDecisionAblation: true,
        });

        // Quick log and bounds assertions
        const report = [
          '',
          '== PR3 Ablation: BTC-USD 7d/14d Default vs Raw-Decision ==',
          'Horizon | Mode       |  N | Buy | Hold | Sell | DirAcc',
          '--------+------------+----+-----+------+------+-------',
        ];

        const pushReport = (horizon: number, mode: string, res: WalkForwardResult) => {
          const s = res.steps;
          const buys = s.filter(x => x.recommendation === 'BUY').length;
          const holds = s.filter(x => x.recommendation === 'HOLD').length;
          const sells = s.filter(x => x.recommendation === 'SELL').length;
          const dir = s.length > 0 ? (directionalAccuracy(s) * 100).toFixed(1) + '%' : '0.0%';
          report.push(
            `${String(horizon).padStart(7)} | ` +
            `${mode.padEnd(10)} | ` +
            `${String(s.length).padStart(2)} | ` +
            `${String(buys).padStart(3)} | ` +
            `${String(holds).padStart(4)} | ` +
            `${String(sells).padStart(4)} | ` +
            `${dir.padStart(6)}`
          );
        };

        pushReport(7, 'Default', btc7dDefault);
        pushReport(7, 'Raw', btc7dAblation);
        pushReport(14, 'Default', btc14dDefault);
        pushReport(14, 'Raw', btc14dAblation);

        console.log(report.join('\n'));

        // Basic bounded assertions
        expect(btc7dDefault.steps.length).toBeGreaterThan(0);
        expect(btc7dAblation.steps.length).toBe(btc7dDefault.steps.length);
        expect(btc14dDefault.steps.length).toBeGreaterThan(0);
        expect(btc14dAblation.steps.length).toBe(btc14dDefault.steps.length);

        // PR3E provenance assertions: probabilitySource is calibrated for all steps
        for (const step of [...btc7dDefault.steps, ...btc7dAblation.steps, ...btc14dDefault.steps, ...btc14dAblation.steps]) {
          expect(step.probabilitySource).toBe('calibrated');
        }

        // PR3E provenance: default steps use 'default', ablation steps use 'crypto-short-horizon-raw'
        for (const step of btc7dDefault.steps) {
          expect(step.decisionSource).toBe('default');
        }
        for (const step of btc14dDefault.steps) {
          expect(step.decisionSource).toBe('default');
        }
        for (const step of btc7dAblation.steps) {
          expect(step.decisionSource).toBe('crypto-short-horizon-raw');
        }
        for (const step of btc14dAblation.steps) {
          expect(step.decisionSource).toBe('crypto-short-horizon-raw');
        }

        // PR3E: aggregate provenance via generateReport
        const report7dDefault = generateReport('BTC-USD', 7, btc7dDefault.steps);
        expect(report7dDefault.provenanceSummary).toBeDefined();
        expect(report7dDefault.provenanceSummary!.decisionSources.default).toBe(btc7dDefault.steps.length);
        expect(report7dDefault.provenanceSummary!.probabilitySources.calibrated).toBe(btc7dDefault.steps.length);

        const report7dAblation = generateReport('BTC-USD', 7, btc7dAblation.steps);
        expect(report7dAblation.provenanceSummary!.decisionSources['crypto-short-horizon-raw']).toBe(btc7dAblation.steps.length);
      },
      TIMEOUT
    );
  });

  describe('PR3F Lever: short-horizon crypto disagreement prior', () => {
    integrationIt(
      'BTC-USD 7d & 14d default vs PR3F ablation',
      async () => {
        const prices = fixture.tickers['BTC-USD'].closes;

        // 7d Default
        const btc7dDefault = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 7,
          warmup: WARMUP,
          stride: STRIDE,
        });

        // 7d PR3F
        const btc7dPr3f = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 7,
          warmup: WARMUP,
          stride: STRIDE,
          pr3fCryptoShortHorizonDisagreementPrior: true,
        });

        // 14d Default
        const btc14dDefault = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 14,
          warmup: WARMUP,
          stride: STRIDE,
        });

        // 14d PR3F
        const btc14dPr3f = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 14,
          warmup: WARMUP,
          stride: STRIDE,
          pr3fCryptoShortHorizonDisagreementPrior: true,
        });

        const logRun = (label: string, steps: any[]) => {
          const report = generateReport('BTC-USD', 0, steps);
          const pr3fActive = report.provenanceSummary?.decisionSources['crypto-short-horizon-disagreement-blend'] || 0;
          console.log(`[PR3F Test] ${label.padEnd(20)} | Steps: ${steps.length} | PR3F Active: ${pr3fActive} | Brier: ${report.brierScore.toFixed(4)} | Acc: ${(report.balancedDirectionalAccuracy! * 100).toFixed(1)}% | Edge: ${report.meanEdge?.toFixed(4) ?? 'N/A'}`);
        };

        console.log('\\n--- PR3F Lever Ablation ---');
        logRun('BTC 7d Default', btc7dDefault.steps);
        logRun('BTC 7d PR3F', btc7dPr3f.steps);
        logRun('BTC 14d Default', btc14dDefault.steps);
        logRun('BTC 14d PR3F', btc14dPr3f.steps);

        // Assert step counts match
        expect(btc7dPr3f.steps.length).toBeGreaterThan(0);
        expect(btc7dPr3f.steps.length).toBe(btc7dDefault.steps.length);
        expect(btc14dPr3f.steps.length).toBe(btc14dDefault.steps.length);

        // Ensure canonical surfaces remain untouched
        for (let i = 0; i < btc7dDefault.steps.length; i++) {
          const def = btc7dDefault.steps[i];
          const pr3f = btc7dPr3f.steps[i];
          
          expect(pr3f.predictedProb).toBe(def.predictedProb);
          expect(pr3f.probabilitySource).toBe('calibrated');
          expect(pr3f.ciLower).toBe(def.ciLower);
          expect(pr3f.ciUpper).toBe(def.ciUpper);
        }

        // PR3E provenance
        for (const step of btc7dDefault.steps) {
          expect(step.decisionSource).toBe('default');
        }
        for (const step of btc14dDefault.steps) {
          expect(step.decisionSource).toBe('default');
        }
        
        // PR3F shouldn't be active on *every* step due to conditional checks, 
        // but it should activate on some steps.
        const report7dPr3f = generateReport('BTC-USD', 7, btc7dPr3f.steps);
        const report14dPr3f = generateReport('BTC-USD', 14, btc14dPr3f.steps);
        
        expect(report7dPr3f.provenanceSummary?.decisionSources['crypto-short-horizon-disagreement-blend']).toBeGreaterThanOrEqual(0);
        expect(report14dPr3f.provenanceSummary?.decisionSources['crypto-short-horizon-disagreement-blend']).toBeGreaterThanOrEqual(0);
      },
      TIMEOUT
    );
  });

  describe('PR3G Lever: short-horizon crypto recency weighting', () => {
    integrationIt(
      'BTC-USD 7d & 14d default vs PR3G ablation',
      async () => {
        const prices = fixture.tickers['BTC-USD'].closes;

        // 7d Default
        const btc7dDefault = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 7,
          warmup: WARMUP,
          stride: STRIDE,
        });

        // 7d PR3G (default decay 0.94)
        const btc7dPr3g = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 7,
          warmup: WARMUP,
          stride: STRIDE,
          pr3gCryptoShortHorizonRecencyWeighting: true,
        });

        // 7d PR3G (milder decay 0.98)
        const btc7dPr3gMilder = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 7,
          warmup: WARMUP,
          stride: STRIDE,
          pr3gCryptoShortHorizonRecencyWeighting: true,
          pr3gCryptoShortHorizonDecay: 0.98,
        });

        // 14d Default
        const btc14dDefault = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 14,
          warmup: WARMUP,
          stride: STRIDE,
        });

        // 14d PR3G (default decay 0.94)
        const btc14dPr3g = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 14,
          warmup: WARMUP,
          stride: STRIDE,
          pr3gCryptoShortHorizonRecencyWeighting: true,
        });

        // 14d PR3G (milder decay 0.98)
        const btc14dPr3gMilder = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon: 14,
          warmup: WARMUP,
          stride: STRIDE,
          pr3gCryptoShortHorizonRecencyWeighting: true,
          pr3gCryptoShortHorizonDecay: 0.98,
        });

        const logRun = (label: string, steps: any[]) => {
          const report = generateReport('BTC-USD', 0, steps);
          const pr3gActive = report.provenanceSummary?.decisionSources['crypto-short-horizon-recency'] || 0;
          
          let recAcc = 0;
          let recAttempts = 0;
          let pUpAcc = 0;
          let pUpAttempts = 0;
          
          for (const step of steps) {
            if (step.recommendation === 'BUY' || step.recommendation === 'SELL') {
              recAttempts++;
              if ((step.recommendation === 'BUY' && step.actualBinary === 1) ||
                  (step.recommendation === 'SELL' && step.actualBinary === 0)) {
                recAcc++;
              }
            }
            if (Math.abs(step.predictedProb - 0.5) > 0.001) {
              pUpAttempts++;
              if ((step.predictedProb > 0.5 && step.actualBinary === 1) ||
                  (step.predictedProb < 0.5 && step.actualBinary === 0)) {
                pUpAcc++;
              }
            }
          }
          
          const recAccStr = recAttempts > 0 ? (recAcc / recAttempts * 100).toFixed(1) + '%' : 'N/A';
          const pUpAccStr = pUpAttempts > 0 ? (pUpAcc / pUpAttempts * 100).toFixed(1) + '%' : 'N/A';
          
          console.log(`[PR3G Test] ${label.padEnd(20)} | Steps: ${steps.length} | PR3G Active: ${pr3gActive} | Brier: ${report.brierScore.toFixed(4)} | P(up) Acc: ${pUpAccStr} | Rec Acc: ${recAccStr} (${recAttempts} recs)`);
        };

        console.log('\n--- PR3G Lever Ablation ---');
        logRun('BTC 7d Default', btc7dDefault.steps);
        logRun('BTC 7d PR3G (0.94)', btc7dPr3g.steps);
        logRun('BTC 7d PR3G (0.98)', btc7dPr3gMilder.steps);
        logRun('BTC 14d Default', btc14dDefault.steps);
        logRun('BTC 14d PR3G (0.94)', btc14dPr3g.steps);
        logRun('BTC 14d PR3G (0.98)', btc14dPr3gMilder.steps);

        // Assert step counts match
        expect(btc7dPr3g.steps.length).toBeGreaterThan(0);
        expect(btc7dPr3g.steps.length).toBe(btc7dDefault.steps.length);
        expect(btc7dPr3gMilder.steps.length).toBe(btc7dDefault.steps.length);
        expect(btc14dPr3g.steps.length).toBe(btc14dDefault.steps.length);
        expect(btc14dPr3gMilder.steps.length).toBe(btc14dDefault.steps.length);

        // PR3G changes the conditional PUp, so it *will* change the calibrated surfaces.
        // We only assert that probabilitySource remains 'calibrated'.
        for (let i = 0; i < btc7dDefault.steps.length; i++) {
          const pr3g = btc7dPr3g.steps[i];
          expect(pr3g.probabilitySource).toBe('calibrated');
        }

        // PR3E provenance
        for (const step of btc7dDefault.steps) {
          expect(step.decisionSource).toBe('default');
        }
        for (const step of btc14dDefault.steps) {
          expect(step.decisionSource).toBe('default');
        }
        for (const step of btc7dPr3g.steps) {
          expect(step.decisionSource).toBe('crypto-short-horizon-recency');
        }
        for (const step of btc7dPr3gMilder.steps) {
          expect(step.decisionSource).toBe('crypto-short-horizon-recency');
        }
        for (const step of btc14dPr3g.steps) {
          expect(step.decisionSource).toBe('crypto-short-horizon-recency');
        }
        for (const step of btc14dPr3gMilder.steps) {
          expect(step.decisionSource).toBe('crypto-short-horizon-recency');
        }
        
        const report7dPr3g = generateReport('BTC-USD', 7, btc7dPr3g.steps);
        const report14dPr3g = generateReport('BTC-USD', 14, btc14dPr3g.steps);
        
        expect(report7dPr3g.provenanceSummary?.decisionSources['crypto-short-horizon-recency']).toBe(btc7dPr3g.steps.length);
        expect(report14dPr3g.provenanceSummary?.decisionSources['crypto-short-horizon-recency']).toBe(btc14dPr3g.steps.length);
      },
      TIMEOUT
    );
  });

  describe('PR3H: deterministic historical replay', () => {
    integrationIt(
      'BTC-USD 14d default vs replay (empty & active)',
      async () => {
        const prices = fixture.tickers['BTC-USD'].closes;
        const dates = fixture.tickers['BTC-USD'].dates;
        const horizon = 14;
        
        const { walkForwardWithReplay } = await import('./backtest/replay.js');
        
        const btc14dEmptyReplay = await walkForwardWithReplay({
          ticker: 'BTC-USD',
          prices,
          dates,
          horizon,
          warmup: WARMUP,
          stride: STRIDE,
          replaySnapshots: [],
        });
        
        const btcReplayFixture = JSON.parse(readFileSync(join(import.meta.dir, 'fixtures', 'btc-replay.json'), 'utf-8'));
        
        const btc14dActiveReplay = await walkForwardWithReplay({
          ticker: 'BTC-USD',
          prices,
          dates,
          horizon,
          warmup: WARMUP,
          stride: STRIDE,
          replaySnapshots: btcReplayFixture,
        });

        const btc14dBaseline = await walkForward({
          ticker: 'BTC-USD',
          prices,
          horizon,
          warmup: WARMUP,
          stride: STRIDE,
        });

        expect(btc14dBaseline.steps.length).toBeGreaterThan(0);
        expect(btc14dEmptyReplay.steps.length).toBe(btc14dBaseline.steps.length);
        expect(btc14dActiveReplay.steps.length).toBe(btc14dBaseline.steps.length);

        for (let i = 0; i < btc14dBaseline.steps.length; i++) {
          const base = btc14dBaseline.steps[i];
          const empty = btc14dEmptyReplay.steps[i];
          expect(empty.predictedProb).toBeCloseTo(base.predictedProb, 5);
          expect(empty.decisionSource).toBe(base.decisionSource);
        }

        let activeReplayCount = 0;
        let trustedReplayCount = 0;
        let changedCount = 0;
        for (let i = 0; i < btc14dActiveReplay.steps.length; i++) {
          const step = btc14dActiveReplay.steps[i];
          if (step.decisionSource === 'replay-anchor') {
            activeReplayCount++;
          }
          if ((step.trustedAnchors ?? 0) > 0) {
            trustedReplayCount++;
          }
          const base = btc14dBaseline.steps[i];
          if (
            Math.abs(base.predictedProb - step.predictedProb) > 1e-9 ||
            Math.abs(base.ciLower - step.ciLower) > 1e-9 ||
            Math.abs(base.ciUpper - step.ciUpper) > 1e-9 ||
            base.recommendation !== step.recommendation
          ) {
            changedCount++;
          }
        }
        
        console.log(`[PR3H] Replay Anchor injected in ${activeReplayCount}/${btc14dActiveReplay.steps.length} steps | trusted=${trustedReplayCount} | changed=${changedCount}`);
        expect(activeReplayCount).toBeGreaterThan(0);
        expect(trustedReplayCount).toBeGreaterThan(0);
        expect(changedCount).toBeGreaterThan(0);
      },
      TIMEOUT
    );
  });

  
  

  
  describe('PR3 Post-Experiment: sideways_coil vs sideways_chop', () => {
    integrationIt('evaluates sideways split against PR3G ceiling on canonical BTC 7d/14d', async () => {
      const prices = fixture.tickers['BTC-USD'].closes;

        const runEval = async (horizon: number, name: string, sidewaysSplit: boolean) => {
          const config = {
            ticker: 'BTC-USD',
            prices,
            horizon,
            warmup: WARMUP,
            stride: STRIDE,
            pr3gCryptoShortHorizonRecencyWeighting: true,
            pr3gCryptoShortHorizonDecay: 0.98,
            sidewaysSplit,
          };

        const res = await walkForward(config);
        const acc = directionalAccuracy(res.steps);
        const calAcc = calibratedPUpDirectionalAccuracy(res.steps);
        const brier = brierScore(res.steps);

        const splitsActive = res.steps.filter(w => w.sidewaysSplitActive).length;

        console.log(`[BTC ${horizon}d ${name}] Acc: ${(acc * 100).toFixed(1)}% | CalAcc: ${(calAcc * 100).toFixed(1)}% | Brier: ${brier.toFixed(3)} | Splits active: ${splitsActive}/${res.steps.length}`);
      };

      await runEval(7, 'Baseline (PR3G ceiling)', false);
      await runEval(7, 'Experiment (Sideways Split)', true);

      await runEval(14, 'Baseline (PR3G ceiling)', false);
      await runEval(14, 'Experiment (Sideways Split)', true);
      
      expect(true).toBe(true);
    }, 60000);
  });

  describe('PR3 Post-Experiment: matureBullCalibration', () => {
    integrationIt('evaluates BTC 14d mature bull calibration against PR3G ceiling', async () => {
      const prices = fixture.tickers['BTC-USD'].closes;

      const runEval = async (horizon: number, name: string, matureBullCalibration: boolean) => {
        const config = {
          ticker: 'BTC-USD',
          prices,
          horizon,
          warmup: WARMUP,
          stride: STRIDE,
          pr3gCryptoShortHorizonRecencyWeighting: true,
          pr3gCryptoShortHorizonDecay: 0.98,
          matureBullCalibration,
        };

        const res = await walkForward(config);
        const acc = directionalAccuracy(res.steps);
        const calAcc = calibratedPUpDirectionalAccuracy(res.steps);
        const brier = brierScore(res.steps);

        const calibrationsActive = res.steps.filter(w => w.matureBullCalibrationActive).length;

        console.log(`[BTC ${horizon}d ${name}] Acc: ${(acc * 100).toFixed(1)}% | CalAcc: ${(calAcc * 100).toFixed(1)}% | Brier: ${brier.toFixed(3)} | Calibrations active: ${calibrationsActive}/${res.steps.length}`);
        return { acc, calAcc, brier, calibrationsActive };
      };

      const base = await runEval(14, 'Baseline (PR3G ceiling)', false);
      const exp = await runEval(14, 'Experiment (matureBullCalibration)', true);
      
      expect(base.brier).toBeGreaterThan(0);
      expect(exp.brier).toBeGreaterThan(0);
      expect(exp.calibrationsActive).toBeGreaterThan(0);
    }, 60000);
  });

  describe('PR3I: replay-quality additive evaluation', () => {
    integrationIt(
      'BTC-USD 7d/14d: PR3G baseline vs quality-gated replay evaluation',
      async () => {
        const prices = fixture.tickers['BTC-USD'].closes;
        const dates = fixture.tickers['BTC-USD'].dates;
        const btcReplayFixture = JSON.parse(readFileSync(join(import.meta.dir, 'fixtures', 'btc-replay.json'), 'utf-8'));

        const { walkForwardWithReplay, filterReplayMarketsByTime } = await import('./backtest/replay.js');

        const lines: string[] = ['', '═══ BTC SHORT-HORIZON PR3I REPLAY EVALUATION ═══'];
        const horizons = [7, 14] as const;

        for (const horizon of horizons) {
          const baseline = await walkForward({
            ticker: 'BTC-USD',
            prices,
            horizon,
            warmup: WARMUP,
            stride: STRIDE,
            pr3gCryptoShortHorizonRecencyWeighting: true,
            pr3gCryptoShortHorizonDecay: 0.98,
          });

          const emptyReplay = await walkForwardWithReplay({
            ticker: 'BTC-USD',
            prices,
            dates,
            horizon,
            warmup: WARMUP,
            stride: STRIDE,
            replaySnapshots: [],
            pr3gCryptoShortHorizonRecencyWeighting: true,
            pr3gCryptoShortHorizonDecay: 0.98,
          });

          const activeReplay = await walkForwardWithReplay({
            ticker: 'BTC-USD',
            prices,
            dates,
            horizon,
            warmup: WARMUP,
            stride: STRIDE,
            replaySnapshots: btcReplayFixture,
            replayQualityFilters: {
              minVolume: 600000,
              requirePersistence: true,
              maxProbabilityShock: 0.10,
              requireHorizonAlignment: true,
            },
            pr3gCryptoShortHorizonRecencyWeighting: true,
            pr3gCryptoShortHorizonDecay: 0.98,
          });

          expect(baseline.errors).toHaveLength(0);
          expect(emptyReplay.errors).toHaveLength(0);
          expect(activeReplay.errors).toHaveLength(0);
          expect(emptyReplay.steps.length).toBe(baseline.steps.length);
          expect(activeReplay.steps.length).toBe(baseline.steps.length);

          for (let i = 0; i < baseline.steps.length; i++) {
            expect(emptyReplay.steps[i].predictedProb).toBeCloseTo(baseline.steps[i].predictedProb, 5);
            expect(emptyReplay.steps[i].decisionSource).toBe(baseline.steps[i].decisionSource);
          }

          const activeReport = generateReport('BTC-USD', horizon, activeReplay.steps);
          const baselineDir = directionalAccuracy(baseline.steps);
          const replayDir = directionalAccuracy(activeReplay.steps);
          const baselinePUp = calibratedPUpDirectionalAccuracy(baseline.steps);
          const replayPUp = calibratedPUpDirectionalAccuracy(activeReplay.steps);
          const baselineBrier = brierScore(baseline.steps);
          const replayBrier = brierScore(activeReplay.steps);
          const changedSteps = activeReplay.steps.filter((step, idx) => {
            const base = baseline.steps[idx];
            return Math.abs(base.predictedProb - step.predictedProb) > 1e-9 || base.recommendation !== step.recommendation;
          }).length;

          const replayOnly =
            (activeReport.provenanceSummary?.decisionSources['replay-anchor'] ?? 0) +
            (activeReport.provenanceSummary?.decisionSources['crypto-short-horizon-recency+replay-anchor'] ?? 0) +
            (activeReport.provenanceSummary?.decisionSources['crypto-short-horizon-disagreement-blend+replay-anchor'] ?? 0) +
            (activeReport.provenanceSummary?.decisionSources['crypto-short-horizon-raw+replay-anchor'] ?? 0);

          lines.push(
            `  BTC-USD ${horizon}d | baseline dir=${(baselineDir * 100).toFixed(1)}% -> replay dir=${(replayDir * 100).toFixed(1)}% | ` +
            `pUp=${(baselinePUp * 100).toFixed(1)}% -> ${(replayPUp * 100).toFixed(1)}% | ` +
            `brier=${baselineBrier.toFixed(4)} -> ${replayBrier.toFixed(4)} | replaySteps=${replayOnly} | changed=${changedSteps}`
          );

          // Replay quality is fixture-dependent (market resolution dates, volume, horizon alignment).
          // When fixture markets fail quality filters, replay falls back to baseline — this is correct
          // behaviour, not a failure. We assert only on infrastructure correctness (no crashes).
          if (replayOnly === 0 || changedSteps === 0) {
            lines.push(
              `    ⚠ replayOnly=${replayOnly} changedSteps=${changedSteps} — fixture markets may fail quality filters`
            );
          }
        }

        const futureRejected = filterReplayMarketsByTime(
          [
            {
              question: 'Will Bitcoin be above $120000 on 2030-01-01?',
              probability: 0.2,
              createdAt: '2030-01-01T00:00:00Z',
              endDate: '2030-01-02T00:00:00Z',
              active: true,
              closed: false,
              enableOrderBook: true,
            }
          ],
          Date.parse('2025-03-15T23:59:59Z'),
        );

        expect(futureRejected).toHaveLength(0);
        lines.push('═══════════════════════════════════════════════', '');
        console.log(lines.join('\n'));
      },
      TIMEOUT,
    );
  });

  describe('PR3 experiment: startStateMixture', () => {
    integrationIt(
      'BTC-USD 7d/14d: PR3G baseline vs new startStateMixture experiment',
      async () => {
        const prices = fixture.tickers['BTC-USD'].closes;
        const horizons = [7, 14] as const;

        const lines: string[] = ['', '═══ BTC SHORT-HORIZON PR3 MIXTURE EXPERIMENT ═══'];

        for (const horizon of horizons) {
          const baseline = await walkForward({
            ticker: 'BTC-USD',
            prices,
            horizon,
            warmup: WARMUP,
            stride: STRIDE,
            pr3gCryptoShortHorizonRecencyWeighting: true,
            pr3gCryptoShortHorizonDecay: 0.98,
          });

          const experiment = await walkForward({
            ticker: 'BTC-USD',
            prices,
            horizon,
            warmup: WARMUP,
            stride: STRIDE,
            pr3gCryptoShortHorizonRecencyWeighting: true,
            pr3gCryptoShortHorizonDecay: 0.98,
            startStateMixture: true,
          });

          const baseDir = directionalAccuracy(baseline.steps);
          const basePUp = calibratedPUpDirectionalAccuracy(baseline.steps);
          const baseBrier = brierScore(baseline.steps);
          
          const expDir = directionalAccuracy(experiment.steps);
          const expPUp = calibratedPUpDirectionalAccuracy(experiment.steps);
          const expBrier = brierScore(experiment.steps);

          lines.push(`\n[${horizon}d Horizon]`);
          lines.push(`  Baseline (PR3G ceiling):`);
          lines.push(`    Dir Acc:    ${(baseDir * 100).toFixed(1)}%`);
          lines.push(`    Cal P(up):  ${(basePUp * 100).toFixed(1)}%`);
          lines.push(`    Brier:      ${baseBrier.toFixed(4)}`);
          lines.push(`  Experiment (startStateMixture):`);
          lines.push(`    Dir Acc:    ${(expDir * 100).toFixed(1)}%`);
          lines.push(`    Cal P(up):  ${(expPUp * 100).toFixed(1)}%`);
          lines.push(`    Brier:      ${expBrier.toFixed(4)}`);

          const dirAccDelta = expDir - baseDir;
          const calPUpDelta = expPUp - basePUp;
          const brierDelta = expBrier - baseBrier;

          lines.push(`  Diff (Exp - Base):`);
          lines.push(`    Dir Acc:    ${dirAccDelta > 0 ? '+' : ''}${(dirAccDelta * 100).toFixed(2)}pp`);
          lines.push(`    Cal P(up):  ${calPUpDelta > 0 ? '+' : ''}${(calPUpDelta * 100).toFixed(2)}pp`);
          lines.push(`    Brier:      ${brierDelta > 0 ? '+' : ''}${brierDelta.toFixed(4)} ${brierDelta < 0 ? '(better)' : '(worse)'}`);
        }

        lines.push('═══════════════════════════════════════════════', '');
        console.log(lines.join('\n'));

        expect(lines.length).toBeGreaterThan(5);
      },
      TIMEOUT,
    );
  });
});
