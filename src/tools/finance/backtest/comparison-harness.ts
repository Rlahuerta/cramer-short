/**
 * Shared harness for BTC multi-horizon comparison backtests.
 * Provides common metric calculation, report generation, and file writing utilities.
 */

import { mkdirSync, writeFileSync } from 'fs';
import { join } from 'path';
import { walkForward, type WalkForwardConfig, type WalkForwardResult } from './walk-forward.js';
import type { BacktestStep } from './metrics.js';
import {
  brierScore,
  bootstrapDirectionalCI,
  ciCoverage,
  directionalAccuracy,
  meanEdge,
  selectiveDirectionalAccuracy,
  sharpness,
} from './metrics.js';

/**
 * Common metric block structure used across all comparison scripts.
 */
export interface MetricBlock {
  n: number;
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  meanEdge: number;
  sharpness: number;
  avgConfidence: number;
  directionalCi: { lower: number; median: number; upper: number; nResamples: number };
}

export interface ArmHorizon {
  horizon: number;
  metrics: MetricBlock;
}

export interface ArmReport {
  label: string;
  perHorizon: ArmHorizon[];
}

export interface DeltaRow {
  horizon: number;
  n: number;
  directionalAccuracyDelta: number;
  brierScoreDelta: number;
  ciCoverageDelta: number;
  meanEdgeDelta: number;
  sharpnessDelta: number;
}

export interface CalibrationSummary {
  n: number;
  ciCoverage: number;
  coverageError: number;
  breakCount: number;
  breakCiCoverage: number | null;
  breakCoverageError: number | null;
  sharpness: number;
  brierScore: number;
  directionalAccuracy: number;
  selectiveDirectionalAccuracy045: number;
  selectiveCoverage045: number;
}

export interface WalkForwardHorizonResult {
  horizon: number;
  result: WalkForwardResult;
}

/**
 * Summarizes backtest steps into a standard metric block.
 */
export function summarizeBacktestSteps(steps: BacktestStep[]): MetricBlock {
  if (steps.length === 0) {
    return {
      n: 0,
      directionalAccuracy: 0,
      brierScore: 0,
      ciCoverage: 0,
      meanEdge: 0,
      sharpness: 0,
      avgConfidence: 0,
      directionalCi: { lower: 0, median: 0, upper: 0, nResamples: 0 },
    };
  }

  const n = steps.length;
  const da = directionalAccuracy(steps);
  const bs = brierScore(steps);
  const cic = ciCoverage(steps);
  const edge = meanEdge(steps);
  const sharp = sharpness(steps);
  const avgConf = steps.reduce((sum, s) => sum + s.confidence, 0) / n;
  const dirCi = bootstrapDirectionalCI(steps, 500);

  return {
    n,
    directionalAccuracy: da,
    brierScore: bs,
    ciCoverage: cic,
    meanEdge: edge,
    sharpness: sharp,
    avgConfidence: avgConf,
    directionalCi: dirCi,
  };
}

/**
 * Summarize calibration diagnostics for harnesses that need raw coverage metrics.
 */
export function summarizeCalibrationSteps(
  steps: BacktestStep[],
  options: {
    targetCiCoverage: number;
    isBreakStep?: (step: BacktestStep) => boolean;
  },
): CalibrationSummary {
  const breakSteps = options.isBreakStep ? steps.filter(options.isBreakStep) : [];
  const selective045 = selectiveDirectionalAccuracy(steps, 0.45);
  const overallCoverage = ciCoverage(steps);
  const breakCoverage = breakSteps.length > 0 ? ciCoverage(breakSteps) : null;
  return {
    n: steps.length,
    ciCoverage: overallCoverage,
    coverageError: Math.abs(overallCoverage - options.targetCiCoverage),
    breakCount: breakSteps.length,
    breakCiCoverage: breakCoverage,
    breakCoverageError: breakCoverage === null ? null : Math.abs(breakCoverage - options.targetCiCoverage),
    sharpness: sharpness(steps),
    brierScore: brierScore(steps),
    directionalAccuracy: directionalAccuracy(steps),
    selectiveDirectionalAccuracy045: selective045.accuracy,
    selectiveCoverage045: selective045.coverage,
  };
}

/**
 * Run walk-forward across horizons and return raw results for custom summaries.
 */
export async function runWalkForwardHorizons(
  label: string,
  options: {
    ticker: string;
    prices: number[];
    horizons: readonly number[];
    warmup: number;
    stride: number;
    flags: Partial<WalkForwardConfig>;
  },
): Promise<WalkForwardHorizonResult[]> {
  const results: WalkForwardHorizonResult[] = [];
  for (const horizon of options.horizons) {
    const result = await walkForward({
      ticker: options.ticker,
      prices: options.prices,
      horizon,
      warmup: options.warmup,
      stride: options.stride,
      ...options.flags,
    });
    if (result.errors.length > 0) {
      console.error(`[${label} h=${horizon}] ${result.errors.length} errors (first: ${result.errors[0]?.error ?? 'unknown'})`);
    }
    results.push({ horizon, result });
  }
  return results;
}

/**
 * Run walk-forward for one arm across horizons and summarize each horizon.
 */
export async function runMultiHorizonArm(
  label: string,
  options: {
    ticker: string;
    prices: number[];
    horizons: readonly number[];
    warmup: number;
    stride: number;
    flags: Partial<WalkForwardConfig>;
  },
): Promise<ArmReport> {
  const perHorizon: ArmHorizon[] = [];
  for (const { horizon, result } of await runWalkForwardHorizons(label, options)) {
    const metrics = summarizeBacktestSteps(result.steps);
    perHorizon.push({ horizon, metrics });
    console.log(
      `[${label} h=${horizon}] n=${result.steps.length} dirAcc=${metrics.directionalAccuracy.toFixed(3)} brier=${metrics.brierScore.toFixed(3)} cov=${metrics.ciCoverage.toFixed(3)} edge=${metrics.meanEdge.toFixed(4)}`,
    );
  }
  return { label, perHorizon };
}

/**
 * Compute metric deltas between two multi-horizon backtest arms.
 */
export function computeMetricDelta(base: ArmReport, ext: ArmReport): DeltaRow[] {
  return base.perHorizon.map((b, i) => {
    const e = ext.perHorizon[i];
    return {
      horizon: b.horizon,
      n: b.metrics.n,
      directionalAccuracyDelta: e.metrics.directionalAccuracy - b.metrics.directionalAccuracy,
      brierScoreDelta: e.metrics.brierScore - b.metrics.brierScore,
      ciCoverageDelta: e.metrics.ciCoverage - b.metrics.ciCoverage,
      meanEdgeDelta: e.metrics.meanEdge - b.metrics.meanEdge,
      sharpnessDelta: e.metrics.sharpness - b.metrics.sharpness,
    };
  });
}

function fmtSign(x: number, digits = 3): string {
  return `${x >= 0 ? '+' : ''}${x.toFixed(digits)}`;
}

/**
 * Render the standard arm metrics table used by BTC comparison reports.
 */
export function renderArmMetricTable(arm: ArmReport): string[] {
  const lines: string[] = [];
  lines.push('| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |');
  lines.push('|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|');
  for (const row of arm.perHorizon) {
    const m = row.metrics;
    lines.push(
      `| ${row.horizon} | ${m.n} | ${m.directionalAccuracy.toFixed(3)} | [${m.directionalCi.lower.toFixed(3)}, ${m.directionalCi.upper.toFixed(3)}] | ${m.brierScore.toFixed(3)} | ${m.ciCoverage.toFixed(3)} | ${m.meanEdge.toFixed(4)} | ${m.sharpness.toFixed(4)} | ${m.avgConfidence.toFixed(3)} |`,
    );
  }
  return lines;
}

/**
 * Render the standard delta table used by BTC comparison reports.
 */
export function renderDeltaMetricTable(rows: DeltaRow[]): string[] {
  const lines: string[] = [];
  lines.push('| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |');
  lines.push('|------:|--:|--------:|-------:|----------:|----------:|-----------:|');
  for (const r of rows) {
    lines.push(
      `| ${r.horizon} | ${r.n} | ${fmtSign(r.directionalAccuracyDelta)} | ${fmtSign(r.brierScoreDelta)} | ${fmtSign(r.ciCoverageDelta)} | ${fmtSign(r.meanEdgeDelta, 4)} | ${fmtSign(r.sharpnessDelta, 4)} |`,
    );
  }
  return lines;
}

/**
 * Writes dated JSON and markdown report files under docs/.
 */
export function writeBacktestDocs(
  basename: string,
  artifact: unknown,
  markdown: string,
): { jsonPath: string; mdPath: string } {
  const outDir = join(process.cwd(), 'docs');
  mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 10);
  const jsonPath = join(outDir, `${basename}-${stamp}.json`);
  const mdPath = join(outDir, `${basename}-${stamp}.md`);
  writeFileSync(jsonPath, JSON.stringify(artifact, null, 2));
  writeFileSync(mdPath, markdown);
  console.log(`Wrote ${jsonPath}`);
  console.log(`Wrote ${mdPath}`);
  return { jsonPath, mdPath };
}

/**
 * Writes a JSON artifact to the backtest comparison directory.
 */
export function writeBacktestArtifact(filename: string, artifact: unknown): void {
  const dir = join('src', 'tools', 'finance', 'backtest', 'comparison-artifacts');
  mkdirSync(dir, { recursive: true });
  const path = join(dir, filename);
  writeFileSync(path, JSON.stringify(artifact, null, 2));
  console.log(`Wrote artifact → ${path}`);
}

/**
 * Writes a markdown report to the docs directory.
 */
export function writeBacktestReport(filename: string, markdown: string): void {
  const dir = join('docs');
  mkdirSync(dir, { recursive: true });
  const path = join(dir, filename);
  writeFileSync(path, markdown);
  console.log(`Wrote report → ${path}`);
}

/**
 * Formats a metric block as a markdown table row.
 */
export function formatMetricRow(m: MetricBlock): string {
  return [
    `n=${m.n}`,
    `DA=${(m.directionalAccuracy * 100).toFixed(1)}%`,
    `Brier=${m.brierScore.toFixed(4)}`,
    `CI-Cov=${(m.ciCoverage * 100).toFixed(1)}%`,
    `Edge=${(m.meanEdge * 100).toFixed(2)}%`,
    `Sharp=${m.sharpness.toFixed(4)}`,
  ].join(' | ');
}

/**
 * Formats delta values for comparison reports.
 */
export function formatDelta(delta: number, format: 'percent' | 'bps' | 'raw' = 'raw'): string {
  const sign = delta > 0 ? '+' : '';
  if (format === 'percent') {
    return `${sign}${(delta * 100).toFixed(2)}%`;
  } else if (format === 'bps') {
    return `${sign}${(delta * 10000).toFixed(0)} bps`;
  } else {
    return `${sign}${delta.toFixed(4)}`;
  }
}

/**
 * Generates a markdown header for comparison reports.
 */
export function generateReportHeader(
  title: string,
  ticker: string,
  config: { warmup: number; stride: number; horizons: number[] },
  fixtureRange: { startDate: string; endDate: string; days: number },
): string {
  return `# ${title}

**Ticker**: ${ticker}
**Data**: ${fixtureRange.startDate} to ${fixtureRange.endDate} (${fixtureRange.days} days)
**Config**: warmup=${config.warmup}, stride=${config.stride}, horizons=[${config.horizons.join(', ')}]
**Generated**: ${new Date().toISOString()}

`;
}
