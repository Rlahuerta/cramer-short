/**
 * Multi-horizon BTC backtest: baseline (defaults only) vs. "improved" (every
 * wired W2/W3 toggle on) across horizons 1, 2, 3, 7, 14, 30 days.
 *
 * Generates a JSON artifact and a markdown summary under docs/.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { walkForward, type WalkForwardConfig } from './walk-forward.js';
import {
  brierScore,
  bootstrapDirectionalCI,
  ciCoverage,
  directionalAccuracy,
  meanEdge,
  sharpness,
  type BacktestStep,
} from './metrics.js';

interface FixtureTickerData {
  closes: number[];
  dates: string[];
}
interface FixtureData {
  tickers: Record<string, FixtureTickerData>;
}

const TICKER = 'BTC-USD';
const HORIZONS = [1, 2, 3, 7, 14, 30] as const;
const WARMUP = 180; // ensure enough warmup for all horizons
const STRIDE = 3;   // dense sampling

interface MetricBlock {
  n: number;
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  meanEdge: number;
  sharpness: number;
  avgConfidence: number;
  directionalCi: { lower: number; median: number; upper: number; nResamples: number };
}

interface ArmHorizon {
  horizon: number;
  metrics: MetricBlock;
}

interface ArmReport {
  label: string;
  perHorizon: ArmHorizon[];
}

interface DeltaRow {
  horizon: number;
  n: number;
  directionalAccuracyDelta: number;
  brierScoreDelta: number;
  ciCoverageDelta: number;
  meanEdgeDelta: number;
  sharpnessDelta: number;
}

export interface BtcMultiHorizonArtifact {
  generatedAt: string;
  ticker: string;
  fixtureRange: { startDate: string; endDate: string; days: number };
  config: { warmup: number; stride: number; horizons: number[] };
  baseline: ArmReport;
  improved: ArmReport;
  delta: DeltaRow[];
}

function summarize(steps: BacktestStep[]): MetricBlock {
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
  return {
    n: steps.length,
    directionalAccuracy: directionalAccuracy(steps),
    brierScore: brierScore(steps),
    ciCoverage: ciCoverage(steps),
    meanEdge: meanEdge(steps),
    sharpness: sharpness(steps),
    avgConfidence: steps.reduce((s, x) => s + x.confidence, 0) / steps.length,
    directionalCi: bootstrapDirectionalCI(steps, 500, 12345),
  };
}

const baselineFlags: Partial<WalkForwardConfig> = {
  // pure defaults, no experimental toggles
};

const improvedFlags: Partial<WalkForwardConfig> = {
  sidewaysSplit: true,
  matureBullCalibration: true,
  startStateMixture: true,
  postBreakShortWindow: true,
  postBreakWindowSize: 60,
  trendPenaltyOnlyBreakConfidence: true,
  divergenceWeightedBreakConfidence: true,
  regimeSpecificSigma: true,
  pr3gCryptoShortHorizonRecencyWeighting: true,
  pr3fCryptoShortHorizonDisagreementPrior: true,
};

async function runArm(
  label: string,
  prices: number[],
  flags: Partial<WalkForwardConfig>,
): Promise<ArmReport> {
  const perHorizon: ArmHorizon[] = [];
  for (const horizon of HORIZONS) {
    const result = await walkForward({
      ticker: TICKER,
      prices,
      horizon,
      warmup: WARMUP,
      stride: STRIDE,
      ...flags,
    });
    if (result.errors.length > 0) {
      console.error(`[${label} h=${horizon}] ${result.errors.length} errors (first: ${result.errors[0].error})`);
    }
    perHorizon.push({ horizon, metrics: summarize(result.steps) });
    console.log(
      `[${label} h=${horizon}] n=${result.steps.length} dirAcc=${perHorizon[perHorizon.length - 1].metrics.directionalAccuracy.toFixed(3)} brier=${perHorizon[perHorizon.length - 1].metrics.brierScore.toFixed(3)} cov=${perHorizon[perHorizon.length - 1].metrics.ciCoverage.toFixed(3)}`,
    );
  }
  return { label, perHorizon };
}

function computeDelta(baseline: ArmReport, improved: ArmReport): DeltaRow[] {
  return baseline.perHorizon.map((b, i) => {
    const e = improved.perHorizon[i];
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

function renderMarkdown(artifact: BtcMultiHorizonArtifact): string {
  const lines: string[] = [];
  lines.push(`# BTC Multi-Horizon Backtest — Baseline vs Improved`);
  lines.push('');
  lines.push(`Generated: ${artifact.generatedAt}`);
  lines.push('');
  lines.push(`**Ticker:** ${artifact.ticker}  `);
  lines.push(`**Fixture:** ${artifact.fixtureRange.startDate} → ${artifact.fixtureRange.endDate} (${artifact.fixtureRange.days} daily closes)  `);
  lines.push(`**Walk-forward:** warmup=${artifact.config.warmup} days, stride=${artifact.config.stride} days  `);
  lines.push(`**Horizons:** ${artifact.config.horizons.join(', ')} days`);
  lines.push('');
  lines.push('## Configuration');
  lines.push('');
  lines.push('- **Baseline arm:** `walkForward` with defaults only (no experimental flags).');
  lines.push('- **Improved arm:** all wired W2/W3 toggles ON: `sidewaysSplit`, `matureBullCalibration`, `startStateMixture`, `postBreakShortWindow`, `trendPenaltyOnlyBreakConfidence`, `divergenceWeightedBreakConfidence`, `regimeSpecificSigma`, `pr3gCryptoShortHorizonRecencyWeighting`, `pr3fCryptoShortHorizonDisagreementPrior`.');
  lines.push('- W3 Hawkes/ADWIN are not wired into `markov-distribution.ts` yet — they appear as standalone modules only.');
  lines.push('');
  lines.push('## Baseline (defaults)');
  lines.push('');
  lines.push('| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |');
  lines.push('|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|');
  for (const row of artifact.baseline.perHorizon) {
    const m = row.metrics;
    lines.push(
      `| ${row.horizon} | ${m.n} | ${m.directionalAccuracy.toFixed(3)} | [${m.directionalCi.lower.toFixed(3)}, ${m.directionalCi.upper.toFixed(3)}] | ${m.brierScore.toFixed(3)} | ${m.ciCoverage.toFixed(3)} | ${m.meanEdge.toFixed(4)} | ${m.sharpness.toFixed(4)} | ${m.avgConfidence.toFixed(3)} |`,
    );
  }
  lines.push('');
  lines.push('## Improved (W2/W3 toggles on)');
  lines.push('');
  lines.push('| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |');
  lines.push('|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|');
  for (const row of artifact.improved.perHorizon) {
    const m = row.metrics;
    lines.push(
      `| ${row.horizon} | ${m.n} | ${m.directionalAccuracy.toFixed(3)} | [${m.directionalCi.lower.toFixed(3)}, ${m.directionalCi.upper.toFixed(3)}] | ${m.brierScore.toFixed(3)} | ${m.ciCoverage.toFixed(3)} | ${m.meanEdge.toFixed(4)} | ${m.sharpness.toFixed(4)} | ${m.avgConfidence.toFixed(3)} |`,
    );
  }
  lines.push('');
  lines.push('## Delta (Improved − Baseline)');
  lines.push('');
  lines.push('Positive ΔdirAcc / Δedge / Δcoverage and negative ΔBrier mean the improved arm wins.');
  lines.push('');
  lines.push('| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |');
  lines.push('|------:|--:|--------:|-------:|----------:|----------:|-----------:|');
  for (const row of artifact.delta) {
    lines.push(
      `| ${row.horizon} | ${row.n} | ${row.directionalAccuracyDelta >= 0 ? '+' : ''}${row.directionalAccuracyDelta.toFixed(3)} | ${row.brierScoreDelta >= 0 ? '+' : ''}${row.brierScoreDelta.toFixed(3)} | ${row.ciCoverageDelta >= 0 ? '+' : ''}${row.ciCoverageDelta.toFixed(3)} | ${row.meanEdgeDelta >= 0 ? '+' : ''}${row.meanEdgeDelta.toFixed(4)} | ${row.sharpnessDelta >= 0 ? '+' : ''}${row.sharpnessDelta.toFixed(4)} |`,
    );
  }
  lines.push('');
  lines.push('## Interpretation guide');
  lines.push('');
  lines.push('- **Directional accuracy** (best ↑): fraction of HOLD-vs-directional decisions that match the realized outcome at the threshold = 0.03 horizon-return cutoff.');
  lines.push('- **Brier score** (best ↓): mean squared error of the calibrated P(up) against the realized binary outcome.');
  lines.push('- **CI coverage** (best ≈ 0.90): fraction of realized prices that fell inside the model\'s conservative survival interval.');
  lines.push('- **meanEdge** (best ↑): average expected return from the action signal across all steps.');
  lines.push('- **sharpness** (best ↑): standard deviation of the calibrated P(up) — higher = more decisive predictions.');
  lines.push('- The **bootstrap 95% CI on directional accuracy** indicates whether observed deltas are likely real signal vs. resampling noise.');
  lines.push('');
  lines.push('## Notes');
  lines.push('');
  lines.push('- This backtest uses real BTC daily closes from the project fixture; Polymarket anchors are intentionally disabled (`polymarketMarkets: []`) to isolate the Markov-side effect of the toggles.');
  lines.push('- The "improved" arm activates every W2/W3 flag accepted by `WalkForwardConfig`. Some of these flags only fire on certain horizons (e.g. `matureBullCalibration` is BTC-14d-only, recency weighting is crypto h≤14), so deltas at h ∈ {1, 2, 3, 30} should naturally be smaller than at h ∈ {7, 14}.');
  lines.push('- W3 Hawkes (jump intensity) and ADWIN (drift detector) ship as standalone, fully tested modules but are not yet wired into `markov-distribution.ts`. They cannot influence this backtest until that wiring lands behind a feature flag.');
  return lines.join('\n');
}

async function main(): Promise<void> {
  const fixturePath = join(process.cwd(), 'src/tools/finance/fixtures/backtest-prices.json');
  const raw = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData & { startDate?: string; endDate?: string };
  const tdata = raw.tickers[TICKER];
  if (!tdata) throw new Error(`No fixture data for ${TICKER}`);
  console.log(`Loaded ${tdata.closes.length} BTC closes (${raw.startDate} → ${raw.endDate})`);

  console.log('— Running BASELINE arm —');
  const baseline = await runArm('baseline', tdata.closes, baselineFlags);

  console.log('— Running IMPROVED arm —');
  const improved = await runArm('improved', tdata.closes, improvedFlags);

  const artifact: BtcMultiHorizonArtifact = {
    generatedAt: new Date().toISOString(),
    ticker: TICKER,
    fixtureRange: {
      startDate: raw.startDate ?? 'unknown',
      endDate: raw.endDate ?? 'unknown',
      days: tdata.closes.length,
    },
    config: { warmup: WARMUP, stride: STRIDE, horizons: [...HORIZONS] },
    baseline,
    improved,
    delta: computeDelta(baseline, improved),
  };

  const outDir = join(process.cwd(), 'docs');
  mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 10);
  const jsonPath = join(outDir, `btc-multi-horizon-backtest-${stamp}.json`);
  const mdPath = join(outDir, `btc-multi-horizon-backtest-${stamp}.md`);
  writeFileSync(jsonPath, JSON.stringify(artifact, null, 2));
  writeFileSync(mdPath, renderMarkdown(artifact));
  console.log(`Wrote ${jsonPath}`);
  console.log(`Wrote ${mdPath}`);
}

if (import.meta.main) {
  await main();
}
