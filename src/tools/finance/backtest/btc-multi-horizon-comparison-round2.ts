/**
 * Multi-horizon BTC backtest (Round-2): adds a third arm wiring W3R2 ADWIN +
 * Hawkes intensity behind feature flags, on top of the Round-1 W2/W3 toggles.
 *
 * Three arms compared across horizons {1, 2, 3, 7, 14, 30}:
 *   - baseline:      defaults only
 *   - improved:      all wired W2/W3 toggles ON (Round-1)
 *   - improved+HA:   improved + ADWIN trim + Hawkes intensity amplification
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
const WARMUP = 180;
const STRIDE = 3;

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
interface ArmHorizon { horizon: number; metrics: MetricBlock; }
interface ArmReport  { label: string; perHorizon: ArmHorizon[]; }
interface DeltaRow {
  horizon: number; n: number;
  directionalAccuracyDelta: number; brierScoreDelta: number;
  ciCoverageDelta: number; meanEdgeDelta: number; sharpnessDelta: number;
}

export interface BtcMultiHorizonRound2Artifact {
  generatedAt: string;
  ticker: string;
  fixtureRange: { startDate: string; endDate: string; days: number };
  config: { warmup: number; stride: number; horizons: number[] };
  baseline: ArmReport;
  improved: ArmReport;
  improvedHawkesAdwin: ArmReport;
  deltaImprovedVsBaseline: DeltaRow[];
  deltaHawkesAdwinVsImproved: DeltaRow[];
  deltaHawkesAdwinVsBaseline: DeltaRow[];
}

function summarize(steps: BacktestStep[]): MetricBlock {
  if (steps.length === 0) {
    return {
      n: 0, directionalAccuracy: 0, brierScore: 0, ciCoverage: 0,
      meanEdge: 0, sharpness: 0, avgConfidence: 0,
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

const baselineFlags: Partial<WalkForwardConfig> = {};

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

const improvedHawkesAdwinFlags: Partial<WalkForwardConfig> = {
  ...improvedFlags,
  enableAdwinTrim: true,
  adwinDelta: 0.05,
  enableHawkesIntensity: true,
  hawkesSigmaThreshold: 3.0,
};

async function runArm(
  label: string,
  prices: number[],
  flags: Partial<WalkForwardConfig>,
): Promise<ArmReport> {
  const perHorizon: ArmHorizon[] = [];
  for (const horizon of HORIZONS) {
    const result = await walkForward({
      ticker: TICKER, prices, horizon, warmup: WARMUP, stride: STRIDE, ...flags,
    });
    if (result.errors.length > 0) {
      console.error(`[${label} h=${horizon}] ${result.errors.length} errors (first: ${result.errors[0].error})`);
    }
    const m = summarize(result.steps);
    perHorizon.push({ horizon, metrics: m });
    console.log(
      `[${label} h=${horizon}] n=${result.steps.length} dirAcc=${m.directionalAccuracy.toFixed(3)} brier=${m.brierScore.toFixed(3)} cov=${m.ciCoverage.toFixed(3)} edge=${m.meanEdge.toFixed(4)}`,
    );
  }
  return { label, perHorizon };
}

function computeDelta(base: ArmReport, ext: ArmReport): DeltaRow[] {
  return base.perHorizon.map((b, i) => {
    const e = ext.perHorizon[i];
    return {
      horizon: b.horizon, n: b.metrics.n,
      directionalAccuracyDelta: e.metrics.directionalAccuracy - b.metrics.directionalAccuracy,
      brierScoreDelta:          e.metrics.brierScore          - b.metrics.brierScore,
      ciCoverageDelta:          e.metrics.ciCoverage          - b.metrics.ciCoverage,
      meanEdgeDelta:            e.metrics.meanEdge            - b.metrics.meanEdge,
      sharpnessDelta:           e.metrics.sharpness           - b.metrics.sharpness,
    };
  });
}

function fmtSign(x: number, digits = 3): string {
  return `${x >= 0 ? '+' : ''}${x.toFixed(digits)}`;
}

function renderArmTable(arm: ArmReport): string[] {
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

function renderDeltaTable(rows: DeltaRow[]): string[] {
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

function renderMarkdown(art: BtcMultiHorizonRound2Artifact): string {
  const lines: string[] = [];
  lines.push('# BTC Multi-Horizon Backtest — Round 2 (W3 Hawkes + ADWIN wired)');
  lines.push('');
  lines.push(`Generated: ${art.generatedAt}`);
  lines.push('');
  lines.push(`**Ticker:** ${art.ticker}  `);
  lines.push(`**Fixture:** ${art.fixtureRange.startDate} → ${art.fixtureRange.endDate} (${art.fixtureRange.days} daily closes)  `);
  lines.push(`**Walk-forward:** warmup=${art.config.warmup} days, stride=${art.config.stride} days  `);
  lines.push(`**Horizons:** ${art.config.horizons.join(', ')} days`);
  lines.push('');
  lines.push('## TL;DR (vs Round-1)');
  lines.push('');
  lines.push('Compare the third delta table (Hawkes+ADWIN vs Improved). Positive ΔdirAcc / Δedge and negative ΔBrier on top of the already-improved arm = the W3R2 wiring adds *real* signal beyond what the Round-1 toggles already captured.');
  lines.push('');
  lines.push('## Configuration');
  lines.push('');
  lines.push('- **baseline arm:** `walkForward` with defaults (no experimental flags).');
  lines.push('- **improved arm:** all Round-1 W2/W3 toggles ON: `sidewaysSplit`, `matureBullCalibration`, `startStateMixture`, `postBreakShortWindow`, `trendPenaltyOnlyBreakConfidence`, `divergenceWeightedBreakConfidence`, `regimeSpecificSigma`, `pr3gCryptoShortHorizonRecencyWeighting`, `pr3fCryptoShortHorizonDisagreementPrior`.');
  lines.push('- **improved+HA arm:** improved + W3R2 wirings ON: `enableAdwinTrim` (δ=0.05), `enableHawkesIntensity` (σ=3.0).');
  lines.push('');
  lines.push('## Baseline (defaults)');
  lines.push('');
  for (const l of renderArmTable(art.baseline)) lines.push(l);
  lines.push('');
  lines.push('## Improved (Round-1 W2/W3 toggles ON)');
  lines.push('');
  for (const l of renderArmTable(art.improved)) lines.push(l);
  lines.push('');
  lines.push('## Improved + Hawkes + ADWIN (W3R2)');
  lines.push('');
  for (const l of renderArmTable(art.improvedHawkesAdwin)) lines.push(l);
  lines.push('');
  lines.push('## Δ Improved − Baseline (recap)');
  lines.push('');
  for (const l of renderDeltaTable(art.deltaImprovedVsBaseline)) lines.push(l);
  lines.push('');
  lines.push('## Δ Hawkes+ADWIN − Improved  *(this is the W3R2 ablation that matters)*');
  lines.push('');
  for (const l of renderDeltaTable(art.deltaHawkesAdwinVsImproved)) lines.push(l);
  lines.push('');
  lines.push('## Δ Hawkes+ADWIN − Baseline (cumulative gain over defaults)');
  lines.push('');
  for (const l of renderDeltaTable(art.deltaHawkesAdwinVsBaseline)) lines.push(l);
  lines.push('');
  lines.push('## Notes');
  lines.push('');
  lines.push('- This backtest uses the same BTC fixture (real daily closes) and same warmup/stride as Round-1, so the deltas are comparable.');
  lines.push('- `polymarketMarkets` is intentionally empty so the Hawkes path can only fire if it synthesizes an *endogenous* jump from clustered 3σ moves in the BTC return series itself.');
  lines.push('- ADWIN trimming uses δ=0.05 with a 60-bar safety floor so the model never runs out of history.');
  lines.push('- If the W3R2 ablation is neutral or negative across most horizons, that\'s a signal these wirings need different defaults (looser σ, smaller δ) or shouldn\'t be promoted to defaults yet.');
  return lines.join('\n');
}

async function main(): Promise<void> {
  const fixturePath = join(process.cwd(), 'src/tools/finance/fixtures/backtest-prices.json');
  const raw = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData & { startDate?: string; endDate?: string };
  const tdata = raw.tickers[TICKER];
  if (!tdata) throw new Error(`No fixture data for ${TICKER}`);
  console.log(`Loaded ${tdata.closes.length} BTC closes (${raw.startDate} → ${raw.endDate})`);

  console.log('— BASELINE —');
  const baseline = await runArm('baseline', tdata.closes, baselineFlags);
  console.log('— IMPROVED —');
  const improved = await runArm('improved', tdata.closes, improvedFlags);
  console.log('— IMPROVED + HAWKES + ADWIN —');
  const improvedHA = await runArm('improvedHA', tdata.closes, improvedHawkesAdwinFlags);

  const artifact: BtcMultiHorizonRound2Artifact = {
    generatedAt: new Date().toISOString(),
    ticker: TICKER,
    fixtureRange: {
      startDate: raw.startDate ?? 'unknown',
      endDate: raw.endDate ?? 'unknown',
      days: tdata.closes.length,
    },
    config: { warmup: WARMUP, stride: STRIDE, horizons: [...HORIZONS] },
    baseline, improved, improvedHawkesAdwin: improvedHA,
    deltaImprovedVsBaseline:    computeDelta(baseline, improved),
    deltaHawkesAdwinVsImproved: computeDelta(improved, improvedHA),
    deltaHawkesAdwinVsBaseline: computeDelta(baseline, improvedHA),
  };

  const outDir = join(process.cwd(), 'docs');
  mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 10);
  const jsonPath = join(outDir, `btc-multi-horizon-backtest-round2-${stamp}.json`);
  const mdPath = join(outDir, `btc-multi-horizon-backtest-round2-${stamp}.md`);
  writeFileSync(jsonPath, JSON.stringify(artifact, null, 2));
  writeFileSync(mdPath, renderMarkdown(artifact));
  console.log(`Wrote ${jsonPath}`);
  console.log(`Wrote ${mdPath}`);
}

if (import.meta.main) {
  await main();
}
