/**
 * Multi-horizon BTC backtest (Round-4): adds a fourth arm wiring the three
 * side-effect-free R4 feature flags on top of the Round-2 baseline:
 *   - enableGarchVol   (GARCH(1,1) per-day volatility scalar)
 *   - enableKswinTrim  (KSWIN variance-aware history trim)
 *   - enableCrossAssetBias  (cross-asset Lasso drift bias from SPY/GLD/QQQ)
 *
 * NOTE: Regime-conditional Platt (R4 Idea 3) requires pre-fitted calibration
 * samples from a separate validation pass (to avoid lookahead) and is therefore
 * omitted from this single-pass walk-forward backtest.
 *
 * Four arms compared across horizons {1, 2, 3, 7, 14, 30}:
 *   - baseline:      defaults only
 *   - improved:      all W2/W3 Round-1 toggles ON
 *   - improvedHA:    improved + ADWIN trim + Hawkes intensity (Round-2)
 *   - improvedR4:    improvedHA + GARCH vol + KSWIN trim + cross-asset Lasso
 *
 * Peer assets for cross-asset Lasso: SPY, GLD, QQQ — all present in the
 * BTC daily-close fixture (2024-01-01 → 2025-12-31).  Daily log-returns are
 * computed from the fixture closes and aligned to BTC.
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
  dates?: string[];
}
interface FixtureData {
  tickers: Record<string, FixtureTickerData>;
  startDate?: string;
  endDate?: string;
}

const TICKER = 'BTC-USD';
const PEER_TICKERS = ['SPY', 'GLD', 'QQQ'] as const;
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

export interface BtcMultiHorizonRound4Artifact {
  generatedAt: string;
  ticker: string;
  fixtureRange: { startDate: string; endDate: string; days: number };
  config: { warmup: number; stride: number; horizons: number[] };
  baseline: ArmReport;
  improved: ArmReport;
  improvedHA: ArmReport;
  improvedR4: ArmReport;
  deltaImprovedVsBaseline: DeltaRow[];
  deltaHAVsImproved: DeltaRow[];
  deltaR4VsHA: DeltaRow[];
  deltaR4VsBaseline: DeltaRow[];
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

/** Convert a closes array to an array of daily log-returns (length = closes.length - 1). */
function toLogReturns(closes: number[]): number[] {
  const rets: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    rets.push(Math.log(closes[i] / closes[i - 1]));
  }
  return rets;
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

const improvedHAFlags: Partial<WalkForwardConfig> = {
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

function renderMarkdown(art: BtcMultiHorizonRound4Artifact): string {
  const lines: string[] = [];
  lines.push('# BTC Multi-Horizon Backtest — Round 4 (R4 feature flags)');
  lines.push('');
  lines.push(`Generated: ${art.generatedAt}`);
  lines.push('');
  lines.push(`**Ticker:** ${art.ticker}  `);
  lines.push(`**Fixture:** ${art.fixtureRange.startDate} → ${art.fixtureRange.endDate} (${art.fixtureRange.days} daily closes)  `);
  lines.push(`**Walk-forward:** warmup=${art.config.warmup} days, stride=${art.config.stride} days  `);
  lines.push(`**Horizons:** ${art.config.horizons.join(', ')} days`);
  lines.push('');
  lines.push('## TL;DR');
  lines.push('');
  lines.push('Focus on the **Δ R4 − Hawkes+ADWIN** table. Positive ΔdirAcc / Δedge');
  lines.push('and negative ΔBrier on top of the already-improved+HA arm confirm the R4');
  lines.push('flags add real signal beyond W3R2.');
  lines.push('');
  lines.push('## Configuration');
  lines.push('');
  lines.push('- **baseline:** default flags only.');
  lines.push('- **improved:** all Round-1 W2/W3 toggles: `sidewaysSplit`, `matureBullCalibration`, `startStateMixture`, `postBreakShortWindow`, `trendPenaltyOnlyBreakConfidence`, `divergenceWeightedBreakConfidence`, `regimeSpecificSigma`, `pr3gCryptoShortHorizonRecencyWeighting`, `pr3fCryptoShortHorizonDisagreementPrior`.');
  lines.push('- **improved+HA:** improved + W3R2 wirings: `enableAdwinTrim` (δ=0.05), `enableHawkesIntensity` (σ=3.0).');
  lines.push('- **improved+R4:** improved+HA + Round-4: `enableGarchVol`, `enableKswinTrim` (α=0.005), `enableCrossAssetBias` (peers: SPY/GLD/QQQ, λ=0.005).');
  lines.push('');
  lines.push('> **Note on Regime Platt (R4 Idea 3):** omitted from this single-pass walk-forward');
  lines.push('> because fitting the recalibrator requires a separate validation set of');
  lines.push('> (pUp, regime, realized) triples — unavailable without a two-pass approach. The');
  lines.push('> unit tests verify its correctness; production use fits on a prior window.');
  lines.push('');
  lines.push('> **Cross-asset Lasso peers:** SPY (US equities), GLD (gold), QQQ (tech).  Daily');
  lines.push('> log-returns are computed from the same fixture and passed alongside BTC closes.');
  lines.push('> The Lasso λ=0.005 regularises away noise so only genuine co-movement survives.');
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
  for (const l of renderArmTable(art.improvedHA)) lines.push(l);
  lines.push('');
  lines.push('## Improved + R4 flags (GARCH + KSWIN + Lasso)');
  lines.push('');
  for (const l of renderArmTable(art.improvedR4)) lines.push(l);
  lines.push('');
  lines.push('## Δ Improved − Baseline (Round-1 cumulative gain recap)');
  lines.push('');
  for (const l of renderDeltaTable(art.deltaImprovedVsBaseline)) lines.push(l);
  lines.push('');
  lines.push('## Δ Hawkes+ADWIN − Improved  *(W3R2 ablation)*');
  lines.push('');
  for (const l of renderDeltaTable(art.deltaHAVsImproved)) lines.push(l);
  lines.push('');
  lines.push('## Δ R4 − Hawkes+ADWIN  *(the R4 ablation that matters)*');
  lines.push('');
  for (const l of renderDeltaTable(art.deltaR4VsHA)) lines.push(l);
  lines.push('');
  lines.push('## Δ R4 − Baseline (cumulative gain over defaults)');
  lines.push('');
  for (const l of renderDeltaTable(art.deltaR4VsBaseline)) lines.push(l);
  lines.push('');
  lines.push('## Notes');
  lines.push('');
  lines.push('- Fixture: 2024-01-01 → 2025-12-31 real BTC daily closes, warmup=180 days, stride=3 days.');
  lines.push('- `polymarketMarkets` is empty so Hawkes fires only on endogenous 3σ BTC returns.');
  lines.push('- GARCH: `fitGarch11` on the history window; per-day scalar clamped [0.33, 3.0].');
  lines.push('- KSWIN: operates on |log-return| (variance proxy) at α=0.005; runs after ADWIN.');
  lines.push('- Cross-asset Lasso: per-day bias clipped to [-0.05, +0.05] so no single peer dominates.');
  lines.push('- A neutral or negative Δ on the R4 arm is informative: it means these flags need');
  lines.push('  different defaults or the fixture period lacks the regimes where they help most.');
  return lines.join('\n');
}

async function main(): Promise<void> {
  const fixturePath = join(process.cwd(), 'src/tools/finance/fixtures/backtest-prices.json');
  const raw = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;
  const btcData = raw.tickers[TICKER];
  if (!btcData) throw new Error(`No fixture data for ${TICKER}`);
  console.log(`Loaded ${btcData.closes.length} BTC closes (${raw.startDate} → ${raw.endDate})`);

  // Build peer return series aligned to BTC by date.
  // Equities have ~501 closes (trading days only); BTC has 731 (including weekends).
  // We align by date: for each BTC calendar day with a return, look up the peer
  // return for that same date (0 if the peer doesn't trade that day).
  const btcDates = btcData.dates ?? [];
  const peerLogReturns: Record<string, number[]> = {};
  for (const peer of PEER_TICKERS) {
    const pdata = raw.tickers[peer];
    if (!pdata || !pdata.dates) {
      console.warn(`  peer ${peer}: no data or no dates, skipping`);
      continue;
    }
    // Build a date → 1-day log-return map for the peer.
    const peerRetByDate = new Map<string, number>();
    for (let i = 1; i < pdata.closes.length; i++) {
      const ret = Math.log(pdata.closes[i] / pdata.closes[i - 1]);
      peerRetByDate.set(pdata.dates[i], ret);
    }
    // For each BTC date (starting from index 1, matching BTC return index), look up peer.
    const aligned: number[] = [];
    for (let i = 1; i < btcDates.length; i++) {
      aligned.push(peerRetByDate.get(btcDates[i]) ?? 0);
    }
    peerLogReturns[peer] = aligned;
    const nonZero = aligned.filter(r => r !== 0).length;
    console.log(`  peer ${peer}: ${aligned.length} aligned returns, ${nonZero} trading days`);
  }

  const improvedR4Flags: Partial<WalkForwardConfig> = {
    ...improvedHAFlags,
    enableGarchVol: true,
    enableKswinTrim: true,
    kswinAlpha: 0.005,
    enableCrossAssetBias: true,
    crossAssetReturns: peerLogReturns,
    crossAssetLassoLambda: 0.005,
  };

  console.log('\n— BASELINE —');
  const baseline = await runArm('baseline', btcData.closes, baselineFlags);
  console.log('\n— IMPROVED (Round-1) —');
  const improved = await runArm('improved', btcData.closes, improvedFlags);
  console.log('\n— IMPROVED + HAWKES + ADWIN (Round-2) —');
  const improvedHA = await runArm('improvedHA', btcData.closes, improvedHAFlags);
  console.log('\n— IMPROVED + R4 FLAGS (GARCH + KSWIN + Lasso) —');
  const improvedR4 = await runArm('improvedR4', btcData.closes, improvedR4Flags);

  const artifact: BtcMultiHorizonRound4Artifact = {
    generatedAt: new Date().toISOString(),
    ticker: TICKER,
    fixtureRange: {
      startDate: raw.startDate ?? 'unknown',
      endDate: raw.endDate ?? 'unknown',
      days: btcData.closes.length,
    },
    config: { warmup: WARMUP, stride: STRIDE, horizons: [...HORIZONS] },
    baseline, improved, improvedHA, improvedR4,
    deltaImprovedVsBaseline: computeDelta(baseline, improved),
    deltaHAVsImproved:       computeDelta(improved, improvedHA),
    deltaR4VsHA:             computeDelta(improvedHA, improvedR4),
    deltaR4VsBaseline:       computeDelta(baseline, improvedR4),
  };

  const outDir = join(process.cwd(), 'docs');
  mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 10);
  const jsonPath = join(outDir, `btc-multi-horizon-backtest-round4-${stamp}.json`);
  const mdPath   = join(outDir, `btc-multi-horizon-backtest-round4-${stamp}.md`);
  writeFileSync(jsonPath, JSON.stringify(artifact, null, 2));
  writeFileSync(mdPath, renderMarkdown(artifact));
  console.log(`\nWrote ${jsonPath}`);
  console.log(`Wrote ${mdPath}`);
}

if (import.meta.main) {
  await main();
}
