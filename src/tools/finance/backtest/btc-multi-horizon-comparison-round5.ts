/**
 * Multi-horizon BTC backtest (Round-5): adds a fifth arm wiring the
 * R5 Idea #5 horizon-aware + regime-conditional GARCH clamp on top of
 * the R4 stack.
 *
 * The other Sprint 1 ideas (#3 naive baselines, #11 longshot shrinkage,
 * #14 transition entropy) are exposed as helpers but not yet integrated
 * into the trajectory MC pipeline — see the R5 final report for status.
 *
 * Five arms compared across horizons {1, 2, 3, 7, 14, 30}:
 *   - baseline
 *   - improved
 *   - improvedHA
 *   - improvedR4
 *   - improvedR5    (improvedR4 + garchHorizonCap=7, garchRegimeCeiling={calm:1.5, turbulent:3.0})
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import type { WalkForwardConfig } from './walk-forward.js';
import {
  computeMetricDelta,
  renderArmMetricTable,
  renderDeltaMetricTable,
  runMultiHorizonArm,
  writeBacktestDocs,
  type ArmReport,
  type DeltaRow,
} from './comparison-harness.js';

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

export interface BtcMultiHorizonRound5Artifact {
  generatedAt: string;
  ticker: string;
  fixtureRange: { startDate: string; endDate: string; days: number };
  config: { warmup: number; stride: number; horizons: number[] };
  baseline: ArmReport;
  improved: ArmReport;
  improvedHA: ArmReport;
  improvedR4: ArmReport;
  improvedR5: ArmReport;
  deltaImprovedVsBaseline: DeltaRow[];
  deltaHAVsImproved: DeltaRow[];
  deltaR4VsHA: DeltaRow[];
  deltaR5VsR4: DeltaRow[];
  deltaR5VsBaseline: DeltaRow[];
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

function renderMarkdown(art: BtcMultiHorizonRound5Artifact): string {
  const lines: string[] = [];
  lines.push('# BTC Multi-Horizon Backtest — Round 5 (R5 Idea #5: horizon-aware GARCH)');
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
  lines.push('Focus on the **Δ R5 − R4** table. Negative ΔBrier (especially at h ≥ 7d)');
  lines.push('and non-negative ΔdirAcc confirm Idea #5 fixes the R4 regression where');
  lines.push('GARCH widened CIs unproductively at long horizons.');
  lines.push('');
  lines.push('## Configuration');
  lines.push('');
  lines.push('- **baseline / improved / improvedHA / improvedR4:** identical to Round-4.');
  lines.push('- **improved+R5:** improvedR4 + `garchHorizonCap=7` + `garchRegimeCeiling={calm:1.5, turbulent:3.0}`.');
  lines.push('  Past 7 days the GARCH scalar soft-blends toward 1.0; past 21 days it is 1.0.');
  lines.push('  In calm regimes the scalar is capped at 1.5 instead of 3.0.');
  lines.push('');
  lines.push('## Baseline (defaults)');
  lines.push('');
  for (const l of renderArmMetricTable(art.baseline)) lines.push(l);
  lines.push('');
  lines.push('## Improved (Round-1 W2/W3 toggles ON)');
  lines.push('');
  for (const l of renderArmMetricTable(art.improved)) lines.push(l);
  lines.push('');
  lines.push('## Improved + Hawkes + ADWIN (W3R2)');
  lines.push('');
  for (const l of renderArmMetricTable(art.improvedHA)) lines.push(l);
  lines.push('');
  lines.push('## Improved + R4 flags (GARCH + KSWIN + Lasso)');
  lines.push('');
  for (const l of renderArmMetricTable(art.improvedR4)) lines.push(l);
  lines.push('');
  lines.push('## Improved + R5 (R4 + horizon-aware/regime-clamped GARCH)');
  lines.push('');
  for (const l of renderArmMetricTable(art.improvedR5)) lines.push(l);
  lines.push('');
  lines.push('## Δ Improved − Baseline');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaImprovedVsBaseline)) lines.push(l);
  lines.push('');
  lines.push('## Δ Hawkes+ADWIN − Improved');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaHAVsImproved)) lines.push(l);
  lines.push('');
  lines.push('## Δ R4 − Hawkes+ADWIN');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaR4VsHA)) lines.push(l);
  lines.push('');
  lines.push('## Δ R5 − R4  *(the R5 ablation that matters)*');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaR5VsR4)) lines.push(l);
  lines.push('');
  lines.push('## Δ R5 − Baseline (cumulative gain from defaults)');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaR5VsBaseline)) lines.push(l);
  lines.push('');
  lines.push('## Notes');
  lines.push('');
  lines.push('- Sprint 1 ideas #3 (naive baselines), #11 (longshot shrinkage), #14 (entropy CI)');
  lines.push('  are implemented as standalone helpers but not yet integrated in the predict loop;');
  lines.push('  they are validated by unit tests only.');
  lines.push('- The horizon decay is `blend = max(0, 1 - (d - cap) / (2·cap))` past `cap=7`.');
  lines.push('- Regime detection: rolling 20-obs σ vs full-series σ ⇒ "calm" if recent < full.');
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

  const improvedR5Flags: Partial<WalkForwardConfig> = {
    ...improvedR4Flags,
    garchHorizonCap: 7,
    garchRegimeCeiling: { calm: 1.5, turbulent: 3.0 },
  };

  console.log('\n— BASELINE —');
  const baseline = await runMultiHorizonArm('baseline', { ticker: TICKER, prices: btcData.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: baselineFlags });
  console.log('\n— IMPROVED (Round-1) —');
  const improved = await runMultiHorizonArm('improved', { ticker: TICKER, prices: btcData.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedFlags });
  console.log('\n— IMPROVED + HAWKES + ADWIN (Round-2) —');
  const improvedHA = await runMultiHorizonArm('improvedHA', { ticker: TICKER, prices: btcData.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedHAFlags });
  console.log('\n— IMPROVED + R4 FLAGS (GARCH + KSWIN + Lasso) —');
  const improvedR4 = await runMultiHorizonArm('improvedR4', { ticker: TICKER, prices: btcData.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedR4Flags });
  console.log('\n— IMPROVED + R5 (R4 + horizon-aware/regime-clamped GARCH) —');
  const improvedR5 = await runMultiHorizonArm('improvedR5', { ticker: TICKER, prices: btcData.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedR5Flags });

  const artifact: BtcMultiHorizonRound5Artifact = {
    generatedAt: new Date().toISOString(),
    ticker: TICKER,
    fixtureRange: {
      startDate: raw.startDate ?? 'unknown',
      endDate: raw.endDate ?? 'unknown',
      days: btcData.closes.length,
    },
    config: { warmup: WARMUP, stride: STRIDE, horizons: [...HORIZONS] },
    baseline, improved, improvedHA, improvedR4, improvedR5,
    deltaImprovedVsBaseline: computeMetricDelta(baseline, improved),
    deltaHAVsImproved:       computeMetricDelta(improved, improvedHA),
    deltaR4VsHA:             computeMetricDelta(improvedHA, improvedR4),
    deltaR5VsR4:             computeMetricDelta(improvedR4, improvedR5),
    deltaR5VsBaseline:       computeMetricDelta(baseline, improvedR5),
  };

  writeBacktestDocs('btc-multi-horizon-backtest-round5', artifact, renderMarkdown(artifact));
}

if (import.meta.main) {
  await main();
}
