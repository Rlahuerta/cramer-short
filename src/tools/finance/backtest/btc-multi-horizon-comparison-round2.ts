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
  dates: string[];
}
interface FixtureData {
  tickers: Record<string, FixtureTickerData>;
}

const TICKER = 'BTC-USD';
const HORIZONS = [1, 2, 3, 7, 14, 30] as const;
const WARMUP = 180;
const STRIDE = 3;

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
  for (const l of renderArmMetricTable(art.baseline)) lines.push(l);
  lines.push('');
  lines.push('## Improved (Round-1 W2/W3 toggles ON)');
  lines.push('');
  for (const l of renderArmMetricTable(art.improved)) lines.push(l);
  lines.push('');
  lines.push('## Improved + Hawkes + ADWIN (W3R2)');
  lines.push('');
  for (const l of renderArmMetricTable(art.improvedHawkesAdwin)) lines.push(l);
  lines.push('');
  lines.push('## Δ Improved − Baseline (recap)');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaImprovedVsBaseline)) lines.push(l);
  lines.push('');
  lines.push('## Δ Hawkes+ADWIN − Improved  *(this is the W3R2 ablation that matters)*');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaHawkesAdwinVsImproved)) lines.push(l);
  lines.push('');
  lines.push('## Δ Hawkes+ADWIN − Baseline (cumulative gain over defaults)');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaHawkesAdwinVsBaseline)) lines.push(l);
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
  const baseline = await runMultiHorizonArm('baseline', { ticker: TICKER, prices: tdata.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: baselineFlags });
  console.log('— IMPROVED —');
  const improved = await runMultiHorizonArm('improved', { ticker: TICKER, prices: tdata.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedFlags });
  console.log('— IMPROVED + HAWKES + ADWIN —');
  const improvedHA = await runMultiHorizonArm('improvedHA', { ticker: TICKER, prices: tdata.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedHawkesAdwinFlags });

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
    deltaImprovedVsBaseline:    computeMetricDelta(baseline, improved),
    deltaHawkesAdwinVsImproved: computeMetricDelta(improved, improvedHA),
    deltaHawkesAdwinVsBaseline: computeMetricDelta(baseline, improvedHA),
  };

  writeBacktestDocs('btc-multi-horizon-backtest-round2', artifact, renderMarkdown(artifact));
}

if (import.meta.main) {
  await main();
}
