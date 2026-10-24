/**
 * Multi-horizon BTC backtest (Round-4): adds a fourth arm wiring two
 * side-effect-free R4 feature flags on top of the Round-2 baseline:
 *   - enableGarchVol   (GARCH(1,1) per-day volatility scalar)
 *   - enableKswinTrim  (KSWIN variance-aware history trim)
 *
 * enableCrossAssetBias (cross-asset Lasso drift bias) is off: once it reached
 * the forecast drift it worsened Brier at h >= 7 on this fixture.
 *
 * NOTE: Regime-conditional Platt (R4 Idea 3) requires pre-fitted calibration
 * samples from a separate validation pass (to avoid lookahead) and is therefore
 * omitted from this single-pass walk-forward backtest.
 *
 * Four arms compared across horizons {1, 2, 3, 7, 14, 30}:
 *   - baseline:      defaults only
 *   - improved:      all W2/W3 Round-1 toggles ON
 *   - improvedHA:    improved + ADWIN trim + Hawkes intensity (Round-2)
 *   - improvedR4:    improvedHA + GARCH vol + KSWIN trim
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
const HORIZONS = [1, 2, 3, 7, 14, 30] as const;
const WARMUP = 180;
const STRIDE = 3;

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
  lines.push('- **improved+R4:** improved+HA + Round-4: `enableGarchVol`, `enableKswinTrim` (α=0.005).');
  lines.push('');
  lines.push('> **Note on Regime Platt (R4 Idea 3):** omitted from this single-pass walk-forward');
  lines.push('> because fitting the recalibrator requires a separate validation set of');
  lines.push('> (pUp, regime, realized) triples — unavailable without a two-pass approach. The');
  lines.push('> unit tests verify its correctness; production use fits on a prior window.');
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
  lines.push('## Improved + R4 flags (GARCH + KSWIN)');
  lines.push('');
  for (const l of renderArmMetricTable(art.improvedR4)) lines.push(l);
  lines.push('');
  lines.push('## Δ Improved − Baseline (Round-1 cumulative gain recap)');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaImprovedVsBaseline)) lines.push(l);
  lines.push('');
  lines.push('## Δ Hawkes+ADWIN − Improved  *(W3R2 ablation)*');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaHAVsImproved)) lines.push(l);
  lines.push('');
  lines.push('## Δ R4 − Hawkes+ADWIN  *(the R4 ablation that matters)*');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaR4VsHA)) lines.push(l);
  lines.push('');
  lines.push('## Δ R4 − Baseline (cumulative gain over defaults)');
  lines.push('');
  for (const l of renderDeltaMetricTable(art.deltaR4VsBaseline)) lines.push(l);
  lines.push('');
  lines.push('## Notes');
  lines.push('');
  lines.push('- Fixture: 2024-01-01 → 2025-12-31 real BTC daily closes, warmup=180 days, stride=3 days.');
  lines.push('- `polymarketMarkets` is empty so Hawkes fires only on endogenous 3σ BTC returns.');
  lines.push('- GARCH: `fitGarch11` on the history window; per-day scalar clamped [0.33, 3.0].');
  lines.push('- KSWIN: operates on |log-return| (variance proxy) at α=0.005; runs after ADWIN.');
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

  const improvedR4Flags: Partial<WalkForwardConfig> = {
    ...improvedHAFlags,
    enableGarchVol: true,
    enableKswinTrim: true,
    kswinAlpha: 0.005,
  };

  console.log('\n— BASELINE —');
  const baseline = await runMultiHorizonArm('baseline', { ticker: TICKER, prices: btcData.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: baselineFlags });
  console.log('\n— IMPROVED (Round-1) —');
  const improved = await runMultiHorizonArm('improved', { ticker: TICKER, prices: btcData.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedFlags });
  console.log('\n— IMPROVED + HAWKES + ADWIN (Round-2) —');
  const improvedHA = await runMultiHorizonArm('improvedHA', { ticker: TICKER, prices: btcData.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedHAFlags });
  console.log('\n— IMPROVED + R4 FLAGS (GARCH + KSWIN) —');
  const improvedR4 = await runMultiHorizonArm('improvedR4', { ticker: TICKER, prices: btcData.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedR4Flags });

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
    deltaImprovedVsBaseline: computeMetricDelta(baseline, improved),
    deltaHAVsImproved:       computeMetricDelta(improved, improvedHA),
    deltaR4VsHA:             computeMetricDelta(improvedHA, improvedR4),
    deltaR4VsBaseline:       computeMetricDelta(baseline, improvedR4),
  };

  writeBacktestDocs('btc-multi-horizon-backtest-round4', artifact, renderMarkdown(artifact));
}

if (import.meta.main) {
  await main();
}
