/**
 * Multi-horizon BTC backtest: baseline (defaults only) vs. "improved" (every
 * wired W2/W3 toggle on) across horizons 1, 2, 3, 7, 14, 30 days.
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
const WARMUP = 180; // ensure enough warmup for all horizons
const STRIDE = 3;   // dense sampling

export interface BtcMultiHorizonArtifact {
  generatedAt: string;
  ticker: string;
  fixtureRange: { startDate: string; endDate: string; days: number };
  config: { warmup: number; stride: number; horizons: number[] };
  baseline: ArmReport;
  improved: ArmReport;
  delta: DeltaRow[];
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
  for (const l of renderArmMetricTable(artifact.baseline)) lines.push(l);
  lines.push('');
  lines.push('## Improved (W2/W3 toggles on)');
  lines.push('');
  for (const l of renderArmMetricTable(artifact.improved)) lines.push(l);
  lines.push('');
  lines.push('## Delta (Improved − Baseline)');
  lines.push('');
  lines.push('Positive ΔdirAcc / Δedge / Δcoverage and negative ΔBrier mean the improved arm wins.');
  lines.push('');
  for (const l of renderDeltaMetricTable(artifact.delta)) lines.push(l);
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
  const baseline = await runMultiHorizonArm('baseline', { ticker: TICKER, prices: tdata.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: baselineFlags });

  console.log('— Running IMPROVED arm —');
  const improved = await runMultiHorizonArm('improved', { ticker: TICKER, prices: tdata.closes, horizons: HORIZONS, warmup: WARMUP, stride: STRIDE, flags: improvedFlags });

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
    delta: computeMetricDelta(baseline, improved),
  };

  writeBacktestDocs('btc-multi-horizon-backtest', artifact, renderMarkdown(artifact));
}

if (import.meta.main) {
  await main();
}
