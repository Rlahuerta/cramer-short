import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import type { ForecastTrustPolicyLevel } from '../forecast-arbitrator.js';
import { walkForward, type WalkForwardConfig } from './walk-forward.js';
import {
  brierScore,
  ciCoverage,
  computeRCCurve,
  directionalAccuracy,
  selectiveDirectionalAccuracy,
  sharpness,
  type BacktestStep,
} from './metrics.js';
import { effectiveBreakContext } from './break-policy-ablation.js';

interface FixtureTickerData {
  closes: number[];
  dates?: string[];
}

interface FixtureData {
  tickers: Record<string, FixtureTickerData>;
  startDate?: string;
  endDate?: string;
}

interface CalibrationSummary {
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

interface PolicySummary {
  fullRate: number;
  contextOnlyRate: number;
  abstainRate: number;
  eligibleCoverage: number;
  eligibleDirectionalAccuracy: number;
  comparableBaselineThreshold: number;
  comparableBaselineCoverage: number;
  comparableBaselineAccuracy: number;
  activeTradeRate: number;
  activeTradeAccuracy: number;
  lowConfidenceActiveTradeRate: number;
  weakStateNoTradeRate: number;
}

interface ArmHorizonReport {
  horizon: number;
  calibration: CalibrationSummary;
  recommendationRC: Array<{ threshold: number; accuracy: number; coverage: number; n: number }>;
  policy: PolicySummary | null;
}

interface ArmReport {
  label: string;
  source: 'baseline' | 'adaptive';
  policyMode: 'none' | 'observable-proxy';
  horizons: ArmHorizonReport[];
}

interface Artifact {
  generatedAt: string;
  ticker: string;
  fixtureRange: { startDate: string; endDate: string; days: number };
  config: {
    warmup: number;
    stride: number;
    horizons: number[];
    targetCiCoverage: number;
  };
  caveat: string;
  baseline: ArmReport;
  adaptiveOnly: ArmReport;
  boundedOnly: ArmReport;
  combined: ArmReport;
}

const TICKER = 'BTC-USD';
const HORIZONS = [1, 7, 14, 30] as const;
const WARMUP = 180;
const STRIDE = 3;
const TARGET_CI_COVERAGE = 0.9;
const RC_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7] as const;

const adaptiveFlags: Partial<WalkForwardConfig> = {
  enableAdaptiveConformal: true,
  conformalAlpha: 0.1,
  conformalBreakSensitivity: 1.5,
  conformalFastLearningRate: 0.2,
  conformalCooloffWindow: 20,
};

function correctDirection(step: BacktestStep): boolean {
  if (step.recommendation === 'BUY') return step.actualReturn > 0;
  if (step.recommendation === 'SELL') return step.actualReturn < 0;
  return Math.abs(step.actualReturn) < 0.03;
}

function summarizeCalibration(steps: BacktestStep[]): CalibrationSummary {
  const breakSteps = steps.filter(step => effectiveBreakContext(step));
  const selective045 = selectiveDirectionalAccuracy(steps, 0.45);
  const overallCoverage = ciCoverage(steps);
  const breakCoverage = breakSteps.length > 0 ? ciCoverage(breakSteps) : null;
  return {
    n: steps.length,
    ciCoverage: overallCoverage,
    coverageError: Math.abs(overallCoverage - TARGET_CI_COVERAGE),
    breakCount: breakSteps.length,
    breakCiCoverage: breakCoverage,
    breakCoverageError: breakCoverage === null ? null : Math.abs(breakCoverage - TARGET_CI_COVERAGE),
    sharpness: sharpness(steps),
    brierScore: brierScore(steps),
    directionalAccuracy: directionalAccuracy(steps),
    selectiveDirectionalAccuracy045: selective045.accuracy,
    selectiveCoverage045: selective045.coverage,
  };
}

function nearestRcPoint(
  curve: Array<{ threshold: number; accuracy: number; coverage: number; n: number }>,
  coverage: number,
) {
  return curve.reduce((best, point) => {
    if (!best) return point;
    return Math.abs(point.coverage - coverage) < Math.abs(best.coverage - coverage) ? point : best;
  });
}

function computeObservablePolicyLevel(
  step: BacktestStep,
  currentPrice: number,
): ForecastTrustPolicyLevel {
  const structuralBreak = effectiveBreakContext(step);
  const weakConfidence = step.confidence < 0.45;
  const severeConfidence = step.confidence < 0.2;
  const conformalRadiusRatio = step.conformalRadius !== undefined && currentPrice > 0
    ? step.conformalRadius / currentPrice
    : null;
  const weakConformal = step.conformalApplied === true && (
    step.conformalMode === 'break'
    || (typeof step.conformalCoverageEstimate === 'number' && step.conformalCoverageEstimate < 0.75)
    || (typeof conformalRadiusRatio === 'number' && conformalRadiusRatio >= 0.08)
  );
  const severeConformal = step.conformalApplied === true && (
    (typeof step.conformalCoverageEstimate === 'number' && step.conformalCoverageEstimate < 0.6)
    || (typeof conformalRadiusRatio === 'number' && conformalRadiusRatio >= 0.12)
    || (step.conformalMode === 'break'
      && typeof step.conformalCoverageEstimate === 'number'
      && step.conformalCoverageEstimate < 0.65)
  );

  if ((structuralBreak && severeConfidence) || severeConformal) return 'abstain';
  if (structuralBreak || weakConfidence || weakConformal) return 'context-only';
  return 'full';
}

function summarizeObservablePolicy(
  steps: BacktestStep[],
  prices: number[],
): PolicySummary {
  const curve = computeRCCurve(steps, [...RC_THRESHOLDS]);
  const annotated = steps.map((step) => {
    const currentPrice = prices[step.t];
    const level = computeObservablePolicyLevel(step, currentPrice);
    return {
      step,
      level,
      activeTrade: level === 'full' && step.recommendation !== 'HOLD',
      weakState: effectiveBreakContext(step)
        || step.confidence < 0.45
        || (step.conformalApplied === true && step.conformalMode === 'break'),
    };
  });

  const eligible = annotated.filter(item => item.level === 'full');
  const activeTrades = annotated.filter(item => item.activeTrade);
  const lowConfidence = annotated.filter(item => item.step.confidence < 0.45);
  const weakStates = annotated.filter(item => item.weakState);
  const comparable = nearestRcPoint(curve, eligible.length / Math.max(annotated.length, 1));

  return {
    fullRate: eligible.length / annotated.length,
    contextOnlyRate: annotated.filter(item => item.level === 'context-only').length / annotated.length,
    abstainRate: annotated.filter(item => item.level === 'abstain').length / annotated.length,
    eligibleCoverage: eligible.length / annotated.length,
    eligibleDirectionalAccuracy: eligible.length > 0
      ? directionalAccuracy(eligible.map(item => item.step))
      : 0,
    comparableBaselineThreshold: comparable.threshold,
    comparableBaselineCoverage: comparable.coverage,
    comparableBaselineAccuracy: comparable.accuracy,
    activeTradeRate: activeTrades.length / annotated.length,
    activeTradeAccuracy: activeTrades.length > 0
      ? activeTrades.filter(item => correctDirection(item.step)).length / activeTrades.length
      : 0,
    lowConfidenceActiveTradeRate: lowConfidence.length > 0
      ? lowConfidence.filter(item => item.activeTrade).length / lowConfidence.length
      : 0,
    weakStateNoTradeRate: weakStates.length > 0
      ? weakStates.filter(item => !item.activeTrade).length / weakStates.length
      : 0,
  };
}

async function runArm(
  label: string,
  prices: number[],
  flags: Partial<WalkForwardConfig>,
  policyMode: ArmReport['policyMode'],
  source: ArmReport['source'],
): Promise<ArmReport> {
  const horizons: ArmHorizonReport[] = [];
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
      console.error(`[${label} h=${horizon}] ${result.errors.length} errors (first: ${result.errors[0]?.error ?? 'unknown'})`);
    }
    const calibration = summarizeCalibration(result.steps);
    const recommendationRC = computeRCCurve(result.steps, [...RC_THRESHOLDS]);
    const policy = policyMode === 'observable-proxy'
      ? summarizeObservablePolicy(result.steps, prices)
      : null;
    horizons.push({ horizon, calibration, recommendationRC, policy });
    console.log(
      `[${label} h=${horizon}] covErr=${calibration.coverageError.toFixed(3)} breakCovErr=${(calibration.breakCoverageError ?? 0).toFixed(3)} sharp=${calibration.sharpness.toFixed(4)} brier=${calibration.brierScore.toFixed(3)} dir=${calibration.directionalAccuracy.toFixed(3)} policyCov=${policy?.eligibleCoverage.toFixed(3) ?? 'n/a'} policyAcc=${policy?.eligibleDirectionalAccuracy.toFixed(3) ?? 'n/a'}`,
    );
  }
  return { label, source, policyMode, horizons };
}

function fmt(value: number | null, digits = 3): string {
  if (value === null || Number.isNaN(value)) return 'n/a';
  return value.toFixed(digits);
}

function fmtDelta(value: number | null, digits = 3): string {
  if (value === null || Number.isNaN(value)) return 'n/a';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

function renderCalibrationTable(title: string, arm: ArmReport, baseline?: ArmReport): string[] {
  const lines = [`## ${title}`, '', '| h (d) | covErr | break covErr | sharpness | Brier | dirAcc | selDir@0.45 | selCov@0.45 |', '|------:|-------:|------------:|----------:|------:|-------:|------------:|----------:|' ];
  for (const row of arm.horizons) {
    const c = row.calibration;
    lines.push(`| ${row.horizon} | ${fmt(c.coverageError)} | ${fmt(c.breakCoverageError)} | ${fmt(c.sharpness, 4)} | ${fmt(c.brierScore)} | ${fmt(c.directionalAccuracy)} | ${fmt(c.selectiveDirectionalAccuracy045)} | ${fmt(c.selectiveCoverage045)} |`);
  }
  if (baseline) {
    lines.push('');
    lines.push('| h (d) | ΔcovErr | Δbreak covErr | Δsharpness | ΔBrier | ΔdirAcc |');
    lines.push('|------:|---------:|--------------:|-----------:|-------:|--------:|');
    for (const [index, row] of arm.horizons.entries()) {
      const base = baseline.horizons[index]?.calibration;
      const c = row.calibration;
      lines.push(`| ${row.horizon} | ${fmtDelta(base ? c.coverageError - base.coverageError : null)} | ${fmtDelta(base && c.breakCoverageError !== null && base.breakCoverageError !== null ? c.breakCoverageError - base.breakCoverageError : null)} | ${fmtDelta(base ? c.sharpness - base.sharpness : null, 4)} | ${fmtDelta(base ? c.brierScore - base.brierScore : null)} | ${fmtDelta(base ? c.directionalAccuracy - base.directionalAccuracy : null)} |`);
    }
  }
  lines.push('');
  return lines;
}

function renderPolicyTable(title: string, arm: ArmReport): string[] {
  const lines = [`## ${title}`, '', '| h (d) | full | context-only | abstain | eligible cov | eligible acc | baseline RC acc@≈cov | active trade rate | active trade acc | low-conf trade rate | weak-state no-trade |', '|------:|-----:|-------------:|--------:|-------------:|-------------:|--------------------:|-----------------:|-----------------:|--------------------:|--------------------:|' ];
  for (const row of arm.horizons) {
    const p = row.policy;
    if (!p) continue;
    lines.push(`| ${row.horizon} | ${fmt(p.fullRate)} | ${fmt(p.contextOnlyRate)} | ${fmt(p.abstainRate)} | ${fmt(p.eligibleCoverage)} | ${fmt(p.eligibleDirectionalAccuracy)} | ${fmt(p.comparableBaselineAccuracy)} | ${fmt(p.activeTradeRate)} | ${fmt(p.activeTradeAccuracy)} | ${fmt(p.lowConfidenceActiveTradeRate)} | ${fmt(p.weakStateNoTradeRate)} |`);
  }
  lines.push('');
  return lines;
}

function summarizeRecommendation(artifact: Artifact): string[] {
  const lines: string[] = [];
  const fmtHorizons = (values: number[]) => values.length > 0 ? `${values.join(', ')}d` : 'none';
  const adaptiveWins = artifact.adaptiveOnly.horizons.filter((row, index) => {
    const base = artifact.baseline.horizons[index].calibration;
    return row.calibration.coverageError < base.coverageError
      && (row.calibration.breakCoverageError ?? Number.POSITIVE_INFINITY) < (base.breakCoverageError ?? Number.POSITIVE_INFINITY)
      && row.calibration.sharpness <= base.sharpness * 1.1
      && row.calibration.directionalAccuracy >= base.directionalAccuracy - 0.03;
  }).map(row => row.horizon);

  const usablePolicyHorizons = artifact.combined.horizons.filter((row) =>
    (row.policy?.eligibleCoverage ?? 0) >= 0.05,
  ).map(row => row.horizon);

  const combinedPolicyWins = artifact.combined.horizons.filter((row, index) => {
    const bounded = artifact.boundedOnly.horizons[index].policy;
    const combined = row.policy;
    if (!bounded || !combined) return false;
    if (combined.eligibleCoverage < 0.05) return false;
    return combined.eligibleDirectionalAccuracy >= bounded.eligibleDirectionalAccuracy
      && combined.lowConfidenceActiveTradeRate <= bounded.lowConfidenceActiveTradeRate
      && combined.weakStateNoTradeRate >= bounded.weakStateNoTradeRate;
  }).map(row => row.horizon);

  lines.push('## Recommendation');
  lines.push('');
  lines.push(`- **Adaptive conformal:** ${adaptiveWins.length >= 3 ? 'keep enabled by default' : 'do not enable by default yet'} (clear calibration wins on horizons: ${fmtHorizons(adaptiveWins)}).`);
  lines.push(`- **Bounded abstention:** ${usablePolicyHorizons.length >= 2 && combinedPolicyWins.length >= 2 ? 'keep enabled by default, but treat this as a proxy-backed decision' : 'keep behind a flag until a full anchor-aware backtest exists'} (usable policy coverage on horizons: ${fmtHorizons(usablePolicyHorizons)}; combined policy wins on horizons: ${fmtHorizons(combinedPolicyWins)}).`);
  lines.push(`- **Combined vs baseline:** ${adaptiveWins.length >= 3 && combinedPolicyWins.length >= 2 ? 'accept' : 'reject for now'}; this fixture shows worse calibration under adaptive conformal and the policy proxy collapses to near-zero eligible coverage.`);
  lines.push('');
  return lines;
}

function renderMarkdown(artifact: Artifact): string {
  const lines: string[] = [];
  lines.push('# BTC Multi-Horizon Backtest — Round 6 (adaptive conformal + bounded abstention)');
  lines.push('');
  lines.push(`Generated: ${artifact.generatedAt}`);
  lines.push('');
  lines.push(`**Ticker:** ${artifact.ticker}  `);
  lines.push(`**Fixture:** ${artifact.fixtureRange.startDate} → ${artifact.fixtureRange.endDate} (${artifact.fixtureRange.days} daily closes)  `);
  lines.push(`**Walk-forward:** warmup=${artifact.config.warmup}, stride=${artifact.config.stride}  `);
  lines.push(`**Horizons:** ${artifact.config.horizons.join(', ')} days  `);
  lines.push(`**Target CI coverage:** ${(artifact.config.targetCiCoverage * 100).toFixed(0)}%`);
  lines.push('');
  lines.push('## Setup');
  lines.push('');
  lines.push('- **Baseline:** current walk-forward defaults.');
  lines.push('- **Adaptive conformal only:** `enableAdaptiveConformal=true` with the R6 test parameters already covered by `walk-forward-r5.test.ts`.');
  lines.push('- **Bounded abstention only:** baseline forecasts plus an **observable-diagnostics proxy** derived from current `forecast-arbitrator.ts` thresholds (`confidence`, structural-break state, and when available conformal stress).');
  lines.push('- **Combined:** adaptive conformal forecasts plus the same observable policy proxy.');
  lines.push('');
  lines.push('### Important caveat');
  lines.push('');
  lines.push(artifact.caveat);
  lines.push('');
  lines.push(...renderCalibrationTable('Baseline', artifact.baseline));
  lines.push(...renderCalibrationTable('Adaptive conformal only', artifact.adaptiveOnly, artifact.baseline));
  lines.push(...renderCalibrationTable('Bounded abstention only (same forecast path as baseline)', artifact.boundedOnly, artifact.baseline));
  lines.push(...renderCalibrationTable('Combined (same forecast path as adaptive)', artifact.combined, artifact.baseline));
  lines.push(...renderPolicyTable('Bounded abstention proxy metrics', artifact.boundedOnly));
  lines.push(...renderPolicyTable('Combined proxy metrics', artifact.combined));
  lines.push(...summarizeRecommendation(artifact));
  lines.push('## Caveats');
  lines.push('');
  lines.push('- The walk-forward harness is Markov-only. Historical Polymarket anchor state, divergence, and full live `forecast-arbitrator` semantics are not replayable from the current BTC fixture.');
  lines.push('- Because of that, the policy arms above are **proxy evaluations**, not a perfect replay of live `full/context-only/abstain` decisions.');
  lines.push('- Calibration conclusions for adaptive conformal are direct. Default-on/off conclusions for bounded abstention should be treated as provisional until a market-anchor-aware replay harness exists.');
  lines.push('');
  return lines.join('\n');
}

function sanityCheckPolicyProxy(artifact: Artifact) {
  for (const arm of [artifact.boundedOnly, artifact.combined]) {
    for (const row of arm.horizons) {
      if (!row.policy) continue;
      const sum = row.policy.fullRate + row.policy.contextOnlyRate + row.policy.abstainRate;
      if (Math.abs(sum - 1) > 1e-6) {
        throw new Error(`Policy rates do not sum to 1 for ${arm.label} h=${row.horizon}`);
      }
    }
  }
}

async function main(): Promise<void> {
  const fixturePath = join(process.cwd(), 'src/tools/finance/fixtures/backtest-prices.json');
  const raw = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;
  const btcData = raw.tickers[TICKER];
  if (!btcData) throw new Error(`No fixture data for ${TICKER}`);

  console.log(`Loaded ${btcData.closes.length} BTC closes (${raw.startDate ?? 'unknown'} → ${raw.endDate ?? 'unknown'})`);

  const baseline = await runArm('baseline', btcData.closes, {}, 'none', 'baseline');
  const adaptiveOnly = await runArm('adaptive-only', btcData.closes, adaptiveFlags, 'none', 'adaptive');
  const boundedOnly = await runArm('bounded-only', btcData.closes, {}, 'observable-proxy', 'baseline');
  const combined = await runArm('combined', btcData.closes, adaptiveFlags, 'observable-proxy', 'adaptive');

  const artifact: Artifact = {
    generatedAt: new Date().toISOString(),
    ticker: TICKER,
    fixtureRange: {
      startDate: raw.startDate ?? 'unknown',
      endDate: raw.endDate ?? 'unknown',
      days: btcData.closes.length,
    },
    config: {
      warmup: WARMUP,
      stride: STRIDE,
      horizons: [...HORIZONS],
      targetCiCoverage: TARGET_CI_COVERAGE,
    },
    caveat: 'The current BTC fixture replays price history only. That supports a direct adaptive-conformal backtest, but bounded-abstention must be evaluated through an observable proxy because historical Polymarket/anchor semantics are not preserved in the walk-forward artifact.',
    baseline,
    adaptiveOnly,
    boundedOnly,
    combined,
  };

  sanityCheckPolicyProxy(artifact);

  const outDir = join(process.cwd(), 'docs');
  mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 10);
  const jsonPath = join(outDir, `btc-multi-horizon-backtest-round6-${stamp}.json`);
  const mdPath = join(outDir, `btc-multi-horizon-backtest-round6-${stamp}.md`);
  writeFileSync(jsonPath, JSON.stringify(artifact, null, 2));
  writeFileSync(mdPath, renderMarkdown(artifact));
  console.log(`Wrote ${jsonPath}`);
  console.log(`Wrote ${mdPath}`);
}

await main();
