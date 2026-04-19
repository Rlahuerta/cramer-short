/**
 * Phase 9: Promotion confirmation harness for warmup=90.
 *
 * Confirms the Phase 8 winner (warmup=90 + trendPenaltyOnlyBreakConfidence=true)
 * against the Phase 4 control (warmup=120 + trendPenaltyOnlyBreakConfidence=true)
 * on the 6-ticker 7d/14d/30d fixture universe.
 *
 * This is NOT a production default change. It IS a standalone confirmation
 * package that re-runs the warmup=90 candidate through the same Phase 8
 * guardrails, producing a formal promotion verdict artifact.
 *
 * Evidence basis (from Phase 8):
 *   - warmup=90 passed all Phase 8 guardrails
 *   - warmup=90 improved all per-horizon break-context directional accuracy
 *   - warmup=60 failed on break+sideways RC@0.2 accuracy (-1.2pp < -1.0pp guardrail)
 *   - Phase 9 confirms warmup=90 still passes under identical thresholds
 *
 * Run: bun run src/tools/finance/backtest/phase9-warmup90-promotion.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { walkForward, type WalkForwardResult } from './walk-forward.js';
import {
  brierScore,
  bootstrapDirectionalCI,
  ciCoverage,
  computeRCCurve,
  directionalAccuracy,
  type BacktestStep,
} from './metrics.js';
import { effectiveBreakContext } from './break-policy-ablation.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FixtureTickerData {
  closes: number[];
  dates: string[];
}

interface FixtureData {
  tickers: Record<string, FixtureTickerData>;
}

type AugmentedStep = BacktestStep & { ticker: string; horizon: number };

interface MetricBlock {
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  avgConfidence: number;
  directionalCi: { lower: number; median: number; upper: number; nResamples: number };
}

interface RCPoint {
  threshold: number;
  accuracy: number;
  coverage: number;
  n: number;
}

interface PerHorizonMetrics {
  horizon: number;
  steps: number;
  breakSteps: number;
  overall: MetricBlock;
  breakContext: MetricBlock;
  nonBreak: MetricBlock;
  breakTrending: MetricBlock;
  breakChop: MetricBlock;
  overallRC: RCPoint[];
  breakContextRC: RCPoint[];
  rc020: RCPoint;
  rc030: RCPoint;
}

export interface CandidateResult {
  candidateId: string;
  warmup: number;
  overall: MetricBlock;
  breakContext: MetricBlock;
  nonBreak: MetricBlock;
  breakTrending: MetricBlock;
  breakChop: MetricBlock;
  perHorizon: PerHorizonMetrics[];
  overallRC: RCPoint[];
  breakContextRC: RCPoint[];
  breakTrendingRC: RCPoint[];
  breakChopRC: RCPoint[];
  deltaVsBaseline: {
    overallDirectionalAccuracy: number;
    overallBrier: number;
    overallCiCoverage: number;
    breakContextDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    breakChopRC020Accuracy: number;
    breakChopRC020Coverage: number;
    breakContextRC020Accuracy: number;
    breakContextRC020Coverage: number;
    breakContextRC030Accuracy: number;
    breakContextRC030Coverage: number;
  };
  perHorizonDelta: Array<{
    horizon: number;
    overallDirectionalAccuracy: number;
    breakContextDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    brier: number;
    ciCoverage: number;
    breakContextRC020Accuracy: number;
    breakContextRC030Accuracy: number;
  }>;
  passesThresholds: boolean;
  failureReasons: string[];
}

export interface Phase9Artifact {
  generatedAt: string;
  baseline: { label: string; warmup: number };
  universe: {
    tickers: string[];
    horizons: number[];
    warmup: number;
    stride: number;
  };
  candidate: CandidateResult;
  verdict: { promoted: boolean; reason?: string };
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TICKERS = ['SPY', 'QQQ', 'GLD', 'VOO', 'NVDA', 'MSFT'] as const;
const HORIZONS = [7, 14, 30] as const;
const BASELINE_WARMUP = 120;
const CANDIDATE_WARMUP = 90;
const STRIDE = 5;
const RC_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] as const;

// Same guardrails as Phase 8
const OVERALL_DIRECTIONAL_MAX_LOSS = 0.005;
const NON_BREAK_DIRECTIONAL_MAX_LOSS = 0.005;
const BRIER_MAX_WORSEN = 0.003;
const CI_COVERAGE_MIN = 0.91;
const BREAK_CHOP_RC020_ACC_MAX_LOSS = 0.01;
const BREAK_CHOP_RC020_COV_MAX_LOSS = 0.02;
const HORIZON_BREAK_CONTEXT_MAX_LOSS = 0.015;
const HORIZON_14D_BREAK_CONTEXT_MIN_GAIN = 0.005;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isTrending(step: BacktestStep): boolean {
  return step.regime === 'bull' || step.regime === 'bear';
}

function isChop(step: BacktestStep): boolean {
  return step.regime === 'sideways';
}

function averageConfidence(steps: BacktestStep[]): number {
  return steps.length > 0
    ? steps.reduce((sum, step) => sum + step.confidence, 0) / steps.length
    : 0;
}

function summarizePartition(steps: BacktestStep[]): MetricBlock {
  return {
    directionalAccuracy: steps.length > 0 ? directionalAccuracy(steps) : 0,
    brierScore: steps.length > 0 ? brierScore(steps) : 0,
    ciCoverage: steps.length > 0 ? ciCoverage(steps) : 0,
    avgConfidence: averageConfidence(steps),
    directionalCi: bootstrapDirectionalCI(steps, 500, 12345),
  };
}

function findPoint(points: RCPoint[], threshold: number): RCPoint {
  const found = points.find(point => point.threshold === threshold);
  if (!found) throw new Error(`Missing threshold ${threshold}`);
  return found;
}

function summarizeArm(label: string, steps: AugmentedStep[]) {
  const breakContext = steps.filter(step => effectiveBreakContext(step));
  const nonBreak = steps.filter(step => !effectiveBreakContext(step));
  const breakTrending = breakContext.filter(isTrending);
  const breakChop = breakContext.filter(isChop);

  return {
    totalSteps: steps.length,
    breakSteps: breakContext.length,
    breakTrendingSteps: breakTrending.length,
    breakChopSteps: breakChop.length,
    overall: summarizePartition(steps),
    breakContext: summarizePartition(breakContext),
    nonBreak: summarizePartition(nonBreak),
    breakTrending: summarizePartition(breakTrending),
    breakChop: summarizePartition(breakChop),
    overallRC: computeRCCurve(steps, [...RC_THRESHOLDS]),
    breakContextRC: computeRCCurve(breakContext, [...RC_THRESHOLDS]),
    breakTrendingRC: computeRCCurve(breakTrending, [...RC_THRESHOLDS]),
    breakChopRC: computeRCCurve(breakChop, [...RC_THRESHOLDS]),
  };
}

async function loadSteps(options: {
  fixturePath: string;
  warmup: number;
  trendPenaltyOnlyBreakConfidence?: boolean;
}): Promise<AugmentedStep[]> {
  const fixture = JSON.parse(readFileSync(options.fixturePath, 'utf-8')) as FixtureData;
  const allSteps: AugmentedStep[] = [];

  for (const ticker of TICKERS) {
    const tickerData = fixture.tickers[ticker];
    if (!tickerData || tickerData.closes.length === 0) {
      throw new Error(`Missing fixture data for ${ticker}`);
    }

    for (const horizon of HORIZONS) {
      const result: WalkForwardResult = await walkForward({
        ticker,
        prices: tickerData.closes,
        horizon,
        warmup: options.warmup,
        stride: STRIDE,
        trendPenaltyOnlyBreakConfidence: options.trendPenaltyOnlyBreakConfidence,
      });

      if (result.errors.length > 0) {
        throw new Error(
          `Walk-forward produced errors for ${ticker} ${horizon}d: ${result.errors.map(e => e.error).join('; ')}`,
        );
      }

      allSteps.push(
        ...result.steps.map(step => ({
          ...step,
          ticker,
          horizon,
        })),
      );
    }
  }

  return allSteps;
}

function computePerHorizon(
  steps: AugmentedStep[],
  horizons: readonly number[],
): PerHorizonMetrics[] {
  return horizons.map(horizon => {
    const hSteps = steps.filter(s => s.horizon === horizon);
    const breakContext = hSteps.filter(step => effectiveBreakContext(step));
    const breakTrending = breakContext.filter(isTrending);
    const breakChop = breakContext.filter(isChop);
    const breakContextRC = computeRCCurve(breakContext, [...RC_THRESHOLDS]);

    return {
      horizon,
      steps: hSteps.length,
      breakSteps: breakContext.length,
      overall: summarizePartition(hSteps),
      breakContext: summarizePartition(breakContext),
      nonBreak: summarizePartition(hSteps.filter(step => !effectiveBreakContext(step))),
      breakTrending: summarizePartition(breakTrending),
      breakChop: summarizePartition(breakChop),
      overallRC: computeRCCurve(hSteps, [...RC_THRESHOLDS]),
      breakContextRC,
      rc020: findPoint(breakContextRC, 0.2),
      rc030: findPoint(breakContextRC, 0.3),
    };
  });
}

// ---------------------------------------------------------------------------
// Guardrail evaluation
// ---------------------------------------------------------------------------

export function evaluatePhase9Thresholds(params: {
  candidate: Pick<CandidateResult, 'overall' | 'deltaVsBaseline' | 'perHorizonDelta'>;
  baselinePerHorizon: PerHorizonMetrics[];
}): { passes: boolean; failureReasons: string[] } {
  const { candidate, baselinePerHorizon } = params;
  const failureReasons: string[] = [];

  // Overall guardrails: no significant regression
  if (candidate.deltaVsBaseline.overallDirectionalAccuracy < -OVERALL_DIRECTIONAL_MAX_LOSS) {
    failureReasons.push(
      `overall directional accuracy ${formatDelta(candidate.deltaVsBaseline.overallDirectionalAccuracy)} < ${formatDelta(-OVERALL_DIRECTIONAL_MAX_LOSS)} guardrail`,
    );
  }

  if (candidate.deltaVsBaseline.nonBreakDirectionalAccuracy < -NON_BREAK_DIRECTIONAL_MAX_LOSS) {
    failureReasons.push(
      `non-break directional accuracy ${formatDelta(candidate.deltaVsBaseline.nonBreakDirectionalAccuracy)} < ${formatDelta(-NON_BREAK_DIRECTIONAL_MAX_LOSS)} guardrail`,
    );
  }

  if (candidate.deltaVsBaseline.overallBrier > BRIER_MAX_WORSEN) {
    failureReasons.push(
      `overall Brier ${candidate.deltaVsBaseline.overallBrier.toFixed(4)} > +${BRIER_MAX_WORSEN.toFixed(3)} guardrail`,
    );
  }

  if (candidate.overall.ciCoverage < CI_COVERAGE_MIN) {
    failureReasons.push(
      `overall CI coverage ${formatPct(candidate.overall.ciCoverage)} < ${formatPct(CI_COVERAGE_MIN)} guardrail`,
    );
  }

  if (candidate.deltaVsBaseline.breakChopRC020Accuracy < -BREAK_CHOP_RC020_ACC_MAX_LOSS) {
    failureReasons.push(
      `break+sideways RC@0.2 accuracy ${formatDelta(candidate.deltaVsBaseline.breakChopRC020Accuracy)} < ${formatDelta(-BREAK_CHOP_RC020_ACC_MAX_LOSS)} guardrail`,
    );
  }

  if (candidate.deltaVsBaseline.breakChopRC020Coverage < -BREAK_CHOP_RC020_COV_MAX_LOSS) {
    failureReasons.push(
      `break+sideways RC@0.2 coverage ${formatDelta(candidate.deltaVsBaseline.breakChopRC020Coverage)} < ${formatDelta(-BREAK_CHOP_RC020_COV_MAX_LOSS)} guardrail`,
    );
  }

  // Per-horizon guardrails: no horizon may regress break-context DA > 1.5pp
  const horizonLosses = candidate.perHorizonDelta.filter(
    entry => entry.breakContextDirectionalAccuracy < -HORIZON_BREAK_CONTEXT_MAX_LOSS,
  );
  if (horizonLosses.length > 0) {
    failureReasons.push(
      `horizon guardrail failed: ${horizonLosses.map(entry => `${entry.horizon}d ${formatDelta(entry.breakContextDirectionalAccuracy)}`).join(', ')}`,
    );
  }

  // Evidence-based expectation: warmup=90 should show 14d break-context gain
  const h14 = candidate.perHorizonDelta.find(entry => entry.horizon === 14);
  if (h14 && h14.breakContextDirectionalAccuracy < HORIZON_14D_BREAK_CONTEXT_MIN_GAIN) {
    failureReasons.push(
      `warmup=90 expected 14d break-context gain ≥ ${formatDelta(HORIZON_14D_BREAK_CONTEXT_MIN_GAIN)}, got ${formatDelta(h14.breakContextDirectionalAccuracy)}`,
    );
  }

  return {
    passes: failureReasons.length === 0,
    failureReasons,
  };
}

// ---------------------------------------------------------------------------
// Main comparison
// ---------------------------------------------------------------------------

export async function runPromotion(
  fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json'),
): Promise<Phase9Artifact> {
  // Load baseline (Phase 4 control: warmup=120, trendPenaltyOnly=true)
  const baselineSteps = await loadSteps({
    fixturePath,
    warmup: BASELINE_WARMUP,
    trendPenaltyOnlyBreakConfidence: true,
  });
  const baseline = summarizeArm('phase4-control-w120', baselineSteps);
  const baselinePerHorizon = computePerHorizon(baselineSteps, HORIZONS);

  const baseline020 = findPoint(baseline.breakContextRC, 0.2);
  const baseline030 = findPoint(baseline.breakContextRC, 0.3);
  const baselineBreakChop020 = findPoint(baseline.breakChopRC, 0.2);

  // Load candidate (warmup=90, trendPenaltyOnly=true)
  const steps = await loadSteps({
    fixturePath,
    warmup: CANDIDATE_WARMUP,
    trendPenaltyOnlyBreakConfidence: true,
  });
  const summary = summarizeArm('warmup-90', steps);
  const perHorizon = computePerHorizon(steps, HORIZONS);

  const cand020 = findPoint(summary.breakContextRC, 0.2);
  const cand030 = findPoint(summary.breakContextRC, 0.3);
  const candBreakChop020 = findPoint(summary.breakChopRC, 0.2);

  const perHorizonDelta = perHorizon.map(horizonMetrics => {
    const baseMetrics = baselinePerHorizon.find(
      entry => entry.horizon === horizonMetrics.horizon,
    );
    if (!baseMetrics) {
      throw new Error(`Missing baseline per-horizon metrics for ${horizonMetrics.horizon}d`);
    }

    return {
      horizon: horizonMetrics.horizon,
      overallDirectionalAccuracy:
        horizonMetrics.overall.directionalAccuracy - baseMetrics.overall.directionalAccuracy,
      breakContextDirectionalAccuracy:
        horizonMetrics.breakContext.directionalAccuracy -
        baseMetrics.breakContext.directionalAccuracy,
      nonBreakDirectionalAccuracy:
        horizonMetrics.nonBreak.directionalAccuracy - baseMetrics.nonBreak.directionalAccuracy,
      brier: horizonMetrics.overall.brierScore - baseMetrics.overall.brierScore,
      ciCoverage: horizonMetrics.overall.ciCoverage - baseMetrics.overall.ciCoverage,
      breakContextRC020Accuracy:
        horizonMetrics.rc020.accuracy - baseMetrics.rc020.accuracy,
      breakContextRC030Accuracy:
        horizonMetrics.rc030.accuracy - baseMetrics.rc030.accuracy,
    };
  });

  const deltaVsBaseline = {
    overallDirectionalAccuracy:
      summary.overall.directionalAccuracy - baseline.overall.directionalAccuracy,
    overallBrier: summary.overall.brierScore - baseline.overall.brierScore,
    overallCiCoverage: summary.overall.ciCoverage - baseline.overall.ciCoverage,
    breakContextDirectionalAccuracy:
      summary.breakContext.directionalAccuracy - baseline.breakContext.directionalAccuracy,
    nonBreakDirectionalAccuracy:
      summary.nonBreak.directionalAccuracy - baseline.nonBreak.directionalAccuracy,
    breakChopRC020Accuracy:
      candBreakChop020.accuracy - baselineBreakChop020.accuracy,
    breakChopRC020Coverage:
      candBreakChop020.coverage - baselineBreakChop020.coverage,
    breakContextRC020Accuracy: cand020.accuracy - baseline020.accuracy,
    breakContextRC020Coverage: cand020.coverage - baseline020.coverage,
    breakContextRC030Accuracy: cand030.accuracy - baseline030.accuracy,
    breakContextRC030Coverage: cand030.coverage - baseline030.coverage,
  };

  const thresholdEvaluation = evaluatePhase9Thresholds({
    candidate: {
      overall: summary.overall,
      deltaVsBaseline,
      perHorizonDelta,
    },
    baselinePerHorizon,
  });

  const candidateResult: CandidateResult = {
    candidateId: 'warmup-90',
    warmup: CANDIDATE_WARMUP,
    overall: summary.overall,
    breakContext: summary.breakContext,
    nonBreak: summary.nonBreak,
    breakTrending: summary.breakTrending,
    breakChop: summary.breakChop,
    perHorizon,
    overallRC: summary.overallRC,
    breakContextRC: summary.breakContextRC,
    breakTrendingRC: summary.breakTrendingRC,
    breakChopRC: summary.breakChopRC,
    deltaVsBaseline,
    perHorizonDelta,
    passesThresholds: thresholdEvaluation.passes,
    failureReasons: thresholdEvaluation.failureReasons,
  };

  const verdict = thresholdEvaluation.passes
    ? { promoted: true }
    : { promoted: false, reason: `warmup=90 failed Phase 9 thresholds: ${thresholdEvaluation.failureReasons.join('; ')}` };

  return {
    generatedAt: new Date().toISOString(),
    baseline: { label: 'phase4-control', warmup: BASELINE_WARMUP },
    universe: {
      tickers: [...TICKERS],
      horizons: [...HORIZONS],
      warmup: BASELINE_WARMUP,
      stride: STRIDE,
    },
    candidate: candidateResult,
    verdict,
  };
}

// ---------------------------------------------------------------------------
// Summary printer
// ---------------------------------------------------------------------------

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDelta(delta: number): string {
  const sign = delta >= 0 ? '+' : '';
  return `${sign}${(delta * 100).toFixed(1)}pp`;
}

export function printSummary(artifact: Phase9Artifact): void {
  console.log('Phase 9: warmup=90 Promotion Confirmation');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(
    `Universe: ${artifact.universe.tickers.join(', ')} | horizons=${artifact.universe.horizons.join(', ')}d`,
  );
  console.log(`Baseline: ${artifact.baseline.label} (warmup=${artifact.baseline.warmup})`);
  console.log(`Candidate: warmup-90 (warmup=${artifact.candidate.warmup})`);
  console.log('');

  const c = artifact.candidate;
  console.log('Overall deltas vs baseline:');
  console.log(
    `  warmup-90          | Δdir=${formatDelta(c.deltaVsBaseline.overallDirectionalAccuracy)} | Δbrier=${c.deltaVsBaseline.overallBrier.toFixed(4)} | ΔciCov=${formatDelta(c.deltaVsBaseline.overallCiCoverage)} | ΔbrkCtx=${formatDelta(c.deltaVsBaseline.breakContextDirectionalAccuracy)} | ΔnonBrk=${formatDelta(c.deltaVsBaseline.nonBreakDirectionalAccuracy)} | ${c.passesThresholds ? 'PASS' : 'FAIL'}`,
  );
  if (!c.passesThresholds && c.failureReasons.length > 0) {
    console.log(`    reasons: ${c.failureReasons.join('; ')}`);
  }

  console.log('');
  console.log('Per-horizon break-context deltas:');
  const horizonParts = c.perHorizonDelta.map(
    entry => `${entry.horizon}d=${formatDelta(entry.breakContextDirectionalAccuracy)}`,
  );
  console.log(
    `  warmup-90          | ${horizonParts.join(' | ')}`,
  );

  console.log('');
  console.log('RC@0.2/0.3 deltas (break-context):');
  console.log(
    `  warmup-90          | Δrc020=${formatDelta(c.deltaVsBaseline.breakContextRC020Accuracy)}/${formatDelta(c.deltaVsBaseline.breakContextRC020Coverage)} | Δrc030=${formatDelta(c.deltaVsBaseline.breakContextRC030Accuracy)}/${formatDelta(c.deltaVsBaseline.breakContextRC030Coverage)}`,
  );

  console.log('');
  if (artifact.verdict.promoted) {
    console.log('Verdict: PROMOTED — warmup=90 passes all Phase 9 confirmation thresholds.');
    console.log('  warmup=90 remains experimental / non-default in runtime code.');
  } else {
    console.log('Verdict: NOT PROMOTED');
    if (artifact.verdict.reason) {
      console.log(`  ${artifact.verdict.reason}`);
    }
  }
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

export async function main(): Promise<void> {
  const artifact = await runPromotion();
  const artifactDir = join(process.cwd(), '.sisyphus', 'artifacts');
  const artifactPath = join(artifactDir, 'phase9-warmup90-promotion.json');

  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));

  printSummary(artifact);
  console.log('');
  console.log(`Saved artifact to ${artifactPath}`);
}

if (import.meta.main) {
  main().catch(error => {
    console.error('Fatal:', error);
    process.exit(1);
  });
}