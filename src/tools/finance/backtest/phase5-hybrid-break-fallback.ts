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
import {
  type BreakFallbackCandidate,
} from '../markov-distribution.js';

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
  rc020: RCPoint;
  rc030: RCPoint;
}

interface CandidateResult {
  candidateId: string;
  mode: string;
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
    breakTrendingDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    overallBrier: number;
    overallCiCoverage: number;
    breakChopRC020Accuracy: number;
    breakChopRC020Coverage: number;
    breakContextRC020Accuracy: number;
    breakContextRC020Coverage: number;
  };
  passesThresholds: boolean;
  failureReasons: string[];
}

interface Phase5Artifact {
  generatedAt: string;
  baseline: { candidateId: string };
  universe: {
    tickers: string[];
    horizons: number[];
    warmup: number;
    stride: number;
  };
  candidates: CandidateResult[];
  winner: { candidateId: string | null; reason?: string };
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TICKERS = ['SPY', 'QQQ', 'GLD', 'VOO', 'NVDA', 'MSFT'] as const;
const HORIZONS = [7, 14, 30] as const;
const WARMUP = 120;
const STRIDE = 5;
const RC_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] as const;

const PRIMARY_TARGET_MIN_GAIN = 0.02;
const NON_BREAK_MAX_LOSS = 0.005;
const BRIER_MAX_WORSEN = 0.005;
const CI_COVERAGE_MIN = 0.91;
const BREAK_CHOP_RC020_ACC_MAX_LOSS = 0.01;
const BREAK_CHOP_RC020_COV_MAX_LOSS = 0.02;
const HORIZON_BREAK_CONTEXT_MAX_LOSS = 0.015;
const HORIZON_BREAK_CONTEXT_MIN_GAIN = 0.03;

// ---------------------------------------------------------------------------
// Candidate families (from Phase 5 plan)
// ---------------------------------------------------------------------------

const CANDIDATES: BreakFallbackCandidate[] = [
  // Stage B1 — Conservative-only family
  {
    id: 'C55',
    mode: 'hard',
    conservativeDiagonal: 0.55,
    profileDiagonals: { equity: 0.55, etf: 0.55, commodity: 0.55, crypto: 0.55 },
    conservativeWeight: 1.0,
    severityWeights: { mild: 1.0, medium: 1.0, high: 1.0 },
  },
  {
    id: 'C60',
    mode: 'hard',
    conservativeDiagonal: 0.60,
    profileDiagonals: { equity: 0.60, etf: 0.60, commodity: 0.60, crypto: 0.60 },
    conservativeWeight: 1.0,
    severityWeights: { mild: 1.0, medium: 1.0, high: 1.0 },
  },
  {
    id: 'C65',
    mode: 'hard',
    conservativeDiagonal: 0.65,
    profileDiagonals: { equity: 0.65, etf: 0.65, commodity: 0.65, crypto: 0.65 },
    conservativeWeight: 1.0,
    severityWeights: { mild: 1.0, medium: 1.0, high: 1.0 },
  },

  // Stage B2 — Profile-only family
  {
    id: 'P_BALANCED',
    mode: 'hard',
    conservativeDiagonal: 0.55,
    profileDiagonals: { etf: 0.55, equity: 0.60, commodity: 0.65, crypto: 0.70 },
    conservativeWeight: 0.0,
    severityWeights: { mild: 1.0, medium: 1.0, high: 1.0 },
  },
  {
    id: 'P_STICKY',
    mode: 'hard',
    conservativeDiagonal: 0.60,
    profileDiagonals: { etf: 0.60, equity: 0.65, commodity: 0.70, crypto: 0.75 },
    conservativeWeight: 0.0,
    severityWeights: { mild: 1.0, medium: 1.0, high: 1.0 },
  },

  // Stage B3 — Hybrid family (using severity curve: mild=0.25, medium=0.50, high=0.75)
  {
    id: 'HYB_L025_M050_H075_lambda025',
    mode: 'blended',
    conservativeDiagonal: 0.60,
    profileDiagonals: { etf: 0.55, equity: 0.60, commodity: 0.65, crypto: 0.70 },
    conservativeWeight: 0.25,
    severityWeights: { mild: 0.25, medium: 0.50, high: 0.75 },
  },
  {
    id: 'HYB_L025_M050_H075_lambda050',
    mode: 'blended',
    conservativeDiagonal: 0.60,
    profileDiagonals: { etf: 0.55, equity: 0.60, commodity: 0.65, crypto: 0.70 },
    conservativeWeight: 0.50,
    severityWeights: { mild: 0.25, medium: 0.50, high: 0.75 },
  },
  {
    id: 'HYB_L025_M050_H075_lambda075',
    mode: 'blended',
    conservativeDiagonal: 0.60,
    profileDiagonals: { etf: 0.55, equity: 0.60, commodity: 0.65, crypto: 0.70 },
    conservativeWeight: 0.75,
    severityWeights: { mild: 0.25, medium: 0.50, high: 0.75 },
  },
  {
    id: 'HYB_L025_M050_H075_C060_cap',
    mode: 'blended_capped',
    conservativeDiagonal: 0.60,
    profileDiagonals: { etf: 0.55, equity: 0.60, commodity: 0.65, crypto: 0.70 },
    conservativeWeight: 0.50,
    severityWeights: { mild: 0.25, medium: 0.50, high: 0.75 },
    maxBlendWeight: 0.60,
  },
];

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

function summarizeCandidate(
  label: string,
  steps: AugmentedStep[],
  thresholds: readonly number[] = RC_THRESHOLDS,
): {
  totalSteps: number;
  breakSteps: number;
  breakTrendingSteps: number;
  breakChopSteps: number;
  overall: MetricBlock;
  breakContext: MetricBlock;
  nonBreak: MetricBlock;
  breakTrending: MetricBlock;
  breakChop: MetricBlock;
  overallRC: RCPoint[];
  breakContextRC: RCPoint[];
  breakTrendingRC: RCPoint[];
  breakChopRC: RCPoint[];
} {
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
    overallRC: computeRCCurve(steps, [...thresholds]),
    breakContextRC: computeRCCurve(breakContext, [...thresholds]),
    breakTrendingRC: computeRCCurve(breakTrending, [...thresholds]),
    breakChopRC: computeRCCurve(breakChop, [...thresholds]),
  };
}

async function loadSteps(
  fixturePath: string,
  candidate?: BreakFallbackCandidate,
): Promise<AugmentedStep[]> {
  const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;
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
        warmup: WARMUP,
        stride: STRIDE,
        trendPenaltyOnlyBreakConfidence: true,
        breakFallbackCandidate: candidate,
      });

      if (result.errors.length > 0) {
        throw new Error(`Walk-forward produced errors for ${ticker} ${horizon}d: ${result.errors.map(e => e.error).join('; ')}`);
      }

      allSteps.push(...result.steps.map(step => ({
        ...step,
        ticker,
        horizon,
        breakFallbackCandidateId: candidate?.id ?? undefined,
        breakFallbackMode: candidate?.mode ?? undefined,
      })));
    }
  }

  return allSteps;
}

function computePerHorizon(
  steps: AugmentedStep[],
  horizons: readonly number[],
  thresholds: readonly number[] = RC_THRESHOLDS,
): PerHorizonMetrics[] {
  return horizons.map(horizon => {
    const hSteps = steps.filter(s => s.horizon === horizon);
    const breakContext = hSteps.filter(step => effectiveBreakContext(step));
    const breakTrending = breakContext.filter(isTrending);
    const breakChop = breakContext.filter(isChop);

    return {
      horizon,
      steps: hSteps.length,
      breakSteps: breakContext.length,
      overall: summarizePartition(hSteps),
      breakContext: summarizePartition(breakContext),
      nonBreak: summarizePartition(hSteps.filter(step => !effectiveBreakContext(step))),
      breakTrending: summarizePartition(breakTrending),
      breakChop: summarizePartition(breakChop),
      rc020: findPoint(computeRCCurve(breakContext, [...thresholds]), 0.2),
      rc030: findPoint(computeRCCurve(breakContext, [...thresholds]), 0.3),
    };
  });
}

export function evaluatePhase5Thresholds(params: {
  candidate: Pick<CandidateResult, 'overall' | 'deltaVsBaseline' | 'perHorizon'>;
  baselinePerHorizon: PerHorizonMetrics[];
}): { passes: boolean; failureReasons: string[] } {
  const { candidate, baselinePerHorizon } = params;
  const failureReasons: string[] = [];

  if (candidate.deltaVsBaseline.breakTrendingDirectionalAccuracy < PRIMARY_TARGET_MIN_GAIN) {
    failureReasons.push(
      `break+trending gain ${formatDelta(candidate.deltaVsBaseline.breakTrendingDirectionalAccuracy)} < +2.0pp`,
    );
  }

  if (candidate.deltaVsBaseline.nonBreakDirectionalAccuracy < -NON_BREAK_MAX_LOSS) {
    failureReasons.push(
      `non-break directional accuracy ${formatDelta(candidate.deltaVsBaseline.nonBreakDirectionalAccuracy)} < -0.5pp guardrail`,
    );
  }

  if (candidate.deltaVsBaseline.overallBrier > BRIER_MAX_WORSEN) {
    failureReasons.push(
      `overall Brier ${(candidate.deltaVsBaseline.overallBrier).toFixed(4)} > +0.005 guardrail`,
    );
  }

  if (candidate.overall.ciCoverage < CI_COVERAGE_MIN) {
    failureReasons.push(
      `overall CI coverage ${formatPct(candidate.overall.ciCoverage)} < 91.0% guardrail`,
    );
  }

  if (candidate.deltaVsBaseline.breakChopRC020Accuracy < -BREAK_CHOP_RC020_ACC_MAX_LOSS) {
    failureReasons.push(
      `break+sideways RC@0.2 accuracy ${formatDelta(candidate.deltaVsBaseline.breakChopRC020Accuracy)} < -1.0pp guardrail`,
    );
  }

  if (candidate.deltaVsBaseline.breakChopRC020Coverage < -BREAK_CHOP_RC020_COV_MAX_LOSS) {
    failureReasons.push(
      `break+sideways RC@0.2 coverage ${formatDelta(candidate.deltaVsBaseline.breakChopRC020Coverage)} < -2.0pp guardrail`,
    );
  }

  const perHorizonDeltas = candidate.perHorizon.map(horizonMetrics => {
    const baselineMetrics = baselinePerHorizon.find(entry => entry.horizon === horizonMetrics.horizon);
    if (!baselineMetrics) {
      throw new Error(`Missing baseline per-horizon metrics for ${horizonMetrics.horizon}d`);
    }

    return {
      horizon: horizonMetrics.horizon,
      delta: horizonMetrics.breakContext.directionalAccuracy - baselineMetrics.breakContext.directionalAccuracy,
    };
  });

  const hasStrongOffsettingGain = perHorizonDeltas.some(entry => entry.delta >= HORIZON_BREAK_CONTEXT_MIN_GAIN);
  const horizonLosses = perHorizonDeltas.filter(entry => entry.delta < -HORIZON_BREAK_CONTEXT_MAX_LOSS);
  if (horizonLosses.length > 0 && !hasStrongOffsettingGain) {
    failureReasons.push(
      `horizon guardrail failed: ${horizonLosses.map(entry => `${entry.horizon}d ${formatDelta(entry.delta)}`).join(', ')}`,
    );
  }

  return {
    passes: failureReasons.length === 0,
    failureReasons,
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export async function runComparison(
  fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json'),
): Promise<Phase5Artifact> {
  // Load baseline (Phase 4 control: trendPenaltyOnlyBreakConfidence=true, no fallback candidate)
  const baselineSteps = await loadSteps(fixturePath);
  const baselineSummary = summarizeCandidate('phase4-control', baselineSteps);
  const baselinePerHorizon = computePerHorizon(baselineSteps, HORIZONS);

  const baseline020 = findPoint(baselineSummary.breakContextRC, 0.2);
  const baselineBreakChop020 = findPoint(baselineSummary.breakChopRC, 0.2);

  const candidateResults: CandidateResult[] = [];

  for (const candidate of CANDIDATES) {
    const steps = await loadSteps(fixturePath, candidate);
    const summary = summarizeCandidate(candidate.id, steps);
    const perHorizon = computePerHorizon(steps, HORIZONS);

    const cand020 = findPoint(summary.breakContextRC, 0.2);
    const candBreakChop020 = findPoint(summary.breakChopRC, 0.2);

    const thresholdEvaluation = evaluatePhase5Thresholds({
      candidate: {
        overall: summary.overall,
        deltaVsBaseline: {
          breakTrendingDirectionalAccuracy: summary.breakTrending.directionalAccuracy - baselineSummary.breakTrending.directionalAccuracy,
          nonBreakDirectionalAccuracy: summary.nonBreak.directionalAccuracy - baselineSummary.nonBreak.directionalAccuracy,
          overallBrier: summary.overall.brierScore - baselineSummary.overall.brierScore,
          overallCiCoverage: summary.overall.ciCoverage - baselineSummary.overall.ciCoverage,
          breakChopRC020Accuracy: candBreakChop020.accuracy - baselineBreakChop020.accuracy,
          breakChopRC020Coverage: candBreakChop020.coverage - baselineBreakChop020.coverage,
          breakContextRC020Accuracy: cand020.accuracy - baseline020.accuracy,
          breakContextRC020Coverage: cand020.coverage - baseline020.coverage,
        },
        perHorizon,
      },
      baselinePerHorizon,
    });

    candidateResults.push({
      candidateId: candidate.id,
      mode: candidate.mode,
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
      deltaVsBaseline: {
        breakTrendingDirectionalAccuracy: summary.breakTrending.directionalAccuracy - baselineSummary.breakTrending.directionalAccuracy,
        nonBreakDirectionalAccuracy: summary.nonBreak.directionalAccuracy - baselineSummary.nonBreak.directionalAccuracy,
        overallBrier: summary.overall.brierScore - baselineSummary.overall.brierScore,
        overallCiCoverage: summary.overall.ciCoverage - baselineSummary.overall.ciCoverage,
        breakChopRC020Accuracy: candBreakChop020.accuracy - baselineBreakChop020.accuracy,
        breakChopRC020Coverage: candBreakChop020.coverage - baselineBreakChop020.coverage,
        breakContextRC020Accuracy: cand020.accuracy - baseline020.accuracy,
        breakContextRC020Coverage: cand020.coverage - baseline020.coverage,
      },
      passesThresholds: thresholdEvaluation.passes,
      failureReasons: thresholdEvaluation.failureReasons,
    });
  }

  const qualifyingCandidates = candidateResults.filter(candidate => candidate.passesThresholds);

  const winner = qualifyingCandidates.length > 0
    ? { candidateId: qualifyingCandidates.sort((a, b) =>
        b.deltaVsBaseline.breakTrendingDirectionalAccuracy - a.deltaVsBaseline.breakTrendingDirectionalAccuracy
      )[0].candidateId, reason: undefined }
    : { candidateId: null, reason: 'No candidate passed all Phase 5 thresholds against the Phase 4 baseline' };

  return {
    generatedAt: new Date().toISOString(),
    baseline: { candidateId: 'phase4-control' },
    universe: {
      tickers: [...TICKERS],
      horizons: [...HORIZONS],
      warmup: WARMUP,
      stride: STRIDE,
    },
    candidates: candidateResults,
    winner,
  };
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDelta(delta: number): string {
  const sign = delta >= 0 ? '+' : '';
  return `${sign}${(delta * 100).toFixed(1)}pp`;
}

export function printSummary(artifact: Phase5Artifact): void {
  console.log('Phase 5: Hybrid Structural-Break Fallback Matrix Evaluation');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`Universe: ${artifact.universe.tickers.join(', ')} | horizons=${artifact.universe.horizons.join(', ')}d`);
  console.log(`Baseline: ${artifact.baseline.candidateId}`);
  console.log('');

  console.log('Candidate comparison:');
  for (const c of artifact.candidates) {
    const trendDAAcc = c.breakTrending.directionalAccuracy;
    const nonBrkDAAcc = c.nonBreak.directionalAccuracy;
    console.log(
      `  ${c.candidateId.padEnd(36)} | trend+brk=${formatPct(trendDAAcc)} | non-brk=${formatPct(nonBrkDAAcc)} | Δtrend+brk=${formatDelta(c.deltaVsBaseline.breakTrendingDirectionalAccuracy)} | Δbrier=${(c.deltaVsBaseline.overallBrier).toFixed(4)} | CIcov=${formatPct(c.overall.ciCoverage)}`,
    );
  }

  console.log('');
  console.log('Delta vs baseline (key thresholds):');
  for (const c of artifact.candidates) {
    console.log(
      `  ${c.candidateId.padEnd(36)} | Δtrend+brk=${formatDelta(c.deltaVsBaseline.breakTrendingDirectionalAccuracy)} | ΔnonBrk=${formatDelta(c.deltaVsBaseline.nonBreakDirectionalAccuracy)} | Δbrier=${(c.deltaVsBaseline.overallBrier).toFixed(4)} | ΔchopRC020=${formatDelta(c.deltaVsBaseline.breakChopRC020Accuracy)}/${formatDelta(c.deltaVsBaseline.breakChopRC020Coverage)} | ${c.passesThresholds ? 'PASS' : 'FAIL'}`,
    );
    if (!c.passesThresholds && c.failureReasons.length > 0) {
      console.log(`    reasons: ${c.failureReasons.join('; ')}`);
    }
  }

  console.log('');
  if (artifact.winner && 'candidateId' in artifact.winner && artifact.winner.candidateId) {
    console.log(`Winner: ${artifact.winner.candidateId}`);
  } else {
    console.log('Result: No candidate passed all Phase 5 thresholds.');
  }
}

export async function main(): Promise<void> {
  const artifact = await runComparison();
  const artifactDir = join(process.cwd(), '.sisyphus', 'artifacts');
  const artifactPath = join(artifactDir, 'phase5-hybrid-break-fallback.json');

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
