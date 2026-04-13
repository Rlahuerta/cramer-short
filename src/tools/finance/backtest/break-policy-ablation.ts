import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { walkForward, type WalkForwardResult } from './walk-forward.js';
import {
  brierScore,
  bootstrapDirectionalCI,
  ciCoverage,
  computeRCCurve,
  directionalAccuracy,
  selectivePUpAccuracy,
  selectiveRawPUpAccuracy,
  type BacktestStep,
} from './metrics.js';

interface FixtureTickerData {
  closes: number[];
  dates: string[];
}

interface FixtureData {
  tickers: Record<string, FixtureTickerData>;
}

export interface PolicySummaryPoint {
  threshold: number;
  accuracy: number;
  coverage: number;
  n: number;
}

export interface PolicyPUpSummaryPoint {
  threshold: number;
  accuracy: number;
  coverage: number;
  selected: number;
  total: number;
}

export interface ConfidencePolicy {
  name: string;
  description: string;
  apply: (step: BacktestStep) => number;
}

interface PartitionSummary {
  count: number;
  avgConfidence: number;
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  directionalBand: { lower: number; median: number; upper: number; nResamples: number };
}

interface ConditionalPartitionSummary extends PartitionSummary {
  label: string;
  recommendationRC: PolicySummaryPoint[];
}

export interface PolicyResult {
  name: string;
  description: string;
  overall: PartitionSummary;
  breakContext: PartitionSummary;
  nonBreak: PartitionSummary;
  recommendationRC: {
    breakContext: PolicySummaryPoint[];
    nonBreak: PolicySummaryPoint[];
  };
  pUpRC: {
    breakContext: PolicyPUpSummaryPoint[];
    nonBreak: PolicyPUpSummaryPoint[];
  };
  rawPUpRC: {
    breakContext: PolicyPUpSummaryPoint[];
    nonBreak: PolicyPUpSummaryPoint[];
  };
  conditionalBreakContexts: {
    breakTrending: ConditionalPartitionSummary;
    breakChop: ConditionalPartitionSummary;
    breakLargeMove: ConditionalPartitionSummary;
  };
}

export interface PolicyDelta {
  name: string;
  breakAvgConfidenceDelta: number;
  breakAccuracyDelta: number;
  breakBrierDelta: number;
  breakRecommendationAt020Delta: number;
  breakRecommendationAt020CoverageDelta: number;
  breakRecommendationAt030Delta: number;
  breakRecommendationAt030CoverageDelta: number;
  breakPUpAt020Delta: number;
  breakPUpAt020CoverageDelta: number;
  breakChopAt020Delta: number;
  breakChopAt020CoverageDelta: number;
  breakTrendingAt020Delta: number;
  breakTrendingAt020CoverageDelta: number;
}

export interface AblationArtifact {
  generatedAt: string;
  universe: {
    tickers: string[];
    horizons: number[];
    warmup: number;
    stride: number;
  };
  totalSteps: number;
  breakStepCount: number;
  thresholds: number[];
  policies: PolicyResult[];
  pairwiseVsBaseline: PolicyDelta[];
}

const TICKERS = ['SPY', 'QQQ', 'GLD', 'VOO', 'NVDA', 'MSFT'] as const;
const HORIZONS = [7, 14, 30] as const;
const WARMUP = 120;
const STRIDE = 5;
const CONFIDENCE_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] as const;
const BASE_BREAK_PENALTY = 0.6;

export function effectiveBreakContext(step: BacktestStep): boolean {
  return (step.originalStructuralBreakDetected ?? step.structuralBreakDetected) === true;
}

export function effectiveDivergence(step: BacktestStep): number | null {
  return step.originalStructuralBreakDivergence ?? step.structuralBreakDivergence ?? null;
}

export function reconstructPreBreakConfidence(step: BacktestStep): number {
  if (!effectiveBreakContext(step)) return clamp01(step.confidence);

  const skipBreakPenalty = step.trendPenaltyOnlyBreakConfidenceActive === true && isChop(step);
  if (skipBreakPenalty) return clamp01(step.confidence);

  return clamp01(step.confidence / BASE_BREAK_PENALTY);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function hasLargeMove(step: BacktestStep): boolean {
  return Math.abs(step.actualReturn) >= 0.10;
}

function isTrending(step: BacktestStep): boolean {
  return step.regime === 'bull' || step.regime === 'bear';
}

function isChop(step: BacktestStep): boolean {
  return step.regime === 'sideways';
}

function withConfidence(step: BacktestStep, confidence: number): BacktestStep {
  return { ...step, confidence: clamp01(confidence) };
}

export const CANDIDATE_POLICIES: ConfidencePolicy[] = [
  {
    name: 'baseline',
    description: 'Current production-equivalent break penalty (0.6) preserved from stored confidence.',
    apply: step => step.confidence,
  },
  {
    name: 'reduced_penalty_075',
    description: 'Relax break penalty from 0.6 to 0.75 for all break-context steps.',
    apply: step => effectiveBreakContext(step)
      ? reconstructPreBreakConfidence(step) * 0.75
      : step.confidence,
  },
  {
    name: 'no_break_penalty',
    description: 'Remove break penalty entirely for break-context steps.',
    apply: step => effectiveBreakContext(step)
      ? reconstructPreBreakConfidence(step)
      : step.confidence,
  },
  {
    name: 'aggressive_penalty_040',
    description: 'Strengthen break penalty from 0.6 to 0.4 for all break-context steps.',
    apply: step => effectiveBreakContext(step)
      ? reconstructPreBreakConfidence(step) * 0.40
      : step.confidence,
  },
  {
    name: 'cap_029',
    description: 'Allow break-context confidence to rise only up to 0.29, suppressing high-confidence promotion.',
    apply: step => effectiveBreakContext(step)
      ? Math.min(reconstructPreBreakConfidence(step), 0.29)
      : step.confidence,
  },
  {
    name: 'trend_penalty_only',
    description: 'Keep the 0.6 penalty only in break+trending contexts; remove it in break+chop.',
    apply: step => {
      if (!effectiveBreakContext(step)) return step.confidence;
      const preBreak = reconstructPreBreakConfidence(step);
      return isTrending(step) ? preBreak * 0.6 : preBreak;
    },
  },
  {
    name: 'chop_penalty_only',
    description: 'Keep the 0.6 penalty only in break+chop contexts; remove it in break+trending.',
    apply: step => {
      if (!effectiveBreakContext(step)) return step.confidence;
      const preBreak = reconstructPreBreakConfidence(step);
      return isChop(step) ? preBreak * 0.6 : preBreak;
    },
  },
  {
    name: 'divergence_relief_bucketed',
    description: 'Lighten the break penalty as structural-break divergence increases.',
    apply: step => {
      if (!effectiveBreakContext(step)) return step.confidence;
      const divergence = effectiveDivergence(step) ?? 0.05;
      const penalty = divergence >= 0.20 ? 0.90 : divergence >= 0.10 ? 0.75 : 0.60;
      return reconstructPreBreakConfidence(step) * penalty;
    },
  },
];

function averageConfidence(steps: BacktestStep[]): number {
  return steps.length > 0
    ? steps.reduce((sum, step) => sum + step.confidence, 0) / steps.length
    : 0;
}

function summarizePartition(steps: BacktestStep[]): PartitionSummary {
  return {
    count: steps.length,
    avgConfidence: averageConfidence(steps),
    directionalAccuracy: steps.length > 0 ? directionalAccuracy(steps) : 0,
    brierScore: steps.length > 0 ? brierScore(steps) : 0,
    ciCoverage: steps.length > 0 ? ciCoverage(steps) : 0,
    directionalBand: bootstrapDirectionalCI(steps, 500, 12345),
  };
}

function summarizeConditional(
  label: string,
  steps: BacktestStep[],
  thresholds: readonly number[],
): ConditionalPartitionSummary {
  return {
    label,
    ...summarizePartition(steps),
    recommendationRC: computeRCCurve(steps, [...thresholds]),
  };
}

function findPoint<T extends { threshold: number }>(points: T[], threshold: number): T {
  const found = points.find(point => point.threshold === threshold);
  if (!found) throw new Error(`Missing threshold ${threshold}`);
  return found;
}

export function applyPolicyToSteps(
  steps: BacktestStep[],
  policy: ConfidencePolicy,
): BacktestStep[] {
  return steps.map(step => withConfidence(step, policy.apply(step)));
}

export function simulatePolicy(
  steps: BacktestStep[],
  policy: ConfidencePolicy,
  thresholds: readonly number[] = CONFIDENCE_THRESHOLDS,
): PolicyResult {
  const adjustedSteps = applyPolicyToSteps(steps, policy);
  const breakContextSteps = adjustedSteps.filter(step => effectiveBreakContext(step));
  const nonBreakSteps = adjustedSteps.filter(step => !effectiveBreakContext(step));

  const breakTrending = breakContextSteps.filter(isTrending);
  const breakChop = breakContextSteps.filter(isChop);
  const breakLargeMove = breakContextSteps.filter(hasLargeMove);

  return {
    name: policy.name,
    description: policy.description,
    overall: summarizePartition(adjustedSteps),
    breakContext: summarizePartition(breakContextSteps),
    nonBreak: summarizePartition(nonBreakSteps),
    recommendationRC: {
      breakContext: computeRCCurve(breakContextSteps, [...thresholds]),
      nonBreak: computeRCCurve(nonBreakSteps, [...thresholds]),
    },
    pUpRC: {
      breakContext: thresholds.map(threshold => ({ threshold, ...selectivePUpAccuracy(breakContextSteps, threshold) })),
      nonBreak: thresholds.map(threshold => ({ threshold, ...selectivePUpAccuracy(nonBreakSteps, threshold) })),
    },
    rawPUpRC: {
      breakContext: thresholds.map(threshold => ({ threshold, ...selectiveRawPUpAccuracy(breakContextSteps, threshold) })),
      nonBreak: thresholds.map(threshold => ({ threshold, ...selectiveRawPUpAccuracy(nonBreakSteps, threshold) })),
    },
    conditionalBreakContexts: {
      breakTrending: summarizeConditional('break+trending', breakTrending, thresholds),
      breakChop: summarizeConditional('break+chop', breakChop, thresholds),
      breakLargeMove: summarizeConditional('break+large-move', breakLargeMove, thresholds),
    },
  };
}

export function simulateAllPolicies(
  steps: BacktestStep[],
  policies: ConfidencePolicy[] = CANDIDATE_POLICIES,
  thresholds: readonly number[] = CONFIDENCE_THRESHOLDS,
): PolicyResult[] {
  return policies.map(policy => simulatePolicy(steps, policy, thresholds));
}

export function computePolicyDeltas(results: PolicyResult[]): PolicyDelta[] {
  const baseline = results.find(result => result.name === 'baseline');
  if (!baseline) throw new Error('Missing baseline policy result');

  const baselineBreak020 = findPoint(baseline.recommendationRC.breakContext, 0.2);
  const baselineBreak030 = findPoint(baseline.recommendationRC.breakContext, 0.3);
  const baselineBreakPUp020 = findPoint(baseline.pUpRC.breakContext, 0.2);
  const baselineBreakChop020 = findPoint(baseline.conditionalBreakContexts.breakChop.recommendationRC, 0.2);
  const baselineBreakTrending020 = findPoint(baseline.conditionalBreakContexts.breakTrending.recommendationRC, 0.2);

  return results.map(result => {
    const break020 = findPoint(result.recommendationRC.breakContext, 0.2);
    const break030 = findPoint(result.recommendationRC.breakContext, 0.3);
    const breakPUp020 = findPoint(result.pUpRC.breakContext, 0.2);
    const breakChop020 = findPoint(result.conditionalBreakContexts.breakChop.recommendationRC, 0.2);
    const breakTrending020 = findPoint(result.conditionalBreakContexts.breakTrending.recommendationRC, 0.2);

    return {
      name: result.name,
      breakAvgConfidenceDelta: result.breakContext.avgConfidence - baseline.breakContext.avgConfidence,
      breakAccuracyDelta: result.breakContext.directionalAccuracy - baseline.breakContext.directionalAccuracy,
      breakBrierDelta: result.breakContext.brierScore - baseline.breakContext.brierScore,
      breakRecommendationAt020Delta: break020.accuracy - baselineBreak020.accuracy,
      breakRecommendationAt020CoverageDelta: break020.coverage - baselineBreak020.coverage,
      breakRecommendationAt030Delta: break030.accuracy - baselineBreak030.accuracy,
      breakRecommendationAt030CoverageDelta: break030.coverage - baselineBreak030.coverage,
      breakPUpAt020Delta: breakPUp020.accuracy - baselineBreakPUp020.accuracy,
      breakPUpAt020CoverageDelta: breakPUp020.coverage - baselineBreakPUp020.coverage,
      breakChopAt020Delta: breakChop020.accuracy - baselineBreakChop020.accuracy,
      breakChopAt020CoverageDelta: breakChop020.coverage - baselineBreakChop020.coverage,
      breakTrendingAt020Delta: breakTrending020.accuracy - baselineBreakTrending020.accuracy,
      breakTrendingAt020CoverageDelta: breakTrending020.coverage - baselineBreakTrending020.coverage,
    };
  });
}

export function generateAblationReport(
  steps: BacktestStep[],
  results: PolicyResult[],
  thresholds: readonly number[] = CONFIDENCE_THRESHOLDS,
): AblationArtifact {
  const breakStepCount = steps.filter(step => effectiveBreakContext(step)).length;
  return {
    generatedAt: new Date().toISOString(),
    universe: {
      tickers: [...TICKERS],
      horizons: [...HORIZONS],
      warmup: WARMUP,
      stride: STRIDE,
    },
    totalSteps: steps.length,
    breakStepCount,
    thresholds: [...thresholds],
    policies: results,
    pairwiseVsBaseline: computePolicyDeltas(results),
  };
}

async function loadBacktestSteps(): Promise<BacktestStep[]> {
  const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
  const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;

  const allSteps: BacktestStep[] = [];
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
      });
      if (result.errors.length > 0) {
        throw new Error(`Walk-forward produced errors for ${ticker} ${horizon}d`);
      }
      allSteps.push(...result.steps);
    }
  }

  return allSteps;
}

function printPolicySummary(artifact: AblationArtifact): void {
  console.log('Markov Break-Policy Confidence Ablation');
  console.log('═══════════════════════════════════════════════════');
  console.log(`Universe: ${artifact.universe.tickers.join(', ')} | horizons=${artifact.universe.horizons.join(', ')}d`);
  console.log(`Total steps: ${artifact.totalSteps} | break-context steps: ${artifact.breakStepCount}`);
  console.log('');

  console.log('Policy summary:');
  for (const result of artifact.policies) {
    const break020 = findPoint(result.recommendationRC.breakContext, 0.2);
    const break030 = findPoint(result.recommendationRC.breakContext, 0.3);
    console.log(
      `  ${result.name.padEnd(24)} | break avgConf=${result.breakContext.avgConfidence.toFixed(3)} | dir=${(result.breakContext.directionalAccuracy * 100).toFixed(1).padStart(5)}% | brier=${result.breakContext.brierScore.toFixed(3)} | rc@0.2=${(break020.accuracy * 100).toFixed(1).padStart(5)}%/${(break020.coverage * 100).toFixed(1).padStart(5)}% | rc@0.3=${(break030.accuracy * 100).toFixed(1).padStart(5)}%/${(break030.coverage * 100).toFixed(1).padStart(5)}%`,
    );
  }

  console.log('');
  console.log('Pairwise deltas vs baseline:');
  for (const delta of artifact.pairwiseVsBaseline) {
    console.log(
      `  ${delta.name.padEnd(24)} | ΔavgConf=${delta.breakAvgConfidenceDelta.toFixed(3)} | Δdir=${(delta.breakAccuracyDelta * 100).toFixed(1).padStart(5)}pp | Δbrier=${delta.breakBrierDelta.toFixed(3)} | Δrc@0.2=${(delta.breakRecommendationAt020Delta * 100).toFixed(1).padStart(5)}pp/${(delta.breakRecommendationAt020CoverageDelta * 100).toFixed(1).padStart(5)}pp | Δrc@0.3=${(delta.breakRecommendationAt030Delta * 100).toFixed(1).padStart(5)}pp/${(delta.breakRecommendationAt030CoverageDelta * 100).toFixed(1).padStart(5)}pp`,
    );
  }
}

export async function main(): Promise<void> {
  const artifactDir = join(process.cwd(), '.sisyphus', 'artifacts');
  const artifactPath = join(artifactDir, 'phase3-break-policy-ablation.json');

  const steps = await loadBacktestSteps();
  const results = simulateAllPolicies(steps);
  const artifact = generateAblationReport(steps, results);

  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));

  printPolicySummary(artifact);
  console.log('');
  console.log(`Saved artifact to ${artifactPath}`);
}

if (import.meta.main) {
  main().catch(error => {
    console.error('Fatal:', error);
    process.exit(1);
  });
}
