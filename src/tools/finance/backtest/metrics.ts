/**
 * Backtest calibration metrics for the Markov distribution model.
 *
 * All metrics operate on arrays of BacktestStep records produced by the
 * walk-forward engine. Each step records a predicted probability/CI and
 * the actual realized outcome.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type BacktestRecommendation = 'BUY' | 'HOLD' | 'SELL';

export type AnchorQuality = 'good' | 'sparse' | 'none';

export type ValidationMetric = 'daily_return' | 'horizon_return';

export type ConfidenceBucketLabel =
  | '[0.00, 0.20)'
  | '[0.20, 0.40)'
  | '[0.40, 0.60)'
  | '[0.60, 0.80)'
  | '[0.80, 1.00]';

export type VolatilityBucketLabel = 'low' | 'medium' | 'high';

export type MoveMagnitudeBucketLabel = '<2%' | '2–5%' | '5–10%' | '≥10%';

export type MoveDirectionBucketLabel = 'up' | 'down' | 'flat';

export type TrendVsChopLabel = 'trending' | 'chop';

export type PUpBandLabel = '<0.45' | '0.45–0.50' | '0.50–0.55' | '>0.55';

export type PUpBandSource = 'calibrated' | 'raw';

export type DivergenceBucketLabel = '<0.05' | '0.05–0.10' | '0.10–0.20' | '≥0.20';

export type EntropyCiScaleBucketLabel = 'tightened' | 'neutral' | 'widened';

// ---------------------------------------------------------------------------
// Provenance types (PR3E)
// ---------------------------------------------------------------------------

/** What generated the probability for this step */
export type ProbabilitySource = 'calibrated';

/** What generated the recommendation for this step */
export type DecisionSource =
  | 'default'
  | 'crypto-short-horizon-raw'
  | 'crypto-short-horizon-raw-direction-hybrid'
  | 'crypto-short-horizon-disagreement-blend'
  | 'crypto-short-horizon-recency'
  | 'replay-anchor'
  | 'crypto-short-horizon-raw+replay-anchor'
  | 'crypto-short-horizon-raw-direction-hybrid+replay-anchor'
  | 'crypto-short-horizon-disagreement-blend+replay-anchor'
  | 'crypto-short-horizon-recency+replay-anchor';

export const DEFAULT_PUP_BANDS = [
  { label: '<0.45',      minInclusive: 0.0,  maxExclusive: 0.45 },
  { label: '0.45–0.50', minInclusive: 0.45, maxExclusive: 0.50 },
  { label: '0.50–0.55', minInclusive: 0.50, maxExclusive: 0.55 },
  { label: '>0.55',     minInclusive: 0.55, maxExclusive: 1.0  },
] as const satisfies readonly NumericBucketDefinition<PUpBandLabel>[];

export const DEFAULT_DIVERGENCE_BUCKETS = [
  { label: '<0.05', minInclusive: Number.NEGATIVE_INFINITY, maxExclusive: 0.05 },
  { label: '0.05–0.10', minInclusive: 0.05, maxExclusive: 0.10 },
  { label: '0.10–0.20', minInclusive: 0.10, maxExclusive: 0.20 },
  { label: '≥0.20', minInclusive: 0.20, maxExclusive: Number.POSITIVE_INFINITY },
] as const satisfies readonly NumericBucketDefinition<DivergenceBucketLabel>[];

export type FailureSliceKey =
  | 'tickerHorizon'
  | 'regime'
  | 'confidence'
  | 'volatility'
  | 'moveMagnitude'
  | 'moveDirection'
  | 'anchorQuality'
  | 'recommendation'
  | 'trendVsChop'
  | 'validationMetric'
  | 'structuralBreak'
  | 'divergence'
  | 'hmmConverged'
  | 'ensembleConsensus'
  | 'pUpBand'
  | 'entropyCiScale';

export interface NumericBucketDefinition<Label extends string = string> {
  label: Label;
  minInclusive: number;
  maxExclusive: number;
  includeMax?: boolean;
}

export const DEFAULT_CONFIDENCE_BUCKETS = [
  { label: '[0.00, 0.20)', minInclusive: 0.0, maxExclusive: 0.2 },
  { label: '[0.20, 0.40)', minInclusive: 0.2, maxExclusive: 0.4 },
  { label: '[0.40, 0.60)', minInclusive: 0.4, maxExclusive: 0.6 },
  { label: '[0.60, 0.80)', minInclusive: 0.6, maxExclusive: 0.8 },
  { label: '[0.80, 1.00]', minInclusive: 0.8, maxExclusive: 1.0, includeMax: true },
] as const satisfies readonly NumericBucketDefinition<ConfidenceBucketLabel>[];

export const DEFAULT_VOLATILITY_BUCKET_LABELS = ['low', 'medium', 'high'] as const satisfies readonly VolatilityBucketLabel[];

export const DEFAULT_MOVE_MAGNITUDE_BUCKETS = [
  { label: '<2%', minInclusive: Number.NEGATIVE_INFINITY, maxExclusive: 0.02 },
  { label: '2–5%', minInclusive: 0.02, maxExclusive: 0.05 },
  { label: '5–10%', minInclusive: 0.05, maxExclusive: 0.10 },
  { label: '≥10%', minInclusive: 0.10, maxExclusive: Number.POSITIVE_INFINITY },
] as const satisfies readonly NumericBucketDefinition<MoveMagnitudeBucketLabel>[];

export const MOVE_DIRECTION_BUCKETS = ['up', 'down', 'flat'] as const satisfies readonly MoveDirectionBucketLabel[];

export const TREND_VS_CHOP_BUCKETS = ['trending', 'chop'] as const satisfies readonly TrendVsChopLabel[];

export const RECOMMENDATION_BUCKETS = ['BUY', 'HOLD', 'SELL'] as const satisfies readonly BacktestRecommendation[];

export const BALANCED_DIRECTIONAL_CLASSES = ['BUY', 'HOLD', 'SELL'] as const satisfies readonly BacktestRecommendation[];

export const MEAN_EDGE_HOLD_POLICY = 'penalize_missed_move' as const;

export interface BucketedMetricRow {
  label: string;
  count: number;
  fraction: number;
  directionalAccuracy: number;
  balancedDirectionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  meanEdge: number;
}

export interface FailureSlice {
  key: FailureSliceKey;
  rows: BucketedMetricRow[];
}

export interface FailureDecompositionReport {
  totalSteps: number;
  slices: FailureSlice[];
}

export interface BacktestStep {
  /** Date index (trading day offset from start) */
  t: number;
  /** Predicted P(price > currentPrice at t+horizon) — calibrated */
  predictedProb: number;
  /** Raw (pre-calibration) P(>currentPrice) from the Markov distribution.
   * Present only when the model returned both calibrated and raw distributions.
   * When absent, raw === predictedProb (identical for backtest runs that only
   * have a single distribution object). */
  rawPredictedProb?: number;
  /** Did the actual price exceed currentPrice? (1 = yes, 0 = no) */
  actualBinary: number;
  /** Predicted expected return over the horizon */
  predictedReturn: number;
  /** Actual realized return over the horizon */
  actualReturn: number;
  /** 90% CI lower bound (price) */
  ciLower: number;
  /** 90% CI upper bound (price) */
  ciUpper: number;
  /** Literal central 90% interval lower bound from the forecast distribution when available. */
  central90CiLower?: number;
  /** Literal central 90% interval upper bound from the forecast distribution when available. */
  central90CiUpper?: number;
  /** Literal central 95% interval lower bound from the forecast distribution when available. */
  central95CiLower?: number;
  /** Literal central 95% interval upper bound from the forecast distribution when available. */
  central95CiUpper?: number;
  /** Literal central 99% interval lower bound from the forecast distribution when available. */
  central99CiLower?: number;
  /** Literal central 99% interval upper bound from the forecast distribution when available. */
  central99CiUpper?: number;
  /** Compact forecast distribution summary used for mixture-aware scoring diagnostics. */
  forecastDistribution?: ForecastDistributionPoint[];
  /** Realized price at t + horizon */
  realizedPrice: number;
  /** Recommendation: BUY, HOLD, or SELL */
  recommendation: BacktestRecommendation;
  /** Whether the GOF test passed for this window (null if not computed) */
  gofPasses: boolean | null;
  /** Prediction confidence score (0–1) for selective prediction filtering */
  confidence: number;
  regime?: string;
  anchorQuality?: AnchorQuality;
  trustedAnchors?: number;
  markovWeight?: number;
  anchorWeight?: number;
  validationMetric?: ValidationMetric;
  outOfSampleR2?: number | null;
  structuralBreakDetected?: boolean;
  /** Whether the full-window run triggered a short-window rerun due to a detected break. */
  structuralBreakRerunTriggered?: boolean;
  /** Structural-break flag from the pre-rerun full-window pass when applicable. */
  originalStructuralBreakDetected?: boolean;
  sidewaysSplitActive?: boolean;
  matureBullCalibrationActive?: boolean;
  /** Run-level provenance: the trend-only break-confidence experiment was enabled for this prediction run. */
  trendPenaltyOnlyBreakConfidenceActive?: boolean;
  /** Run-level provenance: the divergence-weighted break-confidence experiment was enabled for this prediction run. */
  divergenceWeightedBreakConfidenceActive?: boolean;
  /** Run-level provenance: the BTC 14d bearish-break selective SELL gate fired for this step. */
  bearishBreakRecommendationGateActive?: boolean;
  /** Phase 5 provenance: which fallback candidate was used for this step (backtest-only). */
  breakFallbackCandidateId?: string;
  /** Phase 5 provenance: which fallback mode was applied for this step (backtest-only). */
  breakFallbackMode?: 'hard' | 'blended' | 'blended_capped';
  /** Phase 7 provenance: whether regime-specific sigma was active for this step (backtest-only). */
  regimeSpecificSigmaActive?: boolean;
  /** R4/R5 provenance: whether GARCH volatility scaling affected the distribution path. */
  garchVolApplied?: boolean;
  /** R5 Idea #14 provenance: stationary-weighted transition entropy in nats. */
  transitionEntropy?: number;
  /** R5 Idea #14 provenance: transition entropy normalized to [0, 1]. */
  transitionEntropyNorm?: number;
  /** R5 Idea #14 provenance: rolling z-score used for CI modulation; null until enough history exists. */
  transitionEntropyZ?: number | null;
  /** R5 Idea #14 provenance: multiplier applied to CI half-width. */
  entropyCiScale?: number;
  /** R5 Idea #14 provenance: true when entropy changed CI width for this step. */
  entropyCiModulationApplied?: boolean;
  /** R6 adaptive conformal provenance: true when the step CI came from online conformal calibration. */
  conformalApplied?: boolean;
  /** R6 adaptive conformal provenance: symmetric radius applied around the forecast center. */
  conformalRadius?: number;
  /** R6 adaptive conformal provenance: running empirical coverage estimate. */
  conformalCoverageEstimate?: number;
  /** R6 adaptive conformal provenance: sample count in the current controller state. */
  conformalSampleCount?: number;
  /** R6 adaptive conformal provenance: normal mode vs break-aware accelerated mode. */
  conformalMode?: 'normal' | 'break';
  /** R6 adaptive conformal provenance: true when a break-triggered forecaster rerun reset the controller state. */
  conformalResetTriggered?: boolean;
  /** Item 2 provenance: true when HMM posterior uncertainty altered confidence or CI width. */
  softRegimeWeightingApplied?: boolean;
  /** Item 2 provenance: normalized entropy of the current HMM posterior. */
  softRegimePosteriorEntropy?: number;
  /** Item 2 provenance: normalized entropy of the horizon-state forecast mixture. */
  softRegimeForecastEntropy?: number;
  /** Item 2 provenance: CI width multiplier derived from posterior uncertainty. */
  softRegimeCiScale?: number;
  /** Item 2 provenance: confidence multiplier derived from posterior uncertainty. */
  softRegimeConfidenceMultiplier?: number;
  structuralBreakDivergence?: number | null;
  /** Structural-break divergence from the pre-rerun full-window pass when applicable. */
  originalStructuralBreakDivergence?: number | null;
  hmmConverged?: boolean | null;
  ensembleConsensus?: number | null;
  /** Provenance: what generated the probability (PR3E) */
  probabilitySource?: ProbabilitySource;
  /** Provenance: what generated the recommendation (PR3E) */
  decisionSource?: DecisionSource;
}

export interface ReliabilityBin {
  /** Lower bound of predicted probability bin (e.g. 0.0, 0.1, ...) */
  binLower: number;
  /** Upper bound of predicted probability bin */
  binUpper: number;
  /** Mean predicted probability in this bin */
  meanPredicted: number;
  /** Actual frequency of positive outcomes in this bin */
  actualFrequency: number;
  /** Number of observations in this bin */
  count: number;
}

export interface ForecastDistributionPoint {
  price: number;
  /** Survival probability P(price > level). */
  probability: number;
}

export interface ProvenanceSummary {
  decisionSources: Record<DecisionSource, number>;
  probabilitySources: Record<ProbabilitySource, number>;
}

export interface MurphyWinklerDecomposition {
  meanWidth: number;
  lowerMissPenalty: number;
  upperMissPenalty: number;
  totalScore: number;
  coverage: number;
}

export interface BacktestReport {
  ticker: string;
  horizon: number;
  totalSteps: number;
  brierScore: number;
  /** Legacy primary interval coverage. In current walk-forward runs this is the conservative helper interval. */
  ciCoverage: number;
  /** Explicit alias for the conservative helper interval coverage used by current walk-forward backtests. */
  conservativeCiCoverage: number;
  /** Literal central-90% coverage when the step payload contains the matching interval. */
  central90CiCoverage: number;
  /** Literal central-95% coverage when the step payload contains the matching interval. */
  central95CiCoverage: number;
  /** Literal central-99% coverage when the step payload contains the matching interval. */
  central99CiCoverage: number;
  directionalAccuracy: number;
  expectedReturnCorrelation: number;
  sharpness: number;
  crps: number;
  mixtureCrps: number;
  scaledCrps: number;
  tailWeightedCrps: number;
  murphyWinklerScore: number;
  murphyWinklerDecomposition: MurphyWinklerDecomposition;
  reliabilityBins: ReliabilityBin[];
  tailReliabilityBins: ReliabilityBin[];
  gofPassRate: number | null;
  balancedDirectionalAccuracy?: number;
  meanEdge?: number;
  failureDecomposition?: FailureDecompositionReport;
  /** Aggregated provenance counts across all steps (PR3E) */
  provenanceSummary?: ProvenanceSummary;
  /** Whether any step in this report came from a run with the trend-only break-confidence experiment enabled. */
  trendPenaltyOnlyBreakConfidenceActive?: boolean;
  /** Whether any step in this report came from a run where the BTC 14d bearish-break SELL gate fired. */
  bearishBreakRecommendationGateActive?: boolean;
}

// ---------------------------------------------------------------------------
// Brier Score
// ---------------------------------------------------------------------------

/**
 * Brier score: mean squared error between predicted probabilities and binary outcomes.
 * Range [0, 1] — lower is better. 0.25 = random coin flip.
 */
export function brierScore(steps: BacktestStep[]): number {
  if (steps.length === 0) return 1;
  const sum = steps.reduce((s, step) => s + (step.predictedProb - step.actualBinary) ** 2, 0);
  return sum / steps.length;
}

// ---------------------------------------------------------------------------
// Reliability Diagram (binned calibration)
// ---------------------------------------------------------------------------

/**
 * Bin predictions into deciles and compute actual frequency per bin.
 * Returns 10 bins: [0,0.1), [0.1,0.2), ..., [0.9,1.0].
 * Perfect calibration: meanPredicted ≈ actualFrequency in each bin.
 */
export function reliabilityBins(steps: BacktestStep[], numBins = 10): ReliabilityBin[] {
  const binWidth = 1 / numBins;
  const bins: ReliabilityBin[] = [];

  for (let i = 0; i < numBins; i++) {
    const lower = i * binWidth;
    const upper = (i + 1) * binWidth;
    const inBin = steps.filter(s =>
      s.predictedProb >= lower && (i === numBins - 1 ? s.predictedProb <= upper : s.predictedProb < upper),
    );

    const meanPred = inBin.length > 0
      ? inBin.reduce((s, st) => s + st.predictedProb, 0) / inBin.length
      : (lower + upper) / 2;
    const actualFreq = inBin.length > 0
      ? inBin.reduce((s, st) => s + st.actualBinary, 0) / inBin.length
      : 0;

    bins.push({
      binLower: lower,
      binUpper: upper,
      meanPredicted: meanPred,
      actualFrequency: actualFreq,
      count: inBin.length,
    });
  }

  return bins;
}

const DEFAULT_TAIL_RELIABILITY_BINS = [
  [0.0, 0.05],
  [0.05, 0.10],
  [0.90, 0.95],
  [0.95, 1.0],
] as const;

/**
 * Reliability bins focused on the extreme probability tails.
 */
export function tailReliabilityBins(steps: BacktestStep[]): ReliabilityBin[] {
  return DEFAULT_TAIL_RELIABILITY_BINS.map(([lower, upper], index) => {
    const isLast = index === DEFAULT_TAIL_RELIABILITY_BINS.length - 1;
    const inBin = steps.filter(
      step => step.predictedProb >= lower && (isLast ? step.predictedProb <= upper : step.predictedProb < upper),
    );
    const meanPredicted = inBin.length > 0
      ? inBin.reduce((sum, step) => sum + step.predictedProb, 0) / inBin.length
      : (lower + upper) / 2;
    const actualFrequency = inBin.length > 0
      ? inBin.reduce((sum, step) => sum + step.actualBinary, 0) / inBin.length
      : 0;

    return {
      binLower: lower,
      binUpper: upper,
      meanPredicted,
      actualFrequency,
      count: inBin.length,
    };
  });
}

/**
 * Maximum absolute deviation between predicted and actual frequency across bins.
 * Only considers bins with ≥ minCount observations.
 */
export function maxReliabilityDeviation(bins: ReliabilityBin[], minCount = 3): number {
  const populated = bins.filter(b => b.count >= minCount);
  if (populated.length === 0) return 0;
  return Math.max(...populated.map(b => Math.abs(b.meanPredicted - b.actualFrequency)));
}

// ---------------------------------------------------------------------------
// CI Coverage
// ---------------------------------------------------------------------------

/**
 * Fraction of steps where the realized price falls within the primary step interval.
 * In the current walk-forward engine this is a conservative helper interval rather
 * than a literal central-90% interval.
 */
export function ciCoverage(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  const covered = steps.filter(s => s.realizedPrice >= s.ciLower && s.realizedPrice <= s.ciUpper);
  return covered.length / steps.length;
}

function resolveCentralIntervalBounds(
  step: BacktestStep,
  level: 0.9 | 0.95 | 0.99,
): { lower: number; upper: number } {
  if (level === 0.9 && step.central90CiLower !== undefined && step.central90CiUpper !== undefined) {
    return { lower: step.central90CiLower, upper: step.central90CiUpper };
  }
  if (level === 0.95 && step.central95CiLower !== undefined && step.central95CiUpper !== undefined) {
    return { lower: step.central95CiLower, upper: step.central95CiUpper };
  }
  if (level === 0.99 && step.central99CiLower !== undefined && step.central99CiUpper !== undefined) {
    return { lower: step.central99CiLower, upper: step.central99CiUpper };
  }
  return { lower: step.ciLower, upper: step.ciUpper };
}

/**
 * Coverage of the literal central interval carried on each step when available.
 * Falls back to the legacy primary interval to keep historical reports defined.
 */
export function centralIntervalCoverage(
  steps: BacktestStep[],
  level: 0.9 | 0.95 | 0.99 = 0.9,
): number {
  if (steps.length === 0) return 0;
  const covered = steps.filter(step => {
    const { lower, upper } = resolveCentralIntervalBounds(step, level);
    return step.realizedPrice >= lower && step.realizedPrice <= upper;
  });
  return covered.length / steps.length;
}

// ---------------------------------------------------------------------------
// CRPS / interval score diagnostics
// ---------------------------------------------------------------------------

const CENTRAL_90_Z = 1.6448536269514722;
const SQRT_PI = Math.sqrt(Math.PI);
const MIN_SCALE = 1e-9;

function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * absX);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-absX * absX);
  return sign * y;
}

function normalPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function normalCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function deriveForecastCenter(step: BacktestStep): number {
  const denominator = 1 + step.actualReturn;
  const currentPrice = Number.isFinite(denominator) && Math.abs(denominator) > MIN_SCALE
    ? step.realizedPrice / denominator
    : (step.ciLower + step.ciUpper) / 2;
  const forecastCenter = currentPrice * (1 + step.predictedReturn);
  return Number.isFinite(forecastCenter) ? forecastCenter : (step.ciLower + step.ciUpper) / 2;
}

function deriveIntervalScale(step: BacktestStep): number {
  const width = Math.abs(step.ciUpper - step.ciLower);
  const sigma = width / (2 * CENTRAL_90_Z);
  return Number.isFinite(sigma) && sigma > MIN_SCALE ? sigma : MIN_SCALE;
}

function normalCrps(observed: number, mean: number, sigma: number): number {
  const z = (observed - mean) / sigma;
  return sigma * (z * (2 * normalCdf(z) - 1) + 2 * normalPdf(z) - 1 / SQRT_PI);
}

function stepCrps(step: BacktestStep): number {
  const score = normalCrps(step.realizedPrice, deriveForecastCenter(step), deriveIntervalScale(step));
  return Number.isFinite(score) ? Math.max(0, score) : 0;
}

function interpolateSurvivalProbability(
  distribution: readonly ForecastDistributionPoint[],
  targetPrice: number,
): number {
  if (distribution.length === 0) return 0.5;
  if (targetPrice <= distribution[0].price) return distribution[0].probability;
  if (targetPrice >= distribution[distribution.length - 1].price) {
    return distribution[distribution.length - 1].probability;
  }

  for (let index = 0; index < distribution.length - 1; index++) {
    const lower = distribution[index];
    const upper = distribution[index + 1];
    if (lower.price <= targetPrice && targetPrice <= upper.price) {
      const span = upper.price - lower.price;
      if (Math.abs(span) <= MIN_SCALE) return lower.probability;
      const frac = (targetPrice - lower.price) / span;
      return lower.probability + frac * (upper.probability - lower.probability);
    }
  }

  return distribution[distribution.length - 1].probability;
}

function mixtureCdf(
  distribution: readonly ForecastDistributionPoint[],
  targetPrice: number,
): number {
  return 1 - interpolateSurvivalProbability(distribution, targetPrice);
}

function stepMixtureCrps(step: BacktestStep): number {
  const distribution = step.forecastDistribution;
  if (!distribution || distribution.length < 2) return stepCrps(step);

  const nodes = Array.from(
    new Set([...distribution.map(point => point.price), step.realizedPrice].sort((a, b) => a - b)),
  );
  if (nodes.length < 2) return stepCrps(step);

  let area = 0;
  for (let index = 0; index < nodes.length - 1; index++) {
    const left = nodes[index];
    const right = nodes[index + 1];
    const leftDiff = mixtureCdf(distribution, left) - (left >= step.realizedPrice ? 1 : 0);
    const rightDiff = mixtureCdf(distribution, right) - (right >= step.realizedPrice ? 1 : 0);
    area += 0.5 * ((leftDiff * leftDiff) + (rightDiff * rightDiff)) * (right - left);
  }

  return Number.isFinite(area) ? Math.max(0, area) : stepCrps(step);
}

function standardizedResidual(step: BacktestStep): number {
  return Math.abs(step.realizedPrice - deriveForecastCenter(step)) / deriveIntervalScale(step);
}

function tailWeight(step: BacktestStep, thresholdZ = CENTRAL_90_Z): number {
  return 1 + Math.max(0, standardizedResidual(step) - thresholdZ);
}

/**
 * Continuous Ranked Probability Score using the current backtest interval as a
 * central-normal approximation. Lower is better.
 */
export function crps(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  return steps.reduce((sum, step) => sum + stepCrps(step), 0) / steps.length;
}

/**
 * Mixture-aware CRPS computed directly from the stored forecast distribution when
 * available; falls back to the normal approximation otherwise.
 */
export function mixtureCrps(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  return steps.reduce((sum, step) => sum + stepMixtureCrps(step), 0) / steps.length;
}

/**
 * Locally scaled CRPS: CRPS divided by the interval-implied scale for each step.
 * Lower is better and less sensitive to heterogeneous forecast units/volatility.
 */
export function scaledCrps(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  const sum = steps.reduce((total, step) => total + stepCrps(step) / deriveIntervalScale(step), 0);
  return sum / steps.length;
}

/**
 * Threshold-weighted CRPS using standardized forecast errors to up-weight tail misses.
 * Lower is better; identical to CRPS when all observations stay inside the threshold.
 */
export function tailWeightedCrps(steps: BacktestStep[], thresholdZ = CENTRAL_90_Z): number {
  if (steps.length === 0) return 0;
  let totalWeight = 0;
  let weightedScore = 0;
  for (const step of steps) {
    const weight = tailWeight(step, thresholdZ);
    totalWeight += weight;
    weightedScore += stepCrps(step) * weight;
  }
  return totalWeight > 0 ? weightedScore / totalWeight : 0;
}

/**
 * Murphy-Winkler interval score decomposition for the primary step interval.
 * In current walk-forward runs this is the conservative helper interval unless a
 * caller populates a different interval into `ciLower` / `ciUpper`.
 */
export function murphyWinklerDecomposition(
  steps: BacktestStep[],
  alpha = 0.10,
): MurphyWinklerDecomposition {
  if (steps.length === 0) {
    return {
      meanWidth: 0,
      lowerMissPenalty: 0,
      upperMissPenalty: 0,
      totalScore: 0,
      coverage: 0,
    };
  }

  const penaltyScale = 2 / alpha;
  let widthSum = 0;
  let lowerPenaltySum = 0;
  let upperPenaltySum = 0;
  let covered = 0;

  for (const step of steps) {
    const lower = Math.min(step.ciLower, step.ciUpper);
    const upper = Math.max(step.ciLower, step.ciUpper);
    widthSum += upper - lower;
    if (step.realizedPrice < lower) {
      lowerPenaltySum += penaltyScale * (lower - step.realizedPrice);
    } else if (step.realizedPrice > upper) {
      upperPenaltySum += penaltyScale * (step.realizedPrice - upper);
    } else {
      covered++;
    }
  }

  const meanWidth = widthSum / steps.length;
  const lowerMissPenalty = lowerPenaltySum / steps.length;
  const upperMissPenalty = upperPenaltySum / steps.length;

  return {
    meanWidth,
    lowerMissPenalty,
    upperMissPenalty,
    totalScore: meanWidth + lowerMissPenalty + upperMissPenalty,
    coverage: covered / steps.length,
  };
}

export function murphyWinklerScore(steps: BacktestStep[], alpha = 0.10): number {
  return murphyWinklerDecomposition(steps, alpha).totalScore;
}

// ---------------------------------------------------------------------------
// Directional Accuracy
// ---------------------------------------------------------------------------

/**
 * Fraction of steps where recommendation matches actual direction.
 * BUY is correct if actualReturn > 0, SELL if actualReturn < 0.
 * HOLD is correct if |actualReturn| < 3% (roughly within noise).
 */
export function directionalAccuracy(steps: BacktestStep[], holdThreshold = 0.03): number {
  if (steps.length === 0) return 0;
  const correct = steps.filter(s => {
    if (s.recommendation === 'BUY')  return s.actualReturn > 0;
    if (s.recommendation === 'SELL') return s.actualReturn < 0;
    // HOLD is correct if price didn't move much
    return Math.abs(s.actualReturn) < holdThreshold;
  });
  return correct.length / steps.length;
}

const UNKNOWN_LABEL = 'unknown';
const ANCHOR_QUALITY_BUCKETS = ['good', 'sparse', 'none', UNKNOWN_LABEL] as const;
const VALIDATION_METRIC_BUCKETS = ['daily_return', 'horizon_return', UNKNOWN_LABEL] as const;

function classifyActualRecommendation(actualReturn: number, holdThreshold: number): BacktestRecommendation {
  if (actualReturn > holdThreshold) return 'BUY';
  if (actualReturn < -holdThreshold) return 'SELL';
  return 'HOLD';
}

function bucketRow(
  label: string,
  bucketSteps: BacktestStep[],
  totalSteps: number,
  holdThreshold: number,
): BucketedMetricRow {
  if (bucketSteps.length === 0) {
    return {
      label,
      count: 0,
      fraction: 0,
      directionalAccuracy: 0,
      balancedDirectionalAccuracy: 0,
      brierScore: 0,
      ciCoverage: 0,
      meanEdge: 0,
    };
  }

  return {
    label,
    count: bucketSteps.length,
    fraction: totalSteps > 0 ? bucketSteps.length / totalSteps : 0,
    directionalAccuracy: directionalAccuracy(bucketSteps, holdThreshold),
    balancedDirectionalAccuracy: balancedDirectionalAccuracy(bucketSteps, holdThreshold),
    brierScore: brierScore(bucketSteps),
    ciCoverage: ciCoverage(bucketSteps),
    meanEdge: meanEdge(bucketSteps),
  };
}

function numericBucketLabel<Label extends string>(
  value: number,
  buckets: readonly NumericBucketDefinition<Label>[],
): Label | null {
  if (!Number.isFinite(value)) return null;
  for (const bucket of buckets) {
    if (value < bucket.minInclusive) continue;
    if (value < bucket.maxExclusive) return bucket.label;
    if (bucket.includeMax && value === bucket.maxExclusive) return bucket.label;
  }
  return null;
}

function summarizeFixedBuckets(
  steps: BacktestStep[],
  labels: readonly string[],
  labeler: (step: BacktestStep) => string,
  holdThreshold: number,
): BucketedMetricRow[] {
  const grouped = new Map<string, BacktestStep[]>();
  for (const label of labels) grouped.set(label, []);

  for (const step of steps) {
    const label = labeler(step);
    const bucket = grouped.get(label);
    if (bucket) {
      bucket.push(step);
      continue;
    }

    const unknownBucket = grouped.get(UNKNOWN_LABEL);
    if (unknownBucket) {
      unknownBucket.push(step);
      continue;
    }

    grouped.set(label, [step]);
  }

  return Array.from(grouped.entries()).map(([label, bucketSteps]) =>
    bucketRow(label, bucketSteps, steps.length, holdThreshold),
  );
}

function summarizeObservedBuckets(
  steps: BacktestStep[],
  labels: string[],
  labeler: (step: BacktestStep) => string,
  holdThreshold: number,
): BucketedMetricRow[] {
  const grouped = new Map<string, BacktestStep[]>();
  for (const label of labels) grouped.set(label, []);
  for (const step of steps) {
    const label = labeler(step);
    const bucket = grouped.get(label);
    if (bucket) {
      bucket.push(step);
    } else {
      grouped.set(label, [step]);
    }
  }
  return Array.from(grouped.entries()).map(([label, bucketSteps]) =>
    bucketRow(label, bucketSteps, steps.length, holdThreshold),
  );
}

export function balancedDirectionalAccuracy(steps: BacktestStep[], holdThreshold = 0.03): number {
  if (steps.length === 0) return 0;

  const recalls: number[] = [];
  for (const label of BALANCED_DIRECTIONAL_CLASSES) {
    const actualClass = steps.filter(step => classifyActualRecommendation(step.actualReturn, holdThreshold) === label);
    if (actualClass.length === 0) continue;
    const correct = actualClass.filter(step => step.recommendation === label).length;
    recalls.push(correct / actualClass.length);
  }

  if (recalls.length === 0) return 0;
  return recalls.reduce((sum, recall) => sum + recall, 0) / recalls.length;
}

export function meanEdge(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;

  const total = steps.reduce((sum, step) => {
    if (step.recommendation === 'BUY') return sum + step.actualReturn;
    if (step.recommendation === 'SELL') return sum - step.actualReturn;
    return sum - Math.abs(step.actualReturn);
  }, 0);

  return total / steps.length;
}

export function bucketByConfidence(
  steps: BacktestStep[],
  buckets: readonly NumericBucketDefinition<ConfidenceBucketLabel>[] = DEFAULT_CONFIDENCE_BUCKETS,
  holdThreshold = 0.03,
): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    buckets.map(bucket => bucket.label),
    step => numericBucketLabel(step.confidence, buckets) ?? UNKNOWN_LABEL,
    holdThreshold,
  );
}

export function bucketByVolatility(
  steps: BacktestStep[],
  holdThreshold = 0.03,
): BucketedMetricRow[] {
  if (steps.length === 0) {
    return DEFAULT_VOLATILITY_BUCKET_LABELS.map(label => bucketRow(label, [], 0, holdThreshold));
  }

  const magnitudes = steps
    .map(step => Math.abs(step.actualReturn))
    .sort((a, b) => a - b);
  const q1 = magnitudes[Math.floor((magnitudes.length - 1) / 3)];
  const q2 = magnitudes[Math.floor((2 * (magnitudes.length - 1)) / 3)];

  return summarizeFixedBuckets(
    steps,
    [...DEFAULT_VOLATILITY_BUCKET_LABELS],
    step => {
      const magnitude = Math.abs(step.actualReturn);
      if (magnitude <= q1) return 'low';
      if (magnitude <= q2) return 'medium';
      return 'high';
    },
    holdThreshold,
  );
}

export function bucketByMoveMagnitude(
  steps: BacktestStep[],
  holdThreshold = 0.03,
  buckets: readonly NumericBucketDefinition<MoveMagnitudeBucketLabel>[] = DEFAULT_MOVE_MAGNITUDE_BUCKETS,
): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    buckets.map(bucket => bucket.label),
    step => numericBucketLabel(Math.abs(step.actualReturn), buckets) ?? UNKNOWN_LABEL,
    holdThreshold,
  );
}

export function bucketByMoveDirection(
  steps: BacktestStep[],
  holdThreshold = 0.03,
): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    [...MOVE_DIRECTION_BUCKETS],
    step => {
      if (step.actualReturn > holdThreshold) return 'up';
      if (step.actualReturn < -holdThreshold) return 'down';
      return 'flat';
    },
    holdThreshold,
  );
}

export function bucketByAnchorQuality(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    [...ANCHOR_QUALITY_BUCKETS],
    step => step.anchorQuality ?? UNKNOWN_LABEL,
    holdThreshold,
  );
}

export function bucketByRegime(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  const observed = Array.from(new Set(steps.map(step => step.regime ?? UNKNOWN_LABEL)));
  const knownLabels = observed.filter(label => label !== UNKNOWN_LABEL).sort((a, b) => a.localeCompare(b));
  const labels = observed.includes(UNKNOWN_LABEL)
    ? [...knownLabels, UNKNOWN_LABEL]
    : knownLabels;
  return summarizeObservedBuckets(steps, labels, step => step.regime ?? UNKNOWN_LABEL, holdThreshold);
}

export function bucketByRecommendation(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    [...RECOMMENDATION_BUCKETS],
    step => step.recommendation,
    holdThreshold,
  );
}

export function bucketByTrendVsChop(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    [...TREND_VS_CHOP_BUCKETS, UNKNOWN_LABEL],
    step => {
      if (!step.regime) return UNKNOWN_LABEL;
      return step.regime === 'sideways' ? 'chop' : 'trending';
    },
    holdThreshold,
  );
}

export function bucketByValidationMetric(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    [...VALIDATION_METRIC_BUCKETS],
    step => step.validationMetric ?? UNKNOWN_LABEL,
    holdThreshold,
  );
}

export function bucketByStructuralBreak(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  const labels = ['true', 'false', UNKNOWN_LABEL] as const;
  return summarizeFixedBuckets(
    steps,
    [...labels],
    step => {
      const effectiveBreak = step.originalStructuralBreakDetected ?? step.structuralBreakDetected;
      return effectiveBreak === undefined ? UNKNOWN_LABEL : String(effectiveBreak);
    },
    holdThreshold,
  );
}

export function bucketByDivergence(
  steps: BacktestStep[],
  holdThreshold = 0.03,
  buckets: readonly NumericBucketDefinition<DivergenceBucketLabel>[] = DEFAULT_DIVERGENCE_BUCKETS,
): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    [...buckets.map(bucket => bucket.label), UNKNOWN_LABEL],
    step => {
      const divergence = step.originalStructuralBreakDivergence
        ?? step.structuralBreakDivergence;
      return divergence === null || divergence === undefined
        ? UNKNOWN_LABEL
        : numericBucketLabel(divergence, buckets) ?? UNKNOWN_LABEL;
    },
    holdThreshold,
  );
}

export function bucketByHmmConverged(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  const labels = ['true', 'false', UNKNOWN_LABEL] as const;
  return summarizeFixedBuckets(
    steps,
    [...labels],
    step => step.hmmConverged === undefined ? UNKNOWN_LABEL : String(step.hmmConverged),
    holdThreshold,
  );
}

export function bucketByEnsembleConsensus(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  if (steps.length === 0) {
    return [UNKNOWN_LABEL, 'low', 'medium', 'high'].map(label =>
      bucketRow(label, [], 0, holdThreshold),
    );
  }
  const labels = ['low', 'medium', 'high', UNKNOWN_LABEL] as const;
  return summarizeFixedBuckets(
    steps,
    [...labels],
    step => {
      if (step.ensembleConsensus === undefined || step.ensembleConsensus === null) return UNKNOWN_LABEL;
      if (step.ensembleConsensus < 0.4) return 'low';
      if (step.ensembleConsensus < 0.7) return 'medium';
      return 'high';
    },
    holdThreshold,
  );
}

export function bucketByPUpBand(
  steps: BacktestStep[],
  holdThreshold = 0.03,
  bands: readonly NumericBucketDefinition<PUpBandLabel>[] = DEFAULT_PUP_BANDS,
  source: PUpBandSource = 'calibrated',
): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    bands.map(b => b.label),
    step => {
      const prob = source === 'raw'
        ? (step.rawPredictedProb ?? step.predictedProb)
        : step.predictedProb;
      return numericBucketLabel(prob, bands) ?? UNKNOWN_LABEL;
    },
    holdThreshold,
  );
}

export function bucketByEntropyCiScale(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  const labels = ['tightened', 'neutral', 'widened'] as const satisfies readonly EntropyCiScaleBucketLabel[];
  return summarizeFixedBuckets(
    steps,
    [...labels],
    step => {
      const scale = step.entropyCiScale ?? 1;
      if (scale < 0.999) return 'tightened';
      if (scale > 1.001) return 'widened';
      return 'neutral';
    },
    holdThreshold,
  );
}

export function computeFailureDecomposition(
  steps: BacktestStep[],
  holdThreshold = 0.03,
): FailureDecompositionReport {
  return {
    totalSteps: steps.length,
    slices: [
      { key: 'regime', rows: bucketByRegime(steps, holdThreshold) },
      { key: 'volatility', rows: bucketByVolatility(steps, holdThreshold) },
      { key: 'moveMagnitude', rows: bucketByMoveMagnitude(steps, holdThreshold) },
      { key: 'moveDirection', rows: bucketByMoveDirection(steps, holdThreshold) },
      { key: 'confidence', rows: bucketByConfidence(steps, DEFAULT_CONFIDENCE_BUCKETS, holdThreshold) },
      { key: 'anchorQuality', rows: bucketByAnchorQuality(steps, holdThreshold) },
      { key: 'recommendation', rows: bucketByRecommendation(steps, holdThreshold) },
      { key: 'trendVsChop', rows: bucketByTrendVsChop(steps, holdThreshold) },
      { key: 'validationMetric', rows: bucketByValidationMetric(steps, holdThreshold) },
      { key: 'structuralBreak', rows: bucketByStructuralBreak(steps, holdThreshold) },
      { key: 'divergence', rows: bucketByDivergence(steps, holdThreshold) },
      { key: 'hmmConverged', rows: bucketByHmmConverged(steps, holdThreshold) },
      { key: 'ensembleConsensus', rows: bucketByEnsembleConsensus(steps, holdThreshold) },
      { key: 'pUpBand', rows: bucketByPUpBand(steps, holdThreshold) },
      { key: 'entropyCiScale', rows: bucketByEntropyCiScale(steps, holdThreshold) },
    ],
  };
}

// ---------------------------------------------------------------------------
// Selective Directional Accuracy (Idea M — sHMM)
// ---------------------------------------------------------------------------

/**
 * Selective directional accuracy: only count predictions above a confidence threshold.
 * Returns both accuracy (on selected predictions) and coverage (fraction selected).
 *
 * Inspired by El-Yaniv & Pidan (NeurIPS 2011): selective prediction achieves
 * lower error by abstaining on uncertain predictions. The RC (risk-coverage)
 * trade-off curve shows monotonically decreasing error with decreasing coverage.
 */
export function selectiveDirectionalAccuracy(
  steps: BacktestStep[],
  minConfidence: number,
  holdThreshold = 0.03,
): { accuracy: number; coverage: number; selected: number; total: number } {
  if (steps.length === 0) return { accuracy: 0, coverage: 0, selected: 0, total: 0 };

  const selected = steps.filter(s => s.confidence >= minConfidence);
  if (selected.length === 0) return { accuracy: 0, coverage: 0, selected: 0, total: steps.length };

  const correct = selected.filter(s => {
    if (s.recommendation === 'BUY')  return s.actualReturn > 0;
    if (s.recommendation === 'SELL') return s.actualReturn < 0;
    return Math.abs(s.actualReturn) < holdThreshold;
  });

  return {
    accuracy: correct.length / selected.length,
    coverage: selected.length / steps.length,
    selected: selected.length,
    total: steps.length,
  };
}

/**
 * Compute the full Risk-Coverage (RC) curve: accuracy at various confidence thresholds.
 * Returns points sorted by decreasing coverage (increasing threshold).
 */
export function computeRCCurve(
  steps: BacktestStep[],
  thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
): Array<{ threshold: number; accuracy: number; coverage: number; n: number }> {
  return thresholds.map(t => {
    const result = selectiveDirectionalAccuracy(steps, t);
    return { threshold: t, accuracy: result.accuracy, coverage: result.coverage, n: result.selected };
  });
}

// ---------------------------------------------------------------------------
// Expected Return Correlation
// ---------------------------------------------------------------------------

/**
 * Pearson correlation between predicted and actual returns.
 * > 0 means the model has some predictive power; > 0.1 is decent.
 */
export function expectedReturnCorrelation(steps: BacktestStep[]): number {
  if (steps.length < 3) return 0;

  const n = steps.length;
  const predMean = steps.reduce((s, st) => s + st.predictedReturn, 0) / n;
  const actMean  = steps.reduce((s, st) => s + st.actualReturn, 0) / n;

  let num = 0, denPred = 0, denAct = 0;
  for (const s of steps) {
    const dp = s.predictedReturn - predMean;
    const da = s.actualReturn - actMean;
    num += dp * da;
    denPred += dp * dp;
    denAct  += da * da;
  }

  const den = Math.sqrt(denPred * denAct);
  return den < 1e-12 ? 0 : num / den;
}

// ---------------------------------------------------------------------------
// Sharpness
// ---------------------------------------------------------------------------

/**
 * Mean CI width as a fraction of current price.
 * Narrower = sharper (more informative). Not a pass/fail metric.
 */
export function sharpness(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  const widths = steps.map(s => (s.ciUpper - s.ciLower) / s.realizedPrice);
  return widths.reduce((a, b) => a + b, 0) / widths.length;
}

// ---------------------------------------------------------------------------
// GOF Pass Rate
// ---------------------------------------------------------------------------

/**
 * Fraction of walk-forward windows where the chi-squared GOF test passed.
 * null if no windows had GOF results.
 */
export function gofPassRate(steps: BacktestStep[]): number | null {
  const withGof = steps.filter(s => s.gofPasses !== null);
  if (withGof.length === 0) return null;
  return withGof.filter(s => s.gofPasses === true).length / withGof.length;
}

// ---------------------------------------------------------------------------
// Aggregate Report
// ---------------------------------------------------------------------------

export function generateReport(
  ticker: string,
  horizon: number,
  steps: BacktestStep[],
): BacktestReport {
  const bins = reliabilityBins(steps);
  const tailBins = tailReliabilityBins(steps);
  const conservativeCoverage = ciCoverage(steps);

  const provenanceSummary: ProvenanceSummary = {
    decisionSources: {
      default: 0,
      'crypto-short-horizon-raw': 0,
      'crypto-short-horizon-raw-direction-hybrid': 0,
      'crypto-short-horizon-disagreement-blend': 0,
      'crypto-short-horizon-recency': 0,
      'replay-anchor': 0,
      'crypto-short-horizon-raw+replay-anchor': 0,
      'crypto-short-horizon-raw-direction-hybrid+replay-anchor': 0,
      'crypto-short-horizon-disagreement-blend+replay-anchor': 0,
      'crypto-short-horizon-recency+replay-anchor': 0,
    },
    probabilitySources: { calibrated: 0 },
  };
  for (const step of steps) {
    if (step.decisionSource) {
      provenanceSummary.decisionSources[step.decisionSource]++;
    } else {
      provenanceSummary.decisionSources.default++;
    }
    if (step.probabilitySource) {
      provenanceSummary.probabilitySources[step.probabilitySource]++;
    } else {
      provenanceSummary.probabilitySources.calibrated++;
    }
  }

  return {
    ticker,
    horizon,
    totalSteps: steps.length,
    brierScore: brierScore(steps),
    ciCoverage: conservativeCoverage,
    conservativeCiCoverage: conservativeCoverage,
    central90CiCoverage: centralIntervalCoverage(steps, 0.9),
    central95CiCoverage: centralIntervalCoverage(steps, 0.95),
    central99CiCoverage: centralIntervalCoverage(steps, 0.99),
    directionalAccuracy: directionalAccuracy(steps),
    expectedReturnCorrelation: expectedReturnCorrelation(steps),
    sharpness: sharpness(steps),
    crps: crps(steps),
    mixtureCrps: mixtureCrps(steps),
    scaledCrps: scaledCrps(steps),
    tailWeightedCrps: tailWeightedCrps(steps),
    murphyWinklerScore: murphyWinklerScore(steps),
    murphyWinklerDecomposition: murphyWinklerDecomposition(steps),
    reliabilityBins: bins,
    tailReliabilityBins: tailBins,
    gofPassRate: gofPassRate(steps),
    balancedDirectionalAccuracy: balancedDirectionalAccuracy(steps),
    meanEdge: meanEdge(steps),
    failureDecomposition: computeFailureDecomposition(steps),
    provenanceSummary,
    trendPenaltyOnlyBreakConfidenceActive: steps.some(
      step => step.trendPenaltyOnlyBreakConfidenceActive === true,
    ),
    bearishBreakRecommendationGateActive: steps.some(
      step => step.bearishBreakRecommendationGateActive === true,
    ),
  };
}

// ---------------------------------------------------------------------------
// Bootstrap Confidence Intervals
// ---------------------------------------------------------------------------

export interface BootstrapCI {
  lower: number;
  median: number;
  upper: number;
  nResamples: number;
}

/**
 * Mulberry32 seeded PRNG — same algorithm used by generate-synthetic.ts.
 * Returns a function that produces uniform [0, 1) values.
 */
export function mulberry32(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Bootstrap confidence interval for any backtest metric.
 *
 * Resamples `steps` with replacement `nResamples` times, evaluates `metricFn`
 * on each resample, then returns the 2.5th/97.5th percentiles (95% CI) and
 * the median.
 *
 * Uses a seeded PRNG (Mulberry32) for reproducibility.
 */
export function bootstrapMetricCI(
  steps: BacktestStep[],
  metricFn: (steps: BacktestStep[]) => number,
  nResamples = 1000,
  seed = 12345,
): BootstrapCI {
  if (steps.length === 0) {
    return { lower: 0, median: 0, upper: 0, nResamples };
  }

  const rand = mulberry32(seed);
  const estimates: number[] = [];

  for (let r = 0; r < nResamples; r++) {
    const sample: BacktestStep[] = [];
    for (let i = 0; i < steps.length; i++) {
      sample.push(steps[Math.floor(rand() * steps.length)]);
    }
    estimates.push(metricFn(sample));
  }

  estimates.sort((a, b) => a - b);

  const pctIdx = (p: number) => Math.min(
    Math.max(Math.floor(p * estimates.length), 0),
    estimates.length - 1,
  );

  return {
    lower: estimates[pctIdx(0.025)],
    median: estimates[pctIdx(0.5)],
    upper: estimates[pctIdx(0.975)],
    nResamples,
  };
}

/** Bootstrap 95% CI for directional accuracy. */
export function bootstrapDirectionalCI(steps: BacktestStep[], nResamples = 1000, seed = 12345): BootstrapCI {
  return bootstrapMetricCI(steps, directionalAccuracy, nResamples, seed);
}

/** Bootstrap 95% CI for Brier score. */
export function bootstrapBrierCI(steps: BacktestStep[], nResamples = 1000, seed = 12345): BootstrapCI {
  return bootstrapMetricCI(steps, brierScore, nResamples, seed);
}

/** Bootstrap 95% CI for CI coverage. */
export function bootstrapCIcoverageCI(steps: BacktestStep[], nResamples = 1000, seed = 12345): BootstrapCI {
  return bootstrapMetricCI(steps, ciCoverage, nResamples, seed);
}

// ---------------------------------------------------------------------------
// Threshold Optimization (development-time calibration tool)
// ---------------------------------------------------------------------------

export interface ThresholdOptResult {
  bestBuyThreshold: number;
  bestSellThreshold: number;
  bestAccuracy: number;
  /** Accuracy at each grid point for analysis */
  grid: Array<{ buy: number; sell: number; accuracy: number }>;
}

/**
 * Grid search for optimal BUY/SELL thresholds that maximize directional accuracy.
 * Recomputes recommendations from each step's `predictedReturn` with different
 * thresholds and finds the pair that maximizes accuracy.
 *
 * This is a development-time calibration tool — NOT used at runtime.
 */
export function optimizeThresholds(
  steps: BacktestStep[],
  buyRange = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07],
  sellRange = [0.005, 0.01, 0.015, 0.02, 0.03],
  holdBand = 0.03,
): ThresholdOptResult {
  let bestAccuracy = 0;
  let bestBuy = 0.03;
  let bestSell = 0.02;
  const grid: ThresholdOptResult['grid'] = [];

  for (const buyThr of buyRange) {
    for (const sellThr of sellRange) {
      let correct = 0;
      for (const step of steps) {
        let rec: 'BUY' | 'HOLD' | 'SELL';
        if (step.predictedReturn > buyThr) {
          rec = 'BUY';
        } else if (step.predictedReturn < -sellThr) {
          rec = 'SELL';
        } else {
          rec = 'HOLD';
        }

        if (rec === 'BUY' && step.actualReturn > 0) correct++;
        else if (rec === 'SELL' && step.actualReturn < 0) correct++;
        else if (rec === 'HOLD' && Math.abs(step.actualReturn) < holdBand) correct++;
      }

      const accuracy = steps.length > 0 ? correct / steps.length : 0;
      grid.push({ buy: buyThr, sell: sellThr, accuracy });

      if (accuracy > bestAccuracy) {
        bestAccuracy = accuracy;
        bestBuy = buyThr;
        bestSell = sellThr;
      }
    }
  }

  return { bestBuyThreshold: bestBuy, bestSellThreshold: bestSell, bestAccuracy, grid };
}

// ---------------------------------------------------------------------------
// P(up)-based Directional Accuracy (no HOLD dead zone)
// ---------------------------------------------------------------------------

/**
 * Calibrated P(up)-based directional accuracy (no HOLD dead zone).
 * Uses the calibrated predictedProb (0.5 → 1 compressed range).
 * Correct when: prob > 0.5 and actualBinary === 1, or prob < 0.5 and actualBinary === 0.
 * Tie (prob === 0.5) counts as correct for actualBinary === 1 (matches base-rate bias).
 */
export function calibratedPUpDirectionalAccuracy(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  const correct = steps.filter(s =>
    (s.predictedProb > 0.5 && s.actualBinary === 1) ||
    (s.predictedProb < 0.5 && s.actualBinary === 0) ||
    (s.predictedProb === 0.5 && s.actualBinary === 1),
  );
  return correct.length / steps.length;
}

/** Backward-compatible alias — pUpDirectionalAccuracy is the calibrated variant. */
export const pUpDirectionalAccuracy = calibratedPUpDirectionalAccuracy;

/**
 * Raw P(up)-based directional accuracy (no HOLD dead zone).
 * Uses the raw (pre-calibration) predicted probability for sign accuracy.
 * This measures whether the model's uncalibrated signal correctly predicted direction.
 * Correct when: rawProb > 0.5 and actualBinary === 1, or rawProb < 0.5 and actualBinary === 0.
 * Tie (rawProb === 0.5) counts as correct for actualBinary === 1.
 */
export function rawPUpDirectionalAccuracy(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  const correct = steps.filter(s => {
    const rawProb = s.rawPredictedProb ?? s.predictedProb;
    return (
      (rawProb > 0.5 && s.actualBinary === 1) ||
      (rawProb < 0.5 && s.actualBinary === 0) ||
      (rawProb === 0.5 && s.actualBinary === 1)
    );
  });
  return correct.length / steps.length;
}

/**
 * Selective P(up)-based directional accuracy with confidence filtering.
 * Uses calibrated predictedProb. Combines confidence-based abstention with P(up) directional check.
 */
export function selectivePUpAccuracy(
  steps: BacktestStep[],
  minConfidence: number,
): { accuracy: number; coverage: number; selected: number; total: number } {
  if (steps.length === 0) return { accuracy: 0, coverage: 0, selected: 0, total: 0 };

  const selected = steps.filter(s => s.confidence >= minConfidence);
  if (selected.length === 0) return { accuracy: 0, coverage: 0, selected: 0, total: steps.length };

  const correct = selected.filter(s =>
    (s.predictedProb > 0.5 && s.actualBinary === 1) ||
    (s.predictedProb < 0.5 && s.actualBinary === 0) ||
    (s.predictedProb === 0.5 && s.actualBinary === 1),
  );

  return {
    accuracy: correct.length / selected.length,
    coverage: selected.length / steps.length,
    selected: selected.length,
    total: steps.length,
  };
}

/**
 * Selective raw P(up)-based directional accuracy with confidence filtering.
 * Uses raw (pre-calibration) predicted probability for sign accuracy.
 */
export function selectiveRawPUpAccuracy(
  steps: BacktestStep[],
  minConfidence: number,
): { accuracy: number; coverage: number; selected: number; total: number } {
  if (steps.length === 0) return { accuracy: 0, coverage: 0, selected: 0, total: 0 };

  const selected = steps.filter(s => s.confidence >= minConfidence);
  if (selected.length === 0) return { accuracy: 0, coverage: 0, selected: 0, total: steps.length };

  const correct = selected.filter(s => {
    const rawProb = s.rawPredictedProb ?? s.predictedProb;
    return (
      (rawProb > 0.5 && s.actualBinary === 1) ||
      (rawProb < 0.5 && s.actualBinary === 0) ||
      (rawProb === 0.5 && s.actualBinary === 1)
    );
  });

  return {
    accuracy: correct.length / selected.length,
    coverage: selected.length / steps.length,
    selected: selected.length,
    total: steps.length,
  };
}
