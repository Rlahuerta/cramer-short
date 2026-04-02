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

export type FailureSliceKey =
  | 'tickerHorizon'
  | 'regime'
  | 'confidence'
  | 'volatility'
  | 'anchorQuality'
  | 'validationMetric'
  | 'structuralBreak'
  | 'hmmConverged'
  | 'ensembleConsensus';

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
  /** Predicted P(price > currentPrice at t+horizon) */
  predictedProb: number;
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
  structuralBreakDivergence?: number | null;
  hmmConverged?: boolean | null;
  ensembleConsensus?: number | null;
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

export interface BacktestReport {
  ticker: string;
  horizon: number;
  totalSteps: number;
  brierScore: number;
  ciCoverage: number;
  directionalAccuracy: number;
  expectedReturnCorrelation: number;
  sharpness: number;
  reliabilityBins: ReliabilityBin[];
  gofPassRate: number | null;
  balancedDirectionalAccuracy?: number;
  meanEdge?: number;
  failureDecomposition?: FailureDecompositionReport;
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
 * Fraction of steps where the realized price falls within the predicted CI.
 * Target for 90% CI: ~0.90 coverage.
 */
export function ciCoverage(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  const covered = steps.filter(s => s.realizedPrice >= s.ciLower && s.realizedPrice <= s.ciUpper);
  return covered.length / steps.length;
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

export function bucketByValidationMetric(steps: BacktestStep[], holdThreshold = 0.03): BucketedMetricRow[] {
  return summarizeFixedBuckets(
    steps,
    [...VALIDATION_METRIC_BUCKETS],
    step => step.validationMetric ?? UNKNOWN_LABEL,
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
      { key: 'confidence', rows: bucketByConfidence(steps, DEFAULT_CONFIDENCE_BUCKETS, holdThreshold) },
      { key: 'anchorQuality', rows: bucketByAnchorQuality(steps, holdThreshold) },
      { key: 'validationMetric', rows: bucketByValidationMetric(steps, holdThreshold) },
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
  return {
    ticker,
    horizon,
    totalSteps: steps.length,
    brierScore: brierScore(steps),
    ciCoverage: ciCoverage(steps),
    directionalAccuracy: directionalAccuracy(steps),
    expectedReturnCorrelation: expectedReturnCorrelation(steps),
    sharpness: sharpness(steps),
    reliabilityBins: bins,
    gofPassRate: gofPassRate(steps),
    balancedDirectionalAccuracy: balancedDirectionalAccuracy(steps),
    meanEdge: meanEdge(steps),
    failureDecomposition: computeFailureDecomposition(steps),
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
function mulberry32(seed: number): () => number {
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
 * Directional accuracy based purely on predicted P(up) vs actual outcome.
 * No HOLD dead zone: P(up) > 0.5 → predicted UP, else predicted DOWN.
 *
 * This gives a cleaner measure of directional skill since the recommendation-based
 * metric penalizes HOLD predictions even when the model correctly signals UP direction
 * but with insufficient conviction to trigger BUY (e.g. E[R]=+0.3% < buy threshold).
 */
export function pUpDirectionalAccuracy(steps: BacktestStep[]): number {
  if (steps.length === 0) return 0;
  const correct = steps.filter(s =>
    (s.predictedProb > 0.5 && s.actualBinary === 1) ||
    (s.predictedProb < 0.5 && s.actualBinary === 0) ||
    (s.predictedProb === 0.5 && s.actualBinary === 1), // tie → UP (matches base-rate bias)
  );
  return correct.length / steps.length;
}

/**
 * Selective P(up)-based directional accuracy with confidence filtering.
 * Combines confidence-based abstention with P(up) directional check.
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
