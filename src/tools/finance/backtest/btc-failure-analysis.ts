import {
  brierScore,
  computeFailureDecomposition,
  directionalAccuracy,
  meanEdge,
  type BacktestStep,
  type BucketedMetricRow,
  type FailureSliceKey,
} from './metrics.js';

const DEFAULT_MIN_BUCKET_N = 5;
const DEFAULT_TOP_K = 10;
const DEFAULT_CI_ALPHA = 0.05;
const MIN_TOTAL_STEPS = 20;
const ACTIONABLE_WEAKNESS_THRESHOLD = 0.05;
const MAX_ACTIONABLE_FRACTION = 0.35;
const ACTIONABLE_DOMINANCE_GAP = 0.03;
const ACTIONABLE_RUNNER_UP_SHARE = 0.85;

type FailureWeaknessFamily =
  | 'market-regime'
  | 'realized-move'
  | 'break-context'
  | 'signal-shape'
  | 'decision-context'
  | 'input-context';

const FAILURE_WEAKNESS_FAMILY: Record<FailureSliceKey, FailureWeaknessFamily> = {
  tickerHorizon: 'input-context',
  regime: 'market-regime',
  confidence: 'signal-shape',
  volatility: 'realized-move',
  moveMagnitude: 'realized-move',
  moveDirection: 'realized-move',
  anchorQuality: 'input-context',
  recommendation: 'decision-context',
  trendVsChop: 'market-regime',
  validationMetric: 'input-context',
  structuralBreak: 'break-context',
  divergence: 'break-context',
  hmmConverged: 'break-context',
  ensembleConsensus: 'signal-shape',
  pUpBand: 'signal-shape',
};

export interface RankedFailureBucket {
  dimension: FailureSliceKey;
  bucket: string;
  n: number;
  fraction: number;
  directionalAccuracy: number;
  accuracyCI: { lower: number; upper: number };
  brierScore: number;
  meanEdge: number;
  accuracyDelta: number;
  brierDelta: number;
  edgeDelta: number;
  weaknessScore: number;
  passesWilsonScreen: boolean;
}

export type BtcFailureAnalysisVerdict = 'actionable' | 'diffuse' | 'insufficient-data';

export interface BtcFailureAnalysisReport {
  horizon: number;
  totalSteps: number;
  aggregateAccuracy: number;
  aggregateBrier: number;
  aggregateEdge: number;
  rankedBuckets: RankedFailureBucket[];
  topCandidate: RankedFailureBucket | null;
  verdict: BtcFailureAnalysisVerdict;
  verdictReason: string;
}

export interface BtcFailureAnalysisConfig {
  minBucketN?: number;
  topK?: number;
  ciAlpha?: number;
}

function clampProbability(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function normalizePositiveInteger(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value)) return fallback;
  return Math.max(1, Math.floor(value));
}

function normalizeAlpha(value: number | undefined): number {
  if (!Number.isFinite(value) || value === undefined || value <= 0 || value >= 1) {
    return DEFAULT_CI_ALPHA;
  }
  return value;
}

function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function formatSignedPct(value: number, digits = 1): string {
  const pct = value * 100;
  return `${pct >= 0 ? '+' : ''}${pct.toFixed(digits)}%`;
}

function formatSignedPp(value: number, digits = 1): string {
  const pctPoints = value * 100;
  return `${pctPoints >= 0 ? '+' : ''}${pctPoints.toFixed(digits)}pp`;
}

function inverseStandardNormal(probability: number): number {
  if (probability <= 0 || probability >= 1) {
    throw new Error(`Probability must be between 0 and 1 exclusive, got ${probability}`);
  }

  const a = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.38357751867269e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
  ] as const;
  const b = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
  ] as const;
  const c = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
  ] as const;
  const d = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
  ] as const;
  const lowerTail = 0.02425;
  const upperTail = 1 - lowerTail;

  if (probability < lowerTail) {
    const q = Math.sqrt(-2 * Math.log(probability));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
      / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }

  if (probability > upperTail) {
    const q = Math.sqrt(-2 * Math.log(1 - probability));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
      / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }

  const q = probability - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function wilsonCI(
  successes: number,
  total: number,
  alpha: number,
): { lower: number; upper: number } {
  if (total <= 0) return { lower: 0, upper: 0 };

  const boundedAlpha = Number.isFinite(alpha) && alpha > 0 && alpha < 1
    ? alpha
    : DEFAULT_CI_ALPHA;
  const z = inverseStandardNormal(1 - boundedAlpha / 2);
  const boundedSuccesses = Math.min(total, Math.max(0, successes));
  const phat = boundedSuccesses / total;
  const denominator = 1 + (z * z) / total;
  const centre = phat + (z * z) / (2 * total);
  const margin = z * Math.sqrt((phat * (1 - phat) + (z * z) / (4 * total)) / total);

  return {
    lower: clampProbability((centre - margin) / denominator),
    upper: clampProbability((centre + margin) / denominator),
  };
}

function computeWeaknessScore(bucket: Omit<RankedFailureBucket, 'weaknessScore' | 'passesWilsonScreen'>): number {
  const accuracyPenalty = Math.max(0, -bucket.accuracyDelta);
  const brierPenalty = Math.max(0, bucket.brierDelta);
  const edgePenalty = Math.max(0, -bucket.edgeDelta);
  const baseScore = (0.5 * accuracyPenalty) + (0.3 * brierPenalty) + (0.2 * edgePenalty);
  const coverageWeight = 0.5 + (0.5 * Math.sqrt(Math.max(0, bucket.fraction)));
  return baseScore * coverageWeight;
}

function failureWeaknessFamily(dimension: FailureSliceKey): FailureWeaknessFamily {
  return FAILURE_WEAKNESS_FAMILY[dimension];
}

function toSuccessCount(row: BucketedMetricRow): number {
  return Math.round(row.directionalAccuracy * row.count);
}

function toRankedBucket(
  dimension: FailureSliceKey,
  row: BucketedMetricRow,
  aggregateAccuracy: number,
  aggregateBrier: number,
  aggregateEdge: number,
  ciAlpha: number,
): RankedFailureBucket {
  const accuracyCI = wilsonCI(toSuccessCount(row), row.count, ciAlpha);
  const rankedBucketBase = {
    dimension,
    bucket: row.label,
    n: row.count,
    fraction: row.fraction,
    directionalAccuracy: row.directionalAccuracy,
    accuracyCI,
    brierScore: row.brierScore,
    meanEdge: row.meanEdge,
    accuracyDelta: row.directionalAccuracy - aggregateAccuracy,
    brierDelta: row.brierScore - aggregateBrier,
    edgeDelta: row.meanEdge - aggregateEdge,
  };

  return {
    ...rankedBucketBase,
    weaknessScore: computeWeaknessScore(rankedBucketBase),
    passesWilsonScreen: accuracyCI.upper < aggregateAccuracy,
  };
}

function sortRankedBuckets(a: RankedFailureBucket, b: RankedFailureBucket): number {
  if (b.weaknessScore !== a.weaknessScore) return b.weaknessScore - a.weaknessScore;
  if (b.n !== a.n) return b.n - a.n;
  if (a.dimension !== b.dimension) return a.dimension.localeCompare(b.dimension);
  return a.bucket.localeCompare(b.bucket);
}

function familyRepresentatives(rankedBuckets: RankedFailureBucket[]): RankedFailureBucket[] {
  const seenFamilies = new Set<FailureWeaknessFamily>();
  const representatives: RankedFailureBucket[] = [];

  for (const bucket of rankedBuckets) {
    const family = failureWeaknessFamily(bucket.dimension);
    if (seenFamilies.has(family)) continue;
    seenFamilies.add(family);
    representatives.push(bucket);
  }

  return representatives;
}

function isComparableIndependentWeakness(
  topBucket: RankedFailureBucket,
  competitor: RankedFailureBucket,
): boolean {
  if (competitor.weaknessScore < ACTIONABLE_WEAKNESS_THRESHOLD) return false;
  if (competitor.fraction > MAX_ACTIONABLE_FRACTION) return false;

  const weaknessGap = topBucket.weaknessScore - competitor.weaknessScore;
  const weaknessShare = topBucket.weaknessScore > 0
    ? competitor.weaknessScore / topBucket.weaknessScore
    : 0;

  return weaknessGap < ACTIONABLE_DOMINANCE_GAP || weaknessShare >= ACTIONABLE_RUNNER_UP_SHARE;
}

function firstComparableIndependentWeakness(
  topBucket: RankedFailureBucket,
  candidates: RankedFailureBucket[],
): RankedFailureBucket | null {
  for (const candidate of candidates) {
    if (isComparableIndependentWeakness(topBucket, candidate)) {
      return candidate;
    }
  }
  return null;
}

export function rankFailureBuckets(
  steps: BacktestStep[],
  horizon: number,
  config: BtcFailureAnalysisConfig = {},
): BtcFailureAnalysisReport {
  const minBucketN = normalizePositiveInteger(config.minBucketN, DEFAULT_MIN_BUCKET_N);
  const topK = normalizePositiveInteger(config.topK, DEFAULT_TOP_K);
  const ciAlpha = normalizeAlpha(config.ciAlpha);
  const aggregateAccuracy = directionalAccuracy(steps);
  const aggregateBrier = brierScore(steps);
  const aggregateEdge = meanEdge(steps);

  const eligibleBuckets = computeFailureDecomposition(steps).slices.flatMap(slice =>
    slice.rows
      .filter(row => row.count >= minBucketN)
      .map(row => toRankedBucket(slice.key, row, aggregateAccuracy, aggregateBrier, aggregateEdge, ciAlpha)),
  );

  if (steps.length < MIN_TOTAL_STEPS) {
    return {
      horizon,
      totalSteps: steps.length,
      aggregateAccuracy,
      aggregateBrier,
      aggregateEdge,
      rankedBuckets: [],
      topCandidate: null,
      verdict: 'insufficient-data',
      verdictReason: `Need at least ${MIN_TOTAL_STEPS} BTC steps before ranking failure slices.`,
    };
  }

  if (eligibleBuckets.length === 0) {
    return {
      horizon,
      totalSteps: steps.length,
      aggregateAccuracy,
      aggregateBrier,
      aggregateEdge,
      rankedBuckets: [],
      topCandidate: null,
      verdict: 'insufficient-data',
      verdictReason: `No slice buckets met the minimum sample threshold (n>=${minBucketN}).`,
    };
  }

  const familyBuckets = familyRepresentatives(eligibleBuckets
    .filter(bucket => bucket.weaknessScore > 0)
    .sort(sortRankedBuckets));
  const topCandidate = familyBuckets[0] ?? null;

  if (!topCandidate) {
    return {
      horizon,
      totalSteps: steps.length,
      aggregateAccuracy,
      aggregateBrier,
      aggregateEdge,
      rankedBuckets: [],
      topCandidate: null,
      verdict: 'diffuse',
      verdictReason: 'No bucket underperformed the aggregate after the minimum-sample filter.',
    };
  }

  const comparableCompetitor = firstComparableIndependentWeakness(topCandidate, familyBuckets.slice(1));
  const rankedBuckets = familyBuckets.slice(0, topK);
  const hasComparableRunnerUp = comparableCompetitor !== null;

  if (
    topCandidate.passesWilsonScreen
    && topCandidate.weaknessScore >= ACTIONABLE_WEAKNESS_THRESHOLD
    && topCandidate.fraction <= MAX_ACTIONABLE_FRACTION
    && !hasComparableRunnerUp
  ) {
    return {
      horizon,
      totalSteps: steps.length,
      aggregateAccuracy,
      aggregateBrier,
      aggregateEdge,
      rankedBuckets,
      topCandidate,
      verdict: 'actionable',
      verdictReason: `Top weakness ${topCandidate.dimension}=${topCandidate.bucket} passes the Wilson screen and clearly dominates the next independent weakness family.`,
    };
  }

  let verdictReason = `Top weakness ${topCandidate.dimension}=${topCandidate.bucket} does not clear the current BTC slice-actionability screen.`;

  if (topCandidate.fraction > MAX_ACTIONABLE_FRACTION) {
    verdictReason = `Top weakness ${topCandidate.dimension}=${topCandidate.bucket} is too broad (${formatPct(topCandidate.fraction)}) to count as a sparse BTC failure slice.`;
  } else if (!topCandidate.passesWilsonScreen) {
    verdictReason = `Top weakness ${topCandidate.dimension}=${topCandidate.bucket} underperforms, but it does not pass the Wilson screen against aggregate directional accuracy.`;
  } else if (topCandidate.weaknessScore < ACTIONABLE_WEAKNESS_THRESHOLD) {
    verdictReason = `Top weakness ${topCandidate.dimension}=${topCandidate.bucket} passes the Wilson screen, but its weakness score stays below the actionable threshold.`;
  } else if (hasComparableRunnerUp && comparableCompetitor) {
    verdictReason = `Weakness is split across independent families: ${topCandidate.dimension}=${topCandidate.bucket} and ${comparableCompetitor.dimension}=${comparableCompetitor.bucket} remain too comparable to justify a single next BTC lever.`;
  }

  return {
    horizon,
    totalSteps: steps.length,
    aggregateAccuracy,
    aggregateBrier,
    aggregateEdge,
    rankedBuckets,
    topCandidate,
    verdict: 'diffuse',
    verdictReason,
  };
}

export function formatFailureAnalysisReport(report: BtcFailureAnalysisReport): string[] {
  const lines = [
    `  BTC-USD ${report.horizon}d failure-slice analysis:`,
    `    aggregate dir=${formatPct(report.aggregateAccuracy)} | brier=${report.aggregateBrier.toFixed(3)} | edge=${formatSignedPct(report.aggregateEdge)} | steps=${report.totalSteps}`,
  ];

  if (report.rankedBuckets.length === 0) {
    lines.push(
      report.verdict === 'insufficient-data'
        ? `    insufficient data — ${report.verdictReason}`
        : '    no independent weak buckets cleared the current ranking filters',
    );
  } else {
    lines.push('    independent weakest buckets:');
    for (const [index, bucket] of report.rankedBuckets.entries()) {
      const marker = index === 0 ? '>' : ' ';
      const screened = bucket.passesWilsonScreen ? ' *' : '';
      lines.push(
        `    ${marker} ${`${bucket.dimension}:${bucket.bucket}`.padEnd(33)} `
        + `n=${String(bucket.n).padStart(3)} | frac=${formatPct(bucket.fraction).padStart(6)} | `
        + `dir=${formatPct(bucket.directionalAccuracy).padStart(6)} [${formatPct(bucket.accuracyCI.lower).padStart(6)}, ${formatPct(bucket.accuracyCI.upper).padStart(6)}] | `
        + `Δdir=${formatSignedPp(bucket.accuracyDelta).padStart(7)} | `
        + `brier=${bucket.brierScore.toFixed(3)} (${bucket.brierDelta >= 0 ? '+' : ''}${bucket.brierDelta.toFixed(3)}) | `
        + `edge=${formatSignedPct(bucket.meanEdge).padStart(7)} (${formatSignedPct(bucket.edgeDelta).padStart(7)}) | `
        + `weak=${bucket.weaknessScore.toFixed(3)}${screened}`,
      );
    }

    if (report.rankedBuckets.some(bucket => bucket.passesWilsonScreen)) {
      lines.push('    * Wilson-screened = bucket CI upper bound stays below aggregate directional accuracy');
    }
  }

  lines.push(`    verdict=${report.verdict} — ${report.verdictReason}`);
  return lines;
}
