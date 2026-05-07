import {
  arbitrateForecast,
  type ForecastArbiterInput,
  type ForecastArbiterResult,
  type ForecastMarketSemantics,
} from '../forecast-arbitrator.js';
import { toForecastArbiterInput, type ArbiterReplayBundle } from '../arbiter-replay.js';

export interface ArbiterReplayEvaluator {
  name: string;
  evaluate(input: ForecastArbiterInput): ForecastArbiterResult;
}

export interface ArbiterReplaySliceMetrics {
  totalRows: number;
  tradedRows: number;
  directionalAccuracy: number | null;
  brierScore: number | null;
  abstainRate: number;
}

export interface ArbiterReplaySemanticBucketMetrics {
  totalRows: number;
  meanPredictedProb: number;
  actualFrequency: number;
}

export interface ArbiterReplayEvaluationRow {
  bundle: ArbiterReplayBundle;
  impliedUpProbability: number;
  actualBinary: 0 | 1;
  traded: boolean;
  directionCorrect: boolean | null;
  primarySemantics: ForecastMarketSemantics;
  whaleSupport: boolean;
  polymarketEvidence: 'none' | 'thin' | 'rich';
  divergent: boolean;
  hasCrossPlatformEvidence: boolean;
  hasFlaggedCrossPlatformEvidence: boolean;
  crossPlatformAdjustmentApplied: boolean;
}

export interface ArbiterReplayEvaluatorReport {
  name: string;
  totalRows: number;
  tradedRows: number;
  abstainRate: number;
  directionalAccuracy: number | null;
  brierScore: number | null;
  semanticBucketCalibration: Record<string, ArbiterReplaySemanticBucketMetrics>;
  disagreementSlice: ArbiterReplaySliceMetrics;
  whaleSupportSlices: {
    withSupport: ArbiterReplaySliceMetrics;
    withoutSupport: ArbiterReplaySliceMetrics;
  };
  polymarketEvidenceSlices: Record<'none' | 'thin' | 'rich', ArbiterReplaySliceMetrics>;
  crossPlatformSlices: {
    withEvidence: ArbiterReplaySliceMetrics;
    withoutEvidence: ArbiterReplaySliceMetrics;
    flaggedDivergence: ArbiterReplaySliceMetrics;
    adjustmentApplied: ArbiterReplaySliceMetrics;
  };
  rows: ArbiterReplayEvaluationRow[];
}

export interface ArbiterReplayRunResult {
  bundleCount: number;
  labeledBundleCount: number;
  skippedBundleCount: number;
  baseline: ArbiterReplayEvaluatorReport;
  candidate?: ArbiterReplayEvaluatorReport;
}

export interface ArbiterReplayGateDecision {
  passed: boolean;
  reasons: string[];
}

function impliedProbability(result: ForecastArbiterResult): number {
  const confidenceMap: Record<ForecastArbiterResult['confidence'], number> = {
    low: 0.58,
    medium: 0.68,
    high: 0.78,
  };

  if (!result.shouldEnterNow || result.preferredDirection === 'neutral' || result.verdict === 'NO_TRADE') {
    return 0.5;
  }

  const directionalProbability = confidenceMap[result.confidence];
  return result.preferredDirection === 'long'
    ? directionalProbability
    : 1 - directionalProbability;
}

function primarySemantics(bundle: ArbiterReplayBundle): ForecastMarketSemantics {
  const counts: Record<ForecastMarketSemantics, number> = {
    terminal: 0,
    barrier_touch: 0,
    range: 0,
    path_dependent: 0,
    unknown: 0,
  };

  for (const market of bundle.polymarket?.selectedMarkets ?? []) {
    counts[market.semantics]++;
  }

  return (Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] as ForecastMarketSemantics | undefined) ?? 'unknown';
}

function polymarketEvidenceRichness(bundle: ArbiterReplayBundle): 'none' | 'thin' | 'rich' {
  const polymarket = bundle.polymarket;
  if (!polymarket || polymarket.selectedMarkets.length === 0) return 'none';
  if (polymarket.selectedMarkets.length >= 2 && (polymarket.qualityScore ?? 0) >= 60 && polymarket.warnings.length === 0) {
    return 'rich';
  }
  return 'thin';
}

function sliceMetrics(rows: ArbiterReplayEvaluationRow[]): ArbiterReplaySliceMetrics {
  if (rows.length === 0) {
    return {
      totalRows: 0,
      tradedRows: 0,
      directionalAccuracy: null,
      brierScore: null,
      abstainRate: 0,
    };
  }

  const tradedRows = rows.filter((row) => row.traded);
  const directionalAccuracy = tradedRows.length > 0
    ? tradedRows.filter((row) => row.directionCorrect === true).length / tradedRows.length
    : null;
  const brierScore = rows.reduce((sum, row) => sum + (row.impliedUpProbability - row.actualBinary) ** 2, 0) / rows.length;

  return {
    totalRows: rows.length,
    tradedRows: tradedRows.length,
    directionalAccuracy,
    brierScore,
    abstainRate: 1 - tradedRows.length / rows.length,
  };
}

function semanticBucketCalibration(rows: ArbiterReplayEvaluationRow[]): Record<string, ArbiterReplaySemanticBucketMetrics> {
  const buckets = new Map<ForecastMarketSemantics, ArbiterReplayEvaluationRow[]>();
  for (const row of rows) {
    const existing = buckets.get(row.primarySemantics) ?? [];
    existing.push(row);
    buckets.set(row.primarySemantics, existing);
  }

  return Object.fromEntries(
    [...buckets.entries()].map(([semantics, bucketRows]) => [
      semantics,
      {
        totalRows: bucketRows.length,
        meanPredictedProb: bucketRows.reduce((sum, row) => sum + row.impliedUpProbability, 0) / bucketRows.length,
        actualFrequency: bucketRows.reduce((sum, row) => sum + row.actualBinary, 0) / bucketRows.length,
      },
    ]),
  );
}

function evaluateRows(
  bundles: ArbiterReplayBundle[],
  evaluator: ArbiterReplayEvaluator,
): ArbiterReplayEvaluatorReport {
  const rows = bundles.map((bundle) => {
    const input = toForecastArbiterInput(bundle);
    const result = evaluator.evaluate(input);
    const actualBinary = bundle.labels?.forecast?.actualBinary ?? 0;
    const traded = result.shouldEnterNow && result.preferredDirection !== 'neutral' && result.verdict !== 'NO_TRADE';
    const directionCorrect = !traded
      ? null
      : result.preferredDirection === 'long'
        ? actualBinary === 1
        : actualBinary === 0;

    return {
      bundle,
      impliedUpProbability: impliedProbability(result),
      actualBinary,
      traded,
      directionCorrect,
      primarySemantics: primarySemantics(bundle),
      whaleSupport: bundle.whale !== undefined && bundle.whale !== null,
      polymarketEvidence: polymarketEvidenceRichness(bundle),
      divergent: result.disagreement.isDivergent,
      hasCrossPlatformEvidence: (bundle.polymarket?.crossPlatformEvidence?.length ?? 0) > 0,
      hasFlaggedCrossPlatformEvidence: (bundle.polymarket?.crossPlatformEvidence ?? []).some((entry) => entry.flagged),
      crossPlatformAdjustmentApplied: bundle.polymarket?.crossPlatformAdjustment?.applied === true,
    };
  });

  return {
    name: evaluator.name,
    ...sliceMetrics(rows),
    semanticBucketCalibration: semanticBucketCalibration(rows),
    disagreementSlice: sliceMetrics(rows.filter((row) => row.divergent)),
    whaleSupportSlices: {
      withSupport: sliceMetrics(rows.filter((row) => row.whaleSupport)),
      withoutSupport: sliceMetrics(rows.filter((row) => !row.whaleSupport)),
    },
    polymarketEvidenceSlices: {
      none: sliceMetrics(rows.filter((row) => row.polymarketEvidence === 'none')),
      thin: sliceMetrics(rows.filter((row) => row.polymarketEvidence === 'thin')),
      rich: sliceMetrics(rows.filter((row) => row.polymarketEvidence === 'rich')),
    },
    crossPlatformSlices: {
      withEvidence: sliceMetrics(rows.filter((row) => row.hasCrossPlatformEvidence)),
      withoutEvidence: sliceMetrics(rows.filter((row) => !row.hasCrossPlatformEvidence)),
      flaggedDivergence: sliceMetrics(rows.filter((row) => row.hasFlaggedCrossPlatformEvidence)),
      adjustmentApplied: sliceMetrics(rows.filter((row) => row.crossPlatformAdjustmentApplied)),
    },
    rows,
  };
}

export function runArbiterReplay(params: {
  bundles: ArbiterReplayBundle[];
  baselineEvaluator?: ArbiterReplayEvaluator;
  candidateEvaluator?: ArbiterReplayEvaluator;
}): ArbiterReplayRunResult {
  const baselineEvaluator = params.baselineEvaluator ?? {
    name: 'baseline',
    evaluate: arbitrateForecast,
  };
  const labeledBundles = params.bundles.filter((bundle) => bundle.labels?.forecast !== undefined);

  return {
    bundleCount: params.bundles.length,
    labeledBundleCount: labeledBundles.length,
    skippedBundleCount: params.bundles.length - labeledBundles.length,
    baseline: evaluateRows(labeledBundles, baselineEvaluator),
    ...(params.candidateEvaluator ? { candidate: evaluateRows(labeledBundles, params.candidateEvaluator) } : {}),
  };
}

export function compareReplayEvaluators(
  baseline: ArbiterReplayEvaluatorReport,
  candidate: ArbiterReplayEvaluatorReport,
  options: {
    maxAbstainRateIncrease?: number;
    maxBrierRegression?: number;
    minDirectionalAccuracyLift?: number;
  } = {},
): ArbiterReplayGateDecision {
  const {
    maxAbstainRateIncrease = 0.05,
    maxBrierRegression = 0,
    minDirectionalAccuracyLift = 0,
  } = options;
  const reasons: string[] = [];

  const baselineDirectionalAccuracy = baseline.directionalAccuracy ?? 0;
  const candidateDirectionalAccuracy = candidate.directionalAccuracy ?? 0;
  if (candidateDirectionalAccuracy < baselineDirectionalAccuracy + minDirectionalAccuracyLift) {
    reasons.push('Candidate directional accuracy did not improve enough over baseline.');
  }

  if (
    baseline.brierScore !== null
    && candidate.brierScore !== null
    && candidate.brierScore > baseline.brierScore + maxBrierRegression
  ) {
    reasons.push('Candidate Brier score regressed beyond the allowed tolerance.');
  }

  if (candidate.abstainRate > baseline.abstainRate + maxAbstainRateIncrease) {
    reasons.push('Candidate abstain rate increased beyond the allowed tolerance.');
  }

  return {
    passed: reasons.length === 0,
    reasons,
  };
}
