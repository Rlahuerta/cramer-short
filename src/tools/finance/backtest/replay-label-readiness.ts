import { readFileSync } from 'node:fs';
import type { ReplayLabelBenchmarkArtifact } from './replay-label-benchmark-pipeline.js';
import type {
  ShortHorizonReplayBenchmarkReport,
  ShortHorizonReplayBenchmarkSlice,
} from './polymarket-short-horizon-benchmark.js';

const READINESS_HORIZON_KEYS = ['1d', '2d', '3d'] as const;

type ReplayLabelReadinessHorizonKey = typeof READINESS_HORIZON_KEYS[number];
type ReplayLabelReadinessMetricByHorizon<T> = Record<ReplayLabelReadinessHorizonKey, T>;

export interface ReplayLabelReadinessThresholds {
  requireReadySlice: boolean;
  minLabeledBundleCountByHorizon: ReplayLabelReadinessMetricByHorizon<number>;
  minTradedRowCountByHorizon: ReplayLabelReadinessMetricByHorizon<number>;
  maxAbstainRateByHorizon: ReplayLabelReadinessMetricByHorizon<number>;
  maxBrierScoreByHorizon: ReplayLabelReadinessMetricByHorizon<number>;
}

export interface ReplayLabelReadinessThresholdOverrides {
  requireReadySlice?: boolean;
  minLabeledBundleCountByHorizon?: Partial<ReplayLabelReadinessMetricByHorizon<number>>;
  minTradedRowCountByHorizon?: Partial<ReplayLabelReadinessMetricByHorizon<number>>;
  maxAbstainRateByHorizon?: Partial<ReplayLabelReadinessMetricByHorizon<number>>;
  maxBrierScoreByHorizon?: Partial<ReplayLabelReadinessMetricByHorizon<number>>;
}

export interface ReplayLabelReadinessHorizonMetrics {
  horizonDays: 1 | 2 | 3;
  bundleCount: number;
  labeledBundleCount: number;
  tradedRowCount: number;
  abstainRate: number;
  directionalAccuracy: number | null;
  brierScore: number | null;
  crossPlatformEvidenceRowCount: number;
  crossPlatformFlaggedRowCount: number;
  crossPlatformAdjustmentAppliedRowCount: number;
  ready: boolean;
  pendingReasons: string[];
}

export interface ReplayLabelReadinessHorizonDecision {
  status: 'pass' | 'hold';
  reasons: string[];
  thresholds: {
    minLabeledBundleCount: number;
    minTradedRowCount: number;
    maxAbstainRate: number;
    maxBrierScore: number;
    requireReadySlice: boolean;
  };
  metrics: ReplayLabelReadinessHorizonMetrics;
}

export interface ReplayLabelReadinessReport {
  formatVersion: 'replay-label-readiness-report.v1';
  generatedAt: string;
  sourceType: 'artifact' | 'benchmark';
  sourcePath: string;
  decision: 'eligible' | 'hold';
  reasons: string[];
  thresholds: ReplayLabelReadinessThresholds;
  horizons: Record<ReplayLabelReadinessHorizonKey, ReplayLabelReadinessHorizonDecision>;
}

export const DEFAULT_REPLAY_LABEL_READINESS_THRESHOLDS: ReplayLabelReadinessThresholds = {
  requireReadySlice: true,
  minLabeledBundleCountByHorizon: {
    '1d': 30,
    '2d': 25,
    '3d': 20,
  },
  minTradedRowCountByHorizon: {
    '1d': 15,
    '2d': 12,
    '3d': 10,
  },
  maxAbstainRateByHorizon: {
    '1d': 0.45,
    '2d': 0.5,
    '3d': 0.55,
  },
  maxBrierScoreByHorizon: {
    '1d': 0.24,
    '2d': 0.25,
    '3d': 0.26,
  },
};

function mergeMetricByHorizon(
  base: ReplayLabelReadinessMetricByHorizon<number>,
  override?: Partial<ReplayLabelReadinessMetricByHorizon<number>>,
): ReplayLabelReadinessMetricByHorizon<number> {
  return {
    '1d': override?.['1d'] ?? base['1d'],
    '2d': override?.['2d'] ?? base['2d'],
    '3d': override?.['3d'] ?? base['3d'],
  };
}

export function resolveReplayLabelReadinessThresholds(
  overrides: ReplayLabelReadinessThresholdOverrides = {},
): ReplayLabelReadinessThresholds {
  return {
    requireReadySlice: overrides.requireReadySlice ?? DEFAULT_REPLAY_LABEL_READINESS_THRESHOLDS.requireReadySlice,
    minLabeledBundleCountByHorizon: mergeMetricByHorizon(
      DEFAULT_REPLAY_LABEL_READINESS_THRESHOLDS.minLabeledBundleCountByHorizon,
      overrides.minLabeledBundleCountByHorizon,
    ),
    minTradedRowCountByHorizon: mergeMetricByHorizon(
      DEFAULT_REPLAY_LABEL_READINESS_THRESHOLDS.minTradedRowCountByHorizon,
      overrides.minTradedRowCountByHorizon,
    ),
    maxAbstainRateByHorizon: mergeMetricByHorizon(
      DEFAULT_REPLAY_LABEL_READINESS_THRESHOLDS.maxAbstainRateByHorizon,
      overrides.maxAbstainRateByHorizon,
    ),
    maxBrierScoreByHorizon: mergeMetricByHorizon(
      DEFAULT_REPLAY_LABEL_READINESS_THRESHOLDS.maxBrierScoreByHorizon,
      overrides.maxBrierScoreByHorizon,
    ),
  };
}

function toHorizonMetrics(slice: ShortHorizonReplayBenchmarkSlice): ReplayLabelReadinessHorizonMetrics {
  return {
    horizonDays: slice.horizonDays,
    bundleCount: slice.bundleCount,
    labeledBundleCount: slice.labeledBundleCount,
    tradedRowCount: slice.tradedRowCount,
    abstainRate: slice.abstainRate,
    directionalAccuracy: slice.directionalAccuracy,
    brierScore: slice.brierScore,
    crossPlatformEvidenceRowCount: slice.crossPlatformEvidenceRowCount,
    crossPlatformFlaggedRowCount: slice.crossPlatformFlaggedRowCount,
    crossPlatformAdjustmentAppliedRowCount: slice.crossPlatformAdjustmentAppliedRowCount,
    ready: slice.ready,
    pendingReasons: [...slice.pendingReasons],
  };
}

function isReplayLabelBenchmarkArtifact(
  value: unknown,
): value is ReplayLabelBenchmarkArtifact {
  return typeof value === 'object'
    && value !== null
    && (value as ReplayLabelBenchmarkArtifact).formatVersion === 'replay-label-benchmark-report.v1'
    && typeof (value as ReplayLabelBenchmarkArtifact).benchmarkReportPath === 'string'
    && typeof (value as ReplayLabelBenchmarkArtifact).benchmark === 'object';
}

function isShortHorizonReplayBenchmarkReport(
  value: unknown,
): value is ShortHorizonReplayBenchmarkReport {
  return typeof value === 'object'
    && value !== null
    && (value as ShortHorizonReplayBenchmarkReport).formatVersion === 'polymarket-short-horizon-benchmark.v1'
    && typeof (value as ShortHorizonReplayBenchmarkReport).sourcePath === 'string'
    && typeof (value as ShortHorizonReplayBenchmarkReport).horizons === 'object';
}

function evaluateHorizon(
  horizonKey: ReplayLabelReadinessHorizonKey,
  slice: ShortHorizonReplayBenchmarkSlice | undefined,
  thresholds: ReplayLabelReadinessThresholds,
): ReplayLabelReadinessHorizonDecision {
  const horizonDays = Number.parseInt(horizonKey, 10) as 1 | 2 | 3;
  const metrics = slice
    ? toHorizonMetrics(slice)
    : {
        horizonDays,
        bundleCount: 0,
        labeledBundleCount: 0,
        tradedRowCount: 0,
        abstainRate: 1,
        directionalAccuracy: null,
        brierScore: null,
        crossPlatformEvidenceRowCount: 0,
        crossPlatformFlaggedRowCount: 0,
        crossPlatformAdjustmentAppliedRowCount: 0,
        ready: false,
        pendingReasons: [`Missing ${horizonKey} benchmark slice.`],
      };

  const reasons: string[] = [];
  const minLabeledBundleCount = thresholds.minLabeledBundleCountByHorizon[horizonKey];
  const minTradedRowCount = thresholds.minTradedRowCountByHorizon[horizonKey];
  const maxAbstainRate = thresholds.maxAbstainRateByHorizon[horizonKey];
  const maxBrierScore = thresholds.maxBrierScoreByHorizon[horizonKey];

  if (!slice) reasons.push(`Missing ${horizonKey} benchmark slice.`);
  if (thresholds.requireReadySlice && !metrics.ready) {
    reasons.push(`${horizonKey} slice is not marked ready.`);
  }
  if (metrics.pendingReasons.length > 0) {
    reasons.push(...metrics.pendingReasons.map((reason) => `${horizonKey}: ${reason}`));
  }
  if (metrics.labeledBundleCount < minLabeledBundleCount) {
    reasons.push(
      `${horizonKey} labeledBundleCount ${metrics.labeledBundleCount} is below minimum ${minLabeledBundleCount}.`,
    );
  }
  if (metrics.tradedRowCount < minTradedRowCount) {
    reasons.push(`${horizonKey} tradedRowCount ${metrics.tradedRowCount} is below minimum ${minTradedRowCount}.`);
  }
  if (!Number.isFinite(metrics.abstainRate) || metrics.abstainRate > maxAbstainRate) {
    reasons.push(`${horizonKey} abstainRate ${metrics.abstainRate} exceeds maximum ${maxAbstainRate}.`);
  }
  if (metrics.brierScore === null) {
    reasons.push(`${horizonKey} brierScore is missing.`);
  } else if (!Number.isFinite(metrics.brierScore) || metrics.brierScore > maxBrierScore) {
    reasons.push(`${horizonKey} brierScore ${metrics.brierScore} exceeds maximum ${maxBrierScore}.`);
  }

  return {
    status: reasons.length === 0 ? 'pass' : 'hold',
    reasons,
    thresholds: {
      minLabeledBundleCount,
      minTradedRowCount,
      maxAbstainRate,
      maxBrierScore,
      requireReadySlice: thresholds.requireReadySlice,
    },
    metrics,
  };
}

export function evaluateReplayLabelReadiness(
  input: ReplayLabelBenchmarkArtifact | ShortHorizonReplayBenchmarkReport,
  options: {
    generatedAt?: string;
    thresholds?: ReplayLabelReadinessThresholdOverrides;
  } = {},
): ReplayLabelReadinessReport {
  const thresholds = resolveReplayLabelReadinessThresholds(options.thresholds);
  const sourceType = isReplayLabelBenchmarkArtifact(input) ? 'artifact' : 'benchmark';
  const benchmark = isReplayLabelBenchmarkArtifact(input) ? input.benchmark : input;
  const sourcePath = isReplayLabelBenchmarkArtifact(input) ? input.benchmarkReportPath : input.sourcePath;
  const horizons = {
    '1d': evaluateHorizon('1d', benchmark.horizons['1d'], thresholds),
    '2d': evaluateHorizon('2d', benchmark.horizons['2d'], thresholds),
    '3d': evaluateHorizon('3d', benchmark.horizons['3d'], thresholds),
  } satisfies Record<ReplayLabelReadinessHorizonKey, ReplayLabelReadinessHorizonDecision>;

  const reasons = Array.from(
    new Set(
      READINESS_HORIZON_KEYS.flatMap((horizonKey) => horizons[horizonKey].reasons),
    ),
  );

  return {
    formatVersion: 'replay-label-readiness-report.v1',
    generatedAt: options.generatedAt ?? new Date().toISOString(),
    sourceType,
    sourcePath,
    decision: reasons.length === 0 ? 'eligible' : 'hold',
    reasons,
    thresholds,
    horizons,
  };
}

export function readReplayLabelReadinessInputFromFile(
  inputPath: string,
): ReplayLabelBenchmarkArtifact | ShortHorizonReplayBenchmarkReport {
  const parsed = JSON.parse(readFileSync(inputPath, 'utf-8')) as unknown;

  if (isReplayLabelBenchmarkArtifact(parsed) || isShortHorizonReplayBenchmarkReport(parsed)) {
    return parsed;
  }

  throw new Error(`replay-label readiness: unsupported benchmark input format in "${inputPath}"`);
}

export function runReplayLabelReadinessFromFile(params: {
  inputPath: string;
  generatedAt?: string;
  thresholds?: ReplayLabelReadinessThresholdOverrides;
}): ReplayLabelReadinessReport {
  const input = readReplayLabelReadinessInputFromFile(params.inputPath);
  return evaluateReplayLabelReadiness(input, {
    generatedAt: params.generatedAt,
    thresholds: params.thresholds,
  });
}
