import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { DEFAULT_ARBITER_REPLAY_BUNDLES_PATH } from '../arbiter-replay.js';
import type { ArbiterReplayEvaluator } from './arbiter-replay-runner.js';
import {
  runShortHorizonReplayBenchmarkFromFile,
  type ShortHorizonReplayBenchmarkReport,
} from './polymarket-short-horizon-benchmark.js';
import {
  DEFAULT_ARBITER_REPLAY_LABELED_PATH,
  type ReplayLabelRunResult,
} from './replay-label-runner.js';
import {
  toReplayLabelBatchReportPath,
  runReplayLabelBatchFromFile,
  type ReplayTickerHistoryLoader,
} from './replay-label-batch-runner.js';
import { assertNoCanonicalPathCollision } from './path-collision-guard.js';

const SHORT_HORIZON_KEYS = ['1d', '2d', '3d'] as const;

type ReplayLabelBenchmarkHorizonKey = typeof SHORT_HORIZON_KEYS[number];

type ShortHorizonBenchmarkRunner = (params?: {
  bundlePath?: string;
  evaluator?: ArbiterReplayEvaluator;
  generatedAt?: string;
}) => ShortHorizonReplayBenchmarkReport;

export interface ReplayLabelBenchmarkHorizonCounts {
  bundleCount: number;
  labeledRowCount: number;
  crossPlatformEvidenceRowCount: number;
  crossPlatformFlaggedRowCount: number;
  crossPlatformAdjustmentAppliedRowCount: number;
}

export interface ReplayLabelBenchmarkArtifact {
  formatVersion: 'replay-label-benchmark-report.v1';
  generatedAt: string;
  labeledOutputPath: string;
  benchmarkReportPath: string;
  horizonCounts: Record<ReplayLabelBenchmarkHorizonKey, ReplayLabelBenchmarkHorizonCounts>;
  benchmark: ShortHorizonReplayBenchmarkReport;
}

export interface ReplayLabelBenchmarkPipelineResult {
  labeling: ReplayLabelRunResult;
  benchmarkArtifact: ReplayLabelBenchmarkArtifact;
}

export function toReplayLabelBenchmarkReportPath(labeledOutputPath: string): string {
  return labeledOutputPath.endsWith('.jsonl')
    ? `${labeledOutputPath.slice(0, -'.jsonl'.length)}.benchmark.report.json`
    : `${labeledOutputPath}.benchmark.report.json`;
}

export const DEFAULT_ARBITER_REPLAY_LABELED_BENCHMARK_REPORT_PATH = toReplayLabelBenchmarkReportPath(
  DEFAULT_ARBITER_REPLAY_LABELED_PATH,
);

export function toReplayLabelBenchmarkHorizonCounts(
  report: ShortHorizonReplayBenchmarkReport,
): Record<ReplayLabelBenchmarkHorizonKey, ReplayLabelBenchmarkHorizonCounts> {
  return Object.fromEntries(
    SHORT_HORIZON_KEYS.map((horizonKey) => {
      const horizon = report.horizons[horizonKey];
      return [
        horizonKey,
        {
          bundleCount: horizon.bundleCount,
          labeledRowCount: horizon.labeledBundleCount,
          crossPlatformEvidenceRowCount: horizon.crossPlatformEvidenceRowCount,
          crossPlatformFlaggedRowCount: horizon.crossPlatformFlaggedRowCount,
          crossPlatformAdjustmentAppliedRowCount: horizon.crossPlatformAdjustmentAppliedRowCount,
        },
      ];
    }),
  ) as Record<ReplayLabelBenchmarkHorizonKey, ReplayLabelBenchmarkHorizonCounts>;
}

function writeReplayLabelBenchmarkArtifact(report: ReplayLabelBenchmarkArtifact, filePath: string): void {
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(report, null, 2)}\n`, 'utf-8');
}

export function runReplayBenchmarkHandoffFromLabeledFile(params: {
  labeledOutputPath?: string;
  benchmarkReportPath?: string;
  evaluator?: ArbiterReplayEvaluator;
  generatedAt?: string;
  runBenchmarkFromFile?: ShortHorizonBenchmarkRunner;
} = {}): ReplayLabelBenchmarkArtifact {
  const labeledOutputPath = params.labeledOutputPath ?? DEFAULT_ARBITER_REPLAY_LABELED_PATH;
  const benchmarkReportPath = params.benchmarkReportPath ?? toReplayLabelBenchmarkReportPath(labeledOutputPath);

  assertNoCanonicalPathCollision(
    'replay-label-benchmark-pipeline',
    'benchmarkReportPath must differ from labeledOutputPath.',
    { labeledOutputPath, benchmarkReportPath },
  );

  const benchmark = (params.runBenchmarkFromFile ?? runShortHorizonReplayBenchmarkFromFile)({
    bundlePath: labeledOutputPath,
    evaluator: params.evaluator,
    generatedAt: params.generatedAt,
  });
  const artifact: ReplayLabelBenchmarkArtifact = {
    formatVersion: 'replay-label-benchmark-report.v1',
    generatedAt: benchmark.generatedAt,
    labeledOutputPath,
    benchmarkReportPath,
    horizonCounts: toReplayLabelBenchmarkHorizonCounts(benchmark),
    benchmark,
  };

  writeReplayLabelBenchmarkArtifact(artifact, benchmarkReportPath);
  return artifact;
}

export async function runReplayLabelBenchmarkPipelineFromFile(params: {
  inputPath?: string;
  outputPath?: string;
  labelReportPath?: string;
  benchmarkReportPath?: string;
  loadHistory: ReplayTickerHistoryLoader;
  labeledAt?: string;
  benchmarkGeneratedAt?: string;
  evaluator?: ArbiterReplayEvaluator;
  runBenchmarkFromFile?: ShortHorizonBenchmarkRunner;
}): Promise<ReplayLabelBenchmarkPipelineResult> {
  const inputPath = params.inputPath ?? DEFAULT_ARBITER_REPLAY_BUNDLES_PATH;
  const outputPath = params.outputPath ?? DEFAULT_ARBITER_REPLAY_LABELED_PATH;
  const labelReportPath = params.labelReportPath ?? toReplayLabelBatchReportPath(outputPath);
  const benchmarkReportPath = params.benchmarkReportPath ?? toReplayLabelBenchmarkReportPath(outputPath);

  assertNoCanonicalPathCollision(
    'replay-label-benchmark-pipeline',
    'inputPath, outputPath, labelReportPath, and benchmarkReportPath must all differ.',
    { inputPath, outputPath, labelReportPath, benchmarkReportPath },
  );

  const labeling = await runReplayLabelBatchFromFile({
    inputPath,
    outputPath,
    reportPath: labelReportPath,
    loadHistory: params.loadHistory,
    labeledAt: params.labeledAt,
  });
  const benchmarkArtifact = runReplayBenchmarkHandoffFromLabeledFile({
    labeledOutputPath: outputPath,
    benchmarkReportPath,
    evaluator: params.evaluator,
    generatedAt: params.benchmarkGeneratedAt,
    runBenchmarkFromFile: params.runBenchmarkFromFile,
  });

  return {
    labeling,
    benchmarkArtifact,
  };
}
