import { describe, expect, it } from 'bun:test';
import type { ReplayLabelBenchmarkArtifact } from './replay-label-benchmark-pipeline.js';
import {
  DEFAULT_REPLAY_LABEL_READINESS_THRESHOLDS,
  evaluateReplayLabelReadiness,
} from './replay-label-readiness.js';

function makeArtifact(
  overrides: Partial<ReplayLabelBenchmarkArtifact['benchmark']['horizons']> = {},
): ReplayLabelBenchmarkArtifact {
  return {
    formatVersion: 'replay-label-benchmark-report.v1',
    generatedAt: '2026-05-05T00:00:00.000Z',
    labeledOutputPath: '.cramer-short/arbiter-replay-bundles-labeled.jsonl',
    benchmarkReportPath: '.cramer-short/arbiter-replay-bundles-labeled.benchmark.report.json',
    horizonCounts: {
      '1d': { bundleCount: 0, labeledRowCount: 0, crossPlatformEvidenceRowCount: 0, crossPlatformFlaggedRowCount: 0, crossPlatformAdjustmentAppliedRowCount: 0 },
      '2d': { bundleCount: 0, labeledRowCount: 0, crossPlatformEvidenceRowCount: 0, crossPlatformFlaggedRowCount: 0, crossPlatformAdjustmentAppliedRowCount: 0 },
      '3d': { bundleCount: 0, labeledRowCount: 0, crossPlatformEvidenceRowCount: 0, crossPlatformFlaggedRowCount: 0, crossPlatformAdjustmentAppliedRowCount: 0 },
    },
    benchmark: {
      formatVersion: 'polymarket-short-horizon-benchmark.v1',
      generatedAt: '2026-05-05T00:00:00.000Z',
      sourcePath: '.cramer-short/arbiter-replay-bundles-labeled.jsonl',
      totalBundleCount: 0,
      shortHorizonBundleCount: 0,
      shortHorizonLabeledBundleCount: 0,
      horizons: {
        '1d': {
          horizonDays: 1,
          bundleCount: 0,
          labeledBundleCount: 0,
          unlabeledBundleCount: 0,
          tradedRowCount: 0,
          abstainRate: 1,
          directionalAccuracy: null,
          brierScore: null,
          crossPlatformEvidenceRowCount: 0,
          crossPlatformFlaggedRowCount: 0,
          crossPlatformAdjustmentAppliedRowCount: 0,
          evaluatorName: 'stub-baseline',
          ready: false,
          pendingReasons: ['No 1d replay bundles were found in the benchmark source.'],
        },
        '2d': {
          horizonDays: 2,
          bundleCount: 0,
          labeledBundleCount: 0,
          unlabeledBundleCount: 0,
          tradedRowCount: 0,
          abstainRate: 1,
          directionalAccuracy: null,
          brierScore: null,
          crossPlatformEvidenceRowCount: 0,
          crossPlatformFlaggedRowCount: 0,
          crossPlatformAdjustmentAppliedRowCount: 0,
          evaluatorName: 'stub-baseline',
          ready: false,
          pendingReasons: ['No 2d replay bundles were found in the benchmark source.'],
        },
        '3d': {
          horizonDays: 3,
          bundleCount: 0,
          labeledBundleCount: 0,
          unlabeledBundleCount: 0,
          tradedRowCount: 0,
          abstainRate: 1,
          directionalAccuracy: null,
          brierScore: null,
          crossPlatformEvidenceRowCount: 0,
          crossPlatformFlaggedRowCount: 0,
          crossPlatformAdjustmentAppliedRowCount: 0,
          evaluatorName: 'stub-baseline',
          ready: false,
          pendingReasons: ['No 3d replay bundles were found in the benchmark source.'],
        },
        ...overrides,
      },
    },
  };
}

describe('evaluateReplayLabelReadiness', () => {
  it('defaults to hold when evidence is thin or empty', () => {
    const artifact = makeArtifact();

    const report = evaluateReplayLabelReadiness(artifact);

    expect(report.decision).toBe('hold');
    expect(report.reasons.length).toBeGreaterThan(0);
    expect(report.horizons['1d']?.status).toBe('hold');
    expect(report.horizons['2d']?.status).toBe('hold');
    expect(report.horizons['3d']?.status).toBe('hold');
    expect(report.thresholds).toEqual(DEFAULT_REPLAY_LABEL_READINESS_THRESHOLDS);
  });

  it('marks sufficiently populated benchmark artifacts as eligible', () => {
    const artifact = makeArtifact({
      '1d': {
        horizonDays: 1,
        bundleCount: 42,
        labeledBundleCount: 42,
        unlabeledBundleCount: 0,
        tradedRowCount: 26,
        abstainRate: 0.38,
        directionalAccuracy: 0.61,
        brierScore: 0.22,
        crossPlatformEvidenceRowCount: 8,
        crossPlatformFlaggedRowCount: 2,
        crossPlatformAdjustmentAppliedRowCount: 2,
        evaluatorName: 'stub-baseline',
        ready: true,
        pendingReasons: [],
      },
      '2d': {
        horizonDays: 2,
        bundleCount: 35,
        labeledBundleCount: 35,
        unlabeledBundleCount: 0,
        tradedRowCount: 22,
        abstainRate: 0.36,
        directionalAccuracy: 0.58,
        brierScore: 0.23,
        crossPlatformEvidenceRowCount: 7,
        crossPlatformFlaggedRowCount: 1,
        crossPlatformAdjustmentAppliedRowCount: 1,
        evaluatorName: 'stub-baseline',
        ready: true,
        pendingReasons: [],
      },
      '3d': {
        horizonDays: 3,
        bundleCount: 30,
        labeledBundleCount: 30,
        unlabeledBundleCount: 0,
        tradedRowCount: 20,
        abstainRate: 0.34,
        directionalAccuracy: 0.57,
        brierScore: 0.24,
        crossPlatformEvidenceRowCount: 6,
        crossPlatformFlaggedRowCount: 1,
        crossPlatformAdjustmentAppliedRowCount: 1,
        evaluatorName: 'stub-baseline',
        ready: true,
        pendingReasons: [],
      },
    });

    const report = evaluateReplayLabelReadiness(artifact);

    expect(report.decision).toBe('eligible');
    expect(report.reasons).toEqual([]);
    expect(report.horizons['1d']?.status).toBe('pass');
    expect(report.horizons['2d']?.status).toBe('pass');
    expect(report.horizons['3d']?.status).toBe('pass');
  });

  it('does not mutate the source benchmark artifact', () => {
    const artifact = makeArtifact({
      '1d': {
        horizonDays: 1,
        bundleCount: 42,
        labeledBundleCount: 42,
        unlabeledBundleCount: 0,
        tradedRowCount: 26,
        abstainRate: 0.38,
        directionalAccuracy: 0.61,
        brierScore: 0.22,
        crossPlatformEvidenceRowCount: 8,
        crossPlatformFlaggedRowCount: 2,
        crossPlatformAdjustmentAppliedRowCount: 2,
        evaluatorName: 'stub-baseline',
        ready: true,
        pendingReasons: [],
      },
      '2d': {
        horizonDays: 2,
        bundleCount: 35,
        labeledBundleCount: 35,
        unlabeledBundleCount: 0,
        tradedRowCount: 22,
        abstainRate: 0.36,
        directionalAccuracy: 0.58,
        brierScore: 0.23,
        crossPlatformEvidenceRowCount: 7,
        crossPlatformFlaggedRowCount: 1,
        crossPlatformAdjustmentAppliedRowCount: 1,
        evaluatorName: 'stub-baseline',
        ready: true,
        pendingReasons: [],
      },
      '3d': {
        horizonDays: 3,
        bundleCount: 30,
        labeledBundleCount: 30,
        unlabeledBundleCount: 0,
        tradedRowCount: 20,
        abstainRate: 0.34,
        directionalAccuracy: 0.57,
        brierScore: 0.24,
        crossPlatformEvidenceRowCount: 6,
        crossPlatformFlaggedRowCount: 1,
        crossPlatformAdjustmentAppliedRowCount: 1,
        evaluatorName: 'stub-baseline',
        ready: true,
        pendingReasons: [],
      },
    });
    const before = JSON.parse(JSON.stringify(artifact));

    evaluateReplayLabelReadiness(artifact);

    expect(artifact).toEqual(before);
  });
});
