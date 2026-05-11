import { afterAll, afterEach, describe, expect, it } from 'bun:test';
import { randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ArbiterReplayBundle } from '../arbiter-replay.js';
import type { ReplayPriceHistory } from '../arbiter-replay-labeler.js';
import type { ForecastArbiterInput, ForecastArbiterResult } from '../forecast-arbitrator.js';
import type { ShortHorizonReplayBenchmarkReport } from './polymarket-short-horizon-benchmark.js';
import {
  runReplayBenchmarkHandoffFromLabeledFile,
  runReplayLabelBenchmarkPipelineFromFile,
} from './replay-label-benchmark-pipeline.js';

const SCRATCH_ROOT = join(import.meta.dir, '__test-scratch__');
const scratchDirs: string[] = [];

function makeScratchDir(): string {
  const dir = join(SCRATCH_ROOT, randomUUID());
  mkdirSync(dir, { recursive: true });
  scratchDirs.push(dir);
  return dir;
}

afterEach(() => {
  for (const dir of scratchDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

afterAll(() => {
  rmSync(SCRATCH_ROOT, { recursive: true, force: true });
});

function makeBundle(params: {
  capturedAt?: string;
  horizonDays: 1 | 2 | 3;
  currentPrice?: number;
  actualBinary?: 0 | 1;
  crossPlatform?: {
    flagged?: boolean;
    applied?: boolean;
  };
}): ArbiterReplayBundle {
  const capturedAt = params.capturedAt ?? '2026-05-01T00:00:00.000Z';
  const currentPrice = params.currentPrice ?? 68_000;
  const endDate = new Date(Date.parse(capturedAt) + params.horizonDays * 86_400_000).toISOString();
  return {
    capturedAt,
    ticker: 'BTC',
    horizonDays: params.horizonDays,
    currentPrice,
    polymarket: {
      querySet: ['bitcoin price'],
      selectedMarketIds: [`btc-${params.horizonDays}d`],
      selectedMarkets: [
        {
          marketId: `btc-${params.horizonDays}d`,
          assetId: `btc-${params.horizonDays}d-yes`,
          question: `Will Bitcoin be above $70,000 in ${params.horizonDays} day${params.horizonDays === 1 ? '' : 's'}?`,
          probability: params.actualBinary === 0 ? 0.38 : 0.62,
          volume24h: 250_000,
          endDate,
          semantics: 'terminal',
          extractedPriceLevels: [70_000],
        },
      ],
      ...(params.crossPlatform
        ? {
            crossPlatformEvidence: [
              {
                source: 'metaforecast' as const,
                kind: 'consensus' as const,
                flagged: params.crossPlatform.flagged === true,
                deltaFromPolymarket: params.crossPlatform.flagged === true ? 0.16 : 0.03,
              },
            ],
            crossPlatformAdjustment: {
              basis: params.crossPlatform.applied === true ? 'metaforecast_divergence' as const : 'none' as const,
              applied: params.crossPlatform.applied === true,
              qualityScoreDelta: params.crossPlatform.applied === true ? -8 : 0,
              sigmaMultiplier: params.crossPlatform.applied === true ? 1.08 : 1,
            },
          }
        : {}),
      warnings: [],
    },
    warnings: [],
    ...(params.actualBinary !== undefined
      ? {
          labels: {
            forecast: {
              realizedPrice: params.actualBinary === 1 ? 71_000 : 66_000,
              realizedReturn: params.actualBinary === 1
                ? (71_000 - currentPrice) / currentPrice
                : (66_000 - currentPrice) / currentPrice,
              actualBinary: params.actualBinary,
              labeledAt: '2026-05-08T12:00:00.000Z',
            },
            semantic: [
              {
                marketId: `btc-${params.horizonDays}d`,
                semantics: 'terminal',
                outcome: params.actualBinary === 1 ? 'yes' : 'no',
                labeledAt: '2026-05-08T12:00:00.000Z',
              },
            ],
          },
        }
      : {}),
  };
}

const BTC_READY_HISTORY: ReplayPriceHistory = {
  points: [
    { at: '2026-05-01T00:00:00.000Z', price: 68_000 },
    { at: '2026-05-02T00:00:00.000Z', price: 71_000 },
    { at: '2026-05-03T00:00:00.000Z', price: 66_000 },
    { at: '2026-05-04T00:00:00.000Z', price: 72_000 },
  ],
};

function makeResult(
  horizonDays: number,
  preferredDirection: ForecastArbiterResult['preferredDirection'],
): ForecastArbiterResult {
  return {
    ticker: 'BTC',
    horizonDays,
    currentPrice: 68_000,
    leverage: 1,
    verdict: preferredDirection === 'long' ? 'LONG' : preferredDirection === 'short' ? 'SHORT' : 'NO_TRADE',
    preferredDirection,
    confidence: 'medium',
    shouldEnterNow: preferredDirection !== 'neutral',
    semanticSummary: {
      primaryPolymarketSemantics: 'terminal',
      counts: { terminal: 1, barrier_touch: 0, range: 0, path_dependent: 0, ambiguous: 0, unknown: 0 },
      barrierPrices: [],
      reconciliation: 'test',
    },
    disagreement: {
      markovDirection: 'long',
      polymarketDirection: 'long',
      whaleDirection: 'neutral',
      isDivergent: false,
      summary: 'test disagreement',
    },
    leverageAssessment: {
      long: { directionalEdgePct: 0.01, riskAdjustedScore: 0.01, leveragePnlPct: 0.01, rr: 1.5, notes: [] },
      short: { directionalEdgePct: -0.01, riskAdjustedScore: -0.01, leveragePnlPct: -0.01, rr: 1.5, notes: [] },
      warning: null,
    },
    conditionalPlan: {
      longTrigger: null,
      shortTrigger: null,
      invalidation: null,
    },
    policy: {
      level: 'full',
      horizonEligible: true,
      tradeEligible: preferredDirection !== 'neutral',
      reasons: [],
    },
    rationale: [],
    rawEvidence: {
      markov: null,
      polymarket: null,
      whale: null,
    },
  };
}

const evaluator = {
  name: 'stub-baseline',
  evaluate(input: ForecastArbiterInput): ForecastArbiterResult {
    if (input.horizon_days === 1) return makeResult(input.horizon_days, 'long');
    if (input.horizon_days === 2) return makeResult(input.horizon_days, 'short');
    return makeResult(input.horizon_days, 'neutral');
  },
};

function makeBenchmarkReport(sourcePath: string): ShortHorizonReplayBenchmarkReport {
  return {
    formatVersion: 'polymarket-short-horizon-benchmark.v1',
    generatedAt: '2026-05-05T00:00:00.000Z',
    sourcePath,
    totalBundleCount: 1,
    shortHorizonBundleCount: 1,
    shortHorizonLabeledBundleCount: 1,
    horizons: {
      '1d': {
        horizonDays: 1,
        bundleCount: 1,
        labeledBundleCount: 1,
        unlabeledBundleCount: 0,
        tradedRowCount: 1,
        abstainRate: 0,
        directionalAccuracy: 1,
        brierScore: 0.1024,
        crossPlatformEvidenceRowCount: 0,
        crossPlatformFlaggedRowCount: 0,
        crossPlatformAdjustmentAppliedRowCount: 0,
        evaluatorName: 'stub-baseline',
        ready: true,
        pendingReasons: [],
      },
      '2d': {
        horizonDays: 2,
        bundleCount: 0,
        labeledBundleCount: 0,
        unlabeledBundleCount: 0,
        tradedRowCount: 0,
        abstainRate: 0,
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
        abstainRate: 0,
        directionalAccuracy: null,
        brierScore: null,
        crossPlatformEvidenceRowCount: 0,
        crossPlatformFlaggedRowCount: 0,
        crossPlatformAdjustmentAppliedRowCount: 0,
        evaluatorName: 'stub-baseline',
        ready: false,
        pendingReasons: ['No 3d replay bundles were found in the benchmark source.'],
      },
    },
  };
}

describe('replay label benchmark pipeline', () => {
  it('hands the staged labeled output path to the short-horizon benchmark handoff', async () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'input.jsonl');
    const outputPath = join(dir, 'staged-labeled.jsonl');
    let seenBundlePath: string | undefined;

    writeFileSync(inputPath, `${JSON.stringify(makeBundle({ horizonDays: 1 }))}\n`, 'utf-8');

    const result = await runReplayLabelBenchmarkPipelineFromFile({
      inputPath,
      outputPath,
      loadHistory: async () => BTC_READY_HISTORY,
      labeledAt: '2026-05-03T00:00:00.000Z',
      benchmarkGeneratedAt: '2026-05-05T00:00:00.000Z',
      runBenchmarkFromFile: (params) => {
        const bundlePath = params?.bundlePath;
        seenBundlePath = bundlePath;
        return makeBenchmarkReport(bundlePath!);
      },
    });

    expect(seenBundlePath).toBe(outputPath);
    expect(result.labeling.summary.newlyLabeled).toBe(1);
    expect(result.benchmarkArtifact.benchmark.sourcePath).toBe(outputPath);
  });

  it('throws when benchmarkReportPath collides with another pipeline artifact path', async () => {
    const cases = [
      {
        name: 'inputPath',
        makePaths: (dir: string) => {
          const inputPath = join(dir, 'input.jsonl');
          const outputPath = join(dir, 'staged-labeled.jsonl');
          const labelReportPath = join(dir, 'staged-labeled.report.json');
          const benchmarkReportPath = `${dir}/./input.jsonl`;
          return { inputPath, outputPath, labelReportPath, benchmarkReportPath };
        },
      },
      {
        name: 'outputPath',
        makePaths: (dir: string) => {
          const inputPath = join(dir, 'input.jsonl');
          const outputPath = join(dir, 'staged-labeled.jsonl');
          const labelReportPath = join(dir, 'staged-labeled.report.json');
          const benchmarkReportPath = `${dir}/./staged-labeled.jsonl`;
          return { inputPath, outputPath, labelReportPath, benchmarkReportPath };
        },
      },
      {
        name: 'labelReportPath',
        makePaths: (dir: string) => {
          const inputPath = join(dir, 'input.jsonl');
          const outputPath = join(dir, 'staged-labeled.jsonl');
          const realDir = join(dir, 'real');
          const aliasedDir = join(dir, 'alias');
          mkdirSync(realDir, { recursive: true });
          symlinkSync(realDir, aliasedDir, 'dir');
          const labelReportPath = join(realDir, 'staged-labeled.report.json');
          const benchmarkReportPath = join(aliasedDir, 'staged-labeled.report.json');
          return { inputPath, outputPath, labelReportPath, benchmarkReportPath };
        },
      },
    ] as const;

    for (const testCase of cases) {
      const dir = makeScratchDir();
      const { inputPath, outputPath, labelReportPath, benchmarkReportPath } = testCase.makePaths(dir);
      writeFileSync(inputPath, `${JSON.stringify(makeBundle({ horizonDays: 1 }))}\n`, 'utf-8');

      await expect(
        runReplayLabelBenchmarkPipelineFromFile({
          inputPath,
          outputPath,
          labelReportPath,
          benchmarkReportPath,
          loadHistory: async () => BTC_READY_HISTORY,
          labeledAt: '2026-05-03T00:00:00.000Z',
          benchmarkGeneratedAt: '2026-05-05T00:00:00.000Z',
        }),
        `expected benchmarkReportPath collision with ${testCase.name}`,
      ).rejects.toThrow();
    }
  });

  it('writes machine-readable horizon counts for labeled rows and cross-platform slices', () => {
    const dir = makeScratchDir();
    const labeledOutputPath = join(dir, 'staged-labeled.jsonl');
    const benchmarkReportPath = join(dir, 'staged-labeled.benchmark.report.json');

    writeFileSync(
      labeledOutputPath,
      [
        JSON.stringify(makeBundle({ horizonDays: 1, actualBinary: 1, crossPlatform: { flagged: true, applied: true } })),
        JSON.stringify(makeBundle({ capturedAt: '2026-05-02T00:00:00.000Z', horizonDays: 2, actualBinary: 0 })),
      ].join('\n') + '\n',
      'utf-8',
    );

    const artifact = runReplayBenchmarkHandoffFromLabeledFile({
      labeledOutputPath,
      benchmarkReportPath,
      evaluator,
      generatedAt: '2026-05-05T00:00:00.000Z',
    });

    expect(artifact.horizonCounts['1d']).toEqual({
      bundleCount: 1,
      labeledRowCount: 1,
      crossPlatformEvidenceRowCount: 1,
      crossPlatformFlaggedRowCount: 1,
      crossPlatformAdjustmentAppliedRowCount: 1,
    });
    expect(artifact.horizonCounts['2d']).toEqual({
      bundleCount: 1,
      labeledRowCount: 1,
      crossPlatformEvidenceRowCount: 0,
      crossPlatformFlaggedRowCount: 0,
      crossPlatformAdjustmentAppliedRowCount: 0,
    });
    expect(JSON.parse(readFileSync(benchmarkReportPath, 'utf-8'))).toEqual(artifact);
  });

  it('writes separate label and benchmark artifacts without mutating the staged labeled JSONL', async () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'input.jsonl');
    const outputPath = join(dir, 'staged-labeled.jsonl');
    const labelReportPath = join(dir, 'staged-labeled.report.json');
    const benchmarkReportPath = join(dir, 'staged-labeled.benchmark.report.json');

    writeFileSync(
      inputPath,
      [
        JSON.stringify(makeBundle({ horizonDays: 1, crossPlatform: { flagged: true, applied: true } })),
        JSON.stringify(makeBundle({ capturedAt: '2026-05-02T00:00:00.000Z', horizonDays: 2 })),
      ].join('\n') + '\n',
      'utf-8',
    );

    await runReplayLabelBenchmarkPipelineFromFile({
      inputPath,
      outputPath,
      labelReportPath,
      benchmarkReportPath,
      loadHistory: async () => BTC_READY_HISTORY,
      labeledAt: '2026-05-03T00:00:00.000Z',
      benchmarkGeneratedAt: '2026-05-05T00:00:00.000Z',
      evaluator,
    });

    const stagedJsonl = readFileSync(outputPath, 'utf-8');
    expect(readFileSync(labelReportPath, 'utf-8')).toContain('replay-label-batch-report.v1');
    expect(readFileSync(benchmarkReportPath, 'utf-8')).toContain('replay-label-benchmark-report.v1');

    const stagedRows = stagedJsonl.trim().split('\n').map((line) => JSON.parse(line));
    expect(stagedRows).toHaveLength(2);
    expect(stagedRows[0]?.labels?.forecast?.labeledAt).toBe('2026-05-03T00:00:00.000Z');
    expect(stagedRows[0]?.benchmark).toBeUndefined();
    expect(stagedRows[1]?.labels?.forecast?.labeledAt).toBe('2026-05-03T00:00:00.000Z');

    runReplayBenchmarkHandoffFromLabeledFile({
      labeledOutputPath: outputPath,
      benchmarkReportPath: join(dir, 'staged-labeled.benchmark.report-2.json'),
      evaluator,
      generatedAt: '2026-05-06T00:00:00.000Z',
    });
    expect(readFileSync(outputPath, 'utf-8')).toBe(stagedJsonl);
  });
});
