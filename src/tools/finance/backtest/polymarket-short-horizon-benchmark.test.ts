import { afterEach, describe, expect, it } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import {
  appendArbiterReplayBundle,
  type ArbiterReplayBundle,
} from '../arbiter-replay.js';
import type {
  ForecastArbiterInput,
  ForecastArbiterResult,
} from '../forecast-arbitrator.js';
import {
  formatShortHorizonReplayBenchmarkReport,
  runShortHorizonReplayBenchmark,
  runShortHorizonReplayBenchmarkFromFile,
} from './polymarket-short-horizon-benchmark.js';

const repoTempDirs: string[] = [];

function makeRepoTempFile(name: string): string {
  const dir = join(
    process.cwd(),
    '.test-artifacts',
    `short-horizon-benchmark-${Date.now()}-${Math.random().toString(36).slice(2)}`,
  );
  mkdirSync(dir, { recursive: true });
  repoTempDirs.push(dir);
  return join(dir, name);
}

function makeBundle(params: {
  capturedAt: string;
  horizonDays: 1 | 2 | 3;
  actualBinary?: 0 | 1;
  crossPlatform?: {
    flagged?: boolean;
    applied?: boolean;
  };
}): ArbiterReplayBundle {
  return {
    capturedAt: params.capturedAt,
    ticker: 'BTC',
    horizonDays: params.horizonDays,
    currentPrice: 68_000,
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
          endDate: '2026-05-08T00:00:00.000Z',
          semantics: 'terminal',
          extractedPriceLevels: [70_000],
        },
      ],
      ...(params.crossPlatform
        ? {
          crossPlatformEvidence: [
            {
              source: 'metaforecast',
              kind: 'consensus',
              flagged: params.crossPlatform.flagged === true,
              deltaFromPolymarket: params.crossPlatform.flagged === true ? 0.16 : 0.03,
            },
          ],
          crossPlatformAdjustment: {
            basis: params.crossPlatform.applied === true ? 'metaforecast_divergence' : 'none',
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
                ? (71_000 - 68_000) / 68_000
                : (66_000 - 68_000) / 68_000,
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

afterEach(() => {
  while (repoTempDirs.length > 0) {
    rmSync(repoTempDirs.pop()!, { recursive: true, force: true });
  }
});

async function runBenchmarkCli(args: string[]): Promise<{
  exitCode: number;
  stdout: string;
  stderr: string;
}> {
  const proc = Bun.spawn({
    cmd: [
      process.execPath,
      'run',
      'src/tools/finance/backtest/polymarket-short-horizon-benchmark.ts',
      ...args,
    ],
    cwd: process.cwd(),
    stdout: 'pipe',
    stderr: 'pipe',
  });

  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);

  return { exitCode, stdout, stderr };
}

describe('polymarket short-horizon replay benchmark', () => {
  it('reports machine-readable 1d/2d/3d accuracy and Brier metrics by horizon', () => {
    const report = runShortHorizonReplayBenchmark({
      bundles: [
        makeBundle({
          capturedAt: '2026-05-01T00:00:00.000Z',
          horizonDays: 1,
          actualBinary: 1,
          crossPlatform: { flagged: true, applied: true },
        }),
        makeBundle({ capturedAt: '2026-05-02T00:00:00.000Z', horizonDays: 2, actualBinary: 0 }),
        makeBundle({ capturedAt: '2026-05-03T00:00:00.000Z', horizonDays: 3 }),
      ],
      evaluator,
      generatedAt: '2026-05-05T00:00:00.000Z',
      sourcePath: 'inline-fixture',
    });

    expect(report).toMatchObject({
      formatVersion: 'polymarket-short-horizon-benchmark.v1',
      generatedAt: '2026-05-05T00:00:00.000Z',
      sourcePath: 'inline-fixture',
      totalBundleCount: 3,
      shortHorizonBundleCount: 3,
      shortHorizonLabeledBundleCount: 2,
    });
    expect(report.horizons['1d']).toMatchObject({
      bundleCount: 1,
      labeledBundleCount: 1,
      tradedRowCount: 1,
      evaluatorName: 'stub-baseline',
      ready: true,
      directionalAccuracy: 1,
      crossPlatformEvidenceRowCount: 1,
      crossPlatformFlaggedRowCount: 1,
      crossPlatformAdjustmentAppliedRowCount: 1,
    });
    expect(report.horizons['1d'].brierScore).toBeCloseTo(0.1024, 6);
    expect(report.horizons['2d']).toMatchObject({
      bundleCount: 1,
      labeledBundleCount: 1,
      tradedRowCount: 1,
      ready: true,
      directionalAccuracy: 1,
      crossPlatformEvidenceRowCount: 0,
      crossPlatformFlaggedRowCount: 0,
      crossPlatformAdjustmentAppliedRowCount: 0,
    });
    expect(report.horizons['2d'].brierScore).toBeCloseTo(0.1024, 6);
    expect(report.horizons['3d'].ready).toBe(false);
    expect(report.horizons['3d'].pendingReasons).toContain(
      'No labeled 3d replay bundles were found; unlabeled bundles are not accuracy proof.',
    );

    expect(JSON.parse(formatShortHorizonReplayBenchmarkReport(report))).toEqual(report);
  });

  it('reads replay bundles from disk and keeps empty horizons machine-readable', () => {
    const bundlePath = makeRepoTempFile('arbiter-replay-bundles.jsonl');
    appendArbiterReplayBundle(
      makeBundle({ capturedAt: '2026-05-01T00:00:00.000Z', horizonDays: 1, actualBinary: 1 }),
      bundlePath,
    );
    appendArbiterReplayBundle(
      makeBundle({ capturedAt: '2026-05-02T00:00:00.000Z', horizonDays: 2 }),
      bundlePath,
    );

    const report = runShortHorizonReplayBenchmarkFromFile({
      bundlePath,
      evaluator,
      generatedAt: '2026-05-05T00:00:00.000Z',
    });

    expect(report.sourcePath).toBe(bundlePath);
    expect(report.shortHorizonBundleCount).toBe(2);
    expect(report.horizons['1d'].ready).toBe(true);
    expect(report.horizons['2d'].ready).toBe(false);
    expect(report.horizons['2d'].pendingReasons).toContain(
      'No labeled 2d replay bundles were found; unlabeled bundles are not accuracy proof.',
    );
    expect(report.horizons['3d'].bundleCount).toBe(0);
    expect(report.horizons['3d'].pendingReasons).toContain(
      'No 3d replay bundles were found in the benchmark source.',
    );
    expect(report.horizons['3d'].pendingReasons).not.toContain(
      'No labeled 3d replay bundles were found; unlabeled bundles are not accuracy proof.',
    );
  });

  it('rejects flag-like values for --bundle-path', async () => {
    const result = await runBenchmarkCli(['--bundle-path', '--help']);

    expect(result.exitCode).toBe(1);
    expect(result.stdout).toBe('');
    expect(result.stderr.trim()).toBe(
      'Invalid value for --bundle-path: expected a path, received flag --help',
    );
  });
});
