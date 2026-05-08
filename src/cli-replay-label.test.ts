import { afterAll, afterEach, describe, expect, it } from 'bun:test';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { createReplayHistoryLoader, runReplayLabelCommand } from './cli-replay-label.js';
import type {
  ReplayLabelCommandOptions,
  ReplayLabelReadinessSummaryOutput,
  ReplayLabelRunSummaryOutput,
} from './cli-replay-label.js';
import type { ReplayLabelBenchmarkPipelineResult } from './tools/finance/backtest/replay-label-benchmark-pipeline.js';
import type { ReplayTickerHistoryLoader } from './tools/finance/backtest/replay-label-batch-runner.js';
import { DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH } from './tools/finance/arbiter-replay.js';
import { DEFAULT_ARBITER_REPLAY_LABELED_PATH } from './tools/finance/backtest/replay-label-runner.js';
import {
  DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH,
  type ReplayLabelPromotionResult,
} from './tools/finance/backtest/replay-label-promotion.js';
import { toReplayLabelBatchReportPath } from './tools/finance/backtest/replay-label-batch-runner.js';
import { toReplayLabelBenchmarkReportPath } from './tools/finance/backtest/replay-label-benchmark-pipeline.js';
import type { ReplayFixturePriceStore } from './tools/finance/backtest/replay-price-history-adapter.js';
import type { ReplayLabelReadinessReport } from './tools/finance/backtest/replay-label-readiness.js';

const SCRATCH_ROOT = join('.cramer-short', 'cli-replay-label-test');
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

// ─── stub helpers ────────────────────────────────────────────────────────────

function makeStubPipelineResult(): ReplayLabelBenchmarkPipelineResult {
  return {
    labeling: {
      summary: {
        total: 5,
        alreadyLabeled: 2,
        newlyLabeled: 3,
        skippedByMissingHistory: 0,
        pending: 0,
        pendingReasons: {},
        perTickerCounts: {},
      },
      labeledAt: '2026-05-01T00:00:00.000Z',
      bundles: [],
    },
    benchmarkArtifact: {
      formatVersion: 'replay-label-benchmark-report.v1',
      generatedAt: '2026-05-01T00:00:00.000Z',
      labeledOutputPath: '.cramer-short/arbiter-replay-bundles-labeled.jsonl',
      benchmarkReportPath: '.cramer-short/arbiter-replay-bundles-labeled.benchmark.report.json',
      horizonCounts: {
        '1d': { bundleCount: 5, labeledRowCount: 5, crossPlatformEvidenceRowCount: 0, crossPlatformFlaggedRowCount: 0, crossPlatformAdjustmentAppliedRowCount: 0 },
        '2d': { bundleCount: 0, labeledRowCount: 0, crossPlatformEvidenceRowCount: 0, crossPlatformFlaggedRowCount: 0, crossPlatformAdjustmentAppliedRowCount: 0 },
        '3d': { bundleCount: 0, labeledRowCount: 0, crossPlatformEvidenceRowCount: 0, crossPlatformFlaggedRowCount: 0, crossPlatformAdjustmentAppliedRowCount: 0 },
      },
      benchmark: {} as ReplayLabelBenchmarkPipelineResult['benchmarkArtifact']['benchmark'],
    },
  };
}

interface Captured {
  inputPath?: string;
  outputPath?: string;
  labelReportPath?: string;
  benchmarkReportPath?: string;
  loadHistory?: ReplayTickerHistoryLoader;
  stagedLabeledPath?: string;
  stagedLabelReportPath?: string;
  stagedBenchmarkReportPath?: string;
  promotedLabeledPath?: string;
  receiptPath?: string;
  readinessInputPath?: string;
}

function makeOptions(overrides: Partial<ReplayLabelCommandOptions> = {}): {
  options: ReplayLabelCommandOptions;
  logs: string[];
  errors: string[];
  exitCodes: number[];
  captured: Captured;
} {
  const logs: string[] = [];
  const errors: string[] = [];
  const exitCodes: number[] = [];
  const captured: Captured = {};

  const options: ReplayLabelCommandOptions = {
    log: (m) => logs.push(m),
    error: (m) => errors.push(m),
    exit: (code) => exitCodes.push(code),
    loaderFactory: () => async () => null,
    runPipeline: async (params) => {
      captured.inputPath = params.inputPath;
      captured.outputPath = params.outputPath;
      captured.labelReportPath = params.labelReportPath;
      captured.benchmarkReportPath = params.benchmarkReportPath;
      captured.loadHistory = params.loadHistory;
      return makeStubPipelineResult();
    },
    promoteArtifacts: (params) => {
      captured.stagedLabeledPath = params.stagedLabeledPath;
      captured.stagedLabelReportPath = params.stagedLabelReportPath;
      captured.stagedBenchmarkReportPath = params.stagedBenchmarkReportPath;
      captured.promotedLabeledPath = params.promotedLabeledPath;
      captured.receiptPath = params.receiptPath;
      return {
        receipt: {
          formatVersion: 'replay-label-promotion-receipt.v1',
          promotedAt: '2026-05-06T00:00:00.000Z',
          receiptPath: params.receiptPath ?? 'receipt.json',
          source: {
            stagedLabeledPath: params.stagedLabeledPath ?? DEFAULT_ARBITER_REPLAY_LABELED_PATH,
            stagedLabelReportPath: params.stagedLabelReportPath ?? toReplayLabelBatchReportPath(DEFAULT_ARBITER_REPLAY_LABELED_PATH),
            stagedBenchmarkReportPath: params.stagedBenchmarkReportPath ?? toReplayLabelBenchmarkReportPath(DEFAULT_ARBITER_REPLAY_LABELED_PATH),
          },
          target: {
            promotedLabeledPath: params.promotedLabeledPath ?? DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH,
            promotedLabelReportPath: toReplayLabelBatchReportPath(
              params.promotedLabeledPath ?? DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH,
            ),
            promotedBenchmarkReportPath: toReplayLabelBenchmarkReportPath(
              params.promotedLabeledPath ?? DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH,
            ),
          },
          bundleCount: 5,
          labeledBundleCount: 5,
        },
        bundles: [],
      } satisfies ReplayLabelPromotionResult;
    },
    runReadiness: (params) => {
      captured.readinessInputPath = params.inputPath;
      return {
        formatVersion: 'replay-label-readiness-report.v1',
        generatedAt: '2026-05-07T00:00:00.000Z',
        sourceType: 'artifact',
        sourcePath: params.inputPath ?? '.cramer-short/arbiter-replay-bundles-labeled.benchmark.report.json',
        decision: 'hold',
        reasons: ['Missing benchmark evidence.'],
        thresholds: {
          requireReadySlice: true,
          minLabeledBundleCountByHorizon: { '1d': 30, '2d': 25, '3d': 20 },
          minTradedRowCountByHorizon: { '1d': 15, '2d': 12, '3d': 10 },
          maxAbstainRateByHorizon: { '1d': 0.45, '2d': 0.5, '3d': 0.55 },
          maxBrierScoreByHorizon: { '1d': 0.24, '2d': 0.25, '3d': 0.26 },
        },
        horizons: {
          '1d': {
            status: 'hold',
            reasons: ['Missing benchmark evidence.'],
            thresholds: {
              minLabeledBundleCount: 30,
              minTradedRowCount: 15,
              maxAbstainRate: 0.45,
              maxBrierScore: 0.24,
              requireReadySlice: true,
            },
            metrics: {
              horizonDays: 1,
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
              pendingReasons: ['Missing benchmark evidence.'],
            },
          },
          '2d': {
            status: 'hold',
            reasons: ['Missing benchmark evidence.'],
            thresholds: {
              minLabeledBundleCount: 25,
              minTradedRowCount: 12,
              maxAbstainRate: 0.5,
              maxBrierScore: 0.25,
              requireReadySlice: true,
            },
            metrics: {
              horizonDays: 2,
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
              pendingReasons: ['Missing benchmark evidence.'],
            },
          },
          '3d': {
            status: 'hold',
            reasons: ['Missing benchmark evidence.'],
            thresholds: {
              minLabeledBundleCount: 20,
              minTradedRowCount: 10,
              maxAbstainRate: 0.55,
              maxBrierScore: 0.26,
              requireReadySlice: true,
            },
            metrics: {
              horizonDays: 3,
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
              pendingReasons: ['Missing benchmark evidence.'],
            },
          },
        },
      } satisfies ReplayLabelReadinessReport;
    },
    ...overrides,
  };

  return { options, logs, errors, exitCodes, captured };
}

// ─── subcommand routing ───────────────────────────────────────────────────────

describe('replay-label subcommand routing', () => {
  it('routes "run" to the run handler without errors', async () => {
    const { options, errors, exitCodes } = makeOptions();
    await runReplayLabelCommand(['run'], options);
    expect(errors).toEqual([]);
    expect(exitCodes).toEqual([]);
  });

  it('routes "promote" to the promotion handler without errors', async () => {
    const { options, errors, exitCodes } = makeOptions();
    await runReplayLabelCommand(['promote'], options);
    expect(errors).toEqual([]);
    expect(exitCodes).toEqual([]);
  });

  it('routes "readiness" to the readiness handler without errors', async () => {
    const { options, errors, exitCodes } = makeOptions();
    await runReplayLabelCommand(['readiness'], options);
    expect(errors).toEqual([]);
    expect(exitCodes).toEqual([]);
  });

  it('prints usage for empty argv', async () => {
    const { options, logs } = makeOptions();
    await runReplayLabelCommand([], options);
    const joined = logs.join('\n');
    expect(joined).toContain('replay-label run');
  });

  it('prints usage for "help"', async () => {
    const { options, logs } = makeOptions();
    await runReplayLabelCommand(['help'], options);
    const joined = logs.join('\n');
    expect(joined).toContain('replay-label run');
  });

  it('prints usage for "--help"', async () => {
    const { options, logs } = makeOptions();
    await runReplayLabelCommand(['--help'], options);
    const joined = logs.join('\n');
    expect(joined).toContain('replay-label run');
  });

  it('exits(1) for unknown subcommand', async () => {
    const { options, errors, exitCodes } = makeOptions();
    await runReplayLabelCommand(['unknown-subcmd'], options);
    expect(exitCodes).toContain(1);
    expect(errors.some((e) => e.includes('unknown-subcmd'))).toBe(true);
  });
});

// ─── default artifact paths ───────────────────────────────────────────────────

describe('replay-label run default artifact paths', () => {
  it('passes DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH as inputPath by default', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['run'], options);
    expect(captured.inputPath).toBe(DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH);
  });

  it('passes DEFAULT_ARBITER_REPLAY_LABELED_PATH as outputPath by default', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['run'], options);
    expect(captured.outputPath).toBe(DEFAULT_ARBITER_REPLAY_LABELED_PATH);
  });

  it('derives labelReportPath from outputPath by default', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['run'], options);
    expect(captured.labelReportPath).toBe(toReplayLabelBatchReportPath(DEFAULT_ARBITER_REPLAY_LABELED_PATH));
  });

  it('derives benchmarkReportPath from outputPath by default', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['run'], options);
    expect(captured.benchmarkReportPath).toBe(toReplayLabelBenchmarkReportPath(DEFAULT_ARBITER_REPLAY_LABELED_PATH));
  });
});

describe('replay-label promote default artifact paths', () => {
  it('passes DEFAULT_ARBITER_REPLAY_LABELED_PATH as the staged labeled path by default', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['promote'], options);
    expect(captured.stagedLabeledPath).toBe(DEFAULT_ARBITER_REPLAY_LABELED_PATH);
  });

  it('passes DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH as promotedLabeledPath by default', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['promote'], options);
    expect(captured.promotedLabeledPath).toBe(DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH);
  });
});

describe('replay-label readiness default artifact paths', () => {
  it('passes the staged benchmark artifact path as inputPath by default', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['readiness'], options);
    expect(captured.readinessInputPath).toBe(toReplayLabelBenchmarkReportPath(DEFAULT_ARBITER_REPLAY_LABELED_PATH));
  });
});

// ─── custom artifact paths ────────────────────────────────────────────────────

describe('replay-label run custom artifact paths', () => {
  it('passes --input path through to pipeline', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['run', '--input', 'custom-in.jsonl'], options);
    expect(captured.inputPath).toBe('custom-in.jsonl');
  });

  it('passes --output path through and derives reports from it', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['run', '--output', 'custom-out.jsonl'], options);
    expect(captured.outputPath).toBe('custom-out.jsonl');
    expect(captured.labelReportPath).toBe(toReplayLabelBatchReportPath('custom-out.jsonl'));
    expect(captured.benchmarkReportPath).toBe(toReplayLabelBenchmarkReportPath('custom-out.jsonl'));
  });

  it('passes explicit --label-report path through', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['run', '--output', 'out.jsonl', '--label-report', 'my-label.report.json'], options);
    expect(captured.labelReportPath).toBe('my-label.report.json');
  });

  it('passes explicit --benchmark-report path through', async () => {
    const { options, captured } = makeOptions();
    await runReplayLabelCommand(['run', '--output', 'out.jsonl', '--benchmark-report', 'my-bench.json'], options);
    expect(captured.benchmarkReportPath).toBe('my-bench.json');
  });
});

// ─── machine-readable summary output ─────────────────────────────────────────

describe('replay-label run machine-readable output', () => {
  it('prints JSON summary with status:ok on success', async () => {
    const { options, logs } = makeOptions();
    await runReplayLabelCommand(['run'], options);
    const jsonLog = logs.find((l) => {
      try {
        JSON.parse(l);
        return true;
      } catch {
        return false;
      }
    });
    expect(jsonLog).toBeDefined();
    const parsed = JSON.parse(jsonLog!) as ReplayLabelRunSummaryOutput;
    expect(parsed.status).toBe('ok');
  });

  it('includes artifact paths in the JSON summary', async () => {
    const { options, logs } = makeOptions();
    await runReplayLabelCommand(['run', '--input', 'in.jsonl', '--output', 'out.jsonl'], options);
    const jsonLog = logs.find((l) => { try { JSON.parse(l); return true; } catch { return false; } });
    const parsed = JSON.parse(jsonLog!) as ReplayLabelRunSummaryOutput;
    expect(parsed.inputPath).toBe('in.jsonl');
    expect(parsed.outputPath).toBe('out.jsonl');
    expect(typeof parsed.labelReportPath).toBe('string');
    expect(typeof parsed.benchmarkReportPath).toBe('string');
  });

  it('includes labelingSummary counts in the JSON output', async () => {
    const { options, logs } = makeOptions();
    await runReplayLabelCommand(['run'], options);
    const jsonLog = logs.find((l) => { try { JSON.parse(l); return true; } catch { return false; } });
    const parsed = JSON.parse(jsonLog!) as ReplayLabelRunSummaryOutput;
    expect(typeof parsed.labelingSummary.total).toBe('number');
    expect(typeof parsed.labelingSummary.newlyLabeled).toBe('number');
    expect(typeof parsed.labelingSummary.alreadyLabeled).toBe('number');
    expect(typeof parsed.labelingSummary.skippedByMissingHistory).toBe('number');
    expect(typeof parsed.labelingSummary.pending).toBe('number');
  });

  it('exits(1) and emits error when pipeline throws', async () => {
    const { options, errors, exitCodes } = makeOptions({
      runPipeline: async () => { throw new Error('pipeline exploded'); },
    });
    await runReplayLabelCommand(['run'], options);
    expect(exitCodes).toContain(1);
    expect(errors.some((e) => e.includes('pipeline exploded'))).toBe(true);
  });
});

describe('replay-label readiness machine-readable output', () => {
  it('prints JSON summary with a readiness report payload', async () => {
    const { options, logs } = makeOptions();
    await runReplayLabelCommand(['readiness', '--input', 'custom-benchmark.json'], options);
    const jsonLog = logs.find((l) => { try { JSON.parse(l); return true; } catch { return false; } });
    expect(jsonLog).toBeDefined();
    const parsed = JSON.parse(jsonLog!) as ReplayLabelReadinessSummaryOutput;
    expect(parsed.status).toBe('ok');
    expect(parsed.inputPath).toBe('custom-benchmark.json');
    expect(parsed.report.decision).toBe('hold');
    expect(parsed.report.thresholds.minLabeledBundleCountByHorizon['1d']).toBeGreaterThan(0);
  });
});

// ─── flag parsing errors ──────────────────────────────────────────────────────

describe('replay-label run flag parsing errors', () => {
  it('exits(1) for unknown flag', async () => {
    const { options, exitCodes, errors } = makeOptions();
    await runReplayLabelCommand(['run', '--no-such-flag'], options);
    expect(exitCodes).toContain(1);
    expect(errors.some((e) => e.includes('--no-such-flag'))).toBe(true);
  });

  it('exits(1) for --input without value', async () => {
    const { options, exitCodes } = makeOptions();
    await runReplayLabelCommand(['run', '--input'], options);
    expect(exitCodes).toContain(1);
  });

  it('exits(1) for --output without value', async () => {
    const { options, exitCodes } = makeOptions();
    await runReplayLabelCommand(['run', '--output'], options);
    expect(exitCodes).toContain(1);
  });
});

// ─── loader mode validation ───────────────────────────────────────────────────

describe('createReplayHistoryLoader', () => {
  it('"fixture" mode returns a loader function without error', () => {
    // Uses bundled backtest-prices.json from the package
    const loader = createReplayHistoryLoader('fixture');
    expect(typeof loader).toBe('function');
  });

  it('"fixture" loader returns null for an unknown ticker', async () => {
    const loader = createReplayHistoryLoader('fixture');
    const result = await loader({
      ticker: 'NONEXISTENT_TICKER_XYZ',
      windowStartAt: '2024-01-01T00:00:00.000Z',
      windowEndAt: '2024-01-31T00:00:00.000Z',
      bundles: [],
    });
    expect(result).toBeNull();
  });

  it('"local:<path>" mode succeeds with a valid fixture JSON file', () => {
    const dir = makeScratchDir();
    const fixturePath = join(dir, 'fixture.json');
    const store: ReplayFixturePriceStore = {
      tickers: {
        'BTC-USD': { type: 'crypto', dates: ['2024-01-01'], closes: [50000], count: 1 },
      },
    };
    writeFileSync(fixturePath, JSON.stringify(store), 'utf-8');
    const loader = createReplayHistoryLoader(`local:${fixturePath}`);
    expect(typeof loader).toBe('function');
  });

  it('"local:<path>" loader resolves a known ticker from the fixture', async () => {
    const dir = makeScratchDir();
    const fixturePath = join(dir, 'fixture.json');
    const store: ReplayFixturePriceStore = {
      tickers: {
        'BTC-USD': { type: 'crypto', dates: ['2024-01-01', '2024-01-08'], closes: [50000, 53000], count: 2 },
      },
    };
    writeFileSync(fixturePath, JSON.stringify(store), 'utf-8');
    const loader = createReplayHistoryLoader(`local:${fixturePath}`);
    const result = await loader({
      ticker: 'BTC-USD',
      windowStartAt: '2024-01-01T00:00:00.000Z',
      windowEndAt: '2024-01-31T00:00:00.000Z',
      bundles: [],
    });
    expect(result).not.toBeNull();
    expect(result!.points.length).toBeGreaterThan(0);
  });

  it('"local:" without path throws at creation time', () => {
    expect(() => createReplayHistoryLoader('local:')).toThrow();
  });

  it('"local:<nonexistent>" throws at creation time', () => {
    expect(() => createReplayHistoryLoader('local:/nonexistent/path/fixture.json')).toThrow();
  });

  it('unknown loader mode throws with descriptive error', () => {
    expect(() => createReplayHistoryLoader('api-backed-mode')).toThrow(/unknown.*loader/i);
  });
});

// ─── loader mode CLI integration ─────────────────────────────────────────────

describe('replay-label run --loader flag', () => {
  it('accepts --loader fixture without error', async () => {
    const { options, errors, exitCodes } = makeOptions();
    await runReplayLabelCommand(['run', '--loader', 'fixture'], options);
    expect(exitCodes).toEqual([]);
    expect(errors).toEqual([]);
  });

  it('exits(1) for unknown --loader mode', async () => {
    const { options, errors, exitCodes } = makeOptions({
      loaderFactory: createReplayHistoryLoader,
    });
    await runReplayLabelCommand(['run', '--loader', 'bad-mode-xyz'], options);
    expect(exitCodes).toContain(1);
    expect(errors.some((e) => e.includes('bad-mode-xyz') || /unknown.*loader/i.test(e))).toBe(true);
  });

  it('exits(1) for "local:" without a path', async () => {
    const { options, exitCodes } = makeOptions({
      loaderFactory: createReplayHistoryLoader,
    });
    await runReplayLabelCommand(['run', '--loader', 'local:'], options);
    expect(exitCodes).toContain(1);
  });

  it('exits(1) for "local:<nonexistent>"', async () => {
    const { options, exitCodes } = makeOptions({
      loaderFactory: createReplayHistoryLoader,
    });
    await runReplayLabelCommand(['run', '--loader', 'local:/nonexistent/file.json'], options);
    expect(exitCodes).toContain(1);
  });
});
