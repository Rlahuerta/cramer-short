import { afterAll, afterEach, describe, expect, it } from 'bun:test';
import { randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  DEFAULT_ARBITER_REPLAY_BUNDLES_PATH,
  DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH,
  readArbiterReplayBundles,
  type ArbiterReplayBundle,
} from '../arbiter-replay.js';
import {
  promoteReplayLabelArtifacts,
  toReplayLabelPromotionReceiptPath,
  type ReplayLabelPromotionReceipt,
} from './replay-label-promotion.js';
import { toReplayLabelBatchReportPath } from './replay-label-batch-runner.js';
import { toReplayLabelBenchmarkReportPath } from './replay-label-benchmark-pipeline.js';

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

function makeLabeledBundle(): ArbiterReplayBundle {
  return {
    capturedAt: '2026-05-01T00:00:00.000Z',
    ticker: 'BTC',
    horizonDays: 1,
    currentPrice: 68_000,
    polymarket: {
      querySet: ['bitcoin price'],
      selectedMarketIds: ['btc-1d'],
      selectedMarkets: [
        {
          marketId: 'btc-1d',
          assetId: 'btc-1d-yes',
          question: 'Will Bitcoin be above $70,000 tomorrow?',
          probability: 0.62,
          volume24h: 250_000,
          endDate: '2026-05-02T00:00:00.000Z',
          semantics: 'terminal',
          extractedPriceLevels: [70_000],
        },
      ],
      warnings: [],
    },
    warnings: [],
    labels: {
      forecast: {
        realizedPrice: 71_000,
        realizedReturn: (71_000 - 68_000) / 68_000,
        actualBinary: 1,
        labeledAt: '2026-05-03T00:00:00.000Z',
      },
      semantic: [
        {
          marketId: 'btc-1d',
          semantics: 'terminal',
          outcome: 'yes',
          labeledAt: '2026-05-03T00:00:00.000Z',
        },
      ],
    },
  };
}

function writePromotionInputs(dir: string): {
  stagedLabeledPath: string;
  stagedLabelReportPath: string;
  stagedBenchmarkReportPath: string;
} {
  const stagedLabeledPath = join(dir, 'staged-labeled.jsonl');
  const stagedLabelReportPath = toReplayLabelBatchReportPath(stagedLabeledPath);
  const stagedBenchmarkReportPath = toReplayLabelBenchmarkReportPath(stagedLabeledPath);

  writeFileSync(stagedLabeledPath, `${JSON.stringify(makeLabeledBundle())}\n`, 'utf-8');
  writeFileSync(stagedLabelReportPath, `${JSON.stringify({ formatVersion: 'replay-label-batch-report.v1' }, null, 2)}\n`, 'utf-8');
  writeFileSync(stagedBenchmarkReportPath, `${JSON.stringify({ formatVersion: 'replay-label-benchmark-report.v1' }, null, 2)}\n`, 'utf-8');

  return { stagedLabeledPath, stagedLabelReportPath, stagedBenchmarkReportPath };
}

describe('replay label promotion', () => {
  it('promotes staged artifacts into the labeled cache target and writes a receipt artifact', () => {
    const dir = makeScratchDir();
    const { stagedLabeledPath, stagedLabelReportPath, stagedBenchmarkReportPath } = writePromotionInputs(dir);
    const promotedLabeledPath = join(dir, 'cache', 'labeled', 'bundles.jsonl');
    const receiptPath = toReplayLabelPromotionReceiptPath(promotedLabeledPath);

    const result = promoteReplayLabelArtifacts({
      stagedLabeledPath,
      stagedLabelReportPath,
      stagedBenchmarkReportPath,
      promotedLabeledPath,
      promotedAt: '2026-05-06T00:00:00.000Z',
    });

    expect(readFileSync(promotedLabeledPath, 'utf-8')).toBe(readFileSync(stagedLabeledPath, 'utf-8'));
    expect(readFileSync(toReplayLabelBatchReportPath(promotedLabeledPath), 'utf-8')).toBe(readFileSync(stagedLabelReportPath, 'utf-8'));
    expect(readFileSync(toReplayLabelBenchmarkReportPath(promotedLabeledPath), 'utf-8')).toBe(readFileSync(stagedBenchmarkReportPath, 'utf-8'));

    const receipt = JSON.parse(readFileSync(receiptPath, 'utf-8')) as ReplayLabelPromotionReceipt;
    expect(receipt).toEqual(result.receipt);
    expect(receipt.promotedAt).toBe('2026-05-06T00:00:00.000Z');
    expect(receipt.bundleCount).toBe(1);
    expect(receipt.labeledBundleCount).toBe(1);
    expect(receipt.source.stagedLabeledPath).toBe(stagedLabeledPath);
    expect(receipt.target.promotedLabeledPath).toBe(promotedLabeledPath);
  });

  it('fails when a required staged artifact is missing', () => {
    const dir = makeScratchDir();
    const { stagedLabeledPath, stagedLabelReportPath, stagedBenchmarkReportPath } = writePromotionInputs(dir);
    rmSync(stagedBenchmarkReportPath);

    expect(() => promoteReplayLabelArtifacts({
      stagedLabeledPath,
      stagedLabelReportPath,
      stagedBenchmarkReportPath,
      promotedLabeledPath: join(dir, 'cache', 'labeled', 'bundles.jsonl'),
    })).toThrow(/benchmark report/i);
  });

  it('refuses to target the raw replay capture bundle path', () => {
    const dir = makeScratchDir();
    const { stagedLabeledPath, stagedLabelReportPath, stagedBenchmarkReportPath } = writePromotionInputs(dir);

    expect(() => promoteReplayLabelArtifacts({
      stagedLabeledPath,
      stagedLabelReportPath,
      stagedBenchmarkReportPath,
      promotedLabeledPath: DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH,
    })).toThrow(/raw replay capture bundle path/i);
  });

  it('refuses to target the legacy raw replay bundle path', () => {
    const dir = makeScratchDir();
    const { stagedLabeledPath, stagedLabelReportPath, stagedBenchmarkReportPath } = writePromotionInputs(dir);

    expect(() => promoteReplayLabelArtifacts({
      stagedLabeledPath,
      stagedLabelReportPath,
      stagedBenchmarkReportPath,
      promotedLabeledPath: DEFAULT_ARBITER_REPLAY_BUNDLES_PATH,
    })).toThrow(/raw replay bundle path/i);
  });

  for (const [label, rawPath, expectedError] of [
    ['stagedLabeledPath', DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH, /raw replay capture bundle path/i],
    ['stagedLabeledPath', DEFAULT_ARBITER_REPLAY_BUNDLES_PATH, /raw replay bundle path/i],
    ['stagedLabelReportPath', DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH, /raw replay capture bundle path/i],
    ['stagedLabelReportPath', DEFAULT_ARBITER_REPLAY_BUNDLES_PATH, /raw replay bundle path/i],
    ['stagedBenchmarkReportPath', DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH, /raw replay capture bundle path/i],
    ['stagedBenchmarkReportPath', DEFAULT_ARBITER_REPLAY_BUNDLES_PATH, /raw replay bundle path/i],
  ] as const) {
    it(`refuses to let ${label} collide with ${rawPath}`, () => {
      const dir = makeScratchDir();
      const { stagedLabeledPath, stagedLabelReportPath, stagedBenchmarkReportPath } = writePromotionInputs(dir);
      const promotedLabeledPath = join(dir, 'cache', 'labeled', 'bundles.jsonl');

      expect(() => promoteReplayLabelArtifacts({
        stagedLabeledPath,
        stagedLabelReportPath,
        stagedBenchmarkReportPath,
        promotedLabeledPath,
        [label]: rawPath,
      })).toThrow(expectedError);
    });
  }

  for (const [label, rawPath, expectedError] of [
    ['promotedLabelReportPath', DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH, /raw replay capture bundle path/i],
    ['promotedLabelReportPath', DEFAULT_ARBITER_REPLAY_BUNDLES_PATH, /raw replay bundle path/i],
    ['promotedBenchmarkReportPath', DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH, /raw replay capture bundle path/i],
    ['promotedBenchmarkReportPath', DEFAULT_ARBITER_REPLAY_BUNDLES_PATH, /raw replay bundle path/i],
    ['receiptPath', DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH, /raw replay capture bundle path/i],
    ['receiptPath', DEFAULT_ARBITER_REPLAY_BUNDLES_PATH, /raw replay bundle path/i],
  ] as const) {
    it(`refuses to let ${label} collide with ${rawPath}`, () => {
      const dir = makeScratchDir();
      const { stagedLabeledPath, stagedLabelReportPath, stagedBenchmarkReportPath } = writePromotionInputs(dir);
      const promotedLabeledPath = join(dir, 'cache', 'labeled', 'bundles.jsonl');

      expect(() => promoteReplayLabelArtifacts({
        stagedLabeledPath,
        stagedLabelReportPath,
        stagedBenchmarkReportPath,
        promotedLabeledPath,
        [label]: rawPath,
      })).toThrow(expectedError);
    });
  }

  it('keeps promoted labeled bundles readable through existing replay bundle readers', () => {
    const dir = makeScratchDir();
    const { stagedLabeledPath, stagedLabelReportPath, stagedBenchmarkReportPath } = writePromotionInputs(dir);
    const promotedLabeledPath = join(dir, 'cache', 'labeled', 'bundles.jsonl');

    promoteReplayLabelArtifacts({
      stagedLabeledPath,
      stagedLabelReportPath,
      stagedBenchmarkReportPath,
      promotedLabeledPath,
    });

    expect(readArbiterReplayBundles(promotedLabeledPath)).toEqual(readArbiterReplayBundles(stagedLabeledPath));
  });
});
