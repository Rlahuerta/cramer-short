import { afterAll, afterEach, describe, expect, it } from 'bun:test';
import { randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ArbiterReplayBundle } from '../arbiter-replay.js';
import type { ReplayPriceHistory } from '../arbiter-replay-labeler.js';
import {
  runReplayLabelBatch,
  runReplayLabelBatchFromFile,
  type ReplayTickerHistoryRequest,
} from './replay-label-batch-runner.js';

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

function makeBundle(overrides: Partial<ArbiterReplayBundle> = {}): ArbiterReplayBundle {
  return {
    capturedAt: '2024-01-01T00:00:00.000Z',
    ticker: 'BTC',
    horizonDays: 7,
    currentPrice: 50000,
    polymarket: {
      querySet: ['bitcoin price'],
      selectedMarketIds: ['pm-1'],
      selectedMarkets: [
        {
          marketId: 'pm-1',
          assetId: 'asset-yes-1',
          question: 'Will Bitcoin be above $52,000 on Jan 8?',
          probability: 0.5,
          volume24h: 100000,
          endDate: '2024-01-08T00:00:00.000Z',
          semantics: 'terminal',
          extractedPriceLevels: [52000],
        },
      ],
      warnings: [],
    },
    warnings: [],
    ...overrides,
  };
}

const FIXED_LABELED_AT = '2024-01-13T00:00:00.000Z';

const BTC_READY_HISTORY: ReplayPriceHistory = {
  points: [
    { at: '2024-01-01T00:00:00.000Z', price: 50000 },
    { at: '2024-01-08T00:00:00.000Z', price: 53000 },
    { at: '2024-01-12T00:00:00.000Z', price: 54000 },
  ],
};

const ETH_PARTIAL_HISTORY: ReplayPriceHistory = {
  points: [
    { at: '2024-01-01T00:00:00.000Z', price: 3000 },
    { at: '2024-01-05T00:00:00.000Z', price: 3100 },
  ],
};

describe('runReplayLabelBatch', () => {
  it('handles mixed ready and not-ready bundles after prefetching histories by ticker', async () => {
    const bundles = [
      makeBundle(),
      makeBundle({
        ticker: 'ETH',
        currentPrice: 3000,
        polymarket: {
          querySet: ['ethereum price'],
          selectedMarketIds: ['pm-eth-1'],
          selectedMarkets: [
            {
              marketId: 'pm-eth-1',
              assetId: 'asset-yes-eth-1',
              question: 'Will Ethereum be above $3,100 on Jan 8?',
              probability: 0.5,
              volume24h: 100000,
              endDate: '2024-01-08T00:00:00.000Z',
              semantics: 'terminal',
              extractedPriceLevels: [3100],
            },
          ],
          warnings: [],
        },
      }),
    ];

    const result = await runReplayLabelBatch({
      bundles,
      labeledAt: FIXED_LABELED_AT,
      loadHistory: async ({ ticker }) => (ticker === 'BTC' ? BTC_READY_HISTORY : ETH_PARTIAL_HISTORY),
    });

    expect(result.summary.total).toBe(2);
    expect(result.summary.newlyLabeled).toBe(1);
    expect(result.summary.pending).toBe(1);
    expect(result.summary.skippedByMissingHistory).toBe(0);
    expect(result.bundles[0]!.labels?.forecast?.labeledAt).toBe(FIXED_LABELED_AT);
    expect(result.bundles[1]!.labels).toBeUndefined();
  });

  it('loads history once per ticker and requests the combined required window', async () => {
    const requests: ReplayTickerHistoryRequest[] = [];
    const bundles = [
      makeBundle({
        capturedAt: '2024-01-01T00:00:00.000Z',
        horizonDays: 7,
        polymarket: {
          querySet: ['bitcoin price'],
          selectedMarketIds: ['pm-1'],
          selectedMarkets: [
            {
              marketId: 'pm-1',
              assetId: 'asset-yes-1',
              question: 'Will Bitcoin be above $52,000 on Jan 9?',
              probability: 0.5,
              volume24h: 100000,
              endDate: '2024-01-09T00:00:00.000Z',
              semantics: 'terminal',
              extractedPriceLevels: [52000],
            },
          ],
          warnings: [],
        },
      }),
      makeBundle({
        capturedAt: '2024-01-05T00:00:00.000Z',
        currentPrice: 51000,
        horizonDays: 2,
        polymarket: {
          querySet: ['bitcoin price'],
          selectedMarketIds: ['pm-2'],
          selectedMarkets: [
            {
              marketId: 'pm-2',
              assetId: 'asset-yes-2',
              question: 'Will Bitcoin be above $53,000 on Jan 12?',
              probability: 0.5,
              volume24h: 120000,
              endDate: '2024-01-12T00:00:00.000Z',
              semantics: 'terminal',
              extractedPriceLevels: [53000],
            },
          ],
          warnings: [],
        },
      }),
    ];

    const result = await runReplayLabelBatch({
      bundles,
      labeledAt: FIXED_LABELED_AT,
      loadHistory: async (request) => {
        requests.push(request);
        return BTC_READY_HISTORY;
      },
    });

    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      ticker: 'BTC',
      windowStartAt: '2024-01-01T00:00:00.000Z',
      windowEndAt: '2024-01-12T00:00:00.000Z',
    });
    expect(requests[0]!.bundles).toHaveLength(2);
    expect(result.summary.newlyLabeled).toBe(2);
  });

  it('preserves already-labeled bundles unchanged and skips history loads for them', async () => {
    const labeled = makeBundle({
      labels: {
        forecast: {
          realizedPrice: 51000,
          realizedReturn: 0.02,
          actualBinary: 1,
          labeledAt: '2024-01-08T12:00:00.000Z',
        },
        semantic: [],
      },
    });
    let loadCalls = 0;

    const result = await runReplayLabelBatch({
      bundles: [labeled],
      labeledAt: FIXED_LABELED_AT,
      loadHistory: async () => {
        loadCalls += 1;
        return BTC_READY_HISTORY;
      },
    });

    expect(loadCalls).toBe(0);
    expect(result.summary.alreadyLabeled).toBe(1);
    expect(result.summary.newlyLabeled).toBe(0);
    expect(result.bundles[0]).toEqual(labeled);
    expect(result.bundles[0]!.labels?.forecast?.labeledAt).toBe('2024-01-08T12:00:00.000Z');
  });
});

describe('runReplayLabelBatchFromFile', () => {
  it('writes the labeled output plus a default report artifact without mutating the input bundle file', async () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'input.jsonl');
    const outputPath = join(dir, 'output.jsonl');
    const reportPath = join(dir, 'output.report.json');
    const alreadyLabeled = makeBundle({
      ticker: 'ETH',
      currentPrice: 3000,
      labels: {
        forecast: {
          realizedPrice: 3050,
          realizedReturn: 3050 / 3000 - 1,
          actualBinary: 1,
          labeledAt: '2024-01-10T00:00:00.000Z',
        },
        semantic: [],
      },
    });

    writeFileSync(
      inputPath,
      `${JSON.stringify(makeBundle())}\n${JSON.stringify(alreadyLabeled)}\n`,
      'utf-8',
    );
    const originalInput = readFileSync(inputPath, 'utf-8');

    const requests: ReplayTickerHistoryRequest[] = [];
    const result = await runReplayLabelBatchFromFile({
      inputPath,
      outputPath,
      labeledAt: FIXED_LABELED_AT,
      loadHistory: async (request) => {
        requests.push(request);
        return request.ticker === 'BTC' ? BTC_READY_HISTORY : ETH_PARTIAL_HISTORY;
      },
    });

    expect(requests).toHaveLength(1);
    expect(requests[0]!.ticker).toBe('BTC');
    expect(result.summary.total).toBe(2);
    expect(result.summary.alreadyLabeled).toBe(1);
    expect(result.summary.newlyLabeled).toBe(1);

    const lines = readFileSync(outputPath, 'utf-8').trim().split('\n');
    expect(lines).toHaveLength(2);
    const writtenReady = JSON.parse(lines[0]!);
    const writtenPreserved = JSON.parse(lines[1]!);
    expect(writtenReady.labels?.forecast?.labeledAt).toBe(FIXED_LABELED_AT);
    expect(writtenPreserved.labels?.forecast?.labeledAt).toBe('2024-01-10T00:00:00.000Z');
    expect(readFileSync(inputPath, 'utf-8')).toBe(originalInput);

    const report = JSON.parse(readFileSync(reportPath, 'utf-8'));
    expect(report.summary).toEqual(result.summary);
  });

  it('writes a report with the run summary and per-ticker history request details', async () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'input.jsonl');
    const outputPath = join(dir, 'output.jsonl');
    const reportPath = join(dir, 'audit.json');
    const ethBundle = makeBundle({
      ticker: 'ETH',
      capturedAt: '2024-01-02T00:00:00.000Z',
      currentPrice: 3000,
      horizonDays: 3,
      polymarket: {
        querySet: ['ethereum price'],
        selectedMarketIds: ['pm-eth-1'],
        selectedMarkets: [
          {
            marketId: 'pm-eth-1',
            assetId: 'asset-yes-eth-1',
            question: 'Will Ethereum be above $3,100 on Jan 5?',
            probability: 0.5,
            volume24h: 100000,
            endDate: '2024-01-05T00:00:00.000Z',
            semantics: 'terminal',
            extractedPriceLevels: [3100],
          },
        ],
        warnings: [],
      },
    });
    const alreadyLabeled = makeBundle({
      ticker: 'SOL',
      currentPrice: 100,
      labels: {
        forecast: {
          realizedPrice: 101,
          realizedReturn: 0.01,
          actualBinary: 1,
          labeledAt: '2024-01-10T00:00:00.000Z',
        },
        semantic: [],
      },
    });

    writeFileSync(
      inputPath,
      `${JSON.stringify(makeBundle())}\n${JSON.stringify(ethBundle)}\n${JSON.stringify(alreadyLabeled)}\n`,
      'utf-8',
    );

    const result = await runReplayLabelBatchFromFile({
      inputPath,
      outputPath,
      reportPath,
      labeledAt: FIXED_LABELED_AT,
      loadHistory: async ({ ticker }) => (ticker === 'BTC' ? BTC_READY_HISTORY : null),
    });

    const report = JSON.parse(readFileSync(reportPath, 'utf-8'));
    expect(report).toEqual({
      formatVersion: 'replay-label-batch-report.v1',
      inputPath,
      outputPath,
      labeledAt: FIXED_LABELED_AT,
      summary: result.summary,
      historyRequests: [
        {
          ticker: 'BTC',
          windowStartAt: '2024-01-01T00:00:00.000Z',
          windowEndAt: '2024-01-08T00:00:00.000Z',
          bundleCount: 1,
          historyFound: true,
          pointCount: 3,
        },
        {
          ticker: 'ETH',
          windowStartAt: '2024-01-02T00:00:00.000Z',
          windowEndAt: '2024-01-05T00:00:00.000Z',
          bundleCount: 1,
          historyFound: false,
          pointCount: 0,
        },
      ],
    });
  });

  it('throws when inputPath and outputPath are lexical aliases of the same file', async () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'bundles.jsonl');
    const outputPath = `${dir}/./bundles.jsonl`;
    writeFileSync(inputPath, `${JSON.stringify(makeBundle())}\n`, 'utf-8');

    await expect(
      runReplayLabelBatchFromFile({
        inputPath,
        outputPath,
        loadHistory: async () => BTC_READY_HISTORY,
      }),
    ).rejects.toThrow();
  });

  it('throws when reportPath resolves through a symlinked parent to the same file as outputPath', async () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'input.jsonl');
    const realDir = join(dir, 'real');
    const aliasedDir = join(dir, 'alias');
    mkdirSync(realDir, { recursive: true });
    symlinkSync(realDir, aliasedDir, 'dir');

    const outputPath = join(realDir, 'output.json');
    const reportPath = join(aliasedDir, 'output.json');
    writeFileSync(inputPath, `${JSON.stringify(makeBundle())}\n`, 'utf-8');

    await expect(
      runReplayLabelBatchFromFile({
        inputPath,
        outputPath,
        reportPath,
        loadHistory: async () => BTC_READY_HISTORY,
      }),
    ).rejects.toThrow();
  });
});
