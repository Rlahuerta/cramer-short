import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { afterEach, afterAll, describe, expect, it, beforeEach, setSystemTime } from 'bun:test';
import { mkdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { ArbiterReplayBundle } from '../arbiter-replay.js';
import type { ReplayPriceHistory } from '../arbiter-replay-labeler.js';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});
import {
  runReplayLabelPass,
  runReplayLabelPassFromFile,
  type ReplayLabelRunSummary,
  type ReplayLabelRunResult,
  type HistoryProvider,
} from './replay-label-runner.js';

// Project-local scratch directory — never /tmp
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

// ---- shared fixtures --------------------------------------------------------

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

/** History that fully covers the 7-day horizon ending 2024-01-08 */
const FULL_HISTORY: ReplayPriceHistory = {
  points: [
    { at: '2024-01-01T00:00:00.000Z', price: 50000 },
    { at: '2024-01-08T00:00:00.000Z', price: 53000 },
  ],
};

/** History that does NOT yet reach the 7-day horizon */
const PARTIAL_HISTORY: ReplayPriceHistory = {
  points: [
    { at: '2024-01-01T00:00:00.000Z', price: 50000 },
    { at: '2024-01-05T00:00:00.000Z', price: 51500 },
  ],
};

const FIXED_LABELED_AT = '2024-01-09T00:00:00.000Z';

// ---- runReplayLabelPass -----------------------------------------------------

describe('runReplayLabelPass', () => {
  it('returns a result that satisfies the required summary shape', () => {
    const result = runReplayLabelPass({
      bundles: [makeBundle()],
      getHistory: () => FULL_HISTORY,
      labeledAt: FIXED_LABELED_AT,
    });

    const s: ReplayLabelRunSummary = result.summary;
    expect(typeof s.total).toBe('number');
    expect(typeof s.alreadyLabeled).toBe('number');
    expect(typeof s.newlyLabeled).toBe('number');
    expect(typeof s.skippedByMissingHistory).toBe('number');
    expect(typeof s.pending).toBe('number');
    expect(typeof s.pendingReasons).toBe('object');
    expect(typeof s.perTickerCounts).toBe('object');
    expect(s.perTickerCounts['BTC']).toBeDefined();
    const tc = s.perTickerCounts['BTC']!;
    expect(typeof tc.total).toBe('number');
    expect(typeof tc.alreadyLabeled).toBe('number');
    expect(typeof tc.newlyLabeled).toBe('number');
    expect(typeof tc.skippedByMissingHistory).toBe('number');
    expect(typeof tc.pending).toBe('number');
  });

  it('counts total correctly across all bundles', () => {
    const bundles = [makeBundle(), makeBundle({ ticker: 'ETH' }), makeBundle({ ticker: 'ETH' })];
    const result = runReplayLabelPass({ bundles, getHistory: () => FULL_HISTORY });
    expect(result.summary.total).toBe(3);
  });

  it('counts already-labeled bundles and skips re-labeling them', () => {
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
    const result = runReplayLabelPass({
      bundles: [labeled],
      getHistory: () => FULL_HISTORY,
      labeledAt: FIXED_LABELED_AT,
    });
    expect(result.summary.alreadyLabeled).toBe(1);
    expect(result.summary.newlyLabeled).toBe(0);
    // The already-labeled bundle must remain unchanged in the output
    expect(result.bundles[0]!.labels!.forecast!.labeledAt).toBe('2024-01-08T12:00:00.000Z');
  });

  it('labels eligible bundles and counts them as newlyLabeled', () => {
    const bundle = makeBundle();
    const result = runReplayLabelPass({
      bundles: [bundle],
      getHistory: () => FULL_HISTORY,
      labeledAt: FIXED_LABELED_AT,
    });
    expect(result.summary.newlyLabeled).toBe(1);
    expect(result.summary.alreadyLabeled).toBe(0);
    expect(result.bundles[0]!.labels?.forecast?.labeledAt).toBe(FIXED_LABELED_AT);
  });

  it('tracks bundles with no available history in skippedByMissingHistory', () => {
    const bundles = [makeBundle(), makeBundle({ ticker: 'ETH' })];
    // ETH has no history, BTC does
    const getHistory: HistoryProvider = (ticker) => (ticker === 'BTC' ? FULL_HISTORY : null);
    const result = runReplayLabelPass({ bundles, getHistory, labeledAt: FIXED_LABELED_AT });

    expect(result.summary.skippedByMissingHistory).toBe(1);
    expect(result.summary.newlyLabeled).toBe(1);
    expect(result.summary.perTickerCounts['ETH']!.skippedByMissingHistory).toBe(1);
    expect(result.summary.perTickerCounts['BTC']!.newlyLabeled).toBe(1);
  });

  it('aggregates pendingReasons for bundles that are not yet eligible', () => {
    const bundles = [makeBundle(), makeBundle()];
    const result = runReplayLabelPass({
      bundles,
      getHistory: () => PARTIAL_HISTORY,
      labeledAt: FIXED_LABELED_AT,
    });

    expect(result.summary.newlyLabeled).toBe(0);
    // Both bundles are pending due to at least one of the standard reasons
    const reasonValues = Object.values(result.summary.pendingReasons);
    expect(reasonValues.length).toBeGreaterThan(0);
    // Each reason count must be a positive integer
    for (const count of reasonValues) {
      expect(count).toBeGreaterThanOrEqual(1);
    }
  });

  it('increments top-level pending count for bundles not yet eligible', () => {
    const bundles = [makeBundle(), makeBundle()];
    const result = runReplayLabelPass({
      bundles,
      getHistory: () => PARTIAL_HISTORY,
      labeledAt: FIXED_LABELED_AT,
    });
    expect(result.summary.pending).toBe(2);
  });

  it('increments per-ticker pending count for pending bundles', () => {
    const bundles = [makeBundle({ ticker: 'BTC' }), makeBundle({ ticker: 'BTC' }), makeBundle({ ticker: 'ETH' })];
    const result = runReplayLabelPass({
      bundles,
      getHistory: () => PARTIAL_HISTORY,
      labeledAt: FIXED_LABELED_AT,
    });
    expect(result.summary.perTickerCounts['BTC']!.pending).toBe(2);
    expect(result.summary.perTickerCounts['ETH']!.pending).toBe(1);
  });

  it('per-ticker counts (alreadyLabeled + newlyLabeled + skippedByMissingHistory + pending) sum to total per ticker', () => {
    const labeled = makeBundle({ ticker: 'BTC', labels: { forecast: { realizedPrice: 51000, realizedReturn: 0.02, actualBinary: 1, labeledAt: '2024-01-08T12:00:00.000Z' }, semantic: [] } });
    const bundles = [
      labeled,
      makeBundle({ ticker: 'BTC' }),         // will be newlyLabeled (full history)
      makeBundle({ ticker: 'BTC', capturedAt: '2024-01-01T00:00:00.000Z' }), // pending (partial)
    ];
    const getHistory: HistoryProvider = (_, bundle) =>
      bundle.labels?.forecast ? FULL_HISTORY : (bundle === bundles[1] ? FULL_HISTORY : PARTIAL_HISTORY);
    const result = runReplayLabelPass({ bundles, getHistory, labeledAt: FIXED_LABELED_AT });
    const tc = result.summary.perTickerCounts['BTC']!;
    const componentSum = tc.alreadyLabeled + tc.newlyLabeled + tc.skippedByMissingHistory + tc.pending;
    expect(componentSum).toBe(tc.total);
  });

  it('does NOT mutate the input bundle objects', () => {
    const bundle = makeBundle();
    const originalJson = JSON.stringify(bundle);
    runReplayLabelPass({ bundles: [bundle], getHistory: () => FULL_HISTORY, labeledAt: FIXED_LABELED_AT });
    expect(JSON.stringify(bundle)).toBe(originalJson);
  });

  it('returns a new bundles array reference, not the same as the input', () => {
    const input = [makeBundle()];
    const result = runReplayLabelPass({ bundles: input, getHistory: () => FULL_HISTORY });
    expect(result.bundles).not.toBe(input);
  });

  it('per-ticker counts sum to the total for multi-ticker input', () => {
    const bundles = [
      makeBundle({ ticker: 'BTC' }),
      makeBundle({ ticker: 'BTC' }),
      makeBundle({ ticker: 'ETH' }),
    ];
    const getHistory: HistoryProvider = () => FULL_HISTORY;
    const result = runReplayLabelPass({ bundles, getHistory });

    const tickerTotals = Object.values(result.summary.perTickerCounts)
      .reduce((sum, tc) => sum + tc.total, 0);
    expect(tickerTotals).toBe(result.summary.total);

    expect(result.summary.perTickerCounts['BTC']!.total).toBe(2);
    expect(result.summary.perTickerCounts['ETH']!.total).toBe(1);
  });

  it('per-ticker newlyLabeled increments match the top-level count', () => {
    const bundles = [makeBundle({ ticker: 'BTC' }), makeBundle({ ticker: 'ETH' })];
    const result = runReplayLabelPass({
      bundles,
      getHistory: () => FULL_HISTORY,
      labeledAt: FIXED_LABELED_AT,
    });

    const perTickerNewlyLabeled = Object.values(result.summary.perTickerCounts)
      .reduce((sum, tc) => sum + tc.newlyLabeled, 0);
    expect(perTickerNewlyLabeled).toBe(result.summary.newlyLabeled);
  });

  it('uses the current timestamp as labeledAt when none is provided', () => {
    const before = FIXED_TEST_NOW_MS;
    const result = runReplayLabelPass({ bundles: [makeBundle()], getHistory: () => FULL_HISTORY });
    const after = FIXED_TEST_NOW_MS;

    const labeledAtMs = Date.parse(result.labeledAt);
    expect(labeledAtMs).toBeGreaterThanOrEqual(before);
    expect(labeledAtMs).toBeLessThanOrEqual(after);
  });

  it('handles an empty bundle array gracefully and returns zero counts', () => {
    const result = runReplayLabelPass({ bundles: [], getHistory: () => null });
    expect(result.summary.total).toBe(0);
    expect(result.summary.alreadyLabeled).toBe(0);
    expect(result.summary.newlyLabeled).toBe(0);
    expect(result.summary.skippedByMissingHistory).toBe(0);
    expect(result.summary.pending).toBe(0);
    expect(Object.keys(result.summary.pendingReasons)).toHaveLength(0);
    expect(Object.keys(result.summary.perTickerCounts)).toHaveLength(0);
    expect(result.bundles).toHaveLength(0);
  });
});

// ---- runReplayLabelPassFromFile ---------------------------------------------

describe('runReplayLabelPassFromFile', () => {
  it('reads bundles from the input file and writes labeled output to the output file', () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'input.jsonl');
    const outputPath = join(dir, 'output.jsonl');

    const bundle = makeBundle();
    writeFileSync(inputPath, `${JSON.stringify(bundle)}\n`, 'utf-8');

    const result = runReplayLabelPassFromFile({
      inputPath,
      outputPath,
      getHistory: () => FULL_HISTORY,
      labeledAt: FIXED_LABELED_AT,
    });

    expect(result.summary.total).toBe(1);
    expect(result.summary.newlyLabeled).toBe(1);

    // Output file must exist and contain labeled bundle
    const lines = require('node:fs').readFileSync(outputPath, 'utf-8').trim().split('\n');
    expect(lines).toHaveLength(1);
    const written = JSON.parse(lines[0]);
    expect(written.labels?.forecast?.labeledAt).toBe(FIXED_LABELED_AT);
  });

  it('preserves already-labeled bundles unchanged in the output file', () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'input.jsonl');
    const outputPath = join(dir, 'output.jsonl');

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
    const unlabeled = makeBundle({ ticker: 'ETH' });
    writeFileSync(inputPath, `${JSON.stringify(labeled)}\n${JSON.stringify(unlabeled)}\n`, 'utf-8');

    const result = runReplayLabelPassFromFile({
      inputPath,
      outputPath,
      getHistory: () => FULL_HISTORY,
      labeledAt: FIXED_LABELED_AT,
    });

    expect(result.summary.total).toBe(2);
    expect(result.summary.alreadyLabeled).toBe(1);
    expect(result.summary.newlyLabeled).toBe(1);

    const lines = require('node:fs').readFileSync(outputPath, 'utf-8').trim().split('\n');
    expect(lines).toHaveLength(2);
  });

  it('throws when inputPath equals outputPath to prevent in-place mutation', () => {
    const dir = makeScratchDir();
    const samePath = join(dir, 'bundles.jsonl');
    writeFileSync(samePath, '', 'utf-8');

    expect(() =>
      runReplayLabelPassFromFile({
        inputPath: samePath,
        outputPath: samePath,
        getHistory: () => null,
      }),
    ).toThrow();
  });

  it('throws when inputPath and outputPath are lexical aliases of the same file', () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'bundles.jsonl');
    const outputPath = `${dir}/./bundles.jsonl`;
    writeFileSync(inputPath, '', 'utf-8');

    expect(() =>
      runReplayLabelPassFromFile({
        inputPath,
        outputPath,
        getHistory: () => null,
      }),
    ).toThrow();
  });

  it('throws when inputPath and outputPath resolve through a symlinked parent to the same file', () => {
    const dir = makeScratchDir();
    const realDir = join(dir, 'real');
    const aliasedDir = join(dir, 'alias');
    mkdirSync(realDir, { recursive: true });
    symlinkSync(realDir, aliasedDir, 'dir');

    const inputPath = join(realDir, 'bundles.jsonl');
    const outputPath = join(aliasedDir, 'bundles.jsonl');
    writeFileSync(inputPath, '', 'utf-8');

    expect(() =>
      runReplayLabelPassFromFile({
        inputPath,
        outputPath,
        getHistory: () => null,
      }),
    ).toThrow();
  });

  it('returns a machine-readable summary result object (not just side effects)', () => {
    const dir = makeScratchDir();
    const inputPath = join(dir, 'input.jsonl');
    const outputPath = join(dir, 'output.jsonl');
    writeFileSync(inputPath, `${JSON.stringify(makeBundle())}\n`, 'utf-8');

    const result: ReplayLabelRunResult = runReplayLabelPassFromFile({
      inputPath,
      outputPath,
      getHistory: () => null,
      labeledAt: FIXED_LABELED_AT,
    });

    expect(result.summary.total).toBe(1);
    expect(result.summary.skippedByMissingHistory).toBe(1);
    expect(typeof result.labeledAt).toBe('string');
    expect(Array.isArray(result.bundles)).toBe(true);
  });
});
