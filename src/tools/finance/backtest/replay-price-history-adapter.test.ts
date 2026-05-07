import { afterAll, afterEach, describe, expect, it } from 'bun:test';
import { randomUUID } from 'node:crypto';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ArbiterReplayBundle } from '../arbiter-replay.js';
import { runReplayLabelPass } from './replay-label-runner.js';
import {
  createReplayPriceHistoryProvider,
  normalizeReplayPriceHistory,
  type ReplayFixturePriceStore,
} from './replay-price-history-adapter.js';

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

const FIXTURE_STORE: ReplayFixturePriceStore = {
  generatedAt: '2026-03-31T15:49:08.043Z',
  startDate: '2024-01-01',
  endDate: '2024-01-31',
  tickers: {
    'BTC-USD': {
      type: 'crypto',
      dates: ['2024-01-01', '2024-01-08', '2024-01-10'],
      closes: [50000, 53000, 54000],
      count: 3,
    },
    AAPL: {
      type: 'stock',
      dates: ['2024-01-02', '2024-01-03'],
      closes: [185.64, 184.25],
      count: 2,
    },
  },
};

describe('normalizeReplayPriceHistory', () => {
  it('passes through ReplayPriceHistory inputs while sorting them into canonical order', () => {
    expect(normalizeReplayPriceHistory({
      points: [
        { at: '2024-01-03T00:00:00.000Z', price: 43000 },
        { at: '2024-01-01T00:00:00.000Z', price: 42000 },
      ],
    })).toEqual({
      points: [
        { at: '2024-01-01T00:00:00.000Z', price: 42000 },
        { at: '2024-01-03T00:00:00.000Z', price: 43000 },
      ],
    });
  });

  it('normalizes fixture close/date series into ReplayPriceHistory points', () => {
    expect(normalizeReplayPriceHistory({
      type: 'crypto',
      dates: ['2024-01-03', '2024-01-01'],
      closes: [43000, 42000],
      count: 2,
    })).toEqual({
      points: [
        { at: '2024-01-01T00:00:00.000Z', price: 42000 },
        { at: '2024-01-03T00:00:00.000Z', price: 43000 },
      ],
    });
  });

  it('normalizes stock/crypto API-style bars that expose close plus date/time fields', () => {
    expect(normalizeReplayPriceHistory([
      { date: '2024-01-02', close: 181 },
      { time: '2024-01-01T00:00:00.000Z', close: 180 },
    ])).toEqual({
      points: [
        { at: '2024-01-01T00:00:00.000Z', price: 180 },
        { at: '2024-01-02T00:00:00.000Z', price: 181 },
      ],
    });
  });

  it('canonicalizes duplicate timestamps so replay labeling stays deterministic', () => {
    const historyA = normalizeReplayPriceHistory({
      points: [
        { at: '2024-01-08T00:00:00.000Z', price: 51000 },
        { at: '2024-01-01T00:00:00.000Z', price: 50000 },
        { at: '2024-01-08T00:00:00.000Z', price: 53000 },
      ],
    });
    const historyB = normalizeReplayPriceHistory({
      points: [
        { at: '2024-01-08T00:00:00.000Z', price: 53000 },
        { at: '2024-01-01T00:00:00.000Z', price: 50000 },
        { at: '2024-01-08T00:00:00.000Z', price: 51000 },
      ],
    });

    expect(historyA).toEqual(historyB);

    const labeledA = runReplayLabelPass({
      bundles: [makeBundle()],
      getHistory: () => historyA,
      labeledAt: '2024-01-09T00:00:00.000Z',
    });
    const labeledB = runReplayLabelPass({
      bundles: [makeBundle()],
      getHistory: () => historyB,
      labeledAt: '2024-01-09T00:00:00.000Z',
    });

    expect(labeledA.bundles[0]!.labels).toEqual(labeledB.bundles[0]!.labels);
  });

  it('returns null for malformed or empty history sources instead of inventing partial data', () => {
    expect(normalizeReplayPriceHistory({
      type: 'crypto',
      dates: ['2024-01-01'],
      closes: [],
      count: 0,
    })).toBeNull();
    expect(normalizeReplayPriceHistory([])).toBeNull();
  });
});

describe('createReplayPriceHistoryProvider', () => {
  it('maps replay tickers like BTC onto fixture symbols like BTC-USD', () => {
    const getHistory = createReplayPriceHistoryProvider({ fixture: FIXTURE_STORE });

    expect(getHistory('BTC', makeBundle())).toEqual({
      points: [
        { at: '2024-01-01T00:00:00.000Z', price: 50000 },
        { at: '2024-01-08T00:00:00.000Z', price: 53000 },
        { at: '2024-01-10T00:00:00.000Z', price: 54000 },
      ],
    });
  });

  it('returns null when the requested ticker has no supported local history', () => {
    const getHistory = createReplayPriceHistoryProvider({ fixture: FIXTURE_STORE });
    expect(getHistory('ETH', makeBundle({ ticker: 'ETH' }))).toBeNull();
  });

  it('loads fixture-backed history from disk and plugs directly into the replay label runner seam', () => {
    const dir = makeScratchDir();
    const fixturePath = join(dir, 'backtest-prices.json');
    writeFileSync(fixturePath, JSON.stringify(FIXTURE_STORE), 'utf-8');

    const result = runReplayLabelPass({
      bundles: [makeBundle()],
      getHistory: createReplayPriceHistoryProvider({ fixturePath }),
      labeledAt: '2024-01-09T00:00:00.000Z',
    });

    expect(result.summary.newlyLabeled).toBe(1);
    expect(result.bundles[0]!.labels?.forecast).toEqual({
      realizedPrice: 53000,
      realizedReturn: (53000 - 50000) / 50000,
      actualBinary: 1,
      labeledAt: '2024-01-09T00:00:00.000Z',
    });
  });
});
