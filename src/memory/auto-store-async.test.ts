/**
 * Tests for the async functions in auto-store.ts.
 * Uses the injectable _getManager parameter to avoid module-level mock contamination.
 */

import { mock, describe, it, expect, beforeEach } from 'bun:test';
// Import from auto-store-core.ts — deliberately separate from auto-store.ts so that
// mock.module('../memory/auto-store.js', stubs) in controller tests never contaminates here.
import { seedWatchlistEntriesCore as seedWatchlistEntries, autoStoreFromRunCore as autoStoreFromRun } from './auto-store-core.js';

// ---------------------------------------------------------------------------
// Shared mock setup
// ---------------------------------------------------------------------------

const mockStoreInsight = mock(async (_opts: unknown) => undefined);
const mockRecallByTicker = mock((_ticker: string): unknown[] => []);

const mockGetFinancialStore = mock(() => ({
  storeInsight: mockStoreInsight,
  recallByTicker: mockRecallByTicker,
}));

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockManagerGet = mock(async () => ({
  getFinancialStore: mockGetFinancialStore,
})) as any;

beforeEach(() => {
  mockStoreInsight.mockClear();
  mockRecallByTicker.mockClear();
  mockGetFinancialStore.mockClear();
  mockManagerGet.mockClear();

  mockRecallByTicker.mockImplementation(() => []);
  mockGetFinancialStore.mockImplementation(() => ({
    storeInsight: mockStoreInsight,
    recallByTicker: mockRecallByTicker,
  }));
  mockManagerGet.mockImplementation(async () => ({
    getFinancialStore: mockGetFinancialStore,
  }));
  mockStoreInsight.mockImplementation(async () => undefined);
});

// ---------------------------------------------------------------------------
// seedWatchlistEntries
// ---------------------------------------------------------------------------

describe('seedWatchlistEntries', () => {
  it('does nothing for an empty entries array', async () => {
    await seedWatchlistEntries([], mockManagerGet);
    expect(mockManagerGet).not.toHaveBeenCalled();
    expect(mockStoreInsight).not.toHaveBeenCalled();
  });

  it('stores an insight for a ticker with no existing record', async () => {
    await seedWatchlistEntries([{ ticker: 'AAPL' }], mockManagerGet);
    expect(mockManagerGet).toHaveBeenCalledTimes(1);
    expect(mockStoreInsight).toHaveBeenCalledTimes(1);
    const call = mockStoreInsight.mock.calls[0]![0] as { ticker: string; source: string; tags: string[] };
    expect(call.ticker).toBe('AAPL');
    expect(call.source).toBe('watchlist');
    expect(call.tags).toContain('source:watchlist');
    expect(call.tags).toContain('ticker:AAPL');
  });

  it('includes cost basis and shares in the content when provided', async () => {
    await seedWatchlistEntries([{ ticker: 'TSLA', costBasis: 250, shares: 10 }], mockManagerGet);
    const call = mockStoreInsight.mock.calls[0]![0] as { content: string };
    expect(call.content).toContain('cost basis $250');
    expect(call.content).toContain('10 shares');
  });

  it('skips tickers that already have a watchlist record', async () => {
    mockRecallByTicker.mockImplementation(() => [
      { source: 'watchlist', tags: ['source:watchlist', 'ticker:MSFT'] },
    ]);
    await seedWatchlistEntries([{ ticker: 'MSFT' }], mockManagerGet);
    expect(mockStoreInsight).not.toHaveBeenCalled();
  });

  it('processes multiple entries, skipping already-stored ones', async () => {
    mockRecallByTicker.mockImplementation((ticker: string) => {
      if (ticker === 'GOOG') return [{ source: 'watchlist', tags: ['source:watchlist'] }];
      return [];
    });
    await seedWatchlistEntries([{ ticker: 'AAPL' }, { ticker: 'GOOG' }, { ticker: 'META' }], mockManagerGet);
    expect(mockStoreInsight).toHaveBeenCalledTimes(2);
  });

  it('does not throw when manager factory rejects', async () => {
    const failGet = mock(async () => { throw new Error('DB unavailable'); }) as () => Promise<never>;
    await expect(seedWatchlistEntries([{ ticker: 'AAPL' }], failGet)).resolves.toBeUndefined();
  });

  it('does not throw when getFinancialStore returns null', async () => {
    mockGetFinancialStore.mockImplementation((() => null) as any);
    await expect(seedWatchlistEntries([{ ticker: 'AAPL' }], mockManagerGet)).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// autoStoreFromRun — guard conditions
// ---------------------------------------------------------------------------

describe('autoStoreFromRun — guard conditions', () => {
  it('does nothing when no financial tool was used', async () => {
    await autoStoreFromRun('Tell me a joke', 'Why did the chicken?', [
      { tool: 'skill', args: {}, result: 'ha' },
    ], mockManagerGet);
    expect(mockManagerGet).not.toHaveBeenCalled();
  });

  it('does nothing when store_financial_insight was already called', async () => {
    await autoStoreFromRun('AAPL analysis', 'Results', [
      { tool: 'get_financials', args: {}, result: 'revenue: $394B' },
      { tool: 'store_financial_insight', args: {}, result: 'stored' },
    ], mockManagerGet);
    expect(mockManagerGet).not.toHaveBeenCalled();
  });

  it('does nothing when no tickers are found in the query', async () => {
    await autoStoreFromRun('what is the weather like today?', 'Sunny', [
      { tool: 'web_search', args: {}, result: 'sunny in SF' },
    ], mockManagerGet);
    expect(mockManagerGet).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// autoStoreFromRun — storage paths
// ---------------------------------------------------------------------------

describe('autoStoreFromRun — storage', () => {
  it('stores an insight for a new ticker', async () => {
    await autoStoreFromRun(
      'What is the revenue of AAPL?',
      'Apple had $394B revenue.',
      [{ tool: 'get_financials', args: { ticker: 'AAPL' }, result: 'revenue: $394B EPS: $6.12' }],
      mockManagerGet,
    );
    expect(mockManagerGet).toHaveBeenCalledTimes(1);
    expect(mockStoreInsight).toHaveBeenCalledTimes(1);
    const call = mockStoreInsight.mock.calls[0]![0] as {
      ticker: string; source: string; tags: string[];
    };
    expect(call.ticker).toBe('AAPL');
    expect(call.source).toBe('auto-run');
    expect(call.tags).toContain('source:auto-run');
    expect(call.tags).toContain('ticker:AAPL');
  });

  it('includes routing tag when routing can be inferred from tool results', async () => {
    await autoStoreFromRun(
      'AAPL stock analysis',
      'Apple stock is up.',
      [{ tool: 'get_financials', args: { ticker: 'AAPL' }, result: 'AAPL price: $178, P/E ratio 28' }],
      mockManagerGet,
    );
    const call = mockStoreInsight.mock.calls[0]![0] as { tags: string[]; routing: string };
    expect(call.tags.some((t: string) => t.startsWith('routing:'))).toBe(true);
    expect(call.routing).toBe('fmp-ok');
  });

  it('includes routing:fmp-premium when result mentions premium', async () => {
    await autoStoreFromRun(
      'TSLA fundamental data',
      'Premium required.',
      [{ tool: 'get_financials', args: { ticker: 'TSLA' }, result: 'TSLA — subscription required for this data' }],
      mockManagerGet,
    );
    const call = mockStoreInsight.mock.calls[0]![0] as { routing: string };
    expect(call.routing).toBe('fmp-premium');
  });

  it('includes routing:web-fallback for web_search results', async () => {
    await autoStoreFromRun(
      'META revenue latest',
      'Meta had $117B revenue.',
      [{ tool: 'web_search', args: {}, result: 'META revenue $117B earnings growth' }],
      mockManagerGet,
    );
    const call = mockStoreInsight.mock.calls[0]![0] as { routing: string };
    expect(call.routing).toBe('web-fallback');
  });

  it('skips tickers that already have a recent record', async () => {
    const recentTime = Date.now() - 60 * 60 * 1000;
    mockRecallByTicker.mockImplementation(() => [{ updatedAt: recentTime }]);
    await autoStoreFromRun(
      'AAPL price check',
      'AAPL is $178.',
      [{ tool: 'get_market_data', args: {}, result: 'AAPL price $178 EPS 6.1' }],
      mockManagerGet,
    );
    expect(mockStoreInsight).not.toHaveBeenCalled();
  });

  it('stores when existing record is older than 24 hours', async () => {
    const oldTime = Date.now() - 25 * 60 * 60 * 1000;
    mockRecallByTicker.mockImplementation(() => [{ updatedAt: oldTime }]);
    await autoStoreFromRun(
      'AAPL deep dive',
      'AAPL analysis complete.',
      [{ tool: 'get_financials', args: {}, result: 'AAPL revenue $394B P/E 28' }],
      mockManagerGet,
    );
    expect(mockStoreInsight).toHaveBeenCalledTimes(1);
  });

  it('processes multiple tickers from a query, up to 6', async () => {
    await autoStoreFromRun(
      'Compare AAPL MSFT GOOG AMZN META NVDA TSLA performance',
      'All up this quarter.',
      [{ tool: 'get_financials', args: {}, result: 'revenue earnings P/E' }],
      mockManagerGet,
    );
    expect(mockStoreInsight.mock.calls.length).toBeLessThanOrEqual(6);
    expect(mockStoreInsight.mock.calls.length).toBeGreaterThan(0);
  });

  it('does not throw when manager factory rejects', async () => {
    const failGet = mock(async () => { throw new Error('DB error'); }) as () => Promise<never>;
    await expect(
      autoStoreFromRun(
        'AAPL data',
        'Answer.',
        [{ tool: 'get_financials', args: {}, result: 'AAPL price $178 earnings $6' }],
        failGet,
      ),
    ).resolves.toBeUndefined();
  });

  it('truncates long answers to 200 characters in the stored content', async () => {
    const longAnswer = 'x'.repeat(300);
    await autoStoreFromRun(
      'AAPL analysis',
      longAnswer,
      [{ tool: 'get_financials', args: {}, result: 'AAPL revenue $394B P/E 28' }],
      mockManagerGet,
    );
    const call = mockStoreInsight.mock.calls[0]![0] as { content: string };
    expect(call.content).toContain('…');
    expect(call.content.length).toBeLessThan(600);
  });
});
