import { describe, test, expect, spyOn, beforeEach, afterEach } from 'bun:test';

const { createPortfolioRiskTool, portfolioRiskTool } = await import('./portfolio-risk.js');
const { api } = await import('./api.js');

/** Generate N monotonically increasing close prices */
function makePrices(n: number, start = 100): number[] {
  return Array.from({ length: n }, (_, i) => start + i * 0.5);
}

let apiSpy: ReturnType<typeof spyOn<typeof api, 'get'>>;
let originalApiKey: string | undefined;

beforeEach(() => {
  originalApiKey = process.env.FINANCIAL_DATASETS_API_KEY;
  process.env.FINANCIAL_DATASETS_API_KEY = 'test-key';
  apiSpy = spyOn(api, 'get').mockResolvedValue({
    data: { prices: makePrices(30).map((close) => ({ close })) },
    url: 'https://api.test/prices/',
  });
});

afterEach(() => {
  apiSpy.mockRestore();
  if (originalApiKey === undefined) {
    delete process.env.FINANCIAL_DATASETS_API_KEY;
  } else {
    process.env.FINANCIAL_DATASETS_API_KEY = originalApiKey;
  }
});

describe('portfolioRiskTool', () => {
  test('returns error when FINANCIAL_DATASETS_API_KEY is not set', async () => {
    delete process.env.FINANCIAL_DATASETS_API_KEY;
    const result = await portfolioRiskTool.invoke({ tickers: ['AAPL'] });
    const parsed = JSON.parse(result);
    expect(parsed.data.error).toBeDefined();
    expect(parsed.data.error.toLowerCase()).toContain('api_key');
  });

  test('returns risk report for explicit tickers', async () => {
    const result = await portfolioRiskTool.invoke({ tickers: ['AAPL', 'MSFT'] });
    const parsed = JSON.parse(result);
    expect(parsed.data).toBeDefined();
    expect(parsed.data.error).toBeUndefined();
  });

  test('uses watchlist entries supplied by caller when tickers are omitted', async () => {
    const result = await portfolioRiskTool.invoke({
      watchlist_entries: [
        { ticker: 'AAPL', addedAt: '2026-01-01' },
        { ticker: 'TSLA', shares: 2, costBasis: 200, addedAt: '2026-01-01' },
      ],
    });
    const parsed = JSON.parse(result);
    expect(parsed.data).toBeDefined();
    expect(parsed.data.error).toBeUndefined();
    expect(apiSpy).toHaveBeenCalledTimes(2);
  });

  test('uses controller-supplied watchlist entries when args omit tickers', async () => {
    const injectedTool = createPortfolioRiskTool({
      watchlistEntries: [
        { ticker: 'AAPL', addedAt: '2026-01-01' },
        { ticker: 'MSFT', addedAt: '2026-01-01' },
      ],
    });

    const result = await injectedTool.invoke({});
    const parsed = JSON.parse(result);
    expect(parsed.data.error).toBeUndefined();
    expect(apiSpy).toHaveBeenCalledTimes(2);
  });

  test('returns error when neither tickers nor watchlist entries are provided', async () => {
    const result = await portfolioRiskTool.invoke({});
    const parsed = JSON.parse(result);
    expect(parsed.data.error).toBeDefined();
    expect(parsed.data.error).toContain('No tickers provided');
  });

  test('returns warning for ticker with insufficient price history', async () => {
    // Return fewer than 20 prices for TINY
    apiSpy.mockImplementation(async (url, params) => {
      const ticker = (params as Record<string, string>).ticker;
      if (ticker === 'TINY') {
        return { data: { prices: makePrices(5).map((c) => ({ close: c })) }, url: 'https://api.test/' };
      }
      return { data: { prices: makePrices(30).map((c) => ({ close: c })) }, url: 'https://api.test/' };
    });

    const result = await portfolioRiskTool.invoke({ tickers: ['TINY', 'AAPL'] });
    const parsed = JSON.parse(result);
    // TINY should produce a warning; AAPL should be in the report
    const warnings = parsed.data.warnings as string[] | undefined;
    expect(warnings?.some((w: string) => w.includes('TINY'))).toBe(true);
  });

  test('returns error when all tickers fail price fetch', async () => {
    apiSpy.mockRejectedValue(new Error('API down'));
    const result = await portfolioRiskTool.invoke({ tickers: ['AAPL', 'MSFT'] });
    const parsed = JSON.parse(result);
    expect(parsed.data.error).toBeDefined();
  });

  test('includes source URLs from price API calls', async () => {
    apiSpy.mockResolvedValue({
      data: { prices: makePrices(30).map((c) => ({ close: c })) },
      url: 'https://api.test/prices/AAPL',
    });
    const result = await portfolioRiskTool.invoke({ tickers: ['AAPL'] });
    const parsed = JSON.parse(result);
    expect(Array.isArray(parsed.sourceUrls)).toBe(true);
    expect(parsed.sourceUrls.length).toBeGreaterThan(0);
  });
});
