import { describe, test, expect, spyOn, beforeEach, afterEach } from 'bun:test';

const { createPortfolioRiskTool, portfolioRiskTool } = await import('./portfolio-risk.js');
const { api } = await import('./api.js');
const { computeReturns, computeVolatility } = await import('../../utils/finance/portfolio-stats.js');

/** Generate N monotonically increasing close prices */
function makePrices(n: number, start = 100): number[] {
  return Array.from({ length: n }, (_, i) => start + i * 0.5);
}

/** Financial Datasets price rows: N consecutive days of monotonically increasing closes */
function makeRows(n: number) {
  return makePrices(n).map((close, i) => ({ time: new Date(Date.UTC(2025, 0, 2 + i)).toISOString(), close }));
}

let apiSpy: ReturnType<typeof spyOn<typeof api, 'get'>>;
let originalApiKey: string | undefined;

beforeEach(() => {
  originalApiKey = process.env.FINANCIAL_DATASETS_API_KEY;
  process.env.FINANCIAL_DATASETS_API_KEY = 'test-key';
  apiSpy = spyOn(api, 'get').mockResolvedValue({
    data: { prices: makeRows(30) },
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
        return { data: { prices: makeRows(5) }, url: 'https://api.test/' };
      }
      return { data: { prices: makeRows(30) }, url: 'https://api.test/' };
    });

    const result = await portfolioRiskTool.invoke({ tickers: ['TINY', 'AAPL'] });
    const parsed = JSON.parse(result);
    // TINY should produce a warning; AAPL should be in the report
    const warnings = parsed.data.warnings as string[] | undefined;
    expect(warnings?.some((w: string) => w.includes('TINY'))).toBe(true);
  });

  test('returns error when all tickers fail price fetch', async () => {
    apiSpy.mockRejectedValue(new Error('API down'));
    const fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(Object.assign(
      async () => new Response('unavailable', { status: 500 }),
      { preconnect: globalThis.fetch.preconnect },
    ));
    try {
      const result = await portfolioRiskTool.invoke({ tickers: ['AAPL', 'MSFT'] });
      const parsed = JSON.parse(result);
      expect(parsed.data.error).toBeDefined();
    } finally {
      fetchSpy.mockRestore();
    }
  });

  test('falls back to Yahoo closes when the Financial Datasets plan has no price history', async () => {
    apiSpy.mockRejectedValue(new Error(
      'FINANCIAL_DATASETS_PREMIUM_REQUIRED: This endpoint requires a paid Financial Datasets plan.',
    ));
    const yahooUrls: string[] = [];
    const fetchMock: typeof fetch = Object.assign(
      async (input: Parameters<typeof fetch>[0]) => {
        const url = input.toString();
        if (!url.includes('query1.finance.yahoo.com')) return new Response('unavailable', { status: 500 });
        yahooUrls.push(url);
        const close = makePrices(250, url.includes('/MSFT') ? 300 : 100).map((p, i) => p + (i % 3));
        const timestamp = close.map((_, i) => Date.UTC(2025, 0, 2 + i, 14, 30) / 1000);
        return new Response(JSON.stringify({ chart: { result: [{ timestamp, indicators: { quote: [{ close }] } }] } }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      },
      { preconnect: globalThis.fetch.preconnect },
    );
    const fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(fetchMock);
    try {
      const parsed = JSON.parse(await portfolioRiskTool.invoke({ tickers: ['AAPL', 'MSFT'] }));
      expect(parsed.data.error).toBeUndefined();
      expect(parsed.data.warnings).toBeUndefined();
      expect(yahooUrls.some((u) => u.includes('/AAPL'))).toBe(true);
      expect(yahooUrls.some((u) => u.includes('/MSFT'))).toBe(true);
    } finally {
      fetchSpy.mockRestore();
    }
  });

  describe('with Financial Datasets down, crypto from Binance and stocks from Yahoo', () => {
    const DAY_MS = 86_400_000;
    /** One deterministic price path shared by every ticker, keyed by UTC day number. */
    const pathPrice = (day: number) => 100 + 10 * ((Math.sin(day * 12.9898) * 43_758.5453) % 1);
    const servedBinanceCloses: Record<string, number[]> = {};
    let fetchSpy: ReturnType<typeof spyOn<typeof globalThis, 'fetch'>>;

    beforeEach(() => {
      apiSpy.mockRejectedValue(new Error(
        'FINANCIAL_DATASETS_PREMIUM_REQUIRED: This endpoint requires a paid Financial Datasets plan.',
      ));
      for (const key of Object.keys(servedBinanceCloses)) delete servedBinanceCloses[key];
      fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(Object.assign(
        async (input: Parameters<typeof fetch>[0]) => {
          const url = new URL(input.toString());
          const params = url.searchParams;
          if (url.hostname === 'api.binance.com') {
            // Calendar-day klines, oldest first from startTime, as Binance serves them.
            const symbol = params.get('symbol') ?? '';
            if (!['BTCUSDT', 'ETHUSDT'].includes(symbol)) return new Response('Invalid symbol', { status: 400 });
            const rows = [];
            for (
              let day = Math.ceil(Number(params.get('startTime')) / DAY_MS);
              day * DAY_MS <= Number(params.get('endTime')) && rows.length < Number(params.get('limit'));
              day++
            ) {
              const close = pathPrice(day) * (symbol === 'ETHUSDT' ? 0.05 : 1);
              (servedBinanceCloses[symbol] ??= []).push(close);
              rows.push([day * DAY_MS, '1', '1', '1', String(close), '1', (day + 1) * DAY_MS - 1, '0', 0, '0', '0', '0']);
            }
            return Response.json(rows);
          }
          if (url.hostname === 'fc.yahoo.com') return new Response('ok', { headers: { 'set-cookie': 'A1=s' } });
          if (url.pathname.includes('getcrumb')) return new Response('crumb-1');
          if (url.pathname.startsWith('/v8/finance/chart/')) {
            // Weekday bars stamped at the 09:30 New York open.
            const endS = Number(params.get('period2') ?? Date.now() / 1000);
            const startS = Number(params.get('period1') ?? endS - 365 * 86_400);
            const days = [];
            for (let day = Math.ceil(startS / 86_400); day * 86_400 <= endS; day++) {
              if (![0, 6].includes(new Date(day * DAY_MS).getUTCDay())) days.push(day);
            }
            return Response.json({
              chart: {
                result: [{
                  meta: { gmtoffset: -18_000 },
                  timestamp: days.map((day) => day * 86_400 + 14.5 * 3_600),
                  indicators: { quote: [{ close: days.map(pathPrice) }] },
                }],
              },
            });
          }
          return new Response('unavailable', { status: 500 });
        },
        { preconnect: globalThis.fetch.preconnect },
      ));
    });

    afterEach(() => fetchSpy.mockRestore());

    test('pairs returns by date, so identical paths correlate fully', async () => {
      const parsed = JSON.parse(await portfolioRiskTool.invoke({ tickers: ['AAPL', 'BTC-USD'] }));

      expect(parsed.data.error).toBeUndefined();
      expect(parsed.data.correlation.AAPL['BTC-USD']).toBeGreaterThan(0.999);
    });

    test('annualises an all-crypto portfolio over 365 calendar days', async () => {
      const parsed = JSON.parse(await portfolioRiskTool.invoke({ tickers: ['BTC-USD', 'ETH-USD'] }));

      const expected = computeVolatility(computeReturns(servedBinanceCloses.BTCUSDT), 365);
      expect(parsed.data.volatility['BTC-USD']).toBeCloseTo(expected, 10);
    });
  });

  test('includes source URLs from price API calls', async () => {
    apiSpy.mockResolvedValue({
      data: { prices: makeRows(30) },
      url: 'https://api.test/prices/AAPL',
    });
    const result = await portfolioRiskTool.invoke({ tickers: ['AAPL'] });
    const parsed = JSON.parse(result);
    expect(Array.isArray(parsed.sourceUrls)).toBe(true);
    expect(parsed.sourceUrls.length).toBeGreaterThan(0);
  });
});
