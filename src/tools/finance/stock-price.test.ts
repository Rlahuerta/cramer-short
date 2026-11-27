import { describe, test, expect, mock, spyOn, beforeEach, afterEach, afterAll } from 'bun:test';

const mockTavilyInvoke = mock(async (_input: unknown): Promise<string> =>
  JSON.stringify({ data: { result: 'AAPL current price $185' }, sourceUrls: [] }),
);

// Snapshot the real modules so afterAll can undo mock.module (Bun never does);
// otherwise these mocks leak into every test file that runs later.
const realTavilyModule = { ...(await import('../search/tavily.js')) };
const realRobinhoodClientModule = { ...(await import('./robinhood-client.js')) };
afterAll(() => {
  mock.module('../search/tavily.js', () => realTavilyModule);
  mock.module('./robinhood-client.js', () => realRobinhoodClientModule);
});

mock.module('../search/tavily.js', () => ({
  tavilySearch: { invoke: mockTavilyInvoke, name: 'web_search' },
}));

const MOCK_RH_QUOTE = {
  symbol: 'AAPL',
  last_trade_price: '186.00',
  bid_price: '185.95',
  ask_price: '186.05',
  bid_size: 100,
  ask_size: 100,
  last_trade_size: 50,
  last_trade_condition: null,
  updated_at: '2025-01-01T00:00:00Z',
  previous_close: '185.00',
  adjusted_previous_close: '185.00',
  trading_halted: false,
  market_state: 'active',
  volume: 50_000_000,
};

const mockGetQuote = mock(async (_ticker: string) => {
  return MOCK_RH_QUOTE;
});
mock.module('./robinhood-client.js', () => ({
  getQuote: mockGetQuote,
  getFundamentals: mock(async () => null),
}));

import { api } from './api.js';
import { makeGetStockPrice, getStockPrices, getStockTickers } from './stock-price.js';
import { _clearCrumbCache } from './yahoo-client.js';

const getStockPrice = makeGetStockPrice(mockGetQuote);

function parseResult(raw: unknown): { data: unknown; sourceUrls?: string[] } {
  return JSON.parse(raw as string);
}

const MOCK_SNAPSHOT = {
  ticker: 'AAPL',
  price: 185.50,
  open: 184.00,
  high: 187.20,
  low: 183.50,
  close: 185.50,
  volume: 55_000_000,
  market_cap: 2_800_000_000_000,
};

describe('getStockPrice', () => {
  beforeEach(() => {
    mockTavilyInvoke.mockClear();
    mockGetQuote.mockClear();
    delete process.env.TAVILY_API_KEY;
  });

  test('tool name is get_stock_price', () => {
    expect(getStockPrice.name).toBe('get_stock_price');
  });

  test('returns snapshot from API', async () => {
    spyOn(api, 'get').mockResolvedValue({
      data: { snapshot: MOCK_SNAPSHOT },
      url: 'https://api.financialdatasets.ai/prices/snapshot/',
    });
    const result = await getStockPrice.invoke({ ticker: 'AAPL' });
    const parsed = parseResult(result);
    expect((parsed.data as typeof MOCK_SNAPSHOT).price).toBe(185.50);
    expect((parsed.data as typeof MOCK_SNAPSHOT).market_cap).toBe(2_800_000_000_000);
  });

  test('normalizes ticker to uppercase', async () => {
    const spy = spyOn(api, 'get').mockResolvedValue({
      data: { snapshot: MOCK_SNAPSHOT },
      url: 'https://api.financialdatasets.ai/prices/snapshot/',
    });
    await getStockPrice.invoke({ ticker: 'aapl' });
    expect(spy).toHaveBeenCalledWith('/prices/snapshot/', { ticker: 'AAPL' });
  });

  test('returns empty object when snapshot is missing', async () => {
    spyOn(api, 'get').mockResolvedValue({
      data: {},
      url: 'https://api.financialdatasets.ai/prices/snapshot/',
    });
    const result = await getStockPrice.invoke({ ticker: 'AAPL' });
    const parsed = parseResult(result);
    expect(parsed.data).toEqual({});
  });

  test('falls back to Robinhood when API throws', async () => {
    spyOn(api, 'get').mockRejectedValue(new Error('API unavailable'));
    const result = await getStockPrice.invoke({ ticker: 'AAPL' });
    expect(mockGetQuote).toHaveBeenCalledWith('AAPL');
    const parsed = parseResult(result);
    expect((parsed.data as { lastTradePrice: string }).lastTradePrice).toBe('186.00');
  });

  test('falls back to Tavily when API throws and Robinhood returns null', async () => {
    process.env.TAVILY_API_KEY = 'test-tavily-key';
    spyOn(api, 'get').mockRejectedValue(new Error('API unavailable'));
    mockGetQuote.mockResolvedValueOnce(null as unknown as typeof MOCK_RH_QUOTE);
    mockTavilyInvoke.mockResolvedValueOnce(
      JSON.stringify({ data: { result: 'AAPL: $185.50' }, sourceUrls: [] }),
    );

    const result = await getStockPrice.invoke({ ticker: 'AAPL' });
    expect(mockTavilyInvoke).toHaveBeenCalled();
    expect(result).toBeDefined();
  });

  test('returns structured error when API and Robinhood both fail and no Tavily key', async () => {
    delete process.env.TAVILY_API_KEY;
    spyOn(api, 'get').mockRejectedValue(new Error('API unavailable'));
    mockGetQuote.mockResolvedValueOnce(null as unknown as typeof MOCK_RH_QUOTE);

    const result = await getStockPrice.invoke({ ticker: 'AAPL' });
    const parsed = parseResult(result);
    expect((parsed.data as { error: string }).error).toContain('AAPL');
    expect(mockTavilyInvoke).not.toHaveBeenCalled();
  });

  test('returns structured error when API, Robinhood, and Tavily all fail', async () => {
    process.env.TAVILY_API_KEY = 'test-tavily-key';
    spyOn(api, 'get').mockRejectedValue(new Error('API unavailable'));
    mockGetQuote.mockResolvedValueOnce(null as unknown as typeof MOCK_RH_QUOTE);
    mockTavilyInvoke.mockRejectedValueOnce(new Error('Tavily unavailable'));

    const result = await getStockPrice.invoke({ ticker: 'AAPL' });
    const parsed = parseResult(result);
    expect((parsed.data as { error: string }).error).toContain('AAPL');
  });
});

describe('getStockPrices', () => {
  test('tool name is get_stock_prices', () => {
    expect(getStockPrices.name).toBe('get_stock_prices');
  });

  test('returns price array from API', async () => {
    const prices = [
      { date: '2025-01-01', open: 180, close: 185 },
      { date: '2025-01-02', open: 185, close: 188 },
    ];
    spyOn(api, 'get').mockResolvedValue({
      data: { prices },
      url: 'https://api.financialdatasets.ai/prices/',
    });
    const result = await getStockPrices.invoke({
      ticker: 'AAPL',
      interval: 'day',
      start_date: '2025-01-01',
      end_date: '2025-01-31',
    });
    const parsed = parseResult(result);
    expect(Array.isArray(parsed.data)).toBe(true);
    expect((parsed.data as typeof prices).length).toBe(2);
  });

  test('returns empty array when prices is missing', async () => {
    spyOn(api, 'get').mockResolvedValue({
      data: {},
      url: 'https://api.financialdatasets.ai/prices/',
    });
    const result = await getStockPrices.invoke({
      ticker: 'AAPL',
      interval: 'day',
      start_date: '2025-01-01',
      end_date: '2025-01-31',
    });
    const parsed = parseResult(result);
    expect(parsed.data).toEqual([]);
  });

  test('passes cacheable=true for fully closed past date range', async () => {
    const spy = spyOn(api, 'get').mockResolvedValue({
      data: { prices: [] },
      url: 'https://api.financialdatasets.ai/prices/',
    });
    spy.mockClear();
    await getStockPrices.invoke({
      ticker: 'AAPL',
      interval: 'day',
      start_date: '2020-01-01',
      end_date: '2020-12-31',
    });
    const opts = spy.mock.calls[0][2] as { cacheable: boolean } | undefined;
    expect(opts?.cacheable).toBe(true);
  });

  describe('when Financial Datasets refuses the request', () => {
    const premiumError = new Error('FINANCIAL_DATASETS_PREMIUM_REQUIRED: This endpoint requires a paid plan.');
    let chartUrl = '';
    let fetchSpy: ReturnType<typeof spyOn<typeof globalThis, 'fetch'>>;

    beforeEach(() => {
      _clearCrumbCache();
      chartUrl = '';
      spyOn(api, 'get').mockRejectedValue(premiumError);
    });

    afterEach(() => fetchSpy.mockRestore());

    function stubYahoo(chart: () => Response) {
      fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(Object.assign(
        async (input: Parameters<typeof fetch>[0]) => {
          const url = input.toString();
          if (url.includes('fc.yahoo.com')) return new Response('ok', { headers: { 'set-cookie': 'A1=s' } });
          if (url.includes('getcrumb')) return new Response('crumb-1');
          chartUrl = url;
          return chart();
        },
        { preconnect: globalThis.fetch.preconnect },
      ));
    }

    test('falls back to Yahoo chart closes for the requested window', async () => {
      stubYahoo(() => Response.json({
        chart: {
          result: [{
            meta: { gmtoffset: -18_000 },
            timestamp: [Date.UTC(2025, 0, 2, 14, 30) / 1000, Date.UTC(2025, 0, 3, 14, 30) / 1000],
            indicators: { quote: [{ close: [243.85, 243.36] }] },
          }],
        },
      }));

      const parsed = parseResult(await getStockPrices.invoke({
        ticker: 'aapl',
        interval: 'day',
        start_date: '2025-01-02',
        end_date: '2025-01-03',
      }));

      expect(parsed.data).toEqual([
        { date: '2025-01-02', close: 243.85 },
        { date: '2025-01-03', close: 243.36 },
      ]);
      const params = new URL(chartUrl).searchParams;
      expect(new URL(chartUrl).pathname).toBe('/v8/finance/chart/AAPL');
      expect(params.get('period1')).toBe(String(Date.UTC(2025, 0, 2) / 1000));
      expect(params.get('period2')).toBe(String(Date.UTC(2025, 0, 4) / 1000));
      expect(params.get('interval')).toBe('1d');
    });

    test('reports both failures when the Yahoo fallback also fails', async () => {
      stubYahoo(() => new Response('Too Many Requests', { status: 429, statusText: 'Too Many Requests' }));

      await expect(getStockPrices.invoke({
        ticker: 'AAPL',
        interval: 'week',
        start_date: '2025-01-01',
        end_date: '2025-03-31',
      })).rejects.toThrow(/FINANCIAL_DATASETS_PREMIUM_REQUIRED.*Yahoo Finance chart request failed for AAPL: 429/);
      expect(new URL(chartUrl).searchParams.get('interval')).toBe('1wk');
    });
  });
});

describe('getStockTickers', () => {
  test('tool name is get_available_stock_tickers', () => {
    expect(getStockTickers.name).toBe('get_available_stock_tickers');
  });

  test('returns tickers array', async () => {
    spyOn(api, 'get').mockResolvedValue({
      data: { tickers: ['AAPL', 'MSFT', 'GOOGL'] },
      url: 'https://api.financialdatasets.ai/prices/snapshot/tickers/',
    });
    const result = await getStockTickers.invoke({});
    const parsed = parseResult(result);
    expect(parsed.data).toEqual(['AAPL', 'MSFT', 'GOOGL']);
  });

  test('returns empty array when tickers is missing', async () => {
    spyOn(api, 'get').mockResolvedValue({
      data: {},
      url: 'https://api.financialdatasets.ai/prices/snapshot/tickers/',
    });
    const result = await getStockTickers.invoke({});
    const parsed = parseResult(result);
    expect(parsed.data).toEqual([]);
  });
});
