import { describe, test, expect, mock, spyOn, beforeEach } from 'bun:test';

const mockTavilyInvoke = mock(async (_input: unknown): Promise<string> =>
  JSON.stringify({ data: { result: 'AAPL current price $185' }, sourceUrls: [] }),
);
mock.module('../search/tavily.js', () => ({
  tavilySearch: { invoke: mockTavilyInvoke, name: 'web_search' },
}));

const MOCK_RH_QUOTE = {
  symbol: 'AAPL',
  lastTradePrice: '186.00',
  bidPrice: '185.95',
  askPrice: '186.05',
  bidSize: 100,
  askSize: 100,
  lastTradeSize: 50,
  lastTradeCondition: null,
  lastUpdatedAt: '2025-01-01T00:00:00Z',
  previousClose: '185.00',
  adjustedPreviousClose: '185.00',
  tradingHalted: false,
  marketState: 'active',
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
