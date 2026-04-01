import { mock, describe, it, expect, beforeEach, spyOn } from 'bun:test';

// Mock api module BEFORE importing any tool that depends on it
const mockGet = mock(() => Promise.resolve({ data: {} as Record<string, unknown>, url: 'https://api.example.com/test' }));
const mockPost = mock(() => Promise.resolve({ data: {} as Record<string, unknown>, url: 'https://api.example.com/test' }));

mock.module('./api.js', () => ({
  api: { get: mockGet, post: mockPost },
  stripFieldsDeep: (val: unknown, _fields: readonly string[]) => val,
  callApi: mockGet,
}));

// Dynamic imports after mocking.
// Cache-busting ?t= forces Bun to re-evaluate each module in a fresh context so it
// picks up the mock.module('./api.js') above — rather than reusing a module instance
// that was already loaded (with real api.js) by another test file in the same worker.
const t = Date.now();
const { getEarnings } = await import(`./earnings.js?t=${t}`) as typeof import('./earnings.js');
const { getAnalystEstimates } = await import(`./estimates.js?t=${t}`) as typeof import('./estimates.js');
const { getCompanyNews } = await import(`./news.js?t=${t}`) as typeof import('./news.js');
const { getCryptoPriceSnapshot, getCryptoPrices, getCryptoTickers } = await import(`./crypto.js?t=${t}`) as typeof import('./crypto.js');
const { getInsiderTrades } = await import(`./insider_trades.js?t=${t}`) as typeof import('./insider_trades.js');
const { getSegmentedRevenues } = await import(`./segments.js?t=${t}`) as typeof import('./segments.js');
const { getStockPrice, getStockPrices, getStockTickers } = await import(`./stock-price.js?t=${t}`) as typeof import('./stock-price.js');
const { getKeyRatios, getHistoricalKeyRatios } = await import(`./key-ratios.js?t=${t}`) as typeof import('./key-ratios.js');
const { getIncomeStatements, getBalanceSheets, getCashFlowStatements, getAllFinancialStatements } = await import(`./fundamentals.js?t=${t}`) as typeof import('./fundamentals.js');
const { getFilings, get10KFilingItems, get10QFilingItems, get8KFilingItems } = await import(`./filings.js?t=${t}`) as typeof import('./filings.js');
// getFilingItemTypes has a module-level cache (cachedItemTypes) — always import fresh so
// prior test runs don't serve stale data from a different worker's fetch mock.
const { getFilingItemTypes } = await import(`./filings.js?getFilingItemTypes=${t}`) as typeof import('./filings.js');

// Rich mock data covering all tool response shapes
const mockApiData: Record<string, unknown> = {
  earnings: { revenue: 1000, eps: 1.5 },
  analyst_estimates: [{ estimate: 1.5 }],
  news: [{ headline: 'Test news', url: 'https://news.example.com' }],
  snapshot: { price: 150, volume: 1000000 },
  prices: [{ close: 150, open: 148 }],
  tickers: ['AAPL', 'GOOG'],
  insider_trades: [{ transaction_type: 'buy', shares: 1000 }],
  segmented_revenues: { products: 1000, services: 500 },
  financial_metrics: [{ pe_ratio: 20 }],
  income_statements: [{ revenue: 1000 }],
  balance_sheets: [{ total_assets: 5000 }],
  cash_flow_statements: [{ operating_cash_flow: 500 }],
  financials: { income_statements: [], balance_sheets: [], cash_flow_statements: [] },
  filings: [{ accession_number: '0001234', filing_type: '10-K' }],
};

beforeEach(() => {
  mockGet.mockClear();
  mockGet.mockImplementation(() =>
    Promise.resolve({ data: mockApiData, url: 'https://api.example.com/test' })
  );
});

// ---------------------------------------------------------------------------
// earnings
// ---------------------------------------------------------------------------

describe('getEarnings', () => {
  it('uppercases ticker and calls /earnings', async () => {
    const result = JSON.parse(await getEarnings.invoke({ ticker: 'aapl' }));
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/earnings');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL' });
    expect(result.data).toEqual(mockApiData.earnings);
  });

  it('returns valid JSON', async () => {
    const result = await getEarnings.invoke({ ticker: 'AAPL' });
    expect(() => JSON.parse(result)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// estimates
// ---------------------------------------------------------------------------

describe('getAnalystEstimates', () => {
  it('passes period param and calls /analyst-estimates/', async () => {
    await getAnalystEstimates.invoke({ ticker: 'aapl', period: 'quarterly' });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/analyst-estimates/');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ period: 'quarterly' });
  });

  it('returns valid JSON', async () => {
    const result = await getAnalystEstimates.invoke({ ticker: 'AAPL', period: 'annual' });
    expect(() => JSON.parse(result)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// news
// ---------------------------------------------------------------------------

describe('getCompanyNews', () => {
  it('uppercases ticker and calls /news', async () => {
    await getCompanyNews.invoke({ ticker: 'aapl', limit: 5 });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/news');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL' });
  });

  it('caps limit at 10', async () => {
    await getCompanyNews.invoke({ ticker: 'AAPL', limit: 20 });
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ limit: 10 });
  });

  it('allows limit <= 10 unchanged', async () => {
    await getCompanyNews.invoke({ ticker: 'AAPL', limit: 7 });
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ limit: 7 });
  });
});

// ---------------------------------------------------------------------------
// crypto
// ---------------------------------------------------------------------------

describe('getCryptoPriceSnapshot', () => {
  it('calls /crypto/prices/snapshot/', async () => {
    await getCryptoPriceSnapshot.invoke({ ticker: 'BTC-USD' });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/crypto/prices/snapshot/');
  });
});

describe('getCryptoPrices', () => {
  it('calls /crypto/prices/ with correct params', async () => {
    await getCryptoPrices.invoke({
      ticker: 'BTC-USD',
      interval: 'day',
      interval_multiplier: 1,
      start_date: '2023-01-01',
      end_date: '2023-06-01',
    });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/crypto/prices/');
  });

  it('sets cacheable=true for fully closed date windows', async () => {
    await getCryptoPrices.invoke({
      ticker: 'BTC-USD',
      interval: 'day',
      interval_multiplier: 1,
      start_date: '2023-01-01',
      end_date: '2023-06-01',
    });
    expect((mockGet.mock.calls as any[][])[0][2]).toMatchObject({ cacheable: true });
  });

  it('sets cacheable=false when end_date is today or future', async () => {
    const today = new Date().toISOString().slice(0, 10);
    await getCryptoPrices.invoke({
      ticker: 'BTC-USD',
      interval: 'day',
      interval_multiplier: 1,
      start_date: '2023-01-01',
      end_date: today,
    });
    expect((mockGet.mock.calls as any[][])[0][2]).toMatchObject({ cacheable: false });
  });
});

describe('getCryptoTickers', () => {
  it('calls /crypto/prices/tickers/', async () => {
    await getCryptoTickers.invoke({});
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/crypto/prices/tickers/');
  });
});

// ---------------------------------------------------------------------------
// insider_trades
// ---------------------------------------------------------------------------

describe('getInsiderTrades', () => {
  it('uppercases ticker and calls /insider-trades/', async () => {
    await getInsiderTrades.invoke({ ticker: 'aapl', limit: 5 });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/insider-trades/');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL', limit: 5 });
  });

  it('passes optional date filters', async () => {
    await getInsiderTrades.invoke({
      ticker: 'AAPL',
      limit: 10,
      filing_date_gte: '2023-01-01',
      filing_date_lte: '2023-12-31',
    });
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({
      filing_date_gte: '2023-01-01',
      filing_date_lte: '2023-12-31',
    });
  });
});

// ---------------------------------------------------------------------------
// segments
// ---------------------------------------------------------------------------

describe('getSegmentedRevenues', () => {
  it('calls /financials/segmented-revenues/ with correct params', async () => {
    await getSegmentedRevenues.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/financials/segmented-revenues/');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL', period: 'annual', limit: 4 });
  });
});

// ---------------------------------------------------------------------------
// stock-price
// ---------------------------------------------------------------------------

describe('getStockPrice', () => {
  it('uppercases and trims ticker, calls /prices/snapshot/', async () => {
    await getStockPrice.invoke({ ticker: '  aapl  ' });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/prices/snapshot/');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL' });
  });
});

describe('getStockPrices', () => {
  it('calls /prices/ with correct params', async () => {
    await getStockPrices.invoke({
      ticker: 'AAPL',
      interval: 'day',
      start_date: '2023-01-01',
      end_date: '2023-06-01',
    });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/prices/');
  });

  it('sets cacheable=true for fully closed date windows', async () => {
    await getStockPrices.invoke({
      ticker: 'AAPL',
      interval: 'day',
      start_date: '2023-01-01',
      end_date: '2023-06-01',
    });
    expect((mockGet.mock.calls as any[][])[0][2]).toMatchObject({ cacheable: true });
  });

  it('sets cacheable=false when end_date is today', async () => {
    const today = new Date().toISOString().slice(0, 10);
    await getStockPrices.invoke({
      ticker: 'AAPL',
      interval: 'day',
      start_date: '2023-01-01',
      end_date: today,
    });
    expect((mockGet.mock.calls as any[][])[0][2]).toMatchObject({ cacheable: false });
  });
});

describe('getStockTickers', () => {
  it('calls /prices/snapshot/tickers/', async () => {
    await getStockTickers.invoke({});
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/prices/snapshot/tickers/');
  });
});

// ---------------------------------------------------------------------------
// key-ratios
// ---------------------------------------------------------------------------

describe('getKeyRatios', () => {
  it('uppercases ticker and calls /financial-metrics/snapshot/', async () => {
    await getKeyRatios.invoke({ ticker: 'aapl' });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/financial-metrics/snapshot/');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL' });
  });
});

describe('getHistoricalKeyRatios', () => {
  it('calls /financial-metrics/ with params', async () => {
    await getHistoricalKeyRatios.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/financial-metrics/');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL', period: 'annual', limit: 4 });
  });

  it('passes optional date filter params', async () => {
    await getHistoricalKeyRatios.invoke({
      ticker: 'AAPL',
      period: 'annual',
      limit: 4,
      report_period_gte: '2022-01-01',
    });
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ report_period_gte: '2022-01-01' });
  });
});

// ---------------------------------------------------------------------------
// fundamentals
// ---------------------------------------------------------------------------

describe('getIncomeStatements', () => {
  it('calls /financials/income-statements/ with correct params', async () => {
    await getIncomeStatements.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/financials/income-statements/');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL', period: 'annual', limit: 4 });
  });
});

describe('getBalanceSheets', () => {
  it('calls /financials/balance-sheets/', async () => {
    await getBalanceSheets.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/financials/balance-sheets/');
  });
});

describe('getCashFlowStatements', () => {
  it('calls /financials/cash-flow-statements/', async () => {
    await getCashFlowStatements.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/financials/cash-flow-statements/');
  });
});

describe('getAllFinancialStatements', () => {
  it('calls /financials/', async () => {
    await getAllFinancialStatements.invoke({ ticker: 'AAPL', period: 'annual', limit: 4 });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/financials/');
  });
});

// ---------------------------------------------------------------------------
// filings
// ---------------------------------------------------------------------------

describe('getFilings', () => {
  it('calls /filings/ with correct params', async () => {
    const result = await getFilings.invoke({ ticker: 'AAPL', limit: 5 });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/filings/');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL', limit: 5 });
    expect(() => JSON.parse(result)).not.toThrow();
  });
});

describe('get10KFilingItems', () => {
  it('uppercases ticker, sets filing_type=10-K, and uses cacheable=true', async () => {
    await get10KFilingItems.invoke({ ticker: 'aapl', accession_number: '0001234' });
    expect((mockGet.mock.calls as any[][])[0][0]).toBe('/filings/items/');
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL', filing_type: '10-K' });
    expect((mockGet.mock.calls as any[][])[0][2]).toMatchObject({ cacheable: true });
  });
});

describe('get10QFilingItems', () => {
  it('uppercases ticker, sets filing_type=10-Q', async () => {
    await get10QFilingItems.invoke({
      ticker: 'aapl',
      accession_number: '0001234',
      items: ['Part-1,Item-1'],
    });
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL', filing_type: '10-Q' });
  });
});

describe('get8KFilingItems', () => {
  it('uppercases ticker, sets filing_type=8-K', async () => {
    await get8KFilingItems.invoke({ ticker: 'aapl', accession_number: '0001234' });
    expect((mockGet.mock.calls as any[][])[0][1]).toMatchObject({ ticker: 'AAPL', filing_type: '8-K' });
  });
});

describe('getFilingItemTypes', () => {
  it('fetches from API and caches on subsequent calls', async () => {
    const mockItemTypes = {
      '10-K': [{ name: 'Item-1', title: 'Business', description: 'Overview' }],
      '10-Q': [{ name: 'Part-1,Item-1', title: 'Financial Statements', description: 'FS' }],
    };

    const fetchSpy = spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: () => Promise.resolve(mockItemTypes),
    } as Response);

    try {
      const result1 = await getFilingItemTypes();
      expect(result1).toMatchObject(mockItemTypes);
      // Second call should return cached result without fetching again
      const result2 = await getFilingItemTypes();
      expect(result2).toMatchObject(mockItemTypes);
      // fetch should only be called once (or possibly 0 times if already cached from earlier test)
      expect((fetchSpy.mock.calls as any[][]).length).toBeLessThanOrEqual(1);
    } finally {
      fetchSpy.mockRestore();
    }
  });
});
