import { describe, test, expect, mock, spyOn, beforeEach, afterEach, afterAll } from 'bun:test';
import { AIMessage } from '@langchain/core/messages';

// Capture real modules before mocking so afterAll can restore
const realLlm = await import('../../model/llm.js');
const realDate = await import('../../utils/date.js');

const mockCallLlm = mock(async () => ({
  response: new AIMessage({ content: '', tool_calls: [] }),
  usage: undefined,
}));

mock.module('../../model/llm.js', () => ({
  ...realLlm,
  callLlm: mockCallLlm,
}));

mock.module('../../utils/date.js', () => ({
  getCurrentDate: () => '2024-01-15',
}));

afterAll(() => {
  mock.module('../../model/llm.js', () => realLlm);
  mock.module('../../utils/date.js', () => realDate);
});

const { createScreenStocks } = await import('./screen-stocks.js');
const { api } = await import('./api.js');

const MOCK_FILTERS = {
  market_cap: { type: 'number', description: 'Market cap in USD' },
  pe_ratio: { type: 'number', description: 'P/E ratio' },
  sector: { type: 'string', description: 'Sector' },
};

const MOCK_SCREENER_RESULTS = [
  { ticker: 'AAPL', market_cap: 2e12, pe_ratio: 28 },
  { ticker: 'MSFT', market_cap: 1.8e12, pe_ratio: 30 },
];

let apiGetSpy: ReturnType<typeof spyOn<typeof api, 'get'>>;
let apiPostSpy: ReturnType<typeof spyOn<typeof api, 'post'>>;

beforeEach(() => {
  mockCallLlm.mockClear();
  apiGetSpy = spyOn(api, 'get').mockImplementation(async (url: string) => {
    if (String(url).includes('filters')) {
      return { data: MOCK_FILTERS, url: 'https://api.test/filters/' };
    }
    return { data: {}, url: 'https://api.test/' };
  });
  apiPostSpy = spyOn(api, 'post').mockResolvedValue({
    data: MOCK_SCREENER_RESULTS as unknown as Record<string, unknown>,
    url: 'https://api.test/screener/',
  });
});

afterEach(() => {
  apiGetSpy.mockRestore();
  apiPostSpy.mockRestore();
});

describe('createScreenStocks', () => {
  test('returns a tool with name stock_screener', () => {
    const tool = createScreenStocks('test-model');
    expect(tool.name).toBe('stock_screener');
  });
});

describe('screen_stocks — happy path', () => {
  test('returns screener results when LLM produces valid filters', async () => {
    mockCallLlm.mockResolvedValueOnce({
      response: {
        filters: [{ field: 'market_cap', operator: 'gt', value: 1e12 }],
        currency: 'USD',
        limit: 10,
      } as any,
      usage: undefined,
    });

    const tool = createScreenStocks('test-model');
    const result = await tool.invoke({ query: 'large cap stocks with market cap over 1 trillion' });
    const parsed = JSON.parse(result);
    expect(parsed.data).toBeDefined();
  });

  test('calls api.post with filters from LLM response', async () => {
    const filters = [{ field: 'pe_ratio', operator: 'lt', value: 15 }];
    mockCallLlm.mockResolvedValueOnce({
      response: { filters, currency: 'USD', limit: 5 } as any,
      usage: undefined,
    });

    const tool = createScreenStocks('test-model');
    await tool.invoke({ query: 'value stocks with low PE' });

    expect(apiPostSpy.mock.calls.length).toBeGreaterThan(0);
  });
});

describe('screen_stocks — error paths', () => {
  test('returns error when screener API call fails', async () => {
    mockCallLlm.mockResolvedValueOnce({
      response: { filters: [], currency: 'USD', limit: 10 } as any,
      usage: undefined,
    });
    apiPostSpy.mockRejectedValueOnce(new Error('API error'));

    const tool = createScreenStocks('test-model');
    const result = await tool.invoke({ query: 'growth stocks' });
    const parsed = JSON.parse(result);
    expect(parsed.data.error).toBeDefined();
  });

  test('returns error when LLM returns invalid filter schema', async () => {
    mockCallLlm.mockResolvedValueOnce({
      response: null as any, // Invalid schema — Zod parse will fail
      usage: undefined,
    });

    const tool = createScreenStocks('test-model');
    const result = await tool.invoke({ query: 'stocks with revenue growth over 20%' });
    const parsed = JSON.parse(result);
    expect(parsed.data.error).toBeDefined();
  });
});
