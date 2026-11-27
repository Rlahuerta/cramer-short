import { describe, test, expect, spyOn, beforeEach, afterEach } from 'bun:test';
import { api } from './api.js';
import { getCompanyNews } from './news.js';
import { tavilySearch } from '../search/tavily.js';

function parseResult(raw: unknown): { data: unknown; sourceUrls?: string[] } {
  return JSON.parse(raw as string);
}

const MOCK_NEWS = [
  { title: 'Apple Reports Record Revenue', source: 'Bloomberg', date: '2025-01-28', url: 'https://bloomberg.com/apple-q1' },
  { title: 'Apple Expands AI Features', source: 'Reuters', date: '2025-01-25', url: 'https://reuters.com/apple-ai' },
];

describe('getCompanyNews', () => {
  beforeEach(() => {
    spyOn(api, 'get').mockResolvedValue({
      data: { news: MOCK_NEWS },
      url: 'https://api.financialdatasets.ai/news',
    });
  });

  test('tool name is get_company_news', () => {
    expect(getCompanyNews.name).toBe('get_company_news');
  });

  test('returns news articles from API', async () => {
    const result = await getCompanyNews.invoke({ ticker: 'AAPL', limit: 5 });
    const parsed = parseResult(result);
    expect(Array.isArray(parsed.data)).toBe(true);
    expect((parsed.data as typeof MOCK_NEWS).length).toBe(2);
    expect((parsed.data as typeof MOCK_NEWS)[0].title).toBe('Apple Reports Record Revenue');
  });

  test('normalizes ticker to uppercase', async () => {
    const spy = spyOn(api, 'get').mockResolvedValue({
      data: { news: MOCK_NEWS },
      url: 'https://api.financialdatasets.ai/news',
    });
    spy.mockClear();
    await getCompanyNews.invoke({ ticker: 'aapl', limit: 5 });
    const call = spy.mock.calls[0][1] as { ticker: string; limit: number };
    expect(call.ticker).toBe('AAPL');
  });

  test('clamps limit to 10', async () => {
    const spy = spyOn(api, 'get').mockResolvedValue({
      data: { news: MOCK_NEWS },
      url: 'https://api.financialdatasets.ai/news',
    });
    spy.mockClear();
    await getCompanyNews.invoke({ ticker: 'AAPL', limit: 50 });
    const call = spy.mock.calls[0][1] as { ticker: string; limit: number };
    expect(call.limit).toBe(10);
  });

  test('passes limit below 10 unchanged', async () => {
    const spy = spyOn(api, 'get').mockResolvedValue({
      data: { news: [] },
      url: 'https://api.financialdatasets.ai/news',
    });
    spy.mockClear();
    await getCompanyNews.invoke({ ticker: 'AAPL', limit: 3 });
    const call = spy.mock.calls[0][1] as { ticker: string; limit: number };
    expect(call.limit).toBe(3);
  });

  test('returns empty array when news is missing', async () => {
    spyOn(api, 'get').mockResolvedValue({
      data: {},
      url: 'https://api.financialdatasets.ai/news',
    });
    const result = await getCompanyNews.invoke({ ticker: 'AAPL', limit: 5 });
    const parsed = parseResult(result);
    expect(parsed.data).toEqual([]);
  });

  test('returns empty array when news is null', async () => {
    spyOn(api, 'get').mockResolvedValue({
      data: { news: null },
      url: 'https://api.financialdatasets.ai/news',
    });
    const result = await getCompanyNews.invoke({ ticker: 'AAPL', limit: 5 });
    const parsed = parseResult(result);
    expect(parsed.data).toEqual([]);
  });

  test('includes sourceUrls in response', async () => {
    const result = await getCompanyNews.invoke({ ticker: 'AAPL', limit: 5 });
    const parsed = parseResult(result);
    expect(parsed.sourceUrls).toContain('https://api.financialdatasets.ai/news');
  });
});

describe('getCompanyNews — Financial Datasets unavailable', () => {
  const originalTavilyKey = process.env.TAVILY_API_KEY;
  beforeEach(() => {
    spyOn(api, 'get').mockRejectedValue(new Error('FINANCIAL_DATASETS_PREMIUM_REQUIRED: 402'));
  });
  afterEach(() => {
    if (originalTavilyKey === undefined) delete process.env.TAVILY_API_KEY;
    else process.env.TAVILY_API_KEY = originalTavilyKey;
  });

  test('falls back to Tavily search when TAVILY_API_KEY is set', async () => {
    process.env.TAVILY_API_KEY = 'test-tavily-key';
    const tavilyOutput = JSON.stringify({ data: { results: [] }, sourceUrls: ['https://news.example/aapl'] });
    const tavilySpy = spyOn(tavilySearch, 'invoke').mockResolvedValue(tavilyOutput);
    try {
      const result = await getCompanyNews.invoke({ ticker: 'aapl', limit: 5 });
      expect(result).toBe(tavilyOutput);
      expect((tavilySpy.mock.calls[0][0] as { query: string }).query).toContain('AAPL');
    } finally {
      tavilySpy.mockRestore();
    }
  });

  test('returns an error pointing to web_search when no Tavily key is set', async () => {
    delete process.env.TAVILY_API_KEY;
    const parsed = parseResult(await getCompanyNews.invoke({ ticker: 'AAPL', limit: 5 }));
    expect((parsed.data as { error: string }).error).toContain('web_search');
  });
});
