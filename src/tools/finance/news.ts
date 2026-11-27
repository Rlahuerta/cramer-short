import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { api } from './api.js';
import { formatToolResult } from '../types.js';
import { tavilySearch } from '../search/tavily.js';
import { hasEnv } from '../../utils/env.js';

const CompanyNewsInputSchema = z.object({
  ticker: z
    .string()
    .max(128)
    .describe("The stock ticker symbol to fetch company news for. For example, 'AAPL' for Apple."),
  limit: z
    .number()
    .default(5)
    .describe('Maximum number of news articles to return (default: 5, max: 10).'),
});

/** Fetches recent company news articles. */
export const getCompanyNews = new DynamicStructuredTool({
  name: 'get_company_news',
  description:
    'Retrieves recent company news headlines for a stock ticker, including title, source, publication date, and URL. Use for company catalysts, price move explanations, press releases, and recent announcements.',
  schema: CompanyNewsInputSchema,
  func: async (input) => {
    const ticker = input.ticker.trim().toUpperCase();
    const params: Record<string, string | number | undefined> = {
      ticker,
      limit: Math.min(input.limit, 10),
    };
    try {
      const { data, url } = await api.get('/news', params);
      return formatToolResult((data.news as unknown[]) || [], [url]);
    } catch {
      if (hasEnv('TAVILY_API_KEY')) {
        try {
          return await tavilySearch.invoke({ query: `${ticker} stock latest company news headlines` });
        } catch {
          // Tavily also failed
        }
      }
      return formatToolResult(
        { error: `Company news unavailable for ${ticker}. Use web_search to find recent headlines.` },
        [],
      );
    }
  },
});
