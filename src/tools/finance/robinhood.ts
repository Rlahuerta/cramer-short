import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { getQuote, getFundamentals } from './robinhood-client.js';

const RobinhoodQuoteInputSchema = z.object({
  ticker: z.string().describe("Stock ticker symbol, e.g. 'AAPL'."),
});

export function makeRobinhoodTools(
  quoteFn: typeof getQuote = getQuote,
  fundamentalsFn: typeof getFundamentals = getFundamentals,
) {
  const getRobinhoodQuote = new DynamicStructuredTool({
    name: 'get_robinhood_quote',
    description:
      "Fetches a real-time quote from Robinhood: last trade price, bid/ask, open/high/low/close, volume, and 52-week range. No API key required. " +
      'Use as a fallback when get_stock_price (Financial Datasets) fails.',
    schema: RobinhoodQuoteInputSchema,
    func: async (input) => {
      const ticker = input.ticker.trim().toUpperCase();
      const quote = await quoteFn(ticker);
      if (!quote) {
        return formatToolResult(
          { error: `Robinhood has no quote data for ${ticker}.` },
          [],
        );
      }
      const data = {
        symbol: quote.symbol,
        lastTradePrice: quote.lastTradePrice,
        bidPrice: quote.bidPrice,
        askPrice: quote.askPrice,
        bidSize: quote.bidSize,
        askSize: quote.askSize,
        volume: quote.volume,
        adjustedPreviousClose: quote.adjustedPreviousClose,
        tradingHalted: quote.tradingHalted,
        previousClose: quote.previousClose,
        high52Week: null,
        low52Week: null,
      };
      return formatToolResult(data, [`https://robinhood.com/stocks/${ticker}`]);
    },
  });

  const getRobinhoodFundamentals = new DynamicStructuredTool({
    name: 'get_robinhood_fundamentals',
    description:
      "Fetches basic fundamental metrics from Robinhood: market cap, P/E ratio, EPS, dividend yield, shares outstanding, and 52-week high/low. No API key required. " +
      'Use as a fallback for basic metrics when other sources fail.',
    schema: RobinhoodQuoteInputSchema,
    func: async (input) => {
      const ticker = input.ticker.trim().toUpperCase();
      const fundamentals = await fundamentalsFn(ticker);
      if (!fundamentals) {
        return formatToolResult(
          { error: `Robinhood has no fundamentals data for ${ticker}.` },
          [],
        );
      }
      const data = {
        symbol: fundamentals.symbol,
        marketCap: fundamentals.marketCap,
        adjustedMarketCap: fundamentals.adjustedMarketCap,
        priceEarningsRatio: fundamentals.priceEarningsRatio,
        earningsPerShare: fundamentals.earningsPerShare,
        dividendsYield: fundamentals.dividendsYield,
        dividendsPerShare: fundamentals.dividendsPerShare,
        sharesOutstanding: fundamentals.sharesOutstanding,
        high52Week: fundamentals.high52Week,
        low52Week: fundamentals.low52Week,
        volume: fundamentals.volume,
        averageVolume: fundamentals.averageVolume,
        description: fundamentals.description,
      };
      return formatToolResult(data, [`https://robinhood.com/stocks/${ticker}`]);
    },
  });

  return { getRobinhoodQuote, getRobinhoodFundamentals };
}

const _tools = makeRobinhoodTools();
export const getRobinhoodQuote = _tools.getRobinhoodQuote;
export const getRobinhoodFundamentals = _tools.getRobinhoodFundamentals;
