import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { api } from './api.js';
import { fetchBinanceDailyCloses, fetchBinanceTicker24h, toBinanceSymbol } from './binance.js';
import { formatToolResult } from '../types.js';

const CryptoPriceSnapshotInputSchema = z.object({
  ticker: z
    .string()
    .describe(
      "The crypto ticker symbol to fetch the price snapshot for. For example, 'BTC-USD' for Bitcoin."
    ),
});

export const getCryptoPriceSnapshot = new DynamicStructuredTool({
  name: 'get_crypto_price_snapshot',
  description: `Fetches the most recent price snapshot for a specific cryptocurrency, including the latest price, trading volume, and other open, high, low, and close price data. Ticker format: use 'CRYPTO-USD' for USD prices (e.g., 'BTC-USD') or 'CRYPTO-CRYPTO' for crypto-to-crypto prices (e.g., 'BTC-ETH' for Bitcoin priced in Ethereum).`,
  schema: CryptoPriceSnapshotInputSchema,
  func: async (input) => {
    try {
      const params = { ticker: input.ticker };
      const { data, url } = await api.get('/crypto/prices/snapshot/', params);
      return formatToolResult(data.snapshot || {}, [url]);
    } catch {
      const snapshot = await fetchBinanceTicker24h(input.ticker);
      if (!snapshot) {
        return formatToolResult({ error: `No crypto snapshot available for ${input.ticker}` });
      }
      return formatToolResult({
        ticker: toBinanceSymbol(input.ticker),
        price: snapshot.price,
        change24h: snapshot.change24h,
        changePercent24h: snapshot.changePercent24h,
        volume24h: snapshot.volume24h,
        source: 'binance',
      }, [`https://api.binance.com/api/v3/ticker/24hr?symbol=${toBinanceSymbol(input.ticker)}`]);
    }
  },
});

const CryptoPricesInputSchema = z.object({
  ticker: z
    .string()
    .describe(
      "The crypto ticker symbol to fetch aggregated prices for. For example, 'BTC-USD' for Bitcoin."
    ),
  interval: z
    .enum(['minute', 'day', 'week', 'month', 'year'])
    .default('day')
    .describe("The time interval for price data. Defaults to 'day'."),
  interval_multiplier: z
    .number()
    .default(1)
    .describe('Multiplier for the interval. Defaults to 1.'),
  start_date: z.string().describe('Start date in YYYY-MM-DD format. Required.'),
  end_date: z.string().describe('End date in YYYY-MM-DD format. Required.'),
});

export const getCryptoPrices = new DynamicStructuredTool({
  name: 'get_crypto_prices',
  description: `Retrieves historical price data for a cryptocurrency over a specified date range, including open, high, low, close prices, and volume. Ticker format: use 'CRYPTO-USD' for USD prices (e.g., 'BTC-USD') or 'CRYPTO-CRYPTO' for crypto-to-crypto prices (e.g., 'BTC-ETH' for Bitcoin priced in Ethereum).`,
  schema: CryptoPricesInputSchema,
  func: async (input) => {
    const params = {
      ticker: input.ticker,
      interval: input.interval,
      interval_multiplier: input.interval_multiplier,
      start_date: input.start_date,
      end_date: input.end_date,
    };
    const endDate = new Date(input.end_date + 'T00:00:00');
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    try {
      const { data, url } = await api.get('/crypto/prices/', params, { cacheable: endDate < today });
      return formatToolResult(data.prices || [], [url]);
    } catch {
      const startDate = new Date(input.start_date + 'T00:00:00');
      const days = Math.max(1, Math.round((endDate.getTime() - startDate.getTime()) / 86_400_000) + 1);
      const closes = await fetchBinanceDailyCloses(input.ticker, days);
      if (closes.length === 0) {
        return formatToolResult({ error: `No crypto price history available for ${input.ticker}` });
      }
      return formatToolResult(
        closes.map((close, index) => ({ close, index })),
        [`https://api.binance.com/api/v3/klines?symbol=${toBinanceSymbol(input.ticker)}&interval=1d&limit=${Math.min(days, 1000)}`],
      );
    }
  },
});

export const getCryptoTickers = new DynamicStructuredTool({
  name: 'get_available_crypto_tickers',
  description: `Retrieves the list of available cryptocurrency tickers that can be used with the crypto price tools.`,
  schema: z.object({}),
  func: async () => {
    const { data, url } = await api.get('/crypto/prices/tickers/', {});
    return formatToolResult(data.tickers || [], [url]);
  },
});
